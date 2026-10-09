"""Taste profile — behavior in, sensory centroid out, and the ranking it drives."""

from __future__ import annotations

import importlib
import tempfile

import pytest
from bcd_api.resolver import Resolver
from bcd_api.taste import (
    _rating_weight,
    build_profile,
    load_profile,
    rated_products,
    rebuild_profile,
    save_profile,
    scans_from_events,
    signals_from_events,
)
from bcd_ingest.merge import put_redirect
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    Category,
    ExtractionMethod,
    Product,
    ProductSpec,
    Provenance,
    SensorySource,
    SensoryVector,
    Sourced,
    TasteProfile,
)

PROV = Provenance(source_id="test", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)


def _product(pid, name, cat, style, abv, axes):
    return Product(
        id=pid, brand_id="brand:x", producer_id="prod:x", category=cat, name=name,
        style=Sourced[str](value=style, provenance=PROV),
        spec=ProductSpec(abv_pct=Sourced[float](value=abv, provenance=PROV)),
        sensory=SensoryVector(source=SensorySource.STYLE_PRIOR, confidence=0.25, axes=axes),
    )


@pytest.fixture()
def store():
    s = MedallionStore(root=tempfile.mkdtemp())
    for p in (
        _product("p:ipa", "Lagunitas IPA", Category.BEER, "IPA", 6.5,
                 {"citrus": 0.8, "tropical": 0.7, "piney_resinous": 0.6, "bitterness": 0.8}),
        _product("p:stout", "Guinness", Category.BEER, "Stout", 4.2,
                 {"roasted_coffee_choc": 0.9, "body_fullness": 0.6, "bitterness": 0.4}),
        _product("p:scotch", "Lagavulin 16", Category.SPIRIT, "Islay Single Malt", 43.0,
                 {"smoky_peat": 0.95, "alcohol_warmth": 0.7}),
        # Same style as p:ipa and louder on the axis they share. This is what a dislike
        # inside someone's own favourite style actually looks like.
        _product("p:ddh", "Other Half DDH", Category.BEER, "IPA", 7.0,
                 {"citrus": 0.65, "tropical": 0.75, "piney_resinous": 0.45,
                  "bitterness": 0.4}),
    ):
        s.put_gold(p.id, "product", p.model_dump(mode="json"))
    yield s
    s.close()


def _ev(name="rating_submitted", install="demo", tier="personalization", **props):
    return {"name": name, "install_id": install, "consent_tier": tier, **props}


# ---- signal extraction --------------------------------------------------------------

@pytest.mark.parametrize("rating, weight", [(5.0, 1.0), (4.0, 0.5), (3.0, 0.0), (1.0, -1.0)])
def test_rating_maps_to_signed_weight(rating, weight):
    assert _rating_weight(rating) == weight


def test_neutral_rating_carries_no_signal():
    # 3/5 is "fine" — it must not drag the centroid toward the product.
    assert signals_from_events([_ev(product_id="p:ipa", rating=3.0)], "demo") == {}


def test_consent_tier_is_enforced():
    # Collected under analytics-only consent => may not shape a taste profile.
    events = [_ev(product_id="p:ipa", rating=5.0, tier="analytics")]
    assert signals_from_events(events, "demo") == {}
    ok = [_ev(product_id="p:ipa", rating=5.0, tier="data_sharing")]
    assert signals_from_events(ok, "demo") == {"p:ipa": 1.0}


def test_other_installs_are_ignored():
    events = [_ev(product_id="p:ipa", rating=5.0, install="someone-else")]
    assert signals_from_events(events, "demo") == {}


def test_rerating_supersedes_rather_than_accumulates():
    events = [
        _ev(product_id="p:ipa", rating=5.0),
        _ev(product_id="p:ipa", rating=1.0),  # changed their mind
    ]
    assert signals_from_events(events, "demo") == {"p:ipa": -1.0}


def test_list_add_is_weaker_positive_signal_and_yields_to_a_rating():
    lists = signals_from_events([_ev(name="list_add", product_id="p:ipa",
                                     list_kind="cellar")], "demo")
    assert 0 < lists["p:ipa"] < 1.0
    both = signals_from_events([
        _ev(name="list_add", product_id="p:ipa", list_kind="cellar"),
        _ev(product_id="p:ipa", rating=1.0),
    ], "demo")
    assert both["p:ipa"] == -1.0  # the explicit verdict wins


# ---- centroid -----------------------------------------------------------------------

def test_centroid_leans_toward_what_they_liked(store):
    profile = build_profile("demo", {"p:ipa": 1.0}, store)
    axes = profile.sensory_ideal.axes
    assert axes["citrus"] == pytest.approx(0.8)
    assert "roasted_coffee_choc" not in axes


def test_dislike_pushes_a_shared_axis_down(store):
    """Rocchio: bitterness is on both, so disliking the stout discounts — not erases — it."""
    liked_only = build_profile("demo", {"p:ipa": 1.0}, store)
    with_dislike = build_profile("demo", {"p:ipa": 1.0, "p:stout": -1.0}, store)
    assert liked_only.sensory_ideal.axes["bitterness"] == pytest.approx(0.8)
    # 0.8 - GAMMA(0.4) * 0.4
    assert with_dislike.sensory_ideal.axes["bitterness"] == pytest.approx(0.64, abs=1e-3)
    assert "roasted_coffee_choc" not in with_dislike.sensory_ideal.axes


def test_dislikes_alone_produce_no_ideal(store):
    """"Not that" doesn't locate a taste — better no centroid than a confident wrong one."""
    profile = build_profile("demo", {"p:stout": -1.0}, store)
    assert profile.sensory_ideal is None
    assert profile.style_affinities["Stout"] == -1.0  # the signal isn't lost


def test_confidence_grows_with_evidence(store):
    one = build_profile("demo", {"p:ipa": 1.0}, store)
    three = build_profile("demo", {"p:ipa": 1.0, "p:stout": -1.0, "p:scotch": 1.0}, store)
    assert three.sensory_ideal.confidence > one.sensory_ideal.confidence
    assert three.sensory_ideal.confidence <= 0.9


def test_unknown_product_is_skipped_not_fatal(store):
    profile = build_profile("demo", {"p:ipa": 1.0, "p:not-in-catalog": 1.0}, store)
    assert profile.sensory_ideal.axes["citrus"] == pytest.approx(0.8)


# ---- derived preferences ------------------------------------------------------------

def test_style_affinities_carry_sign(store):
    profile = build_profile("demo", {"p:ipa": 1.0, "p:stout": -0.5}, store)
    assert profile.style_affinities["IPA"] == 1.0
    assert profile.style_affinities["Stout"] == -0.5


def test_abv_band_spans_what_they_liked(store):
    profile = build_profile("demo", {"p:ipa": 1.0, "p:scotch": 1.0}, store)
    assert profile.abv_band_min <= 6.5
    assert profile.abv_band_max >= 43.0


def test_abv_band_needs_more_than_one_point(store):
    assert build_profile("demo", {"p:ipa": 1.0}, store).abv_band_min is None


def test_novelty_reflects_style_spread(store):
    explorer = build_profile("demo", {"p:ipa": 1.0, "p:scotch": 1.0}, store)
    assert explorer.novelty_appetite == 1.0  # two likes, two distinct styles


def test_memo_names_the_driving_axes(store):
    memo = build_profile("demo", {"p:ipa": 1.0, "p:stout": -1.0}, store).memo
    assert "citrus" in memo
    assert "roasted coffee choc" in memo  # prettified, and named as the thing they avoid


def test_a_note_they_love_is_never_named_as_the_one_they_avoid(store):
    """Two bottles of the same style, one liked and one not, share their loud axes.

    The real case: Heady Topper (tropical 0.70) liked, a DDH IPA (tropical 0.75) disliked.
    Naming the disliked drink's loudest axis put "away from tropical" on a card that listed
    tropical among what the drinker goes for.
    """
    profile = build_profile("demo", {"p:ipa": 1.0, "p:ddh": -1.0}, store)
    assert "tropical" in profile.sensory_ideal.axes      # still in the centroid
    assert "away from" not in profile.memo               # and so never disowned


def test_a_dislike_that_isolates_nothing_says_nothing(store):
    """Near-identical bottles leave no axis clearing both bars, so the clause is dropped
    rather than filled with the widest gap in a bad field."""
    memo = build_profile("demo", {"p:ipa": 1.0, "p:ddh": -1.0}, store).memo
    assert memo.startswith("You lean ")
    assert memo.endswith(".")


def test_an_axis_they_are_quiet_on_can_still_be_named(store):
    """The bar is two-sided, not a mute button: a genuinely different dislike still earns
    the clause, which is what keeps the rule from just deleting the feature."""
    memo = build_profile("demo", {"p:ipa": 1.0, "p:scotch": -1.0}, store).memo
    assert "away from smoky peat" in memo


# ---- persistence + the loop ---------------------------------------------------------

def test_profile_round_trips_through_the_store(store):
    profile = build_profile("demo", {"p:ipa": 1.0}, store)
    save_profile(store, profile)
    loaded = load_profile(store, "demo")
    assert loaded.user_id == "demo"
    assert loaded.sensory_ideal.axes["citrus"] == pytest.approx(0.8)


def test_profile_does_not_pollute_the_product_catalog(store):
    save_profile(store, build_profile("demo", {"p:ipa": 1.0}, store))
    assert {p["id"] for p in store.iter_gold("product")} == {"p:ipa", "p:stout", "p:scotch", "p:ddh"}


def test_rebuild_from_events_persists_and_versions(store):
    events = [_ev(product_id="p:ipa", rating=5.0)]
    first = rebuild_profile(store, events, "demo")
    assert first.version == 1
    second = rebuild_profile(store, events, "demo")
    assert second.version == 2  # same signals, new revision
    assert load_profile(store, "demo").version == 2


def test_withdrawn_consent_disappears_on_rebuild(store):
    """The profile is a pure function of consented events, so dropping consent drops the
    influence — no residue left behind in a stored vector."""
    rebuild_profile(store, [_ev(product_id="p:ipa", rating=5.0)], "demo")
    after = rebuild_profile(store, [_ev(product_id="p:ipa", rating=5.0, tier="analytics")],
                            "demo")
    assert after.sensory_ideal is None


def test_learned_profile_reranks_the_catalog(store):
    """The payoff: rating an IPA up makes the IPA outrank the stout for that user."""
    resolver = Resolver(store)
    profile = rebuild_profile(store, [_ev(product_id="p:ipa", rating=5.0)], "demo")
    ipa = Product.model_validate(store.get_gold("p:ipa"))
    stout = Product.model_validate(store.get_gold("p:stout"))
    ipa_score, _, _ = resolver.score(ipa, profile)
    stout_score, _, _ = resolver.score(stout, profile)
    assert ipa_score > stout_score

    # ...and the reverse rating flips the order, i.e. we learned rather than hardcoded.
    flipped = rebuild_profile(store, [_ev(product_id="p:stout", rating=5.0)], "demo")
    assert resolver.score(stout, flipped)[0] > resolver.score(ipa, flipped)[0]


def test_reason_does_not_claim_a_match_it_did_not_find(store):
    """A weak score must not be narrated as a preference match — the overlay's 'why' has
    to agree with its own number."""
    resolver = Resolver(store)
    profile = rebuild_profile(store, [_ev(product_id="p:scotch", rating=5.0)], "demo")
    scotch = Product.model_validate(store.get_gold("p:scotch"))
    ipa = Product.model_validate(store.get_gold("p:ipa"))

    strong_score, strong_reason, _ = resolver.score(scotch, profile)
    weak_score, weak_reason, _ = resolver.score(ipa, profile)
    assert strong_score > weak_score
    assert "matches your smoky peat" in strong_reason
    assert "matches your" not in weak_reason
    assert "outside your usual" in weak_reason


def test_seed_profile_answers_for_any_new_install(monkeypatch):
    """A real install id must still get the seed profile until it has rated something.

    The fallback used to be keyed on the user id, so it only ever matched the literal
    "demo". That was invisible while the client sent no id at all; the moment it started
    sending its own, every fresh install matched no seed, got `None`, and scored a flat
    0.5 on the entire catalog.
    """
    # `bcd_api.__init__` re-exports the FastAPI instance under the name `app`, which
    # shadows the submodule — reach the module itself rather than the instance.
    api = importlib.import_module("bcd_api.app")

    monkeypatch.setitem(api._state, "store", _EmptyStore())
    monkeypatch.setitem(api._state, "profiles", {"demo": api._demo_profile()})

    seeded = api._profile_for("9f3c-not-demo")
    assert seeded is not None
    assert seeded.sensory_ideal is not None


def test_learned_profile_beats_the_seed(monkeypatch):
    """Once an install has a real centroid, the seed must get out of the way — otherwise
    rating something would never change what the next scan shows."""
    # `bcd_api.__init__` re-exports the FastAPI instance under the name `app`, which
    # shadows the submodule — reach the module itself rather than the instance.
    api = importlib.import_module("bcd_api.app")

    learned = TasteProfile(
        user_id="mine",
        version=3,
        sensory_ideal=SensoryVector(source=SensorySource.RECONCILED, confidence=0.9,
                                    axes={"smoky_peat": 0.9}),
    )
    monkeypatch.setitem(api._state, "profiles", {"demo": api._demo_profile()})
    monkeypatch.setattr(api, "load_profile", lambda store, uid: learned if uid == "mine" else None)
    monkeypatch.setitem(api._state, "store", _EmptyStore())

    assert api._profile_for("mine") is learned
    assert api._profile_for("someone-else").user_id == "demo"


class _EmptyStore:
    """Nothing stored, so `load_profile` finds no learned profile."""

    def iter_gold(self, *_a, **_k):
        return iter(())

    def get_gold(self, *_a, **_k):
        return None


# --- which drinks have been judged at all ---------------------------------------------


def test_rated_products_is_this_persons_verdicts_only():
    """What the recommender takes out of the list. Someone else's rating, an unconsented
    one, and a list-add are all not this person's verdict on this drink."""
    store = MedallionStore(root=tempfile.mkdtemp())
    events = [
        _ev(product_id="a", rating=5.0),
        _ev(product_id="b", rating=1.0),
        _ev(install="someone-else", product_id="c", rating=5.0),
        _ev(tier="analytics", product_id="d", rating=5.0),
        _ev(name="list_add", product_id="e", list_kind="wishlist"),
        _ev(product_id=None, rating=4.0),
    ]
    assert rated_products(store, events, "demo") == {"a", "b"}
    store.close()


def test_a_verdict_survives_the_row_being_merged_away():
    """Same reasoning as `_canonical` in the profile rebuild: the catalog being tidied must
    not hand someone back a drink they have already told us they hate."""
    store = MedallionStore(root=tempfile.mkdtemp())
    put_redirect(store, "old", "new")
    assert rated_products(store, [_ev(product_id="old", rating=1.0)], "demo") == {"new"}
    store.close()


# ---- taking a verdict back ----------------------------------------------------------

def test_a_withdrawn_rating_stops_counting():
    """A mis-tap on the five faces is one tap, and until this it was permanent."""
    events = [
        _ev(product_id="p:ipa", rating=5.0),
        _ev(name="rating_withdrawn", product_id="p:ipa"),
    ]
    assert signals_from_events(events, "demo") == {}


def test_withdrawing_leaves_every_other_verdict_alone():
    events = [
        _ev(product_id="p:ipa", rating=5.0),
        _ev(product_id="p:stout", rating=1.0),
        _ev(name="rating_withdrawn", product_id="p:ipa"),
    ]
    assert signals_from_events(events, "demo") == {"p:stout": -1.0}


def test_rating_again_after_withdrawing_stands():
    """Read in order, like a re-rate. Withdrawing is not a tombstone on the product."""
    events = [
        _ev(product_id="p:ipa", rating=5.0),
        _ev(name="rating_withdrawn", product_id="p:ipa"),
        _ev(product_id="p:ipa", rating=4.0),
    ]
    assert signals_from_events(events, "demo") == {"p:ipa": 0.5}


def test_withdrawing_what_was_never_rated_is_a_no_op():
    """The client's reaction log is a disposable cache, so it can ask to withdraw something
    the server never had. That is not an error and must not invent a signal."""
    assert signals_from_events([_ev(name="rating_withdrawn", product_id="p:ipa")], "demo") == {}


def test_withdrawing_a_rating_does_not_withdraw_a_list_add():
    """Separate signals. Saving a bottle is interest and was never the verdict taken back."""
    events = [
        _ev(product_id="p:ipa", rating=5.0),
        _ev(name="list_add", product_id="p:ipa", list_kind="cellar"),
        _ev(name="rating_withdrawn", product_id="p:ipa"),
    ]
    assert signals_from_events(events, "demo") == {"p:ipa": 0.6}


def test_a_withdrawn_rating_is_drinkable_again(store):
    """`rated_products` is what keeps a drink you have judged out of "For you". A verdict
    taken back has to put it back in the pool, or one mis-tap bars a drink for good."""
    rated = [_ev(product_id="p:ipa", rating=5.0)]
    assert rated_products(store, rated, "demo") == {"p:ipa"}
    taken_back = rated + [_ev(name="rating_withdrawn", product_id="p:ipa")]
    assert rated_products(store, taken_back, "demo") == set()


def test_withdrawing_is_consent_gated_like_the_rating(store):
    """A withdrawal collected outside personalization consent may no more change the profile
    than the rating could — and under that consent the rating never counted either."""
    events = [
        _ev(product_id="p:ipa", rating=5.0),
        _ev(name="rating_withdrawn", product_id="p:ipa", tier="analytics"),
    ]
    assert signals_from_events(events, "demo") == {"p:ipa": 1.0}


def test_one_installs_withdrawal_cannot_clear_anothers_verdict():
    events = [
        _ev(product_id="p:ipa", rating=5.0),
        _ev(name="rating_withdrawn", product_id="p:ipa", install="someone-else"),
    ]
    assert signals_from_events(events, "demo") == {"p:ipa": 1.0}



# ---- scans ------------------------------------------------------------------------------

# What someone points a camera at is what they are considering: standing in front of it,
# reading the label, deciding. Weaker than a verdict and weaker than a stated preference, and
# the only one of the three that costs the drinker nothing — so it is what reaches someone who
# skipped the quiz and has rated nothing (2026-10-08).


def _scan(pid, install="demo", tier="personalization"):
    return _ev(name="scan_resolved", install=install, tier=tier, product_id=pid)


def test_a_scanned_product_is_a_small_positive():
    weights = scans_from_events([_scan("p:ipa")], "demo")
    assert weights == {"p:ipa": pytest.approx(0.15)}
    # Positive, always. You cannot read a dislike off a glance.
    assert all(w > 0 for w in weights.values())


def test_a_product_seen_twenty_times_counts_once():
    """The HUD redraws every 350ms and a can sits in the viewfinder for seconds. Keyed by
    product, so a long look is not twenty opinions."""
    assert scans_from_events([_scan("p:ipa")] * 20, "demo") == {"p:ipa": pytest.approx(0.15)}


def test_a_scan_is_worth_far_less_than_a_rating_or_a_quiz_answer():
    """Scanning is how you ask what something IS, so plenty of scans are of drinks that go
    straight back on the shelf. The ordering is the whole safety of the signal."""
    scan = scans_from_events([_scan("p:ipa")], "demo")["p:ipa"]
    chugged = signals_from_events([_ev(product_id="p:ipa", rating=5.0)], "demo")["p:ipa"]
    assert scan < 0.5, "a scan must sit below a quiz answer's weight"
    assert scan < abs(chugged)


def test_scans_are_consent_gated_exactly_as_everything_else_is():
    assert scans_from_events([_scan("p:ipa", tier="analytics")], "demo") == {}


def test_another_installs_scans_are_not_yours():
    assert scans_from_events([_scan("p:ipa", install="someone-else")], "demo") == {}


def test_a_verdict_on_a_scanned_drink_replaces_the_glance_at_it(store):
    """You scanned it, then you drank it and said what you thought. The rating is the better
    evidence and must not be averaged with the look that preceded it."""
    looked = rebuild_profile(store, [_scan("p:ipa")], "demo")
    assert looked.sensory_ideal is not None
    assert looked.sensory_ideal.axes.get("bitterness", 0) > 0

    both = rebuild_profile(store, [_scan("p:ipa"),
                                   _ev(product_id="p:ipa", rating=1.0)], "demo-2")
    # A spat-out IPA is a dislike, not a dislike softened by the look that came before it.
    assert both.sensory_ideal is None


def test_scanning_never_bars_a_drink_from_being_recommended(store):
    """The reason you scanned it is that you might buy it. `rated_products` keeps judged
    drinks out of "For you"; a scanned one must stay in, or the signal would hide the very
    thing it is evidence of."""
    assert rated_products(store, [_scan("p:ipa")], "demo") == set()


def test_scanning_alone_builds_a_profile_for_someone_who_said_nothing(store):
    """The point of the whole signal. No quiz, no ratings — just a camera pointed at two
    IPAs — and there is somewhere to recommend from."""
    profile = rebuild_profile(store, [_scan("p:ipa"), _scan("p:ddh")], "demo")
    assert profile.sensory_ideal is not None
    axes = profile.sensory_ideal.axes
    assert axes.get("citrus", 0) > axes.get("smoky_peat", 0), axes
