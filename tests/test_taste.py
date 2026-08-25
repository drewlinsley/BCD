"""Taste profile — behavior in, sensory centroid out, and the ranking it drives."""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.resolver import Resolver
from bcd_api.taste import (
    _rating_weight,
    build_profile,
    load_profile,
    rebuild_profile,
    save_profile,
    signals_from_events,
)
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


# ---- persistence + the loop ---------------------------------------------------------

def test_profile_round_trips_through_the_store(store):
    profile = build_profile("demo", {"p:ipa": 1.0}, store)
    save_profile(store, profile)
    loaded = load_profile(store, "demo")
    assert loaded.user_id == "demo"
    assert loaded.sensory_ideal.axes["citrus"] == pytest.approx(0.8)


def test_profile_does_not_pollute_the_product_catalog(store):
    save_profile(store, build_profile("demo", {"p:ipa": 1.0}, store))
    assert {p["id"] for p in store.iter_gold("product")} == {"p:ipa", "p:stout", "p:scotch"}


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
