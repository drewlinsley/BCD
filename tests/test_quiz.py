"""The first-run quiz — the cold-start fix.

A drinker who has rated nothing has no centroid. With no centroid `score` falls back to a
style prior and `/v1/recommend` answers from a seed profile the code is careful never to call
"for you", because it is somebody else's taste. The whole point of the app is to recommend,
and it cannot recommend from a blank slate.

These are the tests for what the quiz does to a profile, and for the questions themselves
not drifting away from the catalog's own flavour vectors.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
from bcd_api.taste import build_profile, quiz_from_events
from bcd_ingest.store import MedallionStore
from bcd_schema.quiz import QUIZ_DRINKS, QUIZ_ORDER


@pytest.fixture()
def store():
    s = MedallionStore(root=tempfile.mkdtemp())
    yield s
    s.close()


def _ev(family, weight, install="demo", tier="personalization", name="taste_quiz_answered"):
    return {"name": name, "install_id": install, "consent_tier": tier,
            "family": family, "weight": weight}


# ---- the questions themselves ----------------------------------------------------------

def test_every_question_has_a_flavour_vector_and_a_name_to_ask_by():
    assert len(QUIZ_ORDER) == 8
    for family in QUIZ_ORDER:
        prompt, category, confidence, axes = QUIZ_DRINKS[family]
        # Said the way a person says it. "IPA" is legitimately shouted; "AMERICAN DOUBLE
        # INDIA PALE ALE", the catalog's spelling, is not a thing anyone would answer.
        assert prompt and len(prompt) <= 12, f"{family}: {prompt!r} is a catalog spelling"
        assert prompt == prompt.strip()
        assert category in ("beer", "spirit")
        assert 0 < confidence < 0.45, f"{family}: a style prior must stay below the knowing bar"
        assert axes, f"{family} has no flavour at all"


def test_the_questions_still_match_the_catalogs_own_vectors():
    """`packages/schema/bcd_schema/quiz.py` is generated, because `bcd_enrich` is not on the
    API's import path. Generated means it can drift, so this regenerates and compares — a
    change to the style priors that moves IPA has to move the quiz with it."""
    enrich = os.path.join(os.path.dirname(__file__), "..", "services", "enrich")
    sys.path.insert(0, os.path.abspath(enrich))
    try:
        from bcd_enrich.style_prior import sensory_from_style
    except ImportError:
        pytest.skip("bcd_enrich not importable")
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts")))
    from codegen_quiz_families import QUIZ_FAMILIES

    for family, style, category, prompt in QUIZ_FAMILIES:
        live = sensory_from_style(style, category, style_hint=style)
        assert live is not None, f"{style!r} no longer has a style prior"
        stored_prompt, _cat, stored_conf, stored_axes = QUIZ_DRINKS[family]
        assert stored_prompt == prompt
        assert round(live.confidence, 3) == stored_conf
        assert {k: round(v, 4) for k, v in live.axes.items() if v} == stored_axes, (
            f"{family} has drifted — rerun scripts/codegen_quiz_families.py")


def test_the_questions_are_not_asking_the_same_thing_twice():
    """Porter and stout sit at 0.985 cosine; asking both is one question wearing two hats.
    No two questions may be that close, or the quiz spends a drinker's patience learning
    nothing."""
    import math
    vecs = {f: [QUIZ_DRINKS[f][3].get(a, 0.0) for a in sorted(
        {k for d in QUIZ_DRINKS.values() for k in d[3]})] for f in QUIZ_ORDER}

    def cos(a, b):
        d = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return d / (na * nb) if na and nb else 0.0

    worst = max((cos(vecs[a], vecs[b]), a, b)
                for i, a in enumerate(QUIZ_ORDER) for b in QUIZ_ORDER[i + 1:])
    assert worst[0] < 0.95, f"{worst[1]} and {worst[2]} are the same question ({worst[0]:.2f})"


# ---- reading the answers ---------------------------------------------------------------

def test_an_answer_is_read_as_a_signed_weight():
    assert quiz_from_events([_ev("ipa", 0.5)], "demo") == {"ipa": 0.5}


def test_no_opinion_says_nothing_rather_than_the_middle():
    """Skipping a question must not record a neutral — a drinker who has never had a stout
    is not a drinker who is indifferent to stout."""
    assert quiz_from_events([_ev("stout", 0.0)], "demo") == {}


def test_retaking_the_quiz_supersedes_rather_than_accumulates():
    answers = [_ev("ipa", 0.5), _ev("ipa", -0.5)]
    assert quiz_from_events(answers, "demo") == {"ipa": -0.5}


def test_a_family_the_server_does_not_ask_about_is_ignored():
    assert quiz_from_events([_ev("mead", 0.5)], "demo") == {}


def test_the_quiz_is_consent_gated_exactly_as_ratings_are():
    assert quiz_from_events([_ev("ipa", 0.5, tier="analytics")], "demo") == {}


def test_another_installs_quiz_is_not_yours():
    assert quiz_from_events([_ev("ipa", 0.5, install="someone-else")], "demo") == {}


# ---- what it does to the profile -------------------------------------------------------

def test_the_quiz_alone_builds_a_centroid(store):
    """The whole point. Nothing rated, nothing scanned — and a real place in the flavour
    space to recommend from."""
    blank = build_profile("demo", {}, store)
    assert blank.sensory_ideal is None

    seeded = build_profile("demo", {}, store, quiz={"ipa": 0.5})
    assert seeded.sensory_ideal is not None
    assert seeded.sensory_ideal.axes.get("bitterness", 0) > 0


def test_liking_an_ipa_and_not_a_stout_lands_between_them_not_on_the_stout(store):
    hoppy = build_profile("demo", {}, store, quiz={"ipa": 0.5, "stout": -0.5})
    assert hoppy.sensory_ideal is not None
    axes = hoppy.sensory_ideal.axes
    assert axes.get("bitterness", 0) > axes.get("roasted_coffee_choc", 0), axes


def test_disliking_everything_builds_no_centroid(store):
    """`_centroid`'s rule, which the quiz must not sneak around: "not that" does not locate
    a taste, and a profile invented from dislikes alone would rank the catalog confidently
    wrong."""
    sour = build_profile("demo", {}, store, quiz={"ipa": -0.5, "stout": -0.5})
    assert sour.sensory_ideal is None


def test_the_quiz_names_no_product_so_it_bars_no_drink(store):
    """A quiz answer is a family, never a product. If it were a product the drinker would be
    told they had rated something they never drank, and `rated_products` would keep that
    drink out of the recommendations the quiz exists to make possible."""
    from bcd_api.taste import rated_products
    events = [_ev("ipa", 0.5)]
    assert rated_products(store, events, "demo") == set()


def test_a_real_rating_outweighs_the_quiz_as_they_accumulate(store):
    """Nothing has to expire the quiz. A stated preference is entered at half weight, so
    verdicts on real drinks take the centroid over as they arrive."""
    from bcd_schema import (
        Category,
        Product,
        SensorySource,
        SensoryVector,
    )
    stout = Product(id="p:stout", brand_id="b", producer_id="pr", category=Category.BEER,
                    name="Very Stout", sensory=SensoryVector(
                        source=SensorySource.LLM_PROFILE, confidence=0.9,
                        axes={"roasted_coffee_choc": 1.0, "body_fullness": 1.0}))
    store.put_gold(stout.id, "product", stout.model_dump(mode="json"))

    quiz_only = build_profile("demo", {}, store, quiz={"ipa": 0.5})
    both = build_profile("demo", {"p:stout": 1.0}, store, quiz={"ipa": 0.5})
    assert quiz_only.sensory_ideal is not None and both.sensory_ideal is not None
    assert (both.sensory_ideal.axes.get("roasted_coffee_choc", 0)
            > quiz_only.sensory_ideal.axes.get("roasted_coffee_choc", 0))


# ---- the routes ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    root = tempfile.mkdtemp()
    before = dict(os.environ)
    os.environ.update({"BCD_STORE_BACKEND": "sqlite", "BCD_DATA_ROOT": root,
                       "BCD_LABEL_INDEX": "0"})
    os.environ.pop("BCD_DATABASE_URL", None)
    try:
        from bcd_api.app import app
        with TestClient(app) as c:
            yield c
    finally:
        os.environ.clear()
        os.environ.update(before)


@pytest.fixture()
def token(client):
    return client.post("/v1/auth/anonymous").json()["token"]


def _auth(t):
    return {"Authorization": f"Bearer {t}"}


def test_the_questions_are_served_so_they_can_change_without_a_release(client):
    r = client.get("/v1/taste/quiz")
    assert r.status_code == 200
    drinks = r.json()["drinks"]
    assert [d["family"] for d in drinks] == list(QUIZ_ORDER)
    # The app is told what to ask and what to call it, and never sees a flavour vector.
    assert all(set(d) == {"family", "prompt", "category"} for d in drinks)


def test_answering_the_quiz_is_the_moment_a_profile_starts(client, token):
    assert client.get("/v1/profile", headers=_auth(token)).json()["version"] == 0

    r = client.post("/v1/taste/quiz", headers=_auth(token),
                    json={"answers": [{"family": "ipa", "weight": 1.0},
                                      {"family": "stout", "weight": -1.0}]})
    assert r.status_code == 200
    profile = r.json()
    assert profile["version"] > 0
    assert profile["sensory_ideal"] is not None
    # And it is theirs from then on, not just in the response.
    assert client.get("/v1/profile", headers=_auth(token)).json()["version"] > 0


def test_a_caller_who_does_not_say_who_they_are_cannot_answer(client):
    r = client.post("/v1/taste/quiz", json={"answers": [{"family": "ipa", "weight": 1.0}]})
    assert r.status_code == 401


def test_an_empty_quiz_is_refused(client, token):
    r = client.post("/v1/taste/quiz", headers=_auth(token), json={"answers": []})
    assert r.status_code == 422


def test_a_weight_outside_the_scale_is_refused(client, token):
    r = client.post("/v1/taste/quiz", headers=_auth(token),
                    json={"answers": [{"family": "ipa", "weight": 7.0}]})
    assert r.status_code == 422


def test_a_stale_client_asking_about_a_dropped_drink_is_not_an_error(client, token):
    """The question set is served by this same route, so a client holding an older copy can
    answer about a family that is no longer asked. Drop it, keep the rest."""
    r = client.post("/v1/taste/quiz", headers=_auth(token),
                    json={"answers": [{"family": "mead", "weight": 1.0},
                                      {"family": "ipa", "weight": 1.0}]})
    assert r.status_code == 200
    assert r.json()["sensory_ideal"] is not None


# ---- "sometimes" ------------------------------------------------------------------------

# A third rung, added 2026-10-08. Not the neutral the design leaves out: a neutral is the
# absence of an opinion, and that is still said by not answering. Nobody answers "sometimes"
# meaning never — it is a yes with less conviction, at half a yes, which is the same relation
# `Reaction.weight` gives "pinkie out" against "chugged it".


def test_sometimes_pulls_the_centroid_the_same_way_a_yes_does_only_less(store):
    sometimes = build_profile("demo", {}, store, quiz={"ipa": 0.25})
    yes = build_profile("demo", {}, store, quiz={"ipa": 0.5})

    assert sometimes.sensory_ideal is not None and yes.sensory_ideal is not None
    # Same direction...
    assert sometimes.sensory_ideal.axes.get("bitterness", 0) > 0
    # ...and the quiz's own weight is what separates them, so a stated habit never speaks
    # louder than a stated preference.
    assert (sometimes.sensory_ideal.axes.get("bitterness", 0)
            <= yes.sensory_ideal.axes.get("bitterness", 0))


def test_a_sometimes_is_kept_where_a_skipped_question_is_dropped():
    """The two were the same thing while the quiz had two rungs. They are not now, and the
    difference is the whole reason the rung exists."""
    assert quiz_from_events([_ev("lager", 0.25)], "demo") == {"lager": 0.25}
    assert quiz_from_events([_ev("lager", 0.0)], "demo") == {}


def test_sometimes_everything_still_builds_a_centroid(store):
    """A cautious drinker who answers "sometimes" to all eight has still said plenty. The
    only answer set that builds nothing is one with no positive in it at all."""
    every = dict.fromkeys(QUIZ_ORDER, 0.25)
    assert build_profile("demo", {}, store, quiz=every).sensory_ideal is not None


# ---- the shelves hear it ---------------------------------------------------------------

# A quiz answer names a FAMILY — that is the whole of its design — and Discover is the one
# screen organised by family. Until this, answering "Stout: yes" and then finding the Stout
# shelf dark, under a header calling the list a starting point, was the quiz being heard by
# the ranker and by nothing else (2026-10-08).

def test_a_quiz_yes_speaks_for_that_shelf(store):
    from bcd_api.app import _families_spoken_for
    from bcd_schema.family import Family

    mine = _families_spoken_for(store, judged=(),
                                quiz={"stout": 1.0, "bourbon": 1.0, "ipa": 1.0})
    assert mine == {Family.STOUT, Family.BOURBON, Family.IPA}


def test_a_sometimes_speaks_for_its_shelf_too(store):
    """Someone who sometimes drinks lager does drink lager, and a shelf they drink from is
    one worth ranking for them. The bar is a positive answer, not a loud one."""
    from bcd_api.app import _families_spoken_for
    from bcd_schema.family import Family

    assert _families_spoken_for(store, judged=(), quiz={"lager": 0.25}) == {Family.LAGER}


def test_a_quiz_no_does_not(store):
    """Saying you do not drink gin moves the centroid away, which is worth having. It is not
    a reason to rank the gin shelf FOR you: asking which gin is least unlike the taste of
    someone who just said they do not drink gin is a real cosine and not a recommendation."""
    from bcd_api.app import _families_spoken_for

    assert _families_spoken_for(store, judged=(), quiz={"gin": -1.0, "vodka": -1.0}) == set()


def test_a_family_this_build_does_not_know_is_skipped_not_fatal(store):
    """The questions are served, so the server can ask about a family an older build's
    `Family` enum has never heard of. One unknown question must not take the shelf list
    with it."""
    from bcd_api.app import _families_spoken_for
    from bcd_schema.family import Family

    mine = _families_spoken_for(store, judged=(), quiz={"stout": 1.0, "perry": 1.0})
    assert mine == {Family.STOUT}


def test_ratings_and_quiz_answers_both_count(store):
    """The quiz is an accelerant, not a replacement: a shelf is theirs if they have said
    anything about it, by either route."""
    from bcd_api.app import _families_spoken_for
    from bcd_schema import Category, ExtractionMethod, Product, Provenance, Sourced
    from bcd_schema.family import Family

    prov = Provenance(source_id="t", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)
    store.put_gold("beer:rated", "product", Product(
        id="beer:rated", brand_id="b", producer_id="p", category=Category.BEER,
        name="A Porter", style=Sourced[str](value="Porter", provenance=prov),
    ).model_dump(mode="json"))

    mine = _families_spoken_for(store, judged=["beer:rated"], quiz={"stout": 1.0})
    assert mine == {Family.STOUT, Family.PORTER}
