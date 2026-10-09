"""A score on every door onto the product screen.

The detail screen is reachable four ways and only one of them is the recommender. A drink
found by name, picked out of the scan's "is it this one?" list, or opened from another
drink's Similar profile arrived carrying no score, so the seal read "not scored for you yet"
about beers the server would have called a 91% match. Nothing was wrong with the score;
nothing had asked for it.

These fix what `GET /v1/product/{id}/score` will and will not say: the drinker's own profile
or nothing, a real cosine or nothing.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from bcd_schema import (
    Category,
    ExtractionMethod,
    Product,
    Provenance,
    SensorySource,
    SensoryVector,
    Sourced,
)

PROV = Provenance(source_id="test", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)


def _product(pid, name, *, sensory, style="IPA"):
    return Product(id=pid, brand_id=f"brand:{pid}", producer_id="prod:house",
                   category=Category.BEER, name=name,
                   style=Sourced[str](value=style, provenance=PROV),
                   sensory=sensory).model_dump(mode="json")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    root = tempfile.mkdtemp()
    before = dict(os.environ)
    os.environ.update({"BCD_STORE_BACKEND": "sqlite", "BCD_DATA_ROOT": root,
                       "BCD_LABEL_INDEX": "0"})
    os.environ.pop("BCD_DATABASE_URL", None)
    try:
        from bcd_api.app import _state, app
        with TestClient(app) as c:
            store = _state["store"]
            store.put_gold("prod:house", "producer", {"id": "prod:house", "name": "A House"})
            # A hoppy beer with a profile of its own — the kind a vector can actually be
            # near, so the cosine means something.
            store.put_gold("beer:hoppy", "product", _product(
                "beer:hoppy", "Loud IPA",
                sensory=SensoryVector(source=SensorySource.LLM_PROFILE, confidence=0.9,
                                      axes={"citrus": 0.8, "piney_resinous": 0.7,
                                            "bitterness": 0.6})))
            # And one the catalog knows nothing about beyond its style.
            store.put_gold("beer:blank", "product", _product(
                "beer:blank", "Anonymous Ale", sensory=None))
            yield c
    finally:
        os.environ.clear()
        os.environ.update(before)


@pytest.fixture()
def token(client):
    return client.post("/v1/auth/anonymous").json()["token"]


def _auth(t):
    return {"Authorization": f"Bearer {t}"}


def _answer_quiz(client, token):
    r = client.post("/v1/taste/quiz", headers=_auth(token),
                    json={"answers": [{"family": "ipa", "weight": 1.0}]})
    assert r.status_code == 200
    return r


def test_a_drinker_with_a_profile_gets_the_number_the_recommender_would_give(client, token):
    _answer_quiz(client, token)
    body = client.get("/v1/product/beer:hoppy/score", headers=_auth(token)).json()

    assert body["scored"] is True
    assert body["basis"] == "yours"
    assert 0.0 <= body["personal_score"] <= 1.0
    # An IPA drinker and a loud IPA: this is the case the seal exists for.
    assert body["personal_score"] > 0.5
    assert body["reason"]
    # The same fields a recommendation carries, so the screen draws one the same as the other.
    assert body["evidence"] == "known"
    assert body["cold_start"] is True


def test_the_score_is_the_same_one_the_recommender_computes(client, token):
    """Not a second opinion. If these two could disagree, the detail screen would be arguing
    with the list that sent the drinker to it."""
    from bcd_api.app import _state
    from bcd_api.taste import load_profile

    _answer_quiz(client, token)
    body = client.get("/v1/product/beer:hoppy/score", headers=_auth(token)).json()

    store = _state["store"]
    account = client.get("/v1/profile", headers=_auth(token)).json()["user_id"]
    product = Product.model_validate(store.get_gold("beer:hoppy"))
    expected, reason, cold = _state["resolver"].score(product, load_profile(store, account))

    assert body["personal_score"] == expected
    assert body["reason"] == reason
    assert body["cold_start"] == cold


def test_nobody_is_scored_against_the_seed_profile(client):
    """`_profile_for` falls back to a demo centroid so a fresh install's lists do not look
    dead. That is the right trade for a list of suggestions and the wrong one for a number
    stamped on a label: "matches your citrus preference" has to be about this reader."""
    fresh = client.post("/v1/auth/anonymous").json()["token"]
    body = client.get("/v1/product/beer:hoppy/score", headers=_auth(fresh)).json()

    assert body["scored"] is False
    assert body["basis"] == "no_profile"
    assert body.get("personal_score") is None
    assert body.get("reason") is None


def test_a_product_with_no_vector_is_not_scored_either(client, token):
    """With no vector `Resolver.score` falls through to the style-affinity prior, which is a
    flat 0.5 for any style the drinker has never rated inside -- a number that looks like a
    prediction and is the absence of one."""
    _answer_quiz(client, token)
    body = client.get("/v1/product/beer:blank/score", headers=_auth(token)).json()

    assert body["scored"] is False
    assert body["basis"] == "no_vector"
    assert body.get("personal_score") is None


def test_a_caller_who_does_not_say_who_they_are_gets_no_score(client):
    assert client.get("/v1/product/beer:hoppy/score").status_code == 401


def test_a_product_that_is_not_there_is_a_404_not_an_unscored_answer(client, token):
    """An absent row and an unscored one are different answers, and the screen would draw the
    second as a drink it simply has nothing to say about yet."""
    assert client.get("/v1/product/beer:nope/score",
                      headers=_auth(token)).status_code == 404
