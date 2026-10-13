"""The seed profile does two jobs, and only one of them is honest.

As a probe vector into the index it is load-bearing and invisible: `rank_catalog`'s
no-profile branch reads every gold row -- 534,104 of them on the live catalog, measured at
about 69 minutes before scoring begins -- and the seed's vector is what routes a request for
someone who has said nothing into the ANN path instead.

As a claim about the reader it is a lie, and it was told to exactly one person: whoever
reaches it. Which is not the cold-start drinker it was written for. Answering the quiz grants
personalization and builds a centroid in the same tap, so a consenter never meets the seed;
a skipper grants nothing, and the client drops a personalization-tier event before it is ever
queued, so no amount of scanning or searching will build them a profile. The seed's whole
remaining audience is the person who declined to be learned about, and what it did for them
was stamp "matches your tropical preference" on a stranger's taste.

These fix the split: the order may still come from the seed, the numbers may not.
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


def _product(pid, name, *, axes, style="IPA"):
    return Product(id=pid, brand_id=f"brand:{pid}", producer_id="prod:house",
                   category=Category.BEER, name=name,
                   style=Sourced[str](value=style, provenance=PROV),
                   sensory=SensoryVector(source=SensorySource.LLM_PROFILE, confidence=0.9,
                                         axes=axes)).model_dump(mode="json")


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
            store.put_gold("beer:hoppy", "product", _product(
                "beer:hoppy", "Loud IPA",
                axes={"citrus": 0.8, "piney_resinous": 0.7, "bitterness": 0.6}))
            store.put_gold("beer:soft", "product", _product(
                "beer:soft", "Quiet Wheat", style="Hefeweizen",
                axes={"banana_clove": 0.7, "body_fullness": 0.4}))
            store.put_gold("spirit:peat", "product", _product(
                "spirit:peat", "Peat Monster", style="Islay Scotch",
                axes={"smoky_peat": 0.9, "alcohol_warmth": 0.8}))
            yield c
    finally:
        os.environ.clear()
        os.environ.update(before)


def _auth(t):
    return {"Authorization": f"Bearer {t}"}


def _fresh(client):
    """Someone who has just installed the app and said nothing at all."""
    return client.post("/v1/auth/anonymous").json()["token"]


def _answered(client):
    t = _fresh(client)
    r = client.post("/v1/taste/quiz", headers=_auth(t),
                    json={"answers": [{"family": "ipa", "weight": 1.0}]})
    assert r.status_code == 200
    return t


# ---- the list ----------------------------------------------------------------------------


def test_a_stranger_still_gets_a_list(client):
    """The seed's real job. Removing it outright would drop this caller into a full-catalog
    scan, so the order still comes from it and the list is not empty."""
    body = client.post("/v1/recommend", headers=_auth(_fresh(client))).json()
    assert body["results"], "a cold start must still have somewhere to begin"


def test_a_stranger_gets_no_numbers_on_it(client):
    """What the split is for. A score and "matches your tropical preference" are claims about
    the reader, and this reader has not said anything to be right or wrong about."""
    body = client.post("/v1/recommend", headers=_auth(_fresh(client))).json()
    assert body["basis"] == "no_profile"
    for row in body["results"]:
        assert row["score"] is None, row
        assert row["reason"] is None, row
        assert row["cold_start"] is None, row


def test_the_rows_still_say_what_stands_behind_them(client):
    """Evidence is a fact about the catalog, not a claim about the drinker: "we know this one"
    is true whoever is reading. It is what the starting list has left to order by."""
    body = client.post("/v1/recommend", headers=_auth(_fresh(client))).json()
    assert all(row["evidence"] for row in body["results"])
    assert all(row["name"] for row in body["results"])


def test_someone_who_answered_the_quiz_gets_the_numbers(client):
    """The other side. One quiz answer is a real centroid, so every number on the list is
    about them and the seal, the shelf and the row all say so."""
    body = client.post("/v1/recommend", headers=_auth(_answered(client))).json()
    assert body["basis"] == "yours"
    scored = [r for r in body["results"] if r["score"] is not None]
    assert scored, body["results"]
    assert all(r["reason"] for r in scored)


# ---- the shelves -------------------------------------------------------------------------


def test_cross_style_no_longer_ranks_every_shelf_by_a_stranger(client):
    """The worst of it. `cross_style=true` means "rank the shelves I have not rated on by the
    taste you learned elsewhere" -- and for a caller with no profile the seed WAS the taste
    learned elsewhere, so all twenty-two came back scored and labelled `cross`."""
    body = client.get("/v1/recommend/families?cross_style=true",
                      headers=_auth(_fresh(client))).json()
    shelves = [f for g in body["groups"] for f in g["families"]]
    assert shelves, "the shelves must still be listed"
    assert {s["basis"] for s in shelves} == {"unrated"}
    assert not [r for s in shelves for r in s["results"] if r["score"] is not None]


def test_the_shelves_are_still_there_for_a_stranger(client):
    """Dark, not absent. Somewhere to browse is the point of the screen."""
    body = client.get("/v1/recommend/families", headers=_auth(_fresh(client))).json()
    assert [f for g in body["groups"] for f in g["families"]]


# ---- the chip ----------------------------------------------------------------------------


def test_a_scan_by_a_stranger_carries_no_percentage(client):
    """The number the drinker sees soonest, and the one the seed most clearly misrepresented:
    a chip reading "72% -- matches your tropical preference" over a label, to someone who had
    told the app nothing."""
    r = client.post("/v1/scan/resolve", headers=_auth(_fresh(client)),
                    json={"detections": [{"text": "Loud IPA", "kind": "text"}],
                          "include_score": True})
    assert r.status_code == 200
    for cand in r.json().get("candidates", []):
        assert cand["personal_score"] is None, cand
        assert cand["reason"] is None, cand


def test_a_scan_by_someone_who_answered_does_carry_one(client):
    """Not a regression into never scoring: the signal still reaches the chip when there is
    somebody to score for."""
    r = client.post("/v1/scan/resolve", headers=_auth(_answered(client)),
                    json={"detections": [{"text": "Loud IPA", "kind": "text"}],
                          "include_score": True})
    assert r.status_code == 200
    scored = [c for c in r.json().get("candidates", []) if c["personal_score"] is not None]
    assert scored, r.json()


def test_the_seed_never_decided_which_bottle_it_was(client):
    """Worth pinning down, because it would have been the serious version of this bug. The
    profile only ever reached `Resolver.score`; recognition is the same for everyone, and
    taking the seed away must not move it."""
    detections = {"detections": [{"text": "Peat Monster", "kind": "text"}],
                  "include_score": True}
    stranger = client.post("/v1/scan/resolve", headers=_auth(_fresh(client)),
                           json=detections).json()
    answered = client.post("/v1/scan/resolve", headers=_auth(_answered(client)),
                           json=detections).json()
    named = [[c["resolved"]["product"]["id"] for c in r.get("candidates", [])]
             for r in (stranger, answered)]
    assert named[0], "nothing was recognised, so this proves nothing"
    assert named[0] == named[1], named
