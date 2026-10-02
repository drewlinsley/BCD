"""A drinker telling us about a drink we do not have.

The scan HUD's empty-state is the one place the app authors catalog data rather than reading it,
so these tests are mostly about where a typed name is allowed to land, and what happens when the
same submission arrives twice.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest
from bcd_api.app import CONTRIBUTION_SOURCE


@pytest.fixture(scope="module")
def client():
    """The real app over a throwaway store — same setup as the auth route tests, and for the
    same reason: standing the app up runs its lifespan, which opens a store."""
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


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _crusher(cid: str = "c-1") -> dict:
    return {"id": cid, "name": "The Alchemist Crusher", "category": "beer",
            "maker": "The Alchemist", "abv_pct": 8.0,
            "sightings": ["CRUSHER", "AMERICAN DOUBLE INDIA PALE ALE"]}


# ---- who may contribute ----------------------------------------------------------------------

def test_a_caller_who_does_not_say_who_they_are_cannot_contribute(client):
    assert client.post("/v1/contribute", json=_crusher()).status_code == 401


def test_a_contribution_records_the_account_that_made_it(client, token):
    """Not for display: so a reviewer can tell one person's twenty claims from twenty people's
    one, and so a bad actor is traceable without a second log."""
    made = client.post("/v1/auth/anonymous").json()
    r = client.post("/v1/contribute", json=_crusher("who-1"), headers=_auth(made["token"]))
    assert r.status_code == 200
    from bcd_api.app import _state
    doc = _state["store"].get_bronze(r.json()["doc_id"])
    assert doc is not None and doc.payload["account_id"] == made["account_id"]


# ---- where it lands -------------------------------------------------------------------------

def test_a_contribution_lands_in_bronze_and_not_in_the_catalog(client, token):
    """The whole safety argument. A typed name is a claim, and the catalog is what the resolver
    draws from — if this ever writes a product, any anonymous account can put a row in front of
    every drinker without anyone looking at it."""
    from bcd_api.app import _state
    store = _state["store"]
    before = {row["id"] for row in store.iter_gold("product")}

    r = client.post("/v1/contribute", json=_crusher("land-1"), headers=_auth(token))
    assert r.status_code == 200 and r.json()["accepted"] is True

    doc = store.get_bronze(r.json()["doc_id"])
    assert doc is not None and doc.source_id == CONTRIBUTION_SOURCE
    assert doc.payload["name"] == "The Alchemist Crusher"
    assert {row["id"] for row in store.iter_gold("product")} == before


def test_what_the_camera_saw_is_kept_beside_what_was_typed(client, token):
    """The typed name is the claim; the sightings are the evidence for it. A reviewer with only
    the first has no way to tell a real can from a guess."""
    from bcd_api.app import _state
    r = client.post("/v1/contribute", json=_crusher("saw-1"), headers=_auth(token))
    doc = _state["store"].get_bronze(r.json()["doc_id"])
    assert doc.payload["sightings"] == ["CRUSHER", "AMERICAN DOUBLE INDIA PALE ALE"]


def test_the_server_timestamps_what_it_was_told_and_keeps_the_phones_clock_too(client, token):
    body = _crusher("clock-1") | {"created_at": "2026-10-02T00:00:00Z"}
    from bcd_api.app import _state
    r = client.post("/v1/contribute", json=body, headers=_auth(token))
    doc = _state["store"].get_bronze(r.json()["doc_id"])
    assert doc.payload["created_at"] == "2026-10-02T00:00:00Z"   # theirs, as given
    assert doc.payload["received_at"]                            # ours, and the one to trust


# ---- retries --------------------------------------------------------------------------------

def test_the_same_submission_twice_is_one_contribution(client, token):
    """The phone keeps a contribution until the server confirms it, so a lost response means a
    retry. That must not double the row — which is the whole reason the client sends the id."""
    from bcd_api.app import _state
    store = _state["store"]
    first = client.post("/v1/contribute", json=_crusher("retry-1"), headers=_auth(token)).json()
    again = client.post("/v1/contribute", json=_crusher("retry-1"), headers=_auth(token)).json()
    assert first["doc_id"] == again["doc_id"]
    assert first["duplicate"] is False and again["duplicate"] is True
    assert again["accepted"] is True, "a retry is done, not refused"
    assert sum(1 for d in store.iter_bronze(CONTRIBUTION_SOURCE)
               if d.natural_key.endswith("::retry-1")) == 1


def test_one_account_cannot_overwrite_anothers_contribution_by_guessing_its_id(client):
    """Client ids are a UUID on the phone, not a secret — the document id has to be scoped to
    the account or a guessed id edits someone else's claim."""
    a = client.post("/v1/auth/anonymous").json()
    b = client.post("/v1/auth/anonymous").json()
    mine = client.post("/v1/contribute", json=_crusher("shared-id"),
                       headers=_auth(a["token"])).json()
    theirs = client.post("/v1/contribute",
                         json=_crusher("shared-id") | {"name": "Something Else"},
                         headers=_auth(b["token"])).json()
    assert mine["doc_id"] != theirs["doc_id"]
    assert theirs["duplicate"] is False
    from bcd_api.app import _state
    assert _state["store"].get_bronze(mine["doc_id"]).payload["name"] == "The Alchemist Crusher"


# ---- what it refuses ------------------------------------------------------------------------

@pytest.mark.parametrize("bad,why", [
    ({"id": "", "name": "X", "category": "beer"}, "no id to be idempotent on"),
    ({"id": "b1", "name": "", "category": "beer"}, "a nameless drink is not a claim"),
    ({"id": "b2", "name": "X", "category": "lager"}, "not a category we file under"),
    ({"id": "b3", "name": "X", "category": "beer", "abv_pct": 200.0}, "not a strength"),
    ({"id": "b4", "name": "X" * 300, "category": "beer"}, "not a name"),
    ({"id": "b5", "name": "X", "category": "beer", "note": "n" * 900}, "not a note"),
])
def test_a_contribution_that_could_not_be_curated_is_refused(client, token, bad, why):
    assert client.post("/v1/contribute", json=bad, headers=_auth(token)).status_code == 422, why


def test_a_client_stuck_in_a_loop_is_cut_off_but_its_retries_still_answered(client, token,
                                                                           monkeypatch):
    """The cap exists to stop a broken drain filling bronze. It must not also break the thing the
    drain depends on: re-sending something already accepted has to keep answering, or a capped
    client can never clear its queue."""
    # Reached through `sys.modules`, because `bcd_api/__init__.py` re-exports `app`: both
    # "bcd_api.app.CONTRIBUTION_CAP" and `import bcd_api.app as api` resolve `app` to the
    # FastAPI object, not to the module the constant lives in.
    monkeypatch.setattr(sys.modules["bcd_api.app"], "CONTRIBUTION_CAP", 2)
    made = client.post("/v1/auth/anonymous").json()
    head = _auth(made["token"])
    assert client.post("/v1/contribute", json=_crusher("cap-1"), headers=head).status_code == 200
    assert client.post("/v1/contribute", json=_crusher("cap-2"), headers=head).status_code == 200
    assert client.post("/v1/contribute", json=_crusher("cap-3"), headers=head).status_code == 429
    assert client.post("/v1/contribute", json=_crusher("cap-1"), headers=head).status_code == 200
