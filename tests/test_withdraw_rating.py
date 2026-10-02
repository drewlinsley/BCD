"""Taking a verdict back.

The rating picker is five faces and one tap, and the sheet closes itself the moment a tap
lands — so rating the wrong drink, or the wrong rung, is easy. Until `POST /v1/feedback/withdraw`
the slip was permanent: the picker could move a rating between rungs but never remove one, and
`rated_products` went on barring that drink from "For you" forever.

These are the route's tests. The arithmetic of a withdrawal — that it stops counting, that a
later re-rate stands, that it does not touch a list add — is in `test_taste.py`.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(scope="module")
def client():
    """The real app over a throwaway store — same setup as the other route tests."""
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


def _rate(client, token, pid="p:ipa", rating=5.0):
    return client.post("/v1/feedback", json={"product_id": pid, "rating": rating},
                       headers=_auth(token))


def _withdraw(client, token, pid="p:ipa"):
    return client.post("/v1/feedback/withdraw", json={"product_id": pid}, headers=_auth(token))


# ---- who may withdraw ----------------------------------------------------------------------

def test_a_caller_who_does_not_say_who_they_are_cannot_withdraw(client):
    r = client.post("/v1/feedback/withdraw", json={"product_id": "p:ipa"})
    assert r.status_code == 401


def test_one_account_cannot_withdraw_anothers_verdict(client):
    """The product id is all the body carries, so the only thing standing between two
    drinkers is the token."""
    mine = client.post("/v1/auth/anonymous").json()["token"]
    theirs = client.post("/v1/auth/anonymous").json()["token"]
    _rate(client, mine, "p:shared", 5.0)
    assert _withdraw(client, theirs, "p:shared").status_code == 200
    # Still mine, still counted.
    after = client.get("/v1/profile", headers=_auth(mine)).json()
    assert after["version"] > 0


# ---- what it does --------------------------------------------------------------------------

def test_withdrawing_takes_the_verdict_out_of_the_profile(client, token):
    rated = _rate(client, token, "p:ipa", 5.0).json()
    assert rated["accepted"] and rated["profile"]["version"] > 0

    taken_back = _withdraw(client, token, "p:ipa")
    assert taken_back.status_code == 200
    body = taken_back.json()
    assert body["accepted"]
    # The route hands back the rebuilt profile, so the client need not ask twice.
    assert "profile" in body


def test_an_empty_product_id_is_refused(client, token):
    """The id is the whole request; an empty one would append an event naming nothing."""
    r = client.post("/v1/feedback/withdraw", json={"product_id": ""}, headers=_auth(token))
    assert r.status_code == 422


def test_withdrawing_something_never_rated_is_not_an_error(client, token):
    """The phone's reaction log is a disposable cache, so it can ask to withdraw a verdict
    this server has never heard of. That is a no-op, not a 404."""
    r = _withdraw(client, token, "p:never-rated")
    assert r.status_code == 200
    assert r.json()["accepted"]


def test_withdrawing_twice_is_the_same_as_withdrawing_once(client, token):
    _rate(client, token, "p:twice", 5.0)
    first = _withdraw(client, token, "p:twice").json()
    second = _withdraw(client, token, "p:twice").json()
    assert first["accepted"] and second["accepted"]
    assert first["profile"]["sensory_ideal"] == second["profile"]["sensory_ideal"]


def test_a_withdrawn_drink_can_be_rated_again(client, token):
    """Withdrawing is not a tombstone. A drinker who takes a verdict back and then gives a
    new one must end up with the new one."""
    _rate(client, token, "p:again", 1.0)
    _withdraw(client, token, "p:again")
    again = _rate(client, token, "p:again", 5.0)
    assert again.status_code == 200 and again.json()["accepted"]
