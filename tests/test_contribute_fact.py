"""A drinker telling us something about a drink we DO have.

The other half of `test_contribute`. That route takes a claim towards a row that does not exist;
this one takes a claim about a row that does -- which is the commoner case by three orders of
magnitude, because 517,171 of 534,104 products carry no strength and the card prints a question
mark over every one of them.
"""

from __future__ import annotations

import os
import tempfile

import pytest
from bcd_api.app import USER_FACT_SOURCE


def _product(pid: str, name: str) -> dict:
    return {"id": pid, "brand_id": "brand:house", "producer_id": "prod:house",
            "category": "beer", "name": name, "spec": {}, "recipe": {"ingredients": []},
            "aliases": []}


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
            store.put_gold("beer:blank", "product", _product("beer:blank", "Nameless Pale"))
            yield c
    finally:
        os.environ.clear()
        os.environ.update(before)


@pytest.fixture()
def token(client):
    return client.post("/v1/auth/anonymous").json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _strength(cid: str = "f-1", pid: str = "beer:blank", value: float = 5.2) -> dict:
    return {"id": cid, "product_id": pid, "field": "abv_pct", "value": value}


# ---- who may tell us -------------------------------------------------------------------------

def test_a_caller_who_does_not_say_who_they_are_cannot_tell_us(client):
    assert client.post("/v1/contribute/fact", json=_strength()).status_code == 401


# ---- what it is a fact ABOUT -------------------------------------------------------------------

def test_a_fact_must_be_about_a_row_the_catalog_has(client, token):
    # A claim about a row that does not exist is a claim that a drink exists, which is what
    # `/v1/contribute` is for. Filed here it would sit in bronze with nothing to attach to.
    r = client.post("/v1/contribute/fact", json=_strength("f-404", pid="beer:nope"),
                    headers=_auth(token))
    assert r.status_code == 404


def test_a_strength_lands_in_bronze_under_its_own_source(client, token):
    from bcd_api.app import _state
    r = client.post("/v1/contribute/fact", json=_strength("f-land"), headers=_auth(token))
    assert r.status_code == 200 and r.json()["accepted"] is True

    doc = _state["store"].get_bronze(r.json()["doc_id"])
    assert doc is not None
    assert doc.source_id == USER_FACT_SOURCE, "not filed with claims towards new drinks"
    assert doc.payload["product_id"] == "beer:blank"
    assert doc.payload["field"] == "abv_pct"
    assert doc.payload["value"] == 5.2


def test_the_row_itself_is_untouched(client, token):
    """Bronze, and only bronze. The resolver draws from gold, and one anonymous account must
    not be able to put a number on a card half a million people read."""
    from bcd_api.app import _state
    client.post("/v1/contribute/fact", json=_strength("f-gold"), headers=_auth(token))
    row = _state["store"].get_gold("beer:blank")
    assert (row.get("spec") or {}).get("abv_pct") is None


def test_who_said_so_is_recorded(client, token):
    """Promotion counts ACCOUNTS, so the account has to be on the document."""
    from bcd_api.app import _state
    r = client.post("/v1/contribute/fact", json=_strength("f-who"), headers=_auth(token))
    doc = _state["store"].get_bronze(r.json()["doc_id"])
    assert doc.payload["account_id"]
    assert doc.payload["received_at"], "ours, not the phone's clock"


# ---- saying it twice ---------------------------------------------------------------------------

def test_the_same_submission_twice_leaves_one_document(client, token):
    first = client.post("/v1/contribute/fact", json=_strength("f-retry"),
                        headers=_auth(token)).json()
    again = client.post("/v1/contribute/fact", json=_strength("f-retry"),
                        headers=_auth(token)).json()
    assert again["doc_id"] == first["doc_id"]
    assert again["duplicate"] is True
    assert again["accepted"] is True, "the phone's job is done either way"


def test_two_accounts_with_the_same_client_id_are_two_claims(client):
    """Agreement is counted per account, so one account must not be able to overwrite
    another's by guessing a client id."""
    a = client.post("/v1/auth/anonymous").json()["token"]
    b = client.post("/v1/auth/anonymous").json()["token"]
    one = client.post("/v1/contribute/fact", json=_strength("same"), headers=_auth(a)).json()
    two = client.post("/v1/contribute/fact", json=_strength("same"), headers=_auth(b)).json()
    assert one["doc_id"] != two["doc_id"]


# ---- what a strength may be ---------------------------------------------------------------------

@pytest.mark.parametrize("value", [-0.1, 100.1])
def test_an_impossible_strength_is_refused(client, token, value):
    r = client.post("/v1/contribute/fact", json=_strength("f-bad", value=value),
                    headers=_auth(token))
    assert r.status_code == 422


def test_a_field_this_build_does_not_know_is_refused(client, token):
    body = {**_strength("f-field"), "field": "colour"}
    assert client.post("/v1/contribute/fact", json=body, headers=_auth(token)).status_code == 422
