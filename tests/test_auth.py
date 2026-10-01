"""Who is calling.

Every route used to take `user_id` as a query parameter. `GET /v1/profile?user_id=<someone>`
returned their taste and `POST /v1/feedback?user_id=<someone>` wrote to it, on a server that
binds 0.0.0.0. These are the tests that say that is over.
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest
from bcd_api.auth import (
    AuthError,
    account_for_subject,
    create_account,
    issue_session,
    link_subject,
    principal_for_token,
    verify_id_token,
)
from bcd_ingest.store import MedallionStore


@pytest.fixture()
def store():
    s = MedallionStore(root=tempfile.mkdtemp())
    yield s
    s.close()


# ---- the token is the identity --------------------------------------------------------------

def test_a_token_names_the_account_that_was_issued_it(store):
    acct = create_account(store, "anonymous")
    token = issue_session(store, acct)
    who = principal_for_token(store, token)
    assert who is not None and who.id == acct and who.provider == "anonymous"


def test_a_token_nobody_issued_names_nobody(store):
    create_account(store, "anonymous")
    assert principal_for_token(store, "not-a-real-token") is None


def test_the_token_itself_is_not_stored(store):
    """A dump of the session rows must not be a set of working tokens, so only the hash is
    kept. This is the difference between leaking a database and leaking every account in it."""
    acct = create_account(store, "anonymous")
    token = issue_session(store, acct)
    rows = [r for r in store.iter_gold("session")]
    assert rows, "the session was not written at all"
    assert all(token not in str(r) for r in rows)


def test_an_expired_token_names_nobody(store):
    acct = create_account(store, "anonymous")
    token = issue_session(store, acct)
    row = store.get_gold(f"session:{__import__('hashlib').sha256(token.encode()).hexdigest()}")
    store.put_gold(row["id"], "session", {**row, "expires_at": int(time.time()) - 1})
    assert principal_for_token(store, token) is None


def test_two_accounts_get_different_tokens(store):
    a, b = create_account(store, "anonymous"), create_account(store, "anonymous")
    assert a != b
    assert issue_session(store, a) != issue_session(store, b)
    assert principal_for_token(store, issue_session(store, a)).id == a


# ---- providers ------------------------------------------------------------------------------

def test_signing_in_again_finds_the_same_account(store):
    """The second sign-in from one Google user must not mint a second account, or their taste
    splits in two and each half looks like a quieter drinker."""
    acct = create_account(store, "google", subject="google-sub-1")
    assert account_for_subject(store, "google", "google-sub-1") == acct
    assert account_for_subject(store, "google", "someone-else") is None


def test_a_subject_is_scoped_to_its_provider(store):
    """Apple and Google both hand out opaque subject strings and neither promises anything
    about the other's. Keyed on the subject alone, a collision would hand one drinker another's
    account."""
    create_account(store, "google", subject="shared-string")
    assert account_for_subject(store, "apple", "shared-string") is None


def test_signing_in_claims_the_anonymous_account(store):
    """A drinker who rated ten drinks and then signed in still has ten. The anonymous account
    is linked rather than replaced, so the profile, which is keyed on the account id, survives."""
    anon = create_account(store, "anonymous")
    link_subject(store, anon, "google", "google-sub-2")
    assert account_for_subject(store, "google", "google-sub-2") == anon
    assert principal_for_token(store, issue_session(store, anon)).provider == "google"


def test_a_provider_nobody_configured_refuses_rather_than_verifying_less(store):
    """Without the audience the signature still checks out and the token can be one issued for
    a DIFFERENT app -- which is exactly what `aud` exists to stop."""
    with pytest.raises(AuthError) as exc:
        verify_id_token("google", "irrelevant", audience=None)
    assert exc.value.status_code == 503


def test_an_unknown_provider_is_not_a_provider(store):
    with pytest.raises(AuthError) as exc:
        verify_id_token("facebook", "irrelevant", audience="x")
    assert exc.value.status_code == 400


def test_a_token_that_is_not_a_token_does_not_verify(store):
    with pytest.raises(AuthError) as exc:
        verify_id_token("google", "not.a.jwt", audience="client-id")
    assert exc.value.status_code == 401


# ---- the routes ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """The real app over a throwaway store.

    Module-scoped and pointed at a temp directory on purpose. Standing the app up runs its
    lifespan, which opens a store and builds the label index; per-test, against the real
    `./data`, that both wrote to the dev store and rebuilt the index fourteen times.
    """
    from fastapi.testclient import TestClient
    root = tempfile.mkdtemp()
    # The WHOLE environment is put back, not just the keys set here: the app's lifespan calls
    # `_load_dotenv()`, so merely standing it up loads `.env` into this process. Restoring four
    # names left a vision API key behind and a later test that asserts there is no key failed --
    # only when the suite ran in full, which is the worst way to find out.
    before = dict(os.environ)
    os.environ.update({"BCD_STORE_BACKEND": "sqlite", "BCD_DATA_ROOT": root,
                       "BCD_LABEL_INDEX": "0"})       # these tests never scan
    os.environ.pop("BCD_DATABASE_URL", None)
    try:
        from bcd_api.app import app
        with TestClient(app) as c:
            yield c
    finally:
        os.environ.clear()
        os.environ.update(before)


@pytest.mark.parametrize("method,path", [
    ("get", "/v1/profile"),
    ("post", "/v1/profile/rebuild"),
    ("post", "/v1/recommend"),
    ("get", "/v1/recommend/families"),
    ("post", "/v1/feedback"),
    ("post", "/v1/scan/resolve"),
])
def test_a_route_that_knows_who_you_are_refuses_a_caller_who_does_not_say(client, method, path):
    """The whole point. Before this, each of these took `user_id` as a query parameter and
    believed it."""
    call = getattr(client, method)
    r = call(path, json={}) if method == "post" else call(path)
    assert r.status_code == 401


def test_naming_someone_else_in_the_query_string_no_longer_works(client):
    """`?user_id=` is not an identity any more. It is ignored, and the route still asks for a
    token -- it must not be a back door that merely looks closed."""
    assert client.get("/v1/profile?user_id=someone-elses-id").status_code == 401


def test_a_made_up_token_is_refused(client):
    r = client.get("/v1/profile", headers={"Authorization": "Bearer made-up"})
    assert r.status_code == 401


def test_a_token_without_the_bearer_scheme_is_refused(client):
    tok = client.post("/v1/auth/anonymous").json()["token"]
    assert client.get("/v1/profile", headers={"Authorization": tok}).status_code == 401


def test_the_app_can_get_an_account_and_then_be_itself(client):
    """The first-launch path end to end: ask for an account, keep the token, read your own
    (empty) profile with it."""
    made = client.post("/v1/auth/anonymous").json()
    assert made["provider"] == "anonymous" and made["account_id"].startswith("acct:")
    r = client.get("/v1/profile", headers={"Authorization": f"Bearer {made['token']}"})
    assert r.status_code == 200
    assert r.json()["user_id"] == made["account_id"]


def test_two_installs_do_not_get_the_same_account(client):
    a = client.post("/v1/auth/anonymous").json()
    b = client.post("/v1/auth/anonymous").json()
    assert a["account_id"] != b["account_id"] and a["token"] != b["token"]


def test_a_profile_is_read_under_the_callers_own_account(client):
    """Two accounts, one asking: the answer is the caller's, whatever anyone types in the URL."""
    a = client.post("/v1/auth/anonymous").json()
    b = client.post("/v1/auth/anonymous").json()
    r = client.get(f"/v1/profile?user_id={b['account_id']}",
                   headers={"Authorization": f"Bearer {a['token']}"})
    assert r.status_code == 200
    assert r.json()["user_id"] == a["account_id"]


def test_a_provider_nobody_has_heard_of_is_not_a_route(client):
    r = client.post("/v1/auth/facebook", json={"id_token": "x"})
    assert r.status_code == 404


def test_apple_is_wired_but_unconfigured_and_says_so(client):
    """Sign in with Apple is implemented and will answer as soon as the Developer Program
    membership is paid and `BCD_APPLE_CLIENT_ID` is set. Until then it refuses honestly rather
    than verifying something weaker than it looks."""
    r = client.post("/v1/auth/apple", json={"id_token": "x"})
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"]
