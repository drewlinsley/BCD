"""Who is calling.

Every route used to take `user_id` as a query parameter, defaulting to "demo". That is not an
identity, it is a request to be believed: `GET /v1/profile?user_id=<someone else>` returned their
taste and `POST /v1/feedback?user_id=<someone else>` wrote to it, on a server that binds 0.0.0.0.
The only thing protecting a profile was that install ids are unguessable, which is obscurity.

So the caller presents a bearer token and the server decides who they are. Three ways to get one:

  anonymous  the app asks for a fresh account on first launch and keeps the token. No sign-in,
             no personal data, and the account is as pseudonymous as the install id was -- but
             the server MINTS it, so nobody can ask to be someone else.
  google     an ID token from Google, verified against their JWKS.
  apple      the same, against Apple's. Implemented and tested; the app cannot offer it until
             the Apple Developer Program membership is paid, because Sign in with Apple is a
             capability free provisioning will not sign.

What a sign-in buys the drinker is durability, not secrecy: an anonymous account lives in one
app install, and signing in makes the same profile reachable from a reinstall or a second phone.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

import jwt
from bcd_ingest.store import Store
from fastapi import Header, HTTPException
from jwt import PyJWKClient

# gold entity types. Profiles already live in gold keyed `profile:<id>`, so accounts and sessions
# follow the same shape rather than introducing a second store for three columns.
ACCOUNT = "account"
SESSION = "session"
#: `provider:sub -> account id`, so a second sign-in from the same Google user finds the first
#: account instead of minting another.
LINK = "account_link"

#: A session lasts until it is used once past this, then the client asks for another. Long,
#: because the alternative is a drinker signed out of their own taste for no reason; short
#: enough that a leaked token is not forever.
SESSION_TTL_SECONDS = 60 * 60 * 24 * 90

_GOOGLE_JWKS = "https://www.googleapis.com/oauth2/v3/certs"
_GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
_APPLE_JWKS = "https://appleid.apple.com/auth/keys"
_APPLE_ISSUER = "https://appleid.apple.com"

_jwks: dict[str, PyJWKClient] = {}


def _jwks_client(url: str) -> PyJWKClient:
    """One client per provider: it caches the signing keys, and fetching them per sign-in would
    put a provider's uptime in front of ours."""
    if url not in _jwks:
        _jwks[url] = PyJWKClient(url, cache_keys=True)
    return _jwks[url]


@dataclass(frozen=True)
class Principal:
    """The caller, as the server worked them out. `id` is what every route keys on -- it is the
    `user_id` the profile, the ratings and the recommendations belong to."""

    id: str
    provider: str


class AuthError(HTTPException):
    def __init__(self, detail: str, status: int = 401) -> None:
        super().__init__(status_code=status, detail=detail,
                         headers={"WWW-Authenticate": "Bearer"})


# ---- tokens ---------------------------------------------------------------------------------

def _hash(token: str) -> str:
    """Sessions are stored by hash. A dump of the table is then not a set of working tokens."""
    return hashlib.sha256(token.encode()).hexdigest()


def issue_session(store: Store, account_id: str) -> str:
    """A fresh bearer token for `account_id`. Returned once and never recoverable: only its
    hash is kept."""
    token = secrets.token_urlsafe(32)
    store.put_gold(f"{SESSION}:{_hash(token)}", SESSION, {
        "id": f"{SESSION}:{_hash(token)}",
        "account_id": account_id,
        "issued_at": int(time.time()),
        "expires_at": int(time.time()) + SESSION_TTL_SECONDS,
    })
    return token


def principal_for_token(store: Store, token: str) -> Principal | None:
    rec = store.get_gold(f"{SESSION}:{_hash(token)}")
    if not rec or int(rec.get("expires_at", 0)) < int(time.time()):
        return None
    account = store.get_gold(rec["account_id"])
    if not account:
        return None
    return Principal(id=account["id"], provider=account.get("provider", "anonymous"))


# ---- accounts -------------------------------------------------------------------------------

def create_account(store: Store, provider: str, subject: str | None = None) -> str:
    account_id = f"acct:{secrets.token_hex(16)}"
    store.put_gold(account_id, ACCOUNT, {
        "id": account_id,
        "provider": provider,
        "subject": subject,
        "created_at": int(time.time()),
    })
    if subject:
        key = f"{LINK}:{provider}:{subject}"
        store.put_gold(key, LINK, {"id": key, "account_id": account_id})
    return account_id


def account_for_subject(store: Store, provider: str, subject: str) -> str | None:
    rec = store.get_gold(f"{LINK}:{provider}:{subject}")
    return rec.get("account_id") if rec else None


def link_subject(store: Store, account_id: str, provider: str, subject: str) -> None:
    """Point a provider identity at an existing account, and say on the account which provider
    now owns it. This is what turns the anonymous account a drinker has been using into their
    signed-in one, so their ratings survive the sign-in instead of being left behind."""
    key = f"{LINK}:{provider}:{subject}"
    store.put_gold(key, LINK, {"id": key, "account_id": account_id})
    rec = store.get_gold(account_id) or {"id": account_id, "created_at": int(time.time())}
    store.put_gold(account_id, ACCOUNT, {**rec, "provider": provider, "subject": subject})


# ---- provider ID tokens ---------------------------------------------------------------------

def verify_id_token(provider: str, token: str, *, audience: str | None = None) -> str:
    """The provider's subject (`sub`) for this ID token, or raise.

    `sub` is the only claim taken. An email is a name, not an identity -- Google lets one change
    and Apple hands out a per-app relay address -- and the app has no use for either, so neither
    is read or stored. What the server keeps about a signed-in drinker is an opaque provider id.
    """
    if provider == "google":
        url, issuers = _GOOGLE_JWKS, _GOOGLE_ISSUERS
        audience = audience or os.environ.get("BCD_GOOGLE_CLIENT_ID")
    elif provider == "apple":
        url, issuers = _APPLE_JWKS, (_APPLE_ISSUER,)
        audience = audience or os.environ.get("BCD_APPLE_CLIENT_ID")
    else:
        raise AuthError(f"unknown provider {provider!r}", status=400)
    if not audience:
        # Without the audience the signature still checks out and the token can be one issued
        # for a DIFFERENT app -- which is exactly the attack `aud` exists to stop. Refuse rather
        # than verify something weaker than it looks.
        raise AuthError(f"{provider} sign-in is not configured on this server", status=503)

    try:
        key = _jwks_client(url).get_signing_key_from_jwt(token).key
        claims: dict[str, Any] = jwt.decode(
            token, key, algorithms=["RS256"], audience=audience,
            issuer=list(issuers) if len(issuers) > 1 else issuers[0],
            options={"require": ["sub", "aud", "iss", "exp"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"the {provider} token did not verify: {exc}") from exc
    sub = claims.get("sub")
    if not sub:
        raise AuthError(f"the {provider} token names nobody")
    return str(sub)


# ---- the dependency -------------------------------------------------------------------------

def bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("this route needs a bearer token")
    return authorization.split(" ", 1)[1].strip()


def make_current_principal(get_store):
    """Build the FastAPI dependency. Takes a callable rather than the store so the app can hand
    it the one it opened at startup without this module importing app state."""

    def current_principal(authorization: str | None = Header(default=None)) -> Principal:
        who = principal_for_token(get_store(), bearer(authorization))
        if who is None:
            raise AuthError("that token is not valid, or has expired")
        return who

    return current_principal
