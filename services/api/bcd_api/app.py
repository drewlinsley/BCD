"""BCD API — the surface the iOS app talks to.

Endpoints kept thin; logic lives in Resolver and the services. Reads the local
MedallionStore so the whole thing boots with `make api` after an ingest, no server deps.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
import uuid
from collections.abc import Collection
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

import httpx
from bcd_ingest.merge import get_product
from bcd_ingest.store import BronzeDoc, Store, doc_id, open_store
from bcd_schema import (
    ContributionRequest,
    ContributionResponse,
    FeedbackRequest,
    FeedbackResponse,
    LexiconResponse,
    Product,
    ProductSearchResponse,
    ResolvedProduct,
    ScanResolveRequest,
    ScanResolveResponse,
    ScanVisionRequest,
    ScanVisionResponse,
    TasteProfile,
)
from bcd_schema.api import DetectedText
from bcd_schema.family import (
    FAMILIES,
    GROUPS,
    Family,
    families_in,
    family_of,
    shelf_styles,
    styles_in,
)
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from .auth import (
    Principal,
    account_for_subject,
    bearer,
    create_account,
    issue_session,
    link_subject,
    make_current_principal,
    verify_id_token,
)
from .index import IndexedStore, LabelIndex
from .recommend import rank_catalog, rank_family, shelf_vector, similar_profile
from .resolver import Resolver
from .taste import TASTE_EVENTS, load_profile, rated_products, rebuild_profile
from .telemetry_ingest import TelemetryCollector
from .vision import MAX_IMAGE_BYTES, VisionProvider, provider_from_env

_state: dict = {}


# `.env.example` says "Copy to .env and fill in", and until now nothing read it — every key in
# that file only worked if you also exported it on the command line. The vision provider is the
# first thing whose absence is silent rather than loud (no key, no answers, no error), so the
# documented way to configure it has to actually be the way.
#
# Real environment always wins, the way every dotenv loader behaves: an inline
# `BCD_DATABASE_URL=... uvicorn ...` is an override, not a suggestion.
def _load_dotenv(path: str = ".env") -> None:
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("\"'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_dotenv()
    # Where this server's own files live. An env var rather than a literal so a test can point
    # the whole app at a throwaway directory: with the path baked in, standing the app up at all
    # read and WROTE the real dev store, and rebuilt the label index over it once per test.
    root = os.environ.get("BCD_DATA_ROOT", "./data")
    store = open_store(root=root)
    _state["store"] = store
    # The label index is what makes a scan sub-second: retrieval from memory instead of a
    # trigram scan per line. Built from the store once (cached beside the data and reused
    # while the catalog's row counts are unchanged), then wrapped around the store so the
    # resolver's matching calls answer from it. BCD_LABEL_INDEX=0 keeps the store's own
    # matching, for comparing the two on the same catalog.
    matching: Store = store
    if os.environ.get("BCD_LABEL_INDEX", "1") != "0":
        index = LabelIndex.for_store(store, os.environ.get(
            "BCD_LABEL_INDEX_PATH", os.path.join(root, "label_index.pkl")))
        _state["index"] = index
        matching = IndexedStore(store, index)
    _state["resolver"] = Resolver(matching)
    _state["telemetry"] = TelemetryCollector(root=root)
    # Demo profile so /v1/scan/resolve returns personalized scores out of the box.
    _state["profiles"] = {"demo": _demo_profile()}
    # None without a key. The scan path predates this and has to keep working without it.
    _state["vision"] = provider_from_env()
    yield
    store.close()


app = FastAPI(title="BCD API", version="0.1.0", lifespan=lifespan)


#: Resolves the caller from their bearer token. Built with a getter rather than the store
#: itself, because the store is opened at startup and this module is imported before that.
current_principal = make_current_principal(lambda: _state["store"])

#: Who is calling, on a route that needs to know. Spelled as an `Annotated` alias rather than a
#: `Depends()` default so the type says it once and every route reads `who: Caller`.
Caller = Annotated[Principal, Depends(current_principal)]


@app.get("/healthz")
def healthz() -> dict:
    store: Store = _state["store"]
    out = {"ok": True, "counts": store.counts()}
    index: LabelIndex | None = _state.get("index")
    if index is not None:
        out["index"] = index.stats()
    return out


@app.get("/v1/product/search", response_model=ProductSearchResponse)
def product_search(q: str = Query(..., min_length=1), limit: int = 20) -> ProductSearchResponse:
    store: Store = _state["store"]
    resolver: Resolver = _state["resolver"]
    results: list[ResolvedProduct] = []
    seen: set[str] = set()
    for rec in store.search_gold_products(q, limit=limit):
        if rec["id"] in seen:
            continue
        seen.add(rec["id"])
        hydrated = resolver._hydrate(rec)
        if hydrated:
            results.append(hydrated)
    return ProductSearchResponse(query=q, results=results)


@app.post("/v1/scan/resolve", response_model=ScanResolveResponse)
def scan_resolve(req: ScanResolveRequest,
                 who: Caller) -> ScanResolveResponse:
    user_id = who.id
    resolver: Resolver = _state["resolver"]
    profile = _profile_for(user_id)
    t0 = time.perf_counter()
    resp = resolver.resolve(req, profile=profile)
    resp.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    _log_scan(req, resp)
    return resp


# Off unless BCD_SCAN_LOG names a file. Diagnosing a scan means knowing what the camera
# actually read, and the client's own telemetry cannot be relied on for that during a debug
# session: it batches, and only uploads once twenty events have piled up, so the scan you just
# did is still sitting on the phone. Every real diagnosis so far — a wordmark read as Cyrillic,
# a can whose brand never appeared in 30 lines — came from seeing the raw lines, and each time
# they arrived late or not at all.
_SCAN_LOG = os.environ.get("BCD_SCAN_LOG")


def _log_scan(req: ScanResolveRequest, resp: ScanResolveResponse) -> None:
    if not _SCAN_LOG:
        return
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "ocr": [d.text for d in req.detections],
        "objects": [{"id": o.id, "texts": o.texts, "barcode": o.barcode} for o in req.objects],
        "corroborated": resp.corroborated,
        "latency_ms": resp.latency_ms,
        "verdicts": [
            {"id": o.object_id, "status": o.status,
             "candidates": [{"name": c.resolved.product.name, "score": c.match_score}
                            for c in o.candidates]}
            for o in resp.objects
        ],
        "candidates": [
            {"name": c.resolved.product.name, "producer": c.resolved.producer.name,
             "score": c.match_score}
            for c in resp.candidates[:5]
        ],
    }
    try:
        with open(_SCAN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        pass          # diagnostics must never take the scan path down with them


@app.get("/v1/lexicon", response_model=LexiconResponse)
def lexicon(limit: int = Query(5000, ge=1, le=20000)) -> LexiconResponse:
    """Custom-words hint for the on-device text recognizer: the catalog's identifying
    vocabulary, commonest first. The recognizer already knows "ale"; what it does not
    know is "alchemist", and told about it it stops correcting HEADY into Ready. Cached
    per process; a venue-scoped variant is the obvious next step once menus are live."""
    cache: dict = _state.setdefault("lexicon", {})
    if limit not in cache:
        resolver: Resolver = _state["resolver"]
        cache[limit] = resolver.lexicon(limit=limit)
    return LexiconResponse(words=cache[limit])


_VISION_UNCONFIGURED = (
    "vision is not configured: set ANTHROPIC_API_KEY in .env and restart the API"
)


@app.post("/v1/scan/vision", response_model=ScanVisionResponse)
async def scan_vision(req: ScanVisionRequest,
                      who: Caller) -> ScanVisionResponse:
    """Identify a frame from the picture rather than from what OCR made of it.

    The model only ever supplies *names*. Each one is then resolved by the same `Resolver`
    against the same catalog, under the same guards, and a name the catalog cannot account
    for is dropped — so a model that invents a beer produces no answer rather than a
    confident wrong one. Facts still come from the catalog; the image only improves the query.

    Errors are reported in `detail`, never raised. This runs on the camera's hot path, and a
    timeout at the vision provider must degrade to "no extra answers", not to a failed scan.
    """
    user_id = who.id
    t0 = time.perf_counter()
    provider: VisionProvider | None = _state.get("vision")

    def done(resp: ScanVisionResponse) -> ScanVisionResponse:
        resp.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        _log_vision(req, resp)
        return resp

    if provider is None:
        return done(ScanVisionResponse(detail=_VISION_UNCONFIGURED))
    try:
        image = base64.b64decode(req.image_b64, validate=True)
    except (binascii.Error, ValueError):
        return done(ScanVisionResponse(provider=provider.label,
                                       detail="image_b64 is not valid base64"))
    if not image:
        return done(ScanVisionResponse(provider=provider.label, detail="image is empty"))
    if len(image) > MAX_IMAGE_BYTES:
        return done(ScanVisionResponse(
            provider=provider.label,
            detail=f"image is {len(image)} bytes, over the {MAX_IMAGE_BYTES} limit"))

    try:
        sightings = await provider.identify(image, req.media_type)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return done(ScanVisionResponse(provider=provider.label,
                                       detail=f"{type(exc).__name__}: {exc}"))
    if not sightings:
        return done(ScanVisionResponse(provider=provider.label,
                                       detail="the model read no label in this frame"))

    # The frame the overlays anchor to: one entry per sighting, in the model's own order and
    # carrying whatever box it volunteered. The client never built this, so it comes back with
    # the answers or `detection_index` addresses nothing.
    frame = [DetectedText(text=s.name, kind="text",
                          x=s.box[0] if s.box else None, y=s.box[1] if s.box else None,
                          w=s.box[2] if s.box else None, h=s.box[3] if s.box else None)
             for s in sightings]

    # `resolve_reading`, not `resolve`. A clean name is a query, not a frame: `resolve` is built
    # for OCR — fragmentary, garbled, several lines of one object — and its instruments say the
    # wrong thing here. Containment scores any name wholly inside the reading at a perfect 1.00,
    # so its one-winner-per-line rule handed back `Lawson's` for "Lawson's Sip of Sunshine",
    # `Tree House` for Julius and `Green` for Green City, every one of them a fragment of the
    # name rather than the row it names.
    resolver: Resolver = _state["resolver"]
    profile = _profile_for(user_id)
    kept: list = []
    claimed: set[str] = set()
    for i, sighting in enumerate(sightings):
        cand = resolver.resolve_reading(sighting.name, index=i, profile=profile)
        if cand is None:
            continue          # the catalog has no row that is what the model read
        if cand.resolved.product.id in claimed:
            continue          # two sightings of the same beer are one answer
        claimed.add(cand.resolved.product.id)
        kept.append(cand)

    named = {c.detection_index for c in kept}
    return done(ScanVisionResponse(
        candidates=kept,
        unresolved_indices=[i for i in range(len(sightings)) if i not in named],
        # A kept candidate has cleared two independent bars: a model reading the label off the
        # image, and the catalog holding a row that accounts for that whole reading rather than
        # appearing inside it. That is the corroboration this endpoint can honestly claim, and
        # it is not the same evidence as two OCR lines agreeing — hence computed here.
        corroborated=bool(kept),
        sightings=[s.name for s in sightings],
        provider=provider.label,
        detections=frame,
        detail=None if kept else "the catalog has none of the labels the model read",
    ))


def _log_vision(req: ScanVisionRequest, resp: ScanVisionResponse) -> None:
    if not _SCAN_LOG:
        return
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "path": "vision",
        "ocr": [d.text for d in req.detections],
        "bytes": len(req.image_b64) * 3 // 4,
        "sightings": resp.sightings,
        "provider": resp.provider,
        "detail": resp.detail,
        "latency_ms": resp.latency_ms,
        "candidates": [
            {"name": c.resolved.product.name, "producer": c.resolved.producer.name,
             "score": c.match_score}
            for c in resp.candidates[:5]
        ],
    }
    try:
        with open(_SCAN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        pass


@app.post("/v1/recommend")
def recommend(who: Caller, limit: int = 10) -> dict:
    """Rank the catalog for a user: the store finds the nearest vectors (pgvector on
    Postgres, python cosine on the SQLite dev store), `rank_catalog` scores and orders them
    -- a match before a partial one, what drinkers rated before what we know before what we
    guess, then the score -- and says which of those each result is (`evidence`)."""
    user_id = who.id
    collector: TelemetryCollector = _state["telemetry"]
    store: Store = _state["store"]
    results = rank_catalog(store, _state["resolver"], _profile_for(user_id), limit=limit,
                           exclude=rated_products(store, collector.iter_events(TASTE_EVENTS),
                                                  user_id))
    return {"user_id": user_id, "results": results}


class SignInRequest(BaseModel):
    """An ID token from the provider, obtained by the app. The server never sees a password and
    never talks to the provider on the drinker's behalf -- it checks a signature."""

    id_token: str
    #: The anonymous account the drinker has been using, if any. Its ratings move to the
    #: signed-in account rather than being left behind. Taken from the CALLER'S OWN token, never
    #: from the body -- see the route.


class SignInResponse(BaseModel):
    account_id: str
    token: str
    provider: str
    #: True when this sign-in adopted the anonymous account the caller was already using.
    claimed: bool = False


@app.post("/v1/auth/anonymous", response_model=SignInResponse)
def auth_anonymous() -> SignInResponse:
    """An account with nobody attached, minted by the server on the app's first launch.

    The app used to identify itself with an id it chose. That is what made every other route
    forgeable, so the id is minted here instead and handed back exactly once.
    """
    store: Store = _state["store"]
    account_id = create_account(store, "anonymous")
    return SignInResponse(account_id=account_id, token=issue_session(store, account_id),
                          provider="anonymous")


@app.post("/v1/auth/{provider}", response_model=SignInResponse)
def auth_provider(provider: str, req: SignInRequest,
                  authorization: str | None = Header(default=None)) -> SignInResponse:
    """Sign in with Google or Apple.

    If the caller already holds a token for an anonymous account, that account is CLAIMED: the
    provider identity is linked to it and the drinker keeps the ratings they gave before signing
    in. The account to claim is read from their own bearer token and never from the body, so
    nobody can claim an account by naming it.

    Signing in again later finds the same account through the provider link.
    """
    if provider not in ("google", "apple"):
        raise HTTPException(status_code=404, detail=f"no provider {provider!r}")
    store: Store = _state["store"]
    subject = verify_id_token(provider, req.id_token)

    existing = account_for_subject(store, provider, subject)
    if existing:
        return SignInResponse(account_id=existing, token=issue_session(store, existing),
                              provider=provider)

    # First sign-in with this identity. Adopt the anonymous account the caller is holding, so a
    # drinker who rated ten drinks and then signed in still has ten.
    claimed = False
    account_id = None
    if authorization:
        from .auth import principal_for_token
        who = principal_for_token(store, bearer(authorization))
        if who is not None and who.provider == "anonymous":
            account_id, claimed = who.id, True
    if account_id is None:
        account_id = create_account(store, provider, subject)
    else:
        link_subject(store, account_id, provider, subject)
    return SignInResponse(account_id=account_id, token=issue_session(store, account_id),
                          provider=provider, claimed=claimed)


@app.get("/v1/recommend/families")
def recommend_families(who: Caller, limit: int = 6,
                       cross_style: bool = False) -> dict:
    """Discover, one shelf at a time: gin, bourbon and vodka are three questions.

    `/v1/recommend` can only answer whichever one the drinker's centroid sits nearest. A
    profile built from two IPAs puts the whole nearest neighbourhood inside the IPA shelf, so
    gin never appears in that list -- not ranked low, absent.

    Families the drinker has rated in come back `basis: "yours"` and ranked. The rest come back
    `basis: "unrated"`, ordered by what the catalog knows and carrying no score, because a
    number there would be read as a prediction about someone who has never rated on that shelf.
    `cross_style=true` ranks those by the taste learned elsewhere instead (`basis: "cross"`).

    Ordered rated-in shelves first, then `FAMILIES`' own order, so the list opens on the ones
    that are about the reader.
    """
    user_id = who.id
    collector: TelemetryCollector = _state["telemetry"]
    store: Store = _state["store"]
    profile = _profile_for(user_id)
    events = list(collector.iter_events(TASTE_EVENTS))
    judged = rated_products(store, events, user_id)
    mine = _families_rated_in(store, judged)

    order = sorted(FAMILIES, key=lambda f: (f not in mine, FAMILIES.index(f)))
    # Ordered rated-in shelves first WITHIN their aisle, below; the sort above only decides
    # which shelf leads its own group.
    # Over-fetch: `rank_family` drops whole vector groups the drinker has already judged, and a
    # shelf of five that loses two should be a short section rather than a wrong one.
    vectors = [shelf_vector(profile, rated_in=f in mine, cross_style=cross_style) for f in order]
    spellings = _shelf_styles(store)
    fetched = store.shelves_many([(spellings[f], v) for f, v in zip(order, vectors, strict=True)],
                                 limit=limit * 4)

    shelves: dict[Family, dict] = {}
    for family, vec, rows in zip(order, vectors, fetched, strict=True):
        shelf = rank_family(store, _state["resolver"], profile, rows,
                            personal=vec is not None, rated_in=family in mine,
                            limit=limit, exclude=judged)
        # A shelf with nothing on it is not a shelf. Cider is in the table for completeness
        # and the catalog files all 333 of its rows with a null style.
        if shelf["results"]:
            shelves[family] = {"family": family.value, "label": family.label, **shelf}

    # Two levels: twenty-two shelves in a flat list is a scroll, and beer-or-spirits is a
    # division the drinker already made before opening the screen.
    groups = []
    for group in sorted(GROUPS, key=lambda g: (not (mine & set(families_in(g))),
                                               GROUPS.index(g))):
        inside = [shelves[f] for f in sorted(families_in(group),
                                             key=lambda f: (f not in mine, FAMILIES.index(f)))
                  if f in shelves]
        if inside:
            groups.append({"group": group.value, "label": group.label,
                           "rated_in": any(f in mine for f in families_in(group)),
                           "families": inside})
    return {"user_id": user_id, "rated": len(judged), "groups": groups}


def _shelf_styles(store: Store) -> dict[Family, list[str]]:
    """Which spellings fill each shelf, read from the catalog once and kept.

    `styles_in` names the styles somebody wrote into the table; the shelf query has to match
    the ones that are really in the catalog. `Peated Scotch` reaches the Scotch shelf through
    the last-word rule, and without this the shelf would call itself yours the moment you rated
    one and then not contain a single one.

    Unioned with the table so this can only ever add: if the catalog read comes back thin, the
    shelves are exactly what they were before.
    """
    cached: dict[Family, list[str]] | None = _state.get("shelf_styles")
    if cached is None:
        found = shelf_styles(store.style_catalog())
        cached = {f: sorted(set(found.get(f, ())) | set(styles_in(f))) for f in FAMILIES}
        _state["shelf_styles"] = cached
    return cached


def _families_rated_in(store: Store, judged: Collection[str]) -> set[Family]:
    """The shelves this drinker has passed a verdict on. What separates a ranked family from a
    dark one, so it is read from the verdicts themselves rather than from the profile: the
    profile's style affinities are already averaged and would call a shelf theirs on the
    strength of a style that merely resembles one they rated."""
    out: set[Family] = set()
    for pid in judged:
        rec = get_product(store, pid)
        if not rec:
            continue
        family = family_of((rec.get("style") or {}).get("value"))
        if family:
            out.add(family)
    return out


@app.get("/v1/product/{product_id}/similar")
def similar(product_id: str, limit: int = 6) -> dict:
    """What else tastes like this one. No profile involved -- this is about the bottle, not
    about you, so two people asking of the same row get the same answer.

    Answered from the rows whose vector is their own. A product carrying its style's centroid
    has no profile to be near, and says so (`basis: "style_only"`) rather than listing its
    style back as if that were a similarity.
    """
    store: Store = _state["store"]
    rec = get_product(store, product_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="no such product")
    return {"product_id": rec.get("id", product_id),
            **similar_profile(store, Product.model_validate(rec), limit=limit)}


@app.post("/v1/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest,
             who: Caller) -> FeedbackResponse:
    """A thumbs on one product. Recorded as a real `rating_submitted` event and then
    folded into the profile, so this convenience path and the client's batch telemetry
    upload converge on exactly the same profile."""
    user_id = who.id
    collector: TelemetryCollector = _state["telemetry"]
    store: Store = _state["store"]
    event = {
        "name": "rating_submitted",
        "event_id": str(uuid.uuid4()),
        "ts": datetime.now(UTC).isoformat(),
        "install_id": user_id,
        "consent_tier": "personalization",
        "product_id": req.product_id,
        "rating": req.rating,
    }
    if req.aspects:
        event["aspects"] = req.aspects
    collector.ingest({"events": [event]})
    profile = rebuild_profile(store, collector.iter_events(TASTE_EVENTS), user_id)
    return FeedbackResponse(accepted=True, profile=profile)


@app.get("/v1/profile", response_model=TasteProfile)
def get_profile(who: Caller) -> TasteProfile:
    """What we think of YOUR taste, or an empty profile — never the seed.

    `_profile_for` falls back to the seed so a fresh install still gets a sensible ranking, and
    that is right for ranking and wrong here: this route is read by the "Your taste" card, and
    a card that shows the seed tells a drinker who has rated nothing that they lean hoppy with
    an ABV band of 5-9%. It is about them or it is empty, and `version: 0` is how it says empty.
    """
    learned = load_profile(_state["store"], who.id)
    return learned or TasteProfile(user_id=who.id, version=0)


@app.post("/v1/profile/rebuild", response_model=TasteProfile)
def rebuild(who: Caller) -> TasteProfile:
    """Recompute from the whole event log — the batch job's entry point, and the repair
    path if a profile is ever suspect."""
    user_id = who.id
    collector: TelemetryCollector = _state["telemetry"]
    return rebuild_profile(_state["store"], collector.iter_events(TASTE_EVENTS), user_id)


#: What a contribution's bronze documents are filed under. Its own source, beside `ttb-cola`
#: and `off`, because that is exactly what it is: a source of claims about drinks, with its own
#: reliability. Nothing downstream should ever confuse a typed name with a regulatory filing.
CONTRIBUTION_SOURCE = "user_contribution"

#: Most a single account may contribute while this process is up. A loop guard, not abuse
#: control: it stops a client that retries forever from filling bronze, and it forgets
#: everything on restart. Real abuse control has to be persistent and is not this.
CONTRIBUTION_CAP = 200


@app.post("/v1/contribute", response_model=ContributionResponse)
def contribute(req: ContributionRequest, who: Caller) -> ContributionResponse:
    """Record a drink the catalog could not place, as told by the person who saw the label.

    It lands in **bronze**, not gold. A user-typed name is a claim about a drink, and bronze is
    where claims live until someone promotes them — the same path OFF rows and TTB filings take.
    Writing to gold here would let any anonymous account put a row into the thing the resolver
    draws from, unreviewed, which is a different feature and a much more dangerous one.

    Idempotent on the client's id: the bronze doc id is derived from it, so the phone can retry
    a submission whose response it never saw and still leave one document. The answer says
    whether it was new, because the phone's next move — stop keeping it — is the same either way.
    """
    store: Store = _state["store"]
    submitted = _state.setdefault("contributions_by_account", {})
    seen: set[str] = submitted.setdefault(who.id, set())
    if len(seen) >= CONTRIBUTION_CAP and req.id not in seen:
        raise HTTPException(status_code=429, detail="too many contributions")

    # Keyed on the account too: two installs that happen to generate the same client id are two
    # contributions, and one account cannot overwrite another's by guessing an id.
    natural_key = f"{who.id}::{req.id}"
    did = doc_id(CONTRIBUTION_SOURCE, natural_key)
    duplicate = store.get_bronze(did) is not None

    payload = req.model_dump(mode="json")
    # Who said so, and when we were told. `created_at` in the payload is the phone's clock and
    # is kept as given; `received_at` is ours, and is the one to trust when they disagree.
    payload["account_id"] = who.id
    payload["received_at"] = datetime.now(UTC).isoformat()
    store.put_bronze(BronzeDoc(id=did, source_id=CONTRIBUTION_SOURCE, natural_key=natural_key,
                               fetched_at=payload["received_at"], url=None, payload=payload))
    seen.add(req.id)
    return ContributionResponse(accepted=True, doc_id=did, duplicate=duplicate)


@app.post("/v1/telemetry")
async def telemetry(batch: dict) -> dict:
    """Own-collector ingest. Accepts a gzipped-or-plain batch of events from the client.
    Behavioral data is the monetizable asset, so we never route it to a vendor SDK."""
    collector: TelemetryCollector = _state["telemetry"]
    accepted = collector.ingest(batch)
    # Ratings can arrive in a batch upload, so the flywheel has to turn here too —
    # refresh only the installs this batch actually carried taste signal for.
    touched = {
        ev.get("install_id")
        for ev in accepted
        if ev.get("name") in TASTE_EVENTS and ev.get("install_id")
    }
    for install_id in touched:
        rebuild_profile(_state["store"], collector.iter_events(TASTE_EVENTS), install_id)
    return {"accepted": len(accepted)}


def _profile_for(user_id: str) -> TasteProfile | None:
    """A learned profile beats the seed as soon as it has a real centroid; before that the
    seed answers, so a fresh install still gets personalized-looking scores.

    The fallback is deliberately not keyed on `user_id`. It used to be, which was harmless
    only while every client sent the literal id "demo" — the moment a real install
    identified itself it matched no seed, got no profile, and every product scored a flat
    0.5. The seed is a starting point for anyone who has not rated anything yet, not a
    profile that belongs to one id.
    """
    learned = load_profile(_state["store"], user_id)
    if learned is not None and learned.sensory_ideal is not None:
        return learned
    return _state["profiles"].get(user_id) or _state["profiles"].get("demo")


@app.post("/v1/hooks/parallel")
async def parallel_webhook(payload: dict) -> dict:
    """Receiver for Parallel Monitor/Task webhooks (sentinel hits). Stub: log + ack.
    Real impl verifies the signature, dedupes, and fans out to alert delivery."""
    _state.setdefault("sentinel_hits", []).append(payload)
    return {"received": True}


def _demo_profile() -> TasteProfile:
    from bcd_schema import SensorySource, SensoryVector

    return TasteProfile(
        user_id="demo",
        version=1,
        style_affinities={"Malt Beverage (Ale)": 0.6, "Whisky (Bourbon)": 0.3},
        abv_band_min=5.0,
        abv_band_max=9.0,
        novelty_appetite=0.7,
        sensory_ideal=SensoryVector(
            source=SensorySource.RECONCILED,
            confidence=0.6,
            axes={"citrus": 0.8, "tropical": 0.9, "piney_resinous": 0.6,
                  "bitterness": 0.6, "body_fullness": 0.5},
        ),
    )
