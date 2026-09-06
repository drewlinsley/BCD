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
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from bcd_ingest.store import Store, open_store
from bcd_schema import (
    FeedbackRequest,
    FeedbackResponse,
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
from fastapi import FastAPI, Query

import httpx

from .resolver import Resolver
from .vision import MAX_IMAGE_BYTES, VisionProvider, provider_from_env
from .taste import TASTE_EVENTS, load_profile, rebuild_profile
from .telemetry_ingest import TelemetryCollector

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
    store = open_store(root="./data")
    _state["store"] = store
    _state["resolver"] = Resolver(store)
    _state["telemetry"] = TelemetryCollector(root="./data")
    # Demo profile so /v1/scan/resolve returns personalized scores out of the box.
    _state["profiles"] = {"demo": _demo_profile()}
    # None without a key. The scan path predates this and has to keep working without it.
    _state["vision"] = provider_from_env()
    yield
    store.close()


app = FastAPI(title="BCD API", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    store: Store = _state["store"]
    return {"ok": True, "counts": store.counts()}


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
def scan_resolve(req: ScanResolveRequest, user_id: str = "demo") -> ScanResolveResponse:
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
        "corroborated": resp.corroborated,
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
        pass          # diagnostics must never take the scan path down with them


_VISION_UNCONFIGURED = (
    "vision is not configured: set ANTHROPIC_API_KEY in .env and restart the API"
)


@app.post("/v1/scan/vision", response_model=ScanVisionResponse)
async def scan_vision(req: ScanVisionRequest, user_id: str = "demo") -> ScanVisionResponse:
    """Identify a frame from the picture rather than from what OCR made of it.

    The model only ever supplies *names*. Each one is then resolved by the same `Resolver`
    against the same catalog, under the same guards, and a name the catalog cannot account
    for is dropped — so a model that invents a beer produces no answer rather than a
    confident wrong one. Facts still come from the catalog; the image only improves the query.

    Errors are reported in `detail`, never raised. This runs on the camera's hot path, and a
    timeout at the vision provider must degrade to "no extra answers", not to a failed scan.
    """
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
def recommend(user_id: str = "demo", limit: int = 10) -> dict:
    """Rank the catalog for a user. When the profile has a sensory_ideal the store does
    the candidate generation — pgvector cosine ANN on Postgres, python cosine on the
    SQLite dev store — and we then score + explain each candidate. No profile vector yet
    (cold user) falls back to scanning the catalog."""
    store: Store = _state["store"]
    resolver: Resolver = _state["resolver"]
    profile = _profile_for(user_id)
    if profile is not None and profile.sensory_ideal is not None:
        # over-fetch (limit*3) so the re-score with style/ABV priors has room to reorder
        candidates = store.nearest_by_sensory(profile.sensory_ideal.to_array(), limit=limit * 3)
    else:
        candidates = list(store.iter_gold("product"))
    scored = []
    for rec in candidates:
        product = Product.model_validate(rec)
        s, reason, cold = resolver.score(product, profile)
        scored.append({"product_id": product.id, "name": product.name,
                       "score": s, "reason": reason, "cold_start": cold})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"user_id": user_id, "results": scored[:limit]}


@app.post("/v1/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest, user_id: str = "demo") -> FeedbackResponse:
    """A thumbs on one product. Recorded as a real `rating_submitted` event and then
    folded into the profile, so this convenience path and the client's batch telemetry
    upload converge on exactly the same profile."""
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
def get_profile(user_id: str = "demo") -> TasteProfile:
    """What we think of your taste. Exposed so the client can show it — and so the user
    can see the same thing we rank with, rather than an opaque score."""
    profile = _profile_for(user_id)
    return profile or TasteProfile(user_id=user_id, version=0)


@app.post("/v1/profile/rebuild", response_model=TasteProfile)
def rebuild(user_id: str = "demo") -> TasteProfile:
    """Recompute from the whole event log — the batch job's entry point, and the repair
    path if a profile is ever suspect."""
    collector: TelemetryCollector = _state["telemetry"]
    return rebuild_profile(_state["store"], collector.iter_events(TASTE_EVENTS), user_id)


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
