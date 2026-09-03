"""BCD API — the surface the iOS app talks to.

Endpoints kept thin; logic lives in Resolver and the services. Reads the local
MedallionStore so the whole thing boots with `make api` after an ingest, no server deps.
"""

from __future__ import annotations

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
    TasteProfile,
)
from fastapi import FastAPI, Query

from .resolver import Resolver
from .taste import TASTE_EVENTS, load_profile, rebuild_profile
from .telemetry_ingest import TelemetryCollector

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = open_store(root="./data")
    _state["store"] = store
    _state["resolver"] = Resolver(store)
    _state["telemetry"] = TelemetryCollector(root="./data")
    # Demo profile so /v1/scan/resolve returns personalized scores out of the box.
    _state["profiles"] = {"demo": _demo_profile()}
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
