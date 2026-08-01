"""BCD API — the surface the iOS app talks to.

Endpoints kept thin; logic lives in Resolver and the services. Reads the local
MedallionStore so the whole thing boots with `make api` after an ingest, no server deps.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from bcd_ingest.store import MedallionStore
from bcd_schema import (
    Product,
    ProductSearchResponse,
    ResolvedProduct,
    ScanResolveRequest,
    ScanResolveResponse,
    TasteProfile,
)
from fastapi import FastAPI, Query

from .resolver import Resolver
from .telemetry_ingest import TelemetryCollector

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = MedallionStore(root="./data")
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
    store: MedallionStore = _state["store"]
    return {"ok": True, "counts": store.counts()}


@app.get("/v1/product/search", response_model=ProductSearchResponse)
def product_search(q: str = Query(..., min_length=1), limit: int = 20) -> ProductSearchResponse:
    store: MedallionStore = _state["store"]
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
    profile = _state["profiles"].get(user_id)
    t0 = time.perf_counter()
    resp = resolver.resolve(req, profile=profile)
    resp.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return resp


@app.post("/v1/recommend")
def recommend(user_id: str = "demo", limit: int = 10) -> dict:
    """Stub: rank the whole catalog for a user. Real impl uses pgvector ANN over the
    user's sensory_ideal. Here we score what's in the store and return the top-N."""
    store: MedallionStore = _state["store"]
    resolver: Resolver = _state["resolver"]
    profile = _state["profiles"].get(user_id)
    scored = []
    for rec in store.iter_gold("product"):
        product = Product.model_validate(rec)
        s, reason, cold = resolver.score(product, profile)
        scored.append({"product_id": product.id, "name": product.name,
                       "score": s, "reason": reason, "cold_start": cold})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"user_id": user_id, "results": scored[:limit]}


@app.post("/v1/telemetry")
async def telemetry(batch: dict) -> dict:
    """Own-collector ingest. Accepts a gzipped-or-plain batch of events from the client.
    Behavioral data is the monetizable asset, so we never route it to a vendor SDK."""
    collector: TelemetryCollector = _state["telemetry"]
    n = collector.ingest(batch)
    return {"accepted": n}


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
