# 01 — Architecture

## System at a glance

```
                         ┌─────────────────────────────────────────────┐
   iPhone (BCDApp)       │                 Backend                      │
 ┌──────────────────┐    │                                              │
 │ VisionKit scan   │    │  ┌────────────┐   ┌──────────────┐           │
 │  OCR + barcode   │───►│  │  API (FastAPI)│  │  Resolver     │          │
 │ HUD overlays     │◄───│  │ /scan/resolve│─►│ trigram+vector│          │
 │ provenance recpt │    │  │ /product     │  │ + cold scorer │          │
 │ LLMProvider      │    │  │ /recommend   │  └──────┬───────┘           │
 │ Telemetry queue  │───►│  │ /telemetry   │         │                   │
 └──────────────────┘    │  │ /hooks/paral.│         ▼                   │
                         │  └──────┬───────┘   ┌──────────────┐          │
                         │         │            │ Postgres 16  │          │
   Parallel.ai           │  ┌──────▼───────┐    │ pgvector     │          │
 ┌──────────────┐        │  │  Ingest       │──►│ PostGIS      │          │
 │ Monitor      │───────►│  │ bronze→silver │    │ (gold)       │          │
 │ FindAll      │  hooks │  │      →gold    │    └──────────────┘          │
 │ Task/Search  │◄───────│  │  connectors   │    ┌──────────────┐          │
 └──────────────┘        │  └──────┬───────┘    │ R2/S3 blobs  │          │
                         │         │             │ raw html,    │          │
   Data sources          │  ┌──────▼───────┐    │ label images │          │
 (registry, 90 defs)─────┼─►│ Crawler       │───►└──────────────┘          │
                         │  │ robots+policy │                              │
                         │  │ evidence log  │    ┌──────────────┐          │
                         │  └──────────────┘     │ Enrich        │          │
                         │                        │ chem→sensory │          │
                         │                        └──────────────┘          │
                         └─────────────────────────────────────────────┘
```

## Data flow: a scan

1. **On-device** — `DataScannerViewController` (VisionKit) emits text + barcodes at frame rate with normalized bounding boxes. Zero network. `ScanCoordinator` dedupes stable text so we don't re-query it.
2. **Barcode → cache** — resolved against an on-device SQLite/FTS index of the top ~50k products for a <100ms overlay, offline. (Bar basements have no signal; this is where competitors break.)
3. **Text → `/v1/scan/resolve`** — fresh detections batch to the server.
4. **Resolver** — trigram + pgvector match against the product index → ranked candidates. Each gets a **personal score** from the user's `TasteProfile`. Cold products are scored from **chemistry alone** (the moat) and flagged.
5. **Cold path** — anything unresolved goes to a cloud LLM with the venue's known menu as context.
6. **HUD** — overlays stream onto their bounding boxes, color-coded by predicted enjoyment.

Latency budget: barcode **<100ms** (on-device), text line **<400ms p50** to first overlay.

## Data flow: ingestion (medallion)

`fetch → bronze (immutable raw + fetch metadata) → normalize → silver (per-source) → promote → gold (canonical, resolved)`.

Never lose the raw bytes; **every gold field traces back to a bronze document id.** Entity resolution (normalize → block on brand×producer → embedding similarity → LLM adjudication → human review) happens at the silver→gold boundary, and every merge persists `match_evidence` so it's auditable and reversible. Entity resolution — not the ML — is the real cost; budget time there.

## Storage

| Store | Use | Status |
|---|---|---|
| Postgres 16 + **pgvector** (cosine, HNSW) + **pg_trgm** (trigram) | canonical entities; sensory-vector ANN; fuzzy name match | **live** — [pg_store.py](../services/ingest/bcd_ingest/pg_store.py) |
| Postgres + **PostGIS** | venue geo (`amenity=bar`, `shop=alcohol`) | deferred — venue lat/lon held in plain columns for now |
| R2 / S3 | raw HTML, label images, menu photos (via git-lfs locally) | planned |
| ClickHouse (later) | telemetry at volume; starts Postgres-partitioned | later |
| Redis + `arq` | crawl budgets + sentinel schedules | planned |

Both backends implement one `Store` interface; **`open_store()`** selects Postgres when `BCD_DATABASE_URL` (or `BCD_STORE_BACKEND=postgres`) is set and otherwise falls back to a single-file SQLite medallion store ([store.py](../services/ingest/bcd_ingest/store.py)), so the whole pipeline runs on a laptop with no server. The two search operators are backend-parametric behind the same method names: `match_products` is real `pg_trgm` similarity on Postgres and token-overlap on SQLite; `nearest_by_sensory` is a `pgvector` `<=>` ANN on Postgres and an in-python cosine on SQLite. PostGIS is deferred (its Homebrew bottle targets pg17 while the dev service is pg16), so venue geo lives in plain `lat`/`lon` columns until a geometry column + GiST index land.

## Toolchain ceiling (why it's in the architecture)

The reference machine is a **2018 MacBook Pro** — permanently capped at **Xcode 26.0 / iOS 26 SDK** (it can't run macOS Tahoe, so never Xcode 26.4+/27). iOS 26 covers everything v1 needs. Consequences that shaped the design:

- Core logic lives in **`BCDKit`, a SwiftPM package that builds and tests on the macOS host** — the app is never the only way to verify Swift code.
- Every LLM call is behind **`LLMProvider`** (cloud default, on-device optimization, mock for tests), because Apple Intelligence doesn't run in the Intel Simulator.
- CI must run on an Apple-silicon **`macos-26`** runner so releases aren't hostage to this laptop.

## Key modules

| Concern | Module |
|---|---|
| Canonical model | [packages/schema](../packages/schema/bcd_schema) — `Provenance`, `RecipeGraph`, `SensoryVector`, entities |
| Crawl posture | [packages/crawler/policy.py](../packages/crawler/bcd_crawler/policy.py) |
| Ingest | [services/ingest](../services/ingest/bcd_ingest) |
| Resolve + score | [services/api/resolver.py](../services/api/bcd_api/resolver.py) |
| Cold-start sensory | [services/enrich](../services/enrich/bcd_enrich) |
| Sentinels | [services/sentinel](../services/sentinel/bcd_sentinel) |
| iOS core | [ios/BCDKit](../ios/BCDKit) |
