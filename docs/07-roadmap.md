# 07 — Roadmap

## Done (this scaffold)

- ✅ Canonical data model with provenance, recipe graph, sensory vector ([packages/schema](../packages/schema/bcd_schema)) — 18 python tests
- ✅ Robots-aware crawler policy + evidence log ([packages/crawler](../packages/crawler/bcd_crawler))
- ✅ Medallion ingest + **3 live connectors** (Open Brewery DB, Open Food Facts, TTB COLA)
- ✅ FastAPI: scan/resolve, product/search, recommend, telemetry, Parallel webhook
- ✅ Cold-start chemistry→sensory scorer ([services/enrich](../services/enrich/bcd_enrich))
- ✅ 90-source registry + schema + validator
- ✅ Parallel sentinels wired + key confirmed (Search/Task) — FindAll pending beta access
- ✅ Telemetry spec + codegen (Swift + Python from one file)
- ✅ iOS `BCDKit` (builds + 8 tests on host) + SwiftUI app + XcodeGen manifest

## Immediate next steps (in order)

1. **Rotate both pasted credentials** — GitHub token (github.com/settings/tokens) and the Parallel key (dashboard). The repo is public.
2. **Install Xcode 26.0** on the reference machine (`make toolchain`, then sign in to Xcodes.app) and run `make ios-build` against the iOS 26 SDK.
3. **Get a COLA Cloud quote** — the fastest path to 2.6M SKUs; fall back to scraping TTB directly.
4. **Request Parallel FindAll beta access** so discovery lights up (Search + Task already work).
5. **Decide on a private submodule** for Tier-D registry entries (public-repo exposure — [06-legal.md](06-legal.md)).
6. **Stand up an Apple-silicon `macos-26` CI runner** so releases aren't capped by the Intel laptop.

## Phase 1 — real catalog (weeks)

- Swap the SQLite dev store for **Postgres 16 + pgvector + PostGIS**; port the medallion store behind the same interface.
- Bulk-load TTB/COLA Cloud; ingest Open Food Facts fully; layer Wikidata ownership.
- Build the **entity-resolution pipeline** with the human review queue (the real cost center).
- Train the **ingredient→sensory model** on the SNAP corpora; reconcile with the chemistry prior.
- Ship the **on-device offline index** (top ~50k products, SQLite + FTS5 + quantized embeddings).

## Phase 2 — the loop (weeks)

- Wire **VisionKit** live scanning + Visual Intelligence provider on device.
- Cloud LLM cold path with venue-menu context.
- Weekly evolution job: taste memo + 3 falsifiable predictions → the You card.
- Tier-1 agent: sentinel → push → deep-link checkout.

## Phase 3 — scale & monetize

- Beta = beer + spirits (locked scope); ontology already generic for wine/cider/RTD.
- Distributor data (VIP) under contract; taplist platform integrations.
- Membership tier funding agent token spend; consent-gated ads with provable audience composition.
- ClickHouse for telemetry at volume.

## Known risks (carried from planning)

| Risk | Mitigation |
|---|---|
| Entity resolution is the real cost, not the ML | human review queue + `match_evidence` from day one |
| Hard Xcode ceiling at 26.0 on this machine | iOS 26 covers v1; `macos-26` CI runner for releases |
| Apple Intelligence can't run in the Intel Simulator | `LLMProvider` protocol, cloud default, physical iPhone 15 Pro+ for on-device tests |
| Parallel FindAll gated / 25-per-hour | Search + Task work now; FindAll is a nightly job when provisioned |
| Public-repo crawl-target exposure | private submodule for Tier-D entries |
| COLA Cloud pricing unknown | quote week 1; TTB direct-scrape fallback |
