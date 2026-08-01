# BCD — Bottles, Cans & Draft

**Point your phone at a beer fridge, a bar's tap list, or a shelf of bourbon, and get an instant, personal answer to "what would _I_ like here?"**

Today's apps fail three ways. BCD attacks all three:

| Problem | Today | BCD |
|---|---|---|
| **Interface** | Untappd makes you type, one item at a time | **Camera-first HUD** — live OCR + barcode, overlays resolved in-frame, no typing |
| **Data** | Ratings are an unverifiable popularity average | **Decomposed + verifiable** — every recovered fact carries a source, method, confidence, and the exact supporting quote |
| **Agency** | Nothing acts when an allocated drop lands | **Sentinels + agents** — standing watches on releases you'd care about, then a deep link to buy |

The moat is the data model. We decompose every drink into indexed parts — grain bill, hop chemistry, yeast, barrel, process — and attach provenance to each. Because that structure feeds the recommender, **we can score a beer with zero reviews from its recipe alone.** Nobody else can.

> This is an early scaffold: research deliverables, a machine-readable source registry, a runnable ingest→API pipeline, and a buildable iOS app. See [the roadmap](docs/07-roadmap.md).

---

## What's here, and what actually runs

Everything below runs on the reference machine (an Intel MacBook Pro, macOS 15.7, Python 3.12 + Swift Command Line Tools) **today** — no Xcode required for the core.

```bash
make venv                       # Python 3.12 venv + deps
make ingest SOURCE=openbrewerydb LIMIT=50   # LIVE pull → bronze→silver→gold
make ingest SOURCE=off LIMIT=10             # LIVE barcode→ingredient (Open Food Facts)
make api                        # FastAPI on :8000
make test                       # 18 python + 8 swift tests
make validate-registry          # 90 sources checked against the schema
make sentinel-dryrun            # validate Parallel sentinels (LIVE=1 to ping the key)
make codegen                    # regenerate Swift+Python telemetry from one spec
make ios-gen                    # generate BCDApp.xcodeproj (build needs Xcode 26)
```

A real scan resolution, against live-ingested data:

```bash
curl -s -X POST localhost:8000/v1/scan/resolve -H 'content-type: application/json' \
  -d '{"detections":[{"text":"080244009397","kind":"barcode"},
                     {"text":"Heady Topper Double IPA","kind":"text"}]}'
# → Buffalo Trace (by UPC) + Heady Topper (by OCR), each with a personal score, in ~1ms
```

---

## Layout

```
docs/                 architecture, data sources, data model, iOS, telemetry, legal, roadmap
data/registry/        90 machine-readable source definitions + JSON schema + validator
packages/schema/      pydantic canonical model — the single source of truth (Provenance, RecipeGraph, SensoryVector)
packages/crawler/     robots-aware fetcher + policy module + append-only evidence log
services/ingest/      medallion store + connectors (openbrewerydb, ttb_cola, openfoodfacts)
services/api/         FastAPI: /v1/scan/resolve, /v1/product/search, /v1/recommend, /v1/telemetry
services/enrich/      chemistry→sensory (the cold-start scorer)
services/sentinel/    Parallel Monitor + FindAll orchestration
services/telemetry/   own-collector event ingest
telemetry/events.yaml single source of truth → codegen’d Swift enum + Python allowlist
sentinels/            Parallel job definitions (releases, discovery)
ios/BCDKit/           SwiftPM core — models, APIClient, ScanEngine, LLMProvider, Telemetry (builds+tests on host)
ios/BCDApp/           SwiftUI app — camera HUD, provenance "receipt", weekly evolution
ios/project.yml       XcodeGen manifest (iOS 18 min, iOS 26 SDK features gated)
```

## Two honest constraints

- **This machine caps at Xcode 26.0 / iOS 26 SDK.** A 2018 MacBook Pro can't run macOS Tahoe, so no Xcode 26.4+/27 here ever. iOS 26 covers everything v1 needs (Foundation Models, Visual Intelligence). CI should use an Apple-silicon `macos-26` runner. See [docs/01-architecture.md](docs/01-architecture.md).
- **Apple Intelligence doesn't run in the Intel Simulator.** So the on-device LLM is behind `LLMProvider` with a cloud default and a mock — an optimization, never a dependency.

## Legal posture

The canonical catalog is built from **open, government, and academic** data (Tier A/B) so it stands alone. Consumer-web sources are crawled **robots.txt-gated**, enforced in code ([packages/crawler/policy.py](packages/crawler/bcd_crawler/policy.py)) with an audit trail. ToS-restricted sites (Untappd, BeerAdvocate, RateBeer) are catalogued as `blocked` and not crawled. Full detail in [docs/06-legal.md](docs/06-legal.md).

## Credentials

`.env` (gitignored) holds `PARALLEL_API_KEY` and friends — copy from [.env.example](.env.example). **Never commit real keys.** If a key ever lands in history, rotate it immediately.
