# 02 — Data sources

The machine-readable registry is [data/registry/sources/](../data/registry/sources) — one YAML per source, validated against [schema.json](../data/registry/schema.json) by `make validate-registry`. Regenerate from the curated master list with `make seed-registry` ([scripts/seed_registry.py](../scripts/seed_registry.py)).

**90 sources today**, tiered by how we may use them:

| Tier | Meaning | Count | Posture |
|---|---|---|---|
| **A** | open / government / public-domain | 22 | ingest freely; this is the spine |
| **B** | academic / research release | 12 | ingest under research terms |
| **C** | commercial / licensable | 10 | quote & license; never scrape |
| **D** | consumer web & communities | 46 | crawl **robots.txt-gated**; ToS-restricted → `blocked` |

The canonical catalog is buildable from **Tier A/B alone**, so any Tier-D takedown degrades quality without killing the product.

## The bootstrap spine (Tier A)

| Source | Gives | Why it's foundational |
|---|---|---|
| **TTB Public COLA Registry** | every US label approval since 1999: brand, fanciful name, class/type, ABV, permittee, **label images** | the US SKU universe. Public domain. Highest-authority provenance (`regulatory_filing`). |
| **COLA Cloud** | the same, pre-parsed: **2.6M+** records, barcodes decoded, ABV OCR'd, 100+ fields | buys back ~2 months of scraping+OCR. Priced on contact — quote week 1. |
| **Open Brewery DB** | brewery/brewpub/bottleshop entities + geo | free producer + venue spine, no auth. **Connector implemented.** |
| **Open Food Facts** | barcode → ingredients, ABV, allergens (ODbL) | the only large **open** barcode→ingredient corpus — serves scan-a-can. **Connector implemented.** |
| **EU wine e-labels** | legally-mandated **real ingredient lists** behind per-bottle QR (Reg 2021/2117) | ground-truth ingredient data at scale; a QR our scanner already reads |
| **Vinmonopolet / Systembolaget / Alko** | Nordic monopoly catalogs incl. spirits | clean structured product data |
| **BJCP / Brewers Association** | styles w/ OG/FG/IBU/SRM/ABV + sensory | the prior distribution for every inference |
| **Google Patents / USPTO** | brewing/distilling **process** patents; hop plant patents w/ chemistry | legally-published proprietary detail |
| **Wikidata** | brand/producer entities, aliases, **ownership** | entity-resolution backbone (who owns whom) |
| **OpenStreetMap / Overture** | bars, breweries, bottle shops w/ geo | free venue spine, no POI bill |

## The sensory bridge (Tier B)

The **SNAP BeerAdvocate** (1.5M reviews) and **RateBeer** (~3M) corpora are the only large **aspect-rated** labels linking "what's in it" to "what it tastes like." They train the ingredient→sensory model that lets us score never-reviewed products. Plus recipe corpora (Brewer's Friend ~75k / 180k+, 400k+ BeerXML files), hop-chemistry datasets, and the Meilgaard sensory literature. See [03-data-model.md](03-data-model.md).

## What we license, not scrape (Tier C)

**VIP (Vermont Information Processing)** sells three-tier distributor data directly — the answer to "where do we get inventory?" **Provi/SevenFifty** is buyer-side (needs a licensed retail account). **Untappd for Business**, **BeerMenus**, **DigitalPour** and other taplist platforms each expose live draft feeds under contract. All Q2 line items, not bootstrap paths.

## Crawl targets (Tier D)

Highest-value: **~9,900 US brewery + ~3,000 distillery websites** publishing *stated* ingredients, hop bills, mash bills, barrel programs — first-party fact, not inference. Plus homebrew **clone recipes** (HomebrewTalk, Brewer's Friend) as a shortcut to commercial recipes, spirits communities (Whiskybase, Distiller, Breaking Bourbon), and ingredient-supplier spec sheets (Yakima Chief, yeast labs).

**ToS-restricted (`status: blocked`)**: Untappd, BeerAdvocate, RateBeer. Catalogued for context, **not crawled**. Their historical review data is available *legally* via the SNAP academic release.

## Discovery + sentinels (Parallel.ai)

Our own crawler does volume at ~$0.0001/page; **Parallel** does what it can't:

| Job | API | When |
|---|---|---|
| find new sources | **FindAll** | nightly (25/hr cap), never interactive — gated beta, request access |
| release/collab/drop watches | **Monitor** `event_stream` | webhook → `/v1/hooks/parallel` |
| menu drift | **Monitor** `snapshot` | daily |
| recipe recovery w/ **per-field citations** | **Task** | on demand |
| page fetch (JS/PDF) | **Extract** / **Search** | long tail |

Definitions live in [sentinels/](../sentinels). The key is confirmed working via Search + Task (`make sentinel-dryrun LIVE=1`).
