# 03 — Data model

The product idea in one sentence: **decompose every drink into indexed parts, attach provenance to each part, then prompt models with that structure.** The pydantic definitions are the single source of truth in [packages/schema/bcd_schema](../packages/schema/bcd_schema).

## Entity spine

```
Producer ──< Brand ──< Product ──< Release ──< SKU
(brewery)              (Heady      (2026        (16oz can 4pk,
                        Topper)     batch 3)     UPC — or draft)
```

Everything that isn't a stable identifier is `Sourced[T]` — a value plus a `Provenance`. An `abv_pct` isn't a float, it's a float you can defend.

## Provenance — the verifiability answer

Every recovered fact carries one ([provenance.py](../packages/schema/bcd_schema/provenance.py)):

```python
class Provenance(BaseModel):
    source_id: str
    url: str | None
    quote: str | None            # the exact supporting text
    method: ExtractionMethod     # stated_by_producer → regulatory_filing → label_ocr
                                 #  → retailer_listing → community_clone
                                 #  → review_consensus → llm_inferred_from_style_prior
    confidence: float            # 0–1, CLAMPED to a per-method ceiling
    extracted_at: datetime
```

The clamp is the integrity guarantee: an `llm_inferred_from_style_prior` fact **can never report confidence above 0.4**, so a guess can't masquerade as a stated fact. The app renders this as tappable provenance chips (`ProvenanceChip` / `ProvenanceCard`), color-coded by `trustRank`.

## RecipeGraph — the recoverable process

```python
class RecipeIngredient:            # one indexed part
    role: IngredientRole           # base_malt · dry_hop · yeast · barrel · mash_grain …
    entity_ref: str | None         # → a canonical ingredient entity (once resolved)
    raw_name: str                  # as written on the source
    quantity: float | None         # usually null — that's fine
    timing: Timing | None          # mash · boil@60 · dry-hop day 4 · aging month 18
    provenance: Provenance

class ProcessStep:                 # mash schedule, ferment, barrel spec, packaging
    barrel_ref, warehouse_position, temp_c, duration_minutes, …
```

`RecipeGraph.completeness` (0–1) tempers how much we trust a downstream recommendation and prioritizes enrichment.

## Canonical ingredient entities

Shared, deduplicated rows — one `Citra`, referenced by thousands of beers ([ingredients.py](../packages/schema/bcd_schema/ingredients.py)):

- **Hop** — alpha/beta %, cohumulone, total oil, and oil fractions (myrcene / humulene / caryophyllene / farnesene / linalool / geraniol), thiol potential, descriptors, substitutes
- **Malt** — °L/EBC, extract, diastatic power, descriptors
- **Yeast** — attenuation & temp ranges, flocculation, POF/phenolic, ester profile
- **Barrel** — wood, prior fill (ex-bourbon / oloroso / PX), char #, size, cooperage
- **WaterProfile, GenericIngredient** (fruit, spice, botanical, spirit grain)

## SensoryVector — the bridge, and the moat

25 named axes (0–1) from the ASBC/EBC flavor wheel + spirit axes ([sensory.py](../packages/schema/bcd_schema/sensory.py)). Learned two ways and reconciled:

1. **Bottom-up from chemistry** — hop oils → aroma priors; °L → color/malt descriptors; yeast esters → fruity/spicy. Implemented in [services/enrich](../services/enrich/bcd_enrich): a published Citra+Mosaic bill already yields `tropical > 0.6, citrus > 0.5`. **Needs no reviews.**
2. **Top-down from reviews** — descriptor extraction over the SNAP aspect-rated corpora.

Because (1) is review-free, a brand-new release with a published hop bill gets a real predicted score on **day zero**. That's the cold-start moat, and it's why the `ScoredCandidate.cold_start` flag is surfaced in the HUD.

`to_array()` / `from_array()` serialize to a dense vector for pgvector ANN. **Append axes only at the end** — embeddings depend on the order.

## Entity resolution

The real hard problem (not the ML). Pipeline: normalize → block on brand-trigram × producer → embedding similarity → LLM adjudication for ambiguous pairs → human review queue for low-confidence. Every merge persists a `MatchEvidence` (`method`, `score`, the two refs) so merges are auditable and reversible.
