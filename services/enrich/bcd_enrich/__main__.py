"""`python -m bcd_enrich` — backfill SensoryVectors (and a weak ABV) onto gold products.

For each gold product: derive a `chemistry_prior` vector from its RecipeGraph; where the recipe
carries no signal (most OFF rows — a bare label, no hop bill), fall back to a `style_prior` centroid
so the product is still scoreable. A richer existing source (reconciled/review) is left untouched.
Missing ABVs are filled from a typical-per-style prior, tagged `llm_inferred_from_style_prior` so a
guess never masquerades as a stated fact.

On Postgres the sensory write also populates the `sensory vector(25)` column (put_gold derives it
from the record), so `nearest_by_sensory` and cold-start scoring light up across the whole catalog.
"""

from __future__ import annotations

import argparse

from bcd_ingest.store import open_store
from bcd_schema import (
    ExtractionMethod,
    Product,
    ProductSpec,
    Provenance,
    SensorySource,
    Sourced,
)

from . import sensory_from_recipe
from .style_prior import abv_from_style, sensory_from_style


def run(root: str = "./data") -> int:
    store = open_store(root=root)
    print(f"→ enriching gold products in {store.db_path}")
    chem = style = abv_filled = scanned = 0
    for rec in list(store.iter_gold("product")):
        scanned += 1
        product = Product.model_validate(rec)
        changed = False

        # --- sensory: chemistry first, style prior as the universal fallback ---
        existing = product.sensory
        # Only (re)fill when there's nothing better than a style prior already on the row.
        if existing is None or existing.source == SensorySource.STYLE_PRIOR:
            style_hint = product.style.value if product.style else None
            sv = sensory_from_recipe(product.recipe, style_hint=style_hint)
            source = "chemistry"
            if not sv.axes:
                sv = sensory_from_style(product.name, product.category, style_hint=style_hint)
                source = "style"
            if sv and sv.axes:
                rec["sensory"] = sv.model_dump(mode="json")
                changed = True
                chem, style = (chem + 1, style) if source == "chemistry" else (chem, style + 1)

        # --- ABV: fill only when missing, from the typical-per-style prior ---
        if not (product.spec and product.spec.abv_pct):
            abv = abv_from_style(product.name, product.category)
            if abv is not None:
                prov = Provenance(
                    source_id="style-prior",
                    method=ExtractionMethod.LLM_INFERRED_FROM_STYLE_PRIOR,
                    confidence=0.3,
                    quote="typical ABV for the inferred style",
                )
                spec = rec.get("spec")
                if not isinstance(spec, dict):
                    spec = ProductSpec().model_dump(mode="json")
                spec["abv_pct"] = Sourced[float](value=abv, provenance=prov).model_dump(mode="json")
                rec["spec"] = spec
                changed = True
                abv_filled += 1

        if changed:
            store.put_gold(product.id, "product", rec)

    print("─" * 60)
    print(f"scanned {scanned} products: chemistry={chem} style_prior={style} "
          f"sensory total={chem + style}; abv_backfilled={abv_filled}")
    store.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="bcd_enrich")
    ap.add_argument("--root", default="./data")
    args = ap.parse_args()
    return run(root=args.root)


if __name__ == "__main__":
    raise SystemExit(main())
