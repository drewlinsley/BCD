"""`python -m bcd_enrich` — backfill chemistry-prior SensoryVectors onto gold products.

Loops gold products, computes `sensory_from_recipe()` from each product's RecipeGraph,
and writes the result back into the product record. On Postgres that write also populates
the `sensory vector(25)` column (put_gold derives it from the record), so
`nearest_by_sensory` lights up. Only products whose recipe actually yielded signal are
updated — a bare label with no hop bill stays sensory=None rather than getting a
meaningless all-zero vector.

This closes the loop the whole platform is built around: decompose the recipe → attach
provenance → derive a sensory estimate with ZERO reviews → serve it from a vector index.
"""

from __future__ import annotations

import argparse

from bcd_ingest.store import open_store
from bcd_schema import Product

from . import sensory_from_recipe


def run(root: str = "./data") -> int:
    store = open_store(root=root)
    print(f"→ enriching gold products in {store.db_path}")
    updated = scanned = 0
    for rec in list(store.iter_gold("product")):
        scanned += 1
        product = Product.model_validate(rec)
        style_hint = product.style.value if product.style else None
        sv = sensory_from_recipe(product.recipe, style_hint=style_hint)
        if not sv.axes:
            continue  # no recipe signal — leave sensory unset rather than store zeros
        rec["sensory"] = sv.model_dump(mode="json")
        store.put_gold(product.id, "product", rec)
        updated += 1
        top = sorted(sv.axes.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_s = ", ".join(f"{k}={v:.2f}" for k, v in top)
        print(f"  • {product.name:<28} conf={sv.confidence:.2f}  [{top_s}]")
    print("─" * 60)
    print(f"enriched {updated}/{scanned} products with a chemistry-prior sensory vector")
    store.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="bcd_enrich")
    ap.add_argument("--root", default="./data")
    args = ap.parse_args()
    return run(root=args.root)


if __name__ == "__main__":
    raise SystemExit(main())
