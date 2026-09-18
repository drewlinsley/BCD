"""`python -m bcd_enrich` — backfill SensoryVectors (and a weak ABV) onto gold products.

For each gold product: derive a `chemistry_prior` vector from its RecipeGraph; where the recipe
carries no signal (most OFF rows — a bare label, no hop bill), fall back to a `style_prior` centroid
so the product is still scoreable. A richer existing source (reconciled/review) is left untouched.
With `--abv`, missing ABVs are filled from a typical-per-style prior, tagged
`llm_inferred_from_style_prior`. Off by default: the detail screen prints a strength as a number
with a small provenance chip, and a guessed "5.5" on half a million registry rows reads as a fact
however the chip is drawn. A row with no strength shows its style instead, which is the truth.

The TTB registry files a class/type code, not a style, and half a million rows arrived with one:
`style_prior.class_style` reads it ("Straight Bourbon Whisky" is bourbon, "Other Rum Gold Usb" a
gold rum), and `readable_style` turns it into what a person would call it, which is what the
detail screen prints -- reported from the camera as "Gosling's should be ID'd as rum" when it
printed the code (2026-09-16). Only a style the registry filed is rewritten; a curated one stays.

On Postgres the sensory write also populates the `sensory vector(25)` column (put_gold derives it
from the record), so `nearest_by_sensory` and cold-start scoring light up across the whole catalog.
`--dry-run` prints what would change and writes nothing. The API remembers catalog rows it has
served; restart it after a run so the new vectors are the ones it scores with.
"""

from __future__ import annotations

import argparse
import collections
import os
import time

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
from .style_prior import abv_from_style, detect_style, readable_style, sensory_from_style

#: The provenance method a style carries when the registry filed it (a class code).
_FILED = ExtractionMethod.REGULATORY_FILING.value
#: Rows written per transaction.
_BATCH = 2000


def _flush(store, batch: list[tuple[str, str, dict]]) -> None:
    many = getattr(store, "put_gold_many", None)
    if many is not None:
        many(batch)
    else:
        for gid, kind, rec in batch:
            store.put_gold(gid, kind, rec)
    batch.clear()


def run(root: str = "./data", dry_run: bool = False, limit: int | None = None,
        restyle: bool = True, fill_abv: bool = False) -> int:
    store = open_store(root=root)
    print(f"→ {'would enrich' if dry_run else 'enriching'} gold products in {store.db_path}")
    chem = style = abv_filled = restyled = scanned = written = 0
    styles: collections.Counter[str] = collections.Counter()
    samples: list[tuple[str, str, str]] = []
    batch: list[tuple[str, str, dict]] = []
    t0 = time.time()
    for rec in store.iter_gold("product"):
        if limit is not None and scanned >= limit:
            break
        scanned += 1
        if scanned % 50_000 == 0:
            print(f"  … {scanned} rows, {written} to write, {time.time() - t0:.0f}s", flush=True)
        product = Product.model_validate(rec)
        changed = False
        style_hint = product.style.value if product.style else None

        # --- sensory: chemistry first, style prior as the universal fallback ---
        existing = product.sensory
        # Only (re)fill when there's nothing better than a style prior already on the row.
        if existing is None or existing.source == SensorySource.STYLE_PRIOR:
            sv = sensory_from_recipe(product.recipe, style_hint=style_hint)
            source = "chemistry"
            if not sv.axes:
                sv = sensory_from_style(product.name, product.category, style_hint=style_hint)
                source = "style"
            if sv and sv.axes:
                new = sv.model_dump(mode="json")
                if rec.get("sensory") != new:
                    rec["sensory"] = new
                    changed = True
                chem, style = (chem + 1, style) if source == "chemistry" else (chem, style + 1)
        detected = detect_style(product.name, product.category, class_type=style_hint)
        styles[detected or "<none>"] += 1

        # --- style: the registry's class code, in a person's words ---
        if restyle and product.style is not None and \
                product.style.provenance.method.value == _FILED:
            readable = readable_style(product.style.value, detected)
            if readable and readable != product.style.value:
                st = rec["style"]
                if isinstance(st, dict):
                    prov = st.get("provenance") or {}
                    # The code stays on the record, in the quote, for whoever needs it back.
                    quote = prov.get("quote") or ""
                    if product.style.value not in quote:
                        prov["quote"] = f"{quote} / {product.style.value}".strip(" /")
                    st["value"] = readable
                    st["provenance"] = prov
                    changed = True
                    restyled += 1
                    if len(samples) < 40:
                        samples.append((product.name, product.style.value, readable))

        # --- ABV: fill only when missing, from the typical-per-style prior (opt-in) ---
        if fill_abv and not (product.spec and product.spec.abv_pct):
            abv = abv_from_style(product.name, product.category, style_hint=style_hint)
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
            written += 1
            if not dry_run:
                batch.append((product.id, "product", rec))
                if len(batch) >= _BATCH:
                    _flush(store, batch)

    if batch and not dry_run:
        _flush(store, batch)
    print("─" * 60)
    print(f"scanned {scanned} products: chemistry={chem} style_prior={style} "
          f"sensory total={chem + style}; abv_backfilled={abv_filled}; restyled={restyled}; "
          f"rows {'to write' if dry_run else 'written'}={written} in {time.time() - t0:.0f}s")
    print("styles:", ", ".join(f"{k}={v}" for k, v in styles.most_common(40)))
    if samples:
        print("style rewrites (sample):")
        for name, old, new in samples:
            print(f"   {name!r}: {old!r} -> {new!r}")
    store.close()
    return 0


def _load_dotenv(path: str = ".env") -> None:
    """The API's `.env` reader, so `python -m bcd_enrich` from the repo root finds the same
    catalog the API serves. Without it the store falls back to the SQLite dev files under
    `./data`, and a run there would enrich nothing anyone scans."""
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
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(prog="bcd_enrich")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--dry-run", action="store_true", help="print what would change; write nothing")
    ap.add_argument("--limit", type=int, default=None, help="only the first N products")
    ap.add_argument("--no-restyle", action="store_true",
                    help="leave the registry's class codes as the style text")
    ap.add_argument("--abv", action="store_true",
                    help="also fill a missing ABV with the style's typical value (a guess)")
    args = ap.parse_args()
    return run(root=args.root, dry_run=args.dry_run, limit=args.limit,
               restyle=not args.no_restyle, fill_abv=args.abv)


if __name__ == "__main__":
    raise SystemExit(main())
