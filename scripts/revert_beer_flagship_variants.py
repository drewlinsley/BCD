"""Undo the flagship beer profile on rows that are a DIFFERENT beer from the one it describes.

The 2026-09-23 flagship pass matched a brand and claimed its label variants, but its exclude
list never named the variant qualifiers -- so "Schlitz Dark", "Hamm's Ice", "Guinness Black",
"Michelob Ultra Zero" and 72 others were written with the pale flagship's profile, description
and ABV. The pattern file has since been corrected (its exclude now covers dark/black/red/ice/
dry/gold/light/zero, and skips any of those the pattern's own style, match or summary names);
these rows are the ones it would no longer claim.

Everything that pass wrote carries `source_id: "llm-profile"` provenance, and the style kept the
original in its quote ("was: Beer"), so the undo is exact: drop those fields, restore the filed
style, and put the row back on the style prior -- which now reads a beer's name far better than
it did (PR #28), so "Schlitz Dark" lands on a dark-lager centroid rather than a pale one.

Read-only unless --write is passed. Prints every row it touches.

  python scripts/revert_beer_flagship_variants.py IDFILE            # dry run
  python scripts/revert_beer_flagship_variants.py IDFILE --write
"""
from __future__ import annotations

import argparse
import sys

from bcd_enrich._env import load_dotenv
from bcd_enrich.style_prior import sensory_from_style
from bcd_ingest.store import open_store

WROTE = "llm-profile"


def _ours(node: object) -> bool:
    """Whether this field was written by the profile pass rather than the registry."""
    return (isinstance(node, dict)
            and (node.get("provenance") or {}).get("source_id") == WROTE)


def revert(store, pid: str) -> tuple[str, list[str], dict] | None:
    rec = store.get_gold(pid)
    if rec is None:
        return None
    name, undone = rec.get("name") or pid, []

    # the style: the filed value is in the quote as "was: X"
    style = rec.get("style")
    if _ours(style):
        quote = (style.get("provenance") or {}).get("quote") or ""
        if quote.startswith("was: "):
            was = quote[5:].strip()
            rec["style"] = {"value": was, "provenance": {
                "source_id": "ttb-cola-registry", "method": "regulatory_filing",
                "confidence": 1.0, "quote": was}}
            undone.append(f"style -> {was!r}")
        else:
            rec.pop("style", None)
            undone.append("style dropped")

    if _ours(rec.get("description")):
        rec.pop("description", None)
        undone.append("description")

    spec = rec.get("spec")
    if isinstance(spec, dict) and _ours(spec.get("abv_pct")):
        spec["abv_pct"] = None
        undone.append("abv")

    recipe = rec.get("recipe")
    if isinstance(recipe, dict):
        kept = [i for i in (recipe.get("ingredients") or []) if not _ours(i)]
        if len(kept) != len(recipe.get("ingredients") or []):
            undone.append(f"ingredients -{len(recipe['ingredients']) - len(kept)}")
            recipe["ingredients"] = kept

    # back onto the style prior, recomputed from the name and the style just restored
    hint = (rec.get("style") or {}).get("value")
    sv = sensory_from_style(name, rec.get("category"), style_hint=hint)
    if sv is not None:
        rec["sensory"] = sv.model_dump(mode="json")
        undone.append(f"sensory -> style_prior {sv.confidence}")
    else:
        rec.pop("sensory", None)
        undone.append("sensory dropped")
    return name, undone, rec


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("idfile", help="one product id per line")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args()

    with open(args.idfile, encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    store = open_store(root=args.root)
    print(f"{'REVERTING' if args.write else 'would revert'} {len(ids)} rows in {store.db_path}\n")
    done = missing = 0
    for pid in ids:
        out = revert(store, pid)
        if out is None:
            print(f"  ?? {pid}: no such row")
            missing += 1
            continue
        name, undone, rec = out
        print(f"  {name[:52]:52} {', '.join(undone)}")
        if args.write:
            store.put_gold(pid, "product", rec)
        done += 1

    # The raw answer stays in bronze, and `--reapply` would put it straight back, so it goes
    # too. No store method deletes bronze, so this is the one bit of direct SQL.
    if args.write:
        conn = getattr(store, "_conn", None)
        if conn is None:
            print("\n!! not Postgres -- remove the bronze answers by hand", file=sys.stderr)
        else:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM bronze WHERE source_id = %s AND natural_key = ANY(%s)",
                            (WROTE, ids))
                print(f"\nbronze answers removed: {cur.rowcount}")
            conn.commit()
    print(f"\n{done} rows {'reverted' if args.write else 'would be reverted'}"
          f"{f', {missing} not found' if missing else ''}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
