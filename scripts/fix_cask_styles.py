"""Restyle the spirits whose cask, or a word hiding inside another word, renamed them.

The bug, on two rows filed under the same class ("Other Imported Whisky"):

    Ichiro's Malt & Grain Refill Bourbon Barrel Finish  ->  style "Bourbon"
    Ichiro's Malt & Grain Oloroso Sherry Cask Finish    ->  style "Gold Rum"

The first read its cask as its style. The second matched "oro" inside OLOROSO. Both then led
a Discover shelf they do not belong on, which is how they were found.

`detect_style` is fixed (see `_without_cask` and `_says`). This applies the fix to rows already
in the catalog, and ONLY those rows: the spirits whose style changes because of it.

Why not `scripts/resync_filed_styles.py`, which exists for exactly this and is the right tool
after a rules change: it makes every filed style agree with the rules as they stand, and the
BEER rules still have this same class of bug unfixed -- `wit` matches "With Cherries", `ipa`
matches "DDHIPA", `keller` matches "Mikkeller", `light` matches "Lamplighter". Measured on the
live catalog it would move 6,062 rows, most of them beer being handed a wheat or IPA style by a
word inside another word. Beer needs its own pass, because the same table holds keywords that
MUST match inside a compound (`hefe` in hefeweizen, `bock` in Maibock) and keywords that must
not, so one rule cannot serve it. Until that is measured, resyncing everything would spread a
bug to fix a bug.

    .venv/bin/python scripts/fix_cask_styles.py           # dry run: writes nothing
    .venv/bin/python scripts/fix_cask_styles.py --apply

Only the style VALUE changes, and only where the fix is what changed it. A hand-curated style
is never touched: this reads only rows whose style provenance is `regulatory_filing`.

The style drives the style-prior centroid, so a row that moves shelves is also carrying the
wrong vector. Re-run the enrichment afterwards and restart the API:

    BCD_STORE_BACKEND=postgres .venv/bin/python -m bcd_enrich
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", p)
                for p in ("packages/schema", "services/ingest", "services/enrich")]

import bcd_enrich.style_prior as sp  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402

_FILED = "regulatory_filing"


def _old_says(n: str, keyword: str) -> bool:
    """What the matcher did before the fix: a bare substring, anywhere, inside any word."""
    return keyword.strip().rstrip("*") in n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (for a smoke test)")
    args = ap.parse_args()

    store = open_store()
    moves: collections.Counter = collections.Counter()
    samples: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    changes: list[tuple[str, dict, str]] = []
    seen = 0

    for rec in store.iter_gold("product"):
        style = rec.get("style") or {}
        if (style.get("provenance") or {}).get("method") != _FILED:
            continue
        cat = rec.get("category") or ""
        if cat not in ("spirit", "other"):
            continue            # beer is a separate, unmeasured pass -- see the docstring
        seen += 1
        name = rec.get("name") or ""
        quote = ((style.get("provenance") or {}).get("quote") or "")
        code = quote.split(" / ", 1)[1] if " / " in quote else None

        new = sp.detect_style(name, cat, class_type=code)
        says, without = sp._says, sp._without_cask
        sp._says, sp._without_cask = _old_says, (lambda n: n)
        try:
            old = sp.detect_style(name, cat, class_type=code)
        finally:
            sp._says, sp._without_cask = says, without
        if old == new:
            continue

        want = sp.readable_style(code, new)
        have = style.get("value")
        if not want or want == have:
            continue
        moves[(have, want)] += 1
        if len(samples[(have, want)]) < 3:
            samples[(have, want)].append(name)
        changes.append((rec["id"], rec, want))
        if args.limit and len(changes) >= args.limit:
            break

    print(f"spirit rows with a filed style ..: {seen:,}")
    print(f"restyled by the fix .............: {len(changes):,}\n")
    print("what changes (stored -> what the name says now):")
    for (have, want), n in moves.most_common(25):
        print(f"  {n:5}  {str(have)!r:26} -> {want!r}")
        for s in samples[(have, want)][:2]:
            print(f"           {s[:66]}")

    if not args.apply:
        print("\n(dry run: nothing written; add --apply to write)")
        return 0

    for gid, rec, want in changes:
        rec = dict(rec)
        rec["style"] = {**(rec.get("style") or {}), "value": want}
        store.put_gold(gid, "product", rec)
    print(f"\nwrote {len(changes):,} rows.")
    print("Now re-run the prior so their vectors match, then restart the API:")
    print("  BCD_STORE_BACKEND=postgres .venv/bin/python -m bcd_enrich")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
