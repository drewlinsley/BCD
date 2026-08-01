"""`python -m bcd_ingest SOURCE [--limit N]` — run one connector end to end.

Prints bronze/silver/gold row counts so the verification step has something to assert.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .connectors import get_connector
from .store import MedallionStore


async def _run(source: str, limit: int | None, root: str) -> int:
    store = MedallionStore(root=root)
    try:
        connector = get_connector(store, source)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"→ ingesting '{source}' (limit={limit}) into {store.db_path}")
    summary = await connector.run(limit=limit)
    counts = store.counts()
    print("─" * 48)
    print(f"  run summary : {summary}")
    print(f"  store totals: {counts}")
    print("─" * 48)
    # show a couple of resolved products as proof
    sample = list(store.iter_gold("product"))[:3]
    for p in sample:
        abv = (p.get("spec", {}) or {}).get("abv_pct")
        abv_v = abv["value"] if abv else "—"
        print(f"  • {p['name']:<40} {p['category']:<8} abv={abv_v}")
    store.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="bcd_ingest")
    ap.add_argument("source", help="connector id or alias (openbrewerydb, ttb, off)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--root", default="./data")
    args = ap.parse_args()
    return asyncio.run(_run(args.source, args.limit, args.root))


if __name__ == "__main__":
    raise SystemExit(main())
