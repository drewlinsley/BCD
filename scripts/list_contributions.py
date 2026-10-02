"""What drinkers have told us about drinks the catalog could not place.

`POST /v1/contribute` files every contribution in **bronze**, under the source
`user_contribution`, and stops there on purpose: a typed name is a claim about a drink, not a
catalog row, and promoting one is a curation decision. This script is how you see the queue of
claims waiting on that decision.

Read-only. It writes nothing, and promoting a contribution is still done by hand --
`scripts/add_crusher.py` is the worked example of what that looks like, provenance and all.

    .venv/bin/python scripts/list_contributions.py              # newest first
    .venv/bin/python scripts/list_contributions.py --full       # the whole payload
    .venv/bin/python scripts/list_contributions.py --account acct:b4f2

For each one it prints what was typed, what the camera saw at that moment, and whether the
catalog has since come to hold something by that name -- which is the first thing you want to
know, because a contribution made in September may be a row today.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest"):
    sys.path.insert(0, str(ROOT / p))

from bcd_api.app import CONTRIBUTION_SOURCE, _load_dotenv  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="print each whole payload")
    ap.add_argument("--account", default=None, help="only this account (prefix is enough)")
    args = ap.parse_args(argv)

    _load_dotenv()
    # Same knob the server reads, so this can be pointed at a throwaway store the way the API
    # can -- otherwise the only thing it can ever read is the live one.
    store = open_store(root=os.environ.get("BCD_DATA_ROOT", str(ROOT / "data")))
    try:
        docs = list(store.iter_bronze(CONTRIBUTION_SOURCE))
        if args.account:
            docs = [d for d in docs
                    if str(d.payload.get("account_id", "")).startswith(args.account)]
        docs.sort(key=lambda d: d.payload.get("received_at") or d.fetched_at, reverse=True)
        if not docs:
            print("no contributions yet")
            return 0

        print(f"{len(docs)} contribution(s), newest first\n")
        for d in docs:
            p = d.payload
            abv = f"{p['abv_pct']}%" if p.get("abv_pct") is not None else "-"
            print(f"  {p.get('received_at', d.fetched_at)[:19]}  {p.get('name', '?')!r}")
            print(f"    category {p.get('category')}   maker {p.get('maker') or '-'}   abv {abv}")
            if p.get("note"):
                print(f"    note     {p['note']}")
            if p.get("sightings"):
                print(f"    camera   {' | '.join(p['sightings'][:6])}")
            print(f"    from     {p.get('account_id', '?')}   doc {d.id}")
            # Has the catalog caught up? A name search is how a curator would check, and a hit
            # usually means promoting this is now a merge rather than a new row.
            hits = store.search_gold_products(p.get("name", ""), limit=3) if p.get("name") else []
            if hits:
                names = ", ".join(repr(h.get("name")) for h in hits)
                print(f"    catalog  {len(hits)} already matching: {names}")
            else:
                print("    catalog  nothing by that name yet")
            if args.full:
                print(json.dumps(p, indent=4, ensure_ascii=False))
            print()
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
