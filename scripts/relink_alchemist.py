"""Put The Alchemist's beers under The Alchemist.

The catalog held one brewery as three producers. TTB permit BR-VT-21045 -- twenty of its
beers, Focal Banger, Beelzebub, Luscious, Rapture, Skadoosh among them -- was named "Alena"
by the permit-naming pass, after one of its own labels. The 2018 Heady Topper filing came in
under permit BR-VT-ALC-15000, named "Focal Banger" the same way. And the 2023 filing arrived by
permittee name as "The Alchemist LLC", holding the only Heady Topper row with an ABV and the
barcode. "It's an Alchemist can" therefore reached one beer, and the maker path in the
resolver -- which tells a maker's beers apart by the shape of an unreadable wordmark -- had
nothing to tell apart.

Idempotent: a second run finds the aliases gone and the duplicate already a redirect, and
changes nothing. Writes a backup of every row it touches beside the scan data before the
first write. Run from the repo root with the API's environment (.env is read):

    .venv/bin/python scripts/relink_alchemist.py

Then delete data/label_index.pkl and restart the API, so the label index is rebuilt with
the renamed producer -- it is cached by row count, and a rename does not change the count.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest"):
    sys.path.insert(0, str(ROOT / p))

from bcd_api.app import _load_dotenv  # noqa: E402
from bcd_ingest.merge import merge_producers, merge_products  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402

CANON = "prod:ttb-cola-registry:permit:br-vt-21045"           # the Stowe brewery's permit
FOLD = [
    "prod:ttb-cola-registry:the-alchemist-llc",                # 2023 filing, by permittee name
    "prod:ttb-cola-registry:permit:br-vt-alc-15000",           # 2018 Heady filing, "Focal Banger"
    "prod:openbrewerydb:091edbcc-f9fc-484e-9c04-c18ed505e2ac",  # OpenBreweryDB: Stowe, website
]
HEADY, HEADY_DUP = "ttb:23001001000123", "ttb:18038001000337"  # keep the row with ABV + barcode
TOUCHED = [CANON, *FOLD, HEADY, HEADY_DUP]


def main() -> int:
    os.chdir(ROOT)
    _load_dotenv()
    store = open_store(root="./data")

    canon = store.get_gold(CANON)
    if canon is None:
        print(f"{CANON} is not in the store; nothing to do", file=sys.stderr)
        return 1

    backup = {i: store.get_gold(i) for i in TOUCHED}
    backup["_products_of_canon_before"] = [p["id"] for p in store.products_of(CANON, limit=100)]
    out = Path("data") / "relink_alchemist_backup.json"
    if not out.exists():                       # the first run's picture is the one worth keeping
        out.write_text(json.dumps(backup, indent=1, ensure_ascii=False))
        print(f"backup -> {out}")

    if canon.get("name") != "The Alchemist":
        canon["name"] = "The Alchemist"
        canon["name_is_a_brand"] = True
        canon["aliases"] = sorted(set(canon.get("aliases") or []) | {"Alena"})
        store.put_gold(CANON, "producer", canon)
        print("renamed permit BR-VT-21045 -> The Alchemist (alias Alena)")

    pairs = {a: CANON for a in FOLD if store.get_gold(a) is not None}
    if pairs:
        t = time.time()
        print("merge_producers:", merge_producers(store, pairs), f"({time.time() - t:.0f}s)")
    dup = store.get_gold(HEADY_DUP)
    if dup is not None and not dup.get("redirects_to"):
        t = time.time()
        print("merge_products:", merge_products(store, {HEADY_DUP: HEADY}),
              f"({time.time() - t:.0f}s)")

    canon = store.get_gold(CANON)
    prods = store.products_of(CANON, limit=100)
    heady = store.get_gold(HEADY)
    shown = ("name", "aliases", "city", "region", "website")
    print("\nproducer:", json.dumps({k: canon.get(k) for k in shown}, ensure_ascii=False))
    print(f"beers under it: {len(prods)} ->", sorted(p["name"] for p in prods))
    print("heady:", heady["name"], "| aliases:", heady.get("aliases"),
          "| under The Alchemist:", heady.get("producer_id") == CANON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
