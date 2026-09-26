"""Stop import permits naming the bottles they didn't make.

Searching Lagavulin returned rows whose producer read "Hennessy", "Crown Royal", "Johnnie
Walker" and "The Macallan". None of that is a bug in search. `CT-I-15074` is a basic IMPORTER
permit -- it licences bringing a bottle into the country, which is no claim about who made it --
and `permit_producers.rekey_to_permits` names every permit after its best-filed brand, which is
right for a brewery's notice and false for an importer's. One Diageo import permit holds
Lagavulin, Talisker, Crown Royal and Johnnie Walker, and Johnnie Walker won the vote.

TTB gives no way to name the importer: `permittee` is populated on 5 of 1,049,670 rows and
nothing else in the filing names a company. So the brand becomes the producer -- the best maker
signal the row has -- keyed on the brand rather than the permit, which also gathers a brand's
range back together across the seven importers who file Lagavulin. The country comes with it,
and only where a brand's filings agree on one: the old majority vote across an importer's whole
portfolio is what put "Scotland" on Crown Royal.

`DSP-` and `BR-` permits are left alone even where one holds a thousand unrelated brands,
because those are real plants and nothing in the data tells a co-packer from a house --
`DSP-OH-22` is "Colonial Club" and bottles Jefferson's; `DSP-MN-22` is "Phillips" and bottles
1,770 that really are Phillips's. Both name themselves on about 4% of their own products.

Idempotent: a second run finds every row already pointed at its brand and moves nothing. Run
from the repo root with the API's environment (.env is read):

    .venv/bin/python scripts/fix_importer_producers.py            # dry run: writes nothing
    .venv/bin/python scripts/fix_importer_producers.py --apply    # write it

The dry run takes about 20 seconds and prints what would move, per category, with a sample of
before-and-after producer names. Writing takes about 5 minutes.

Measured on a copy of the catalog first (`createdb -T bcd bcd_scratch`, then this script against
it, then `scripts/replay_scans.py` either side): 113,662 rows move, 82,060 onto their brand and
31,602 to no producer at all; producers go 38,845 -> 51,274. On the 1,927-frame replay log 11
frames change, +6 drawn -- Campari +2, `Colonel E.h. Taylor Small Batch` +2, Modelo +3, and
`The Alchemist Heady Topper` 170 -> 168. `permit_producers` records why those two are the price.

Names change and the producer count changes, so the label index is stale afterwards. Restart the
API and it rebuilds (~80s) because the gold row count no longer matches the cached signature --
check that the rebuild happened before trusting a scan.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest"):
    sys.path.insert(0, str(ROOT / p))

import psycopg  # noqa: E402
from bcd_api.app import _load_dotenv  # noqa: E402
from bcd_ingest.connectors.ttb_cola import _titlecase  # noqa: E402
from bcd_ingest.permit_producers import (  # noqa: E402
    _BRAND_PRODUCER_ID,
    IMPORTER_PERMIT,
    NAMEABLE_BRAND,
    rekey_importers_to_brands,
)
from bcd_ingest.pg_store import PostgresStore  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402

# What the fix will touch, read through the pass's own predicates so the dry run cannot promise
# something the pass then declines to do. `becomes` is the filed brand name -- what the pass
# names the producer after -- or NULL where the brand field holds a product name and there is
# nobody to name.
_PLAN = f"""
CREATE TEMP TABLE brands AS
SELECT DISTINCT {_BRAND_PRODUCER_ID.format(col="payload->>'brand_name'")} AS producer_id
FROM bronze
WHERE source_id = 'ttb-cola-registry'
  AND coalesce(payload->>'fanciful_name','') <> '' AND {NAMEABLE_BRAND};

CREATE TEMP TABLE plan AS
SELECT DISTINCT ON (p.id)
       p.id, p.record->>'category' AS category, p.name AS product,
       pr.name AS was, f.brand AS becomes
FROM gold p
JOIN (SELECT DISTINCT 'ttb:' || (payload->>'ttb_id') AS gid,
             CASE WHEN {_BRAND_PRODUCER_ID.format(col="payload->>'brand_name'")}
                       IN (SELECT producer_id FROM brands)
                  THEN payload->>'brand_name' END AS brand
      FROM bronze
      WHERE source_id = 'ttb-cola-registry' AND {IMPORTER_PERMIT}) f
  ON f.gid = p.id
LEFT JOIN gold pr ON pr.id = p.record->>'producer_id' AND pr.entity_type = 'producer'
WHERE p.entity_type = 'product'
ORDER BY p.id, f.brand NULLS LAST;
"""


# The name the pass will actually write, so the report says what will happen. `initcap` gives
# "Dewar'S" and "President'S"; this is the connector's own apostrophe-aware casing.
def _name(brand: str | None) -> str:
    return _titlecase(brand) or "(unknown)"


#: A row whose producer name is not its own brand -- the ones the fix visibly corrects. The rest
#: are already right (a single-brand import permit) and the pass only re-keys them.
_WRONG = "lower(coalesce(was,'')) IS DISTINCT FROM lower(coalesce(becomes,''))"


def show(cur: psycopg.Cursor) -> int:
    rows, named, wrong = cur.execute(
        f"SELECT count(*), count(becomes), count(*) FILTER (WHERE {_WRONG}) FROM plan").fetchone()
    print(f"rows filed on an import permit: {rows}")
    print(f"  showing a producer that is not their own brand: {wrong}")
    print(f"  moving to their brand: {named}")
    print(f"  brand field holds a product name, so no producer to name: {rows - named}")
    print("\nby category:")
    for cat, n, w, named in cur.execute(
            f"SELECT category, count(*), count(*) FILTER (WHERE {_WRONG}), count(becomes) "
            "FROM plan GROUP BY 1 ORDER BY 2 DESC"):
        print(f"   {cat or '?':8} {n:7}  wrong now: {w:6}  get a brand: {named:6}"
              f"  left unknown: {n - named}")

    # One example each, because a count alone does not show whether the new name is better.
    print("\nthe producer names losing the most bottles they never made:")
    for was, n, eg, brand in cur.execute(
            f"""SELECT was, count(*), min(product), min(becomes) FROM plan
                WHERE {_WRONG} AND was IS NOT NULL AND category = 'spirit' AND becomes IS NOT NULL
                GROUP BY 1 ORDER BY 2 DESC LIMIT 12"""):
        print(f"   {n:6}  {was[:26]:28} e.g. {eg[:32]:34} -> {_name(brand)}")

    # The reported bug, by name: searching Lagavulin returned Hennessy and Johnnie Walker.
    print("\nthe rows this was reported from:")
    for product, was, becomes in cur.execute(
            "SELECT product, was, becomes FROM plan WHERE product ILIKE 'lagavulin%'"
            " ORDER BY product LIMIT 12"):
        print(f"   {product[:40]:42} {was or '-':22} -> {_name(becomes)}")
    return wrong


def counts(cur: psycopg.Cursor) -> tuple[int, int]:
    return cur.execute("SELECT count(*) FILTER (WHERE entity_type='producer'), "
                       "count(*) FROM gold").fetchone()


def main(argv: list[str]) -> int:
    write = "--apply" in argv
    os.chdir(ROOT)
    _load_dotenv()
    store = open_store(root="./data")
    if not isinstance(store, PostgresStore):
        print("this script reads the catalog in Postgres; set BCD_DATABASE_URL", file=sys.stderr)
        return 1

    t = time.time()
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        cur.execute(_PLAN)
        moved = show(cur)
        producers, gold = counts(cur)
    print(f"\nproducers now: {producers}   gold rows: {gold}")
    print(f"(planned in {time.time() - t:.0f}s)")
    if not write:
        print("\n(dry run: nothing written; add --apply to write)")
        return 0
    if not moved:
        print("\nnothing to move")
        return 0

    t = time.time()
    print("\nwriting...")
    for step, n in rekey_importers_to_brands(store.dsn).items():
        print(f"   {step:20} {n}")
    print(f"({time.time() - t:.0f}s)")

    print("\n-- after --")
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        producers, gold = counts(cur)
        print(f"producers now: {producers}   gold rows: {gold}")
        for r in cur.execute(
                "SELECT p.name, pr.name FROM gold p JOIN gold pr ON pr.id=p.record->>'producer_id'"
                " WHERE p.entity_type='product' AND p.name ILIKE 'lagavulin%' LIMIT 6"):
            print(f"   {r[0][:44]:46} {r[1]}")
    print("\nThe label index is stale (the gold row count changed). Restart the API and let it")
    print("rebuild before trusting a scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
