"""Give the TTB rows whose fanciful name was a filer's NONE their real name back.

TTB's export leaves the fanciful-name field empty when a label has none. Some filers type
NONE into the box instead, and the connector read that as a word, so the catalog held
"Dewar's White Label None", "Knockando None", "Sierra Nevada None" -- 62 rows. The same
title-casing pass also turned the "N/A" of a non-alcoholic label into "N/a": "Bud Light N/a",
"10 Barrel Brewing Co. N/a IPA" -- 155 more. Both are fixed in the connector
(`ttb_cola._is_placeholder`, `_KEEP_UPPER`); this script brings the rows already in the
catalog up to what a fresh ingest would land.

It is the reprocess-from-bronze pattern, narrowed to the rows it concerns: for each one,
the raw filing (bronze) is run through the connector's `normalize()` again, the product's
name is re-derived from that, and only the name changes -- every field a later pass added
to the row (sensory vector, aliases, taste profile, curated style) is kept. A row someone
has renamed by hand since is left alone and reported. Silver is rewritten for the touched
rows too, so a later `--promote-only` agrees with what is here.

Idempotent: a second run finds every name already right and changes nothing. Writes a
backup of every gold row it touches to data/fix_ttb_placeholder_names_backup.json before
the first write. Run from the repo root with the API's environment (.env is read):

    .venv/bin/python scripts/fix_ttb_placeholder_names.py            # dry run: writes nothing
    .venv/bin/python scripts/fix_ttb_placeholder_names.py --apply    # write it

`--placeholder-only` limits both modes to the NONE rows and leaves the N/a casing as is.

Then delete data/label_index.pkl and restart the API: the label index is cached by row
count, and a rename does not change the count.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest"):
    sys.path.insert(0, str(ROOT / p))

import psycopg  # noqa: E402
from bcd_api.app import _load_dotenv  # noqa: E402
from bcd_ingest.connectors.ttb_cola import (  # noqa: E402
    TTBColaConnector,
    _display_name,
    _is_placeholder,
)
from bcd_ingest.pg_store import PostgresStore  # noqa: E402
from bcd_ingest.store import BronzeDoc, open_store  # noqa: E402

BACKUP = Path("data") / "fix_ttb_placeholder_names_backup.json"

# A coarse net over the raw filings; the connector's own functions decide below. The
# regexes only keep this from walking 534k products in Python.
_CANDIDATES = """
SELECT s.id, s.record, b.id, b.natural_key, b.fetched_at, b.url, b.payload
FROM silver s JOIN bronze b ON b.id = s.bronze_id
WHERE s.source_id = 'ttb-cola-registry' AND s.entity_type = 'cola'
  AND (b.payload->>'fanciful_name' ~* '^[^[:alnum:]]*none[^[:alnum:]]*$'
       OR b.payload->>'fanciful_name' ~ '^[-–— ]+$'
       OR b.payload->>'fanciful_name' ~ '(^|\\s)N/A(\\s|$)|(^|\\s)N\\.A\\.?(\\s|$)'
       OR b.payload->>'brand_name' ~ '(^|\\s)N/A(\\s|$)|(^|\\s)N\\.A\\.?(\\s|$)')
"""

GROUPS = {
    "placeholder": "fanciful name was a filer's NONE -- the brand stands alone",
    "casing": '"N/A" of a non-alcoholic label was lower-cased to "N/a"',
}


@dataclass
class Fix:
    pid: str
    group: str
    old: str
    new: str
    silver_id: str
    silver: dict


Left = tuple[str, str, str]                     # (id, stored name, why it was left alone)


def _old_display_name(brand: str | None, fanciful: str | None) -> str:
    """`ttb_cola._display_name` as it was before the placeholder gate: what the connector
    wrote for these rows, so a row still carrying that name is one nobody has touched."""
    brand = (brand or "").strip()
    fanciful = (fanciful or "").strip()
    if not fanciful:
        return brand
    if brand and brand.lower() not in fanciful.lower():
        return f"{brand} {fanciful}"
    return fanciful


def plan(store: PostgresStore, conn: TTBColaConnector, *,
         placeholder_only: bool) -> tuple[list[Fix], list[Left]]:
    fixes: list[Fix] = []
    left: list[Left] = []
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        rows = cur.execute(_CANDIDATES).fetchall()
    for silver_id, old_silver, bid, key, fetched, url, payload in rows:
        doc = BronzeDoc(bid, "ttb-cola-registry", key, fetched.isoformat(), url, payload)
        rec = conn.normalize(doc)[0]
        if not rec.get("brand_name"):
            continue                                  # promote() skipped it too
        raw = (payload.get("fanciful_name") or "").strip()
        group = "placeholder" if raw and _is_placeholder(raw) else "casing"
        if placeholder_only and group != "placeholder":
            continue
        new = _display_name(rec["brand_name"], rec.get("fanciful_name"))
        pid = f"ttb:{rec['ttb_id']}"
        gold = store.get_gold(pid)
        if gold is None or gold.get("redirects_to"):
            continue                                  # never promoted, or merged away since
        old = gold.get("name") or ""
        if old == new:
            continue                                  # already right
        # The name the old connector gave this row, from the silver it wrote. A row carrying
        # anything else was renamed by hand since, and a rename by hand outranks a re-derive.
        was = _old_display_name(old_silver.get("brand_name"), old_silver.get("fanciful_name"))
        if old.lower() != was.lower():
            left.append((pid, old, f"renamed since (the connector wrote {was!r})"))
            continue
        fixes.append(Fix(pid, group, old, new, silver_id, rec))
    fixes.sort(key=lambda f: (f.group, f.new.lower()))
    return fixes, left


def show(store: PostgresStore, fixes: list[Fix], left: list[Left]) -> None:
    for group, why in GROUPS.items():
        batch = [f for f in fixes if f.group == group]
        if not batch:
            continue
        print(f"\n== {why} -- {len(batch)} rows")
        for f in batch:
            # A brand-only name may already be in the catalog under another filing; the
            # resolver folds same-brand, same-name rows into one overlay, so this is a note,
            # not a problem -- but worth seeing.
            others = len(store.products_named(f.new, limit=50))
            same = (f"   (joins {others} row{'s' if others != 1 else ''} already named this)"
                    if others else "")
            print(f"{f.pid:<22} {f.old!r}  ->  {f.new!r}{same}")
    if left:
        print(f"\n-- left alone: {len(left)} rows renamed by hand since")
        for pid, name, why in left:
            print(f"{pid:<22} {name!r}: {why}")
    print(f"\n{len(fixes)} rows to rename, {len(left)} left alone")


def apply(store: PostgresStore, fixes: list[Fix]) -> None:
    if not fixes:
        print("nothing to write")
        return
    # The first picture of every row is the one to keep.
    backup = json.loads(BACKUP.read_text()) if BACKUP.exists() else {}
    fresh = {f.pid: store.get_gold(f.pid) for f in fixes if f.pid not in backup}
    if fresh:
        backup.update(fresh)
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        BACKUP.write_text(json.dumps(backup, indent=1, ensure_ascii=False))
        print(f"backup ({len(fresh)} rows added) -> {BACKUP}")

    t = time.time()
    for f in fixes:
        store.put_silver(f.silver_id, "ttb-cola-registry", "cola", f.silver["bronze_id"], f.silver)
        gold = store.get_gold(f.pid)
        gold["name"] = f.new
        store.put_gold(f.pid, "product", gold)
    # `put_gold` leaves the match column alone; a renamed row would go on matching under
    # its old name until this runs.
    n = store.refresh_search_names(ids=[f.pid for f in fixes])
    print(f"renamed {len(fixes)} rows, refreshed {n} search names ({time.time() - t:.0f}s)")

    print("\n-- after --")
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        none = cur.execute(
            "SELECT count(*) FROM gold WHERE entity_type='product' AND id LIKE 'ttb:%' "
            "AND name ~ ' None$' AND name !~* 'for none$'").fetchone()[0]
        na = cur.execute(
            "SELECT count(*) FROM gold WHERE entity_type='product' AND id LIKE 'ttb:%' "
            r"AND (name ~ '(^| )N/a( |$)' OR name ~ '(^| )N\.a\.?( |$)')").fetchone()[0]
    print(f"TTB products still ending in ' None': {none}; still carrying 'N/a' or 'N.a.': {na}")


def main(argv: list[str]) -> int:
    write = "--apply" in argv
    placeholder_only = "--placeholder-only" in argv
    os.chdir(ROOT)
    _load_dotenv()
    store = open_store(root="./data")
    if not isinstance(store, PostgresStore):
        print("this script reads the raw filings in Postgres; set BCD_DATABASE_URL",
              file=sys.stderr)
        return 1
    conn = TTBColaConnector(store=store, use_fixture=False)

    t = time.time()
    fixes, left = plan(store, conn, placeholder_only=placeholder_only)
    show(store, fixes, left)
    print(f"(planned in {time.time() - t:.0f}s)")
    if not write:
        print("\n(dry run: nothing written; add --apply to write)")
        return 0
    apply(store, fixes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
