"""Make every filed style agree with what the class code currently reads to.

The enrichment pass turns a filed TTB class code into a person's words and keeps the code at
the end of the provenance quote. Next run it recovers the code from there -- but trusts it only
when re-reading it reproduces what the row says NOW. That guard is what makes the read safe
without a migration (the quote also holds brand text the pass never wrote), and it has a blind
spot: when a rule change alters what a code reads to, the stored value IS the old answer, so the
guard refuses the code and the pass falls back to reading the stored VALUE as if it were a class
description. Where that old value happens to parse as a class, the row silently keeps it.

Measured on the live catalog (2026-09-25): **417 rows**, all of them filings whose reading the
beer vocabulary changed and whose stored value is still the answer from before it:

    200  filed as an Altbier, still printing "Lager"
     92  filed as a Rye Beer, still printing "Lager"
     72  filed as a Kölsch, still printing "Lager"
     14  Cream Ale, 10 Barleywine, 6 Scottish Ale, 3 Winter Warmer, 2 ESB ...

Nothing said so. The pass reports the restyles it managed and is silent about the ones it could
not reach, so a vocabulary can land for most of a class and quietly miss the rest.

This reads the code and writes what it says, for every product whose style the registry filed.
It is the operation the pass would perform if the guard let it: the same two expressions the
pass uses, so afterwards the guard round-trips and the pass is a fixed point again. A code is
only trusted when `class_style` recognises it as a filing, which is the guard's real intent.

Only the style VALUE changes. The quote keeps the code, a style somebody curated by hand is
never touched (its provenance is not `regulatory_filing`), and every other field on the row is
left alone. Run it after any change to the class or name rules -- that is when it has work.

Idempotent: a second run finds every value already right and changes nothing. Writes a backup of
every gold row it touches to data/resync_filed_styles_backup.json before the first write. Run
from the repo root with the API's environment (.env is read):

    .venv/bin/python scripts/resync_filed_styles.py            # dry run: writes nothing
    .venv/bin/python scripts/resync_filed_styles.py --apply    # write it
    .venv/bin/python -m bcd_enrich                             # then recompute the vectors

`--limit N` plans only the first N products, for a quick look.

No index rebuild: no name changes, so the label index and its row count are untouched. Restart
the API afterwards, which memoizes the rows it serves.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest", "services/enrich"):
    sys.path.insert(0, str(ROOT / p))

import psycopg  # noqa: E402
from bcd_api.app import _load_dotenv  # noqa: E402
from bcd_enrich.style_prior import class_style, detect_style, readable_style  # noqa: E402
from bcd_ingest.pg_store import PostgresStore  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402

BACKUP = Path("data") / "resync_filed_styles_backup.json"
_FILED = "regulatory_filing"

# Only rows this pass wrote a code into: the quote holds "<brand> / <code>".
_CANDIDATES = """
SELECT id, record->>'name', record->>'category', record->'style'->>'value',
       trim(split_part(record->'style'->'provenance'->>'quote', ' / ', -1))
FROM gold
WHERE entity_type = 'product'
  AND record->'style'->'provenance'->>'method' = %s
  AND record->'style'->'provenance'->>'quote' LIKE '%% / %%'
  AND record->'style'->>'value' IS NOT NULL
"""


@dataclass
class Fix:
    pid: str
    name: str
    code: str
    old: str
    new: str


def plan(store: PostgresStore, *, limit: int | None = None) -> list[Fix]:
    """Rows whose stored style disagrees with what their filed code reads to."""
    sql = _CANDIDATES + (f" LIMIT {int(limit)}" if limit else "")
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        rows = cur.execute(sql, (_FILED,)).fetchall()
    fixes: list[Fix] = []
    for pid, name, cat, val, code in rows:
        if not code or class_style(code) is None:
            continue                       # not a class the rules recognise: brand text, not a code
        # the two expressions the pass itself uses, so the guard round-trips afterwards
        want = readable_style(code, detect_style(name or "", cat, class_type=code))
        if want and want != val:
            fixes.append(Fix(pid, name or pid, code, val, want))
    return fixes


def show(fixes: list[Fix], scanned: int) -> None:
    print(f"filed styles carrying a class code: {scanned}")
    print(f"  disagreeing with their code ....: {len(fixes)}")
    if not fixes:
        return
    print("\nwhat changes (stored -> what the filed code says):")
    for (old, new), n in Counter((f.old, f.new) for f in fixes).most_common(30):
        print(f"   {n:5}  {old!r} -> {new!r}")
    print("\nsample:")
    for f in fixes[:12]:
        print(f"   {f.name[:44]:46} [{f.code[:26]:28}] {f.old!r} -> {f.new!r}")


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
    written = 0
    for f in fixes:
        gold = store.get_gold(f.pid)
        if gold is None or gold.get("redirects_to"):
            continue                                    # merged away since the plan was made
        style = gold.get("style")
        if not isinstance(style, dict) or style.get("value") != f.old:
            continue                                    # someone got here first
        style["value"] = f.new                          # the quote keeps the code
        store.put_gold(f.pid, "product", gold)
        written += 1
    print(f"restyled {written} rows ({time.time() - t:.0f}s)")
    print("\nNow recompute the vectors, which the new styles change:")
    print("    .venv/bin/python -m bcd_enrich")


def main(argv: list[str]) -> int:
    write = "--apply" in argv
    limit = None
    for a in argv:
        if a.startswith("--limit"):
            limit = int(a.split("=", 1)[1] if "=" in a else argv[argv.index(a) + 1])
    os.chdir(ROOT)
    _load_dotenv()
    store = open_store(root="./data")
    if not isinstance(store, PostgresStore):
        print("this script reads the catalog in Postgres; set BCD_DATABASE_URL", file=sys.stderr)
        return 1
    t = time.time()
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        scanned = cur.execute("SELECT count(*) FROM gold WHERE entity_type='product' "
                              "AND record->'style'->'provenance'->>'method' = %s "
                              "AND record->'style'->'provenance'->>'quote' LIKE %s",
                              (_FILED, "%% / %%")).fetchone()[0]
    fixes = plan(store, limit=limit)
    show(fixes, scanned)
    print(f"(planned in {time.time() - t:.0f}s)")
    if not write:
        print("\n(dry run: nothing written; add --apply to write)")
        return 0
    apply(store, fixes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
