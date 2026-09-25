"""Take the Islay centroid away from the rows that only ever said "smoked".

`_SPIRIT_RULES` mapped the bare substrings "smoky" and "smoke" to `peated_scotch`, so 520
products reached a peated-Scotch centroid at smoky_peat 0.9 without a word of peat on the
label: "1930 Smoked Rum", "Archrival Smoked Gin", "Alma Loca Original Smoked Margarita",
"Ole Smoky Dill Pickle Moonshine" and the rest of that line. Smoke is a MARK a label states,
not a style, and since the whisky vocabulary landed it reaches the vector through
`_WHISKY_MARKS` instead -- on top of whatever the row actually is, so a smoked bourbon is a
bourbon with smoke in it. The two keywords are gone from the rule.

The code change alone is not enough, and this script is why. The enrichment pass recovers the
filed class code from the end of the provenance quote, but it trusts that code only when
re-reading it reproduces what the row says NOW -- the guard that makes the read safe without a
migration. These rows say "Peated Scotch", which is the OLD answer, so the guard refuses the
code, the pass falls back to reading "Peated Scotch" as if it were a class description, and
`class_style` matches "scotch" inside it. Run the pass without this script first and a dill
pickle moonshine is restyled to "Scotch Whisky".

So the stored value is put back to what the filed code actually reads to, which restores the
invariant the guard wants: the code round-trips, the pass recovers it on the first try, and the
vector is then recomputed from the right style. That ordering is load-bearing -- this script,
then `python -m bcd_enrich`.

Only the style VALUE changes. The quote keeps the filed code, and every field a later pass
added to the row is left alone. The 151 rows that really are peated -- the ones whose label says
peat, or that come from Laphroaig, Lagavulin or Ardbeg -- are recognised and skipped.

Idempotent: a second run finds every value already right and changes nothing. Writes a backup
of every gold row it touches to data/fix_smoke_is_not_peat_backup.json before the first write.
Run from the repo root with the API's environment (.env is read):

    .venv/bin/python scripts/fix_smoke_is_not_peat.py            # dry run: writes nothing
    .venv/bin/python scripts/fix_smoke_is_not_peat.py --apply    # write it
    .venv/bin/python -m bcd_enrich                               # then recompute the vectors

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
from bcd_enrich.style_prior import detect_style, readable_style  # noqa: E402
from bcd_ingest.pg_store import PostgresStore  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402

BACKUP = Path("data") / "fix_smoke_is_not_peat_backup.json"
WRONG = "Peated Scotch"

# Only rows this pass itself wrote, and only the ones still carrying the old answer.
_CANDIDATES = """
SELECT id, record->>'name', record->>'category',
       trim(split_part(record->'style'->'provenance'->>'quote', ' / ', -1))
FROM gold
WHERE entity_type = 'product'
  AND record->'style'->>'value' = %s
  AND record->'style'->'provenance'->>'method' = 'regulatory_filing'
"""


@dataclass
class Fix:
    pid: str
    name: str
    code: str
    new: str


def plan(store: PostgresStore) -> tuple[list[Fix], list[tuple[str, str]]]:
    """Rows whose stored style is the old wrong answer, and what the filed code really reads
    to. A row the new rules still call a peated Scotch is left alone and reported."""
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        rows = cur.execute(_CANDIDATES, (WRONG,)).fetchall()
    fixes: list[Fix] = []
    kept: list[tuple[str, str]] = []
    for pid, name, cat, code in rows:
        if not code:
            continue                                   # nothing to read the class back from
        want = readable_style(code, detect_style(f"{code} {name}", cat, class_type=code))
        if not want or want == WRONG:
            kept.append((name or pid, code))           # really is peated: peat, or an Islay name
            continue
        fixes.append(Fix(pid, name or pid, code, want))
    return fixes, kept


def show(fixes: list[Fix], kept: list[tuple[str, str]]) -> None:
    print(f"rows still carrying {WRONG!r}: {len(fixes) + len(kept)}")
    print(f"  to correct .......... {len(fixes)}")
    print(f"  really peated, kept .. {len(kept)}")
    if kept:
        print("    e.g. " + "; ".join(n for n, _ in kept[:4]))
    if not fixes:
        return
    print("\nwhere they go:")
    for value, n in Counter(f.new for f in fixes).most_common():
        print(f"   {n:5}  {value}")
    print("\nsample:")
    for f in fixes[:15]:
        print(f"   {f.name[:46]:48} [{f.code[:30]:32}] -> {f.new!r}")


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
        gold = store.get_gold(f.pid)
        if gold is None or gold.get("redirects_to"):
            continue                                   # merged away since the plan was made
        style = gold.get("style")
        if not isinstance(style, dict) or style.get("value") != WRONG:
            continue                                   # someone got here first
        style["value"] = f.new                         # the quote keeps the filed code
        store.put_gold(f.pid, "product", gold)
    print(f"restyled {len(fixes)} rows ({time.time() - t:.0f}s)")

    print("\n-- after --")
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        left = cur.execute("SELECT count(*) FROM gold WHERE entity_type='product' "
                           "AND record->'style'->>'value' = %s", (WRONG,)).fetchone()[0]
    print(f"products still called {WRONG!r}: {left} (the ones that are)")
    print("\nNow recompute the vectors, which is the point of all this:")
    print("    .venv/bin/python -m bcd_enrich")


def main(argv: list[str]) -> int:
    write = "--apply" in argv
    os.chdir(ROOT)
    _load_dotenv()
    store = open_store(root="./data")
    if not isinstance(store, PostgresStore):
        print("this script reads the catalog in Postgres; set BCD_DATABASE_URL", file=sys.stderr)
        return 1
    t = time.time()
    fixes, kept = plan(store)
    show(fixes, kept)
    print(f"(planned in {time.time() - t:.0f}s)")
    if not write:
        print("\n(dry run: nothing written; add --apply to write)")
        return 0
    apply(store, fixes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
