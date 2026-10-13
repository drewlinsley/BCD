"""Promote drinks drinkers told us about into the catalog, when they have earned it.

`POST /v1/contribute` files a typed drink in bronze and stops there, which was right and was
also a dead end: nothing promoted one. These are the two gates (user, 2026-10-12):

  1. it must not already be in the catalog, misspellings included;
  2. two different ACCOUNTS must have told us about it.

Three verdicts, not two, because a typed name often cannot decide. One confident catalog match
means it is already there. SEVERAL means the name does not say which, and guessing hangs the
wrong producer, flavour profile and town off the can in somebody's hand -- a bare "Crusher"
covers The Alchemist's, another brewery's own bare "Crusher", and a dozen Bone Crushers. Those
are held and printed for you to look at, along with names that identify a class of drink rather
than a drink.

What a promoted row carries, and nothing beyond it: the name the most people typed, the
category, the maker if anyone gave one, and the ABV only if everyone who gave one agreed. The
sensory vector is `sensory_from_style`, the same call the enrichment pass makes, so the row
says `style_prior` about where it came from and stays off the Discover shelves, which take
known vectors only. No description and no tasting note -- nobody has given us one, and a
sentence invented here would read on the detail screen exactly like the ones that came from
somewhere. Provenance is `user_contributed`, quoting what was typed and by how many accounts.

    .venv/bin/python scripts/promote_contributions.py            # dry run: writes nothing
    .venv/bin/python scripts/promote_contributions.py --verbose  # also every held group's rows
    .venv/bin/python scripts/promote_contributions.py --apply    # write the promotable ones

Then restart the API: the label index is built at startup, so until it is rebuilt a new row is
in the catalog but not yet scannable.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest", "services/enrich"):
    sys.path.insert(0, str(ROOT / p))

from bcd_api.app import CONTRIBUTION_SOURCE, _load_dotenv  # noqa: E402
from bcd_enrich.style_prior import sensory_from_style  # noqa: E402
from bcd_ingest.dedup import _tokens  # noqa: E402
from bcd_ingest.promote import (  # noqa: E402
    IN_CATALOG,
    NEEDS_A_LOOK,
    Contribution,
    plan,
)
from bcd_ingest.store import open_store  # noqa: E402
from bcd_schema import Product  # noqa: E402

#: Rows to pull back per query. Generous on purpose: "how many could this be" is the question
#: being asked, so a cap that bites turns the answer into an artifact of the cap. At 25 a typed
#: "Crusher" reported 25 candidates when the catalog holds 182, and The Alchemist's own Crusher
#: -- the row the drinker most likely meant -- was not among them, because the trigram search
#: ranks the rows named exactly "Crusher" first. This is a curation script run by hand; it can
#: afford the rows.
_PER_QUERY = 250


def _shortlist(store, group) -> tuple[list[dict], bool]:
    """Catalog rows that could be this group's drink.

    Several queries, unioned: the chosen name, every other spelling anybody typed, and each
    long word on its own. The last is what reaches a row through a typo -- "Heady Toper" as a
    whole string is a worse trigram match for "The Alchemist Heady Topper" than "heady" is.
    """
    queries = {group.name, *group.spellings}
    for spelling in {group.name, *group.spellings}:
        queries |= {t for t in _tokens(spelling) if len(t) >= 5}
    if group.maker:
        queries.add(f"{group.maker} {group.name}")

    found: dict[str, dict] = {}
    capped = False
    for q in queries:
        rows = store.search_gold_products(q, limit=_PER_QUERY) or ()
        capped = capped or len(rows) >= _PER_QUERY
        for row in rows:
            rec = row if isinstance(row, dict) else {}
            pid = str(rec.get("id") or "")
            if not pid:
                continue
            found.setdefault(pid, {"id": pid, "name": rec.get("name") or "",
                                   "brand": _brand_words(rec)})
    return list(found.values()), capped


def _brand_words(rec: dict) -> str:
    """The brand as words, for the brand-compatibility gate. Read off `brand_id`'s slug, which
    is already in hand -- the alternative is a lookup per candidate row."""
    bid = str(rec.get("brand_id") or "")
    slug = bid.rsplit(":", 1)[-1] if bid else ""
    return slug.replace("-", " ")


def _slug(name: str) -> str:
    d = unicodedata.normalize("NFKD", name)
    a = "".join(c for c in d if not unicodedata.combining(c)).casefold()
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", a)).strip("-") or "drink"


def _free_id(store, name: str) -> str:
    """`user:<slug>`, so the id says where the row came from the way `ttb:` and `off:` do."""
    base = f"user:{_slug(name)}"
    if store.get_gold(base) is None:
        return base
    for n in range(2, 50):
        pid = f"{base}-{n}"
        if store.get_gold(pid) is None:
            return pid
    raise RuntimeError(f"cannot find a free id for {name!r}")


def _provenance(group, quote: str) -> dict:
    accounts = len(group.accounts)
    return {"source_id": CONTRIBUTION_SOURCE, "url": None, "quote": quote,
            "method": "user_contributed",
            # The ceiling for the method, which is what it is: somebody told us.
            "confidence": 0.7,
            "extracted_at": datetime.now(UTC).isoformat(),
            "extractor_version": f"promote_contributions/{accounts}-accounts"}


def _row(store, group) -> dict:
    pid = _free_id(store, group.name)
    record: dict = {
        "id": pid,
        "name": group.name,
        # Other real names it goes by, so a scan reading one of them still lands. Typos are
        # not among them -- see `Group.aliases`.
        "aliases": list(group.aliases),
        "category": group.category,
        # Empty, not invented. Linking a typed maker to a producer row is its own job, and a
        # guessed producer puts a brewery's name and town on somebody else's beer.
        "brand_id": "",
        "producer_id": "",
    }
    if group.abv_pct is not None:
        record["spec"] = {"abv_pct": {"value": group.abv_pct,
                                      "provenance": _provenance(group, f"{group.abv_pct}%")}}
    sv = sensory_from_style(record["name"], record["category"])
    if sv and sv.axes:
        record["sensory"] = sv.model_dump(mode="json")
    return Product.model_validate(record).model_dump(mode="json")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the promotable rows")
    ap.add_argument("--verbose", action="store_true",
                    help="also print the catalog rows behind every held group")
    args = ap.parse_args(argv)

    _load_dotenv()
    store = open_store(root=str(ROOT / "data"))
    try:
        docs = list(store.iter_bronze(CONTRIBUTION_SOURCE))
        if not docs:
            print("no contributions yet")
            return 0
        contributions = [Contribution.from_bronze(d.id, d.payload) for d in docs]
        accounts = {c.account_id for c in contributions}
        print(f"{len(contributions)} contribution(s) from {len(accounts)} account(s)\n")

        # Grouped first, then each group's shortlist retrieved once, then judged.
        retrieved = {g.name: _shortlist(store, g) for g in plan(contributions, {})}
        shortlist = {name: rows for name, (rows, _) in retrieved.items()}
        groups = plan(contributions, shortlist)

        promotable = [g for g in groups if g.promotable]
        for g in groups:
            mark = "PROMOTE" if g.promotable else "hold   "
            spellings = ", ".join(repr(s) for s in g.spellings)
            print(f"  {mark}  {g.name!r}  [{g.category}]")
            print(f"           {len(g.accounts)} account(s), {len(g.members)} entr"
                  f"{'y' if len(g.members) == 1 else 'ies'}: {spellings}")
            if g.maker:
                print(f"           maker typed: {g.maker!r}")
            if g.abv_pct is not None:
                print(f"           abv agreed:  {g.abv_pct}%")
            if not g.promotable:
                print(f"           why not:     {g.held_because}")
            if g.catalog and (args.verbose or g.verdict in (IN_CATALOG, NEEDS_A_LOOK)):
                for pid, name in g.catalog[:8]:
                    print(f"             catalog  {pid:<30} {name!r}")
                if len(g.catalog) > 8:
                    print(f"             ... and {len(g.catalog) - 8} more")
                # Say so rather than letting the cap read as the answer.
                if retrieved.get(g.name, ((), False))[1]:
                    print(f"             (shortlist hit the {_PER_QUERY}-row cap, so the real "
                          "number is higher)")
            elif args.verbose:
                print(f"           catalog  nothing matched "
                      f"({len(shortlist.get(g.name, ()))} rows considered)")
            print()

        if not promotable:
            print("nothing to promote")
            return 0

        print(f"{len(promotable)} to promote:\n")
        written: list[str] = []
        for g in promotable:
            rec = _row(store, g)
            sensory = rec.get("sensory") or {}
            print(f"  {rec['id']:<34} {rec['name']!r}")
            print(f"    aliases {rec['aliases'] or '-'}")
            print(f"    sensory {sensory.get('source') or '-'} / "
                  f"{len(sensory.get('axes') or {})} axes")
            if not args.apply:
                continue
            store.put_gold(rec["id"], "product", rec)
            written.append(rec["id"])

        if not args.apply:
            print("\n(dry run: nothing written; add --apply to write)")
            return 0
        store.refresh_search_names(written)
        print(f"\nwrote {len(written)} row(s): {', '.join(written)}")
        print("\nNow restart the API so the label index picks them up -- until then they are in\n"
              "the catalog but not yet scannable.")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
