"""Fold the duplicate rows the camera keeps finding into one row each.

Every one of these clusters was watched flicker on the HUD in the 2026-09-14/15 scans: the
same bottle drawn as two names on alternate ticks, or two names at once, because TTB holds a
label-approval filing per importer and per decade, and OFF holds the barcode under a third
name. The resolver collapses rows that share a canonical brand+name and tie-breaks the rest
stably, but it cannot know that `High Life` (filed by an importer in 1983) and `Miller High
Life` (filed by the brewery in 2024) are one beer. A merge is the fix: one row, the other
names kept as aliases (they are what a label prints), every SKU repointed, and a redirect
tombstone under each old id so nothing that holds one -- a rating, a phone's cached answer --
goes dark.

Only rows a person has looked at are here, named one by one. The bulk merge was tried and
abandoned (see the catalog-merge notes): a shorter row wins a similarity contest by saying
less, and the category list holds the very words that tell products apart. `Black Seal` from
1984 is a London dry gin, not the rum, and stays.

Three producers are put right on the way, the same way `relink_alchemist.py` did it: the
permit naming pass called Miller's Milwaukee brewery "Redd's" after one of its 512 labels,
no row at all existed for the Bermuda house that blends Gosling's, whose rum sat under seven
different importers, and Campari sat under Cutty Sark -- which matters to the resolver: a
one-word label is proven by its house's line, and only its own house's.

Idempotent: a second run finds the aliases already redirects, the renames done, the producer
present, and changes nothing. `--dry-run` prints the plan and writes nothing. Writes a backup
of every row it touches beside the scan data before the first write. Run from the repo root
with the API's environment (.env is read):

    .venv/bin/python scripts/merge_scan_dupes.py --dry-run
    .venv/bin/python scripts/merge_scan_dupes.py

Then delete data/label_index.pkl and restart the API. The index is keyed on row COUNTS, and
a merge keeps the gold total constant (a product becomes a redirect under the same id), so
the API would load the old index and go on matching the folded rows by their old names; a
fresh build takes ~80 s on start.

Measured before it was handed over, by running it against a `createdb -T bcd` copy and
replaying the 1204 logged frames (scripts/replay_scans.py) on both catalogs. The first
pass was a lesson: merging the two-word `Goslings Black` away left the rum's one row fifth
and sixth in the per-line matches, and the bottle drew a gin. Three resolver rules came
out of that dry run (a name printed across two lines is found by the frame; a row whose
brand is its whole label needs every word read; a row of another kind than the label
names is not a candidate) and they ship with this script. Run it against the live catalog
only after that resolver is deployed -- on the old one it would make Gosling's worse.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest"):
    sys.path.insert(0, str(ROOT / p))

from bcd_api.app import _load_dotenv  # noqa: E402
from bcd_ingest.merge import merge_producers, merge_products  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402
from bcd_schema import Producer  # noqa: E402


@dataclass
class Cluster:
    """One product, filed several times. `canon` survives; `fold` become redirects to it."""
    what: str
    canon: str
    fold: list[str]
    rename: str | None = None          # the survivor's name, where the filing's is wrong
    producer: str | None = None        # repoint the survivor (and its brand) to this producer
    brand: str | None = None           # ...and this is that brand
    rebrand: str | None = None         # the survivor's brand row, where the filing's is wrong
    aliases: list[str] | None = None   # the survivor's aliases, set outright after the merge:
                                       # a fold's misspellings are not names a label prints


CLUSTERS = [
    Cluster(
        what="Gosling's Black Seal Bermuda Black Rum -- seven filings, seven importers, plus a "
             "1993 one named only `Goslings Black`; drawn as two names on every frame",
        canon="ttb:22028001000163",                # 2022, "Goslings Black Seal", the current logo
        fold=[
            "ttb:99176000000057",                  # 1999 "Gosling's Black Seal"
            "ttb:99350000000027",                  # 1999 "Gosling's Black Seal"
            "ttb:06138001000021",                  # 2006 "Gosling's Black Seal Rum"
            "ttb:97182000000121",                  # 1997 "Gosling's Black Seal Black"
            "ttb:97182000000120",                  # 1997 "Godling's Black Seal Black" (sic)
            "ttb:93122431",                        # 1993 "Goslings Black": what the frame path drew
        ],
        producer="prod:bcd:goslings",
        brand="brand:ttb-cola-registry:goslings-black-seal",
    ),
    Cluster(
        what="Miller High Life -- `High Life` (an importer, 1983), `Miller High Life High Life` "
             "(brand plus label, 2008), and the 1995 filing; `High Life` and `Miller High Life` "
             "alternated on the can",
        canon="ttb:24248001000650",                # 2024, brand Miller, the brewery's own permit
        fold=["ttb:953550061", "ttb:83070219", "ttb:08175001000231"],
    ),
    Cluster(
        what="Lawson's Finest Liquids Sip of Sunshine -- the 2017 filing says IPA, the 2022 one "
             "does not, and Two Roads filed it too (they brew it under contract); the can drew "
             "both Lawson's rows at once",
        canon="ttb:22269001000049",                # the name the can prints as one block
        fold=["ttb:17038001000740", "ttb:14274001000379"],
    ),
    Cluster(
        what="Lawson's Finest Liquids Little Sip -- the same pair: Waitsfield's 2023 filing and "
             "the 2021 contract filing that says IPA; the can alternated between them",
        canon="ttb:23339001000111",
        fold=["ttb:21039001000961"],
    ),
    Cluster(
        what="Ramazzotti Aperitivo Rosato -- filed again in 2026 as `Ramazzotti Rosato`, under "
             "another importer",
        canon="ttb:21335001000307",                # what the label prints
        fold=["ttb:26029001000744"],
    ),
    Cluster(
        what="Campari -- eight filings: `Campari`, `Bitter Campari`, `Campari Bitter Bitter`, "
             "`Campari Aperitivo`, `Campari Aperitico` (sic), `Italy Campari`, `Litter Campari` "
             "(sic), and the OFF row that carries the barcode and the ABV; the bottle drew "
             "nothing, and the row that should have answered was filed under Cutty Sark",
        canon="off:8000040000802",                 # the barcode's row
        fold=[
            "ttb:99140000000165",                  # "Campari", 1999, under Cutty Sark
            "ttb:99153000000210",                  # "Bitter Campari"
            "ttb:88070859",                        # "Campari Bitter Bitter"
            "ttb:90043196",                        # "Campari Aperitivo"
            "ttb:94101955",                        # "Campari Aperitico"
            "ttb:79114305",                        # "Italy Campari"
            "ttb:89031166",                        # "Litter Campari"
        ],
        rename="Campari",                          # the whole of the name on the bottle
        rebrand="brand:ttb-cola-registry:campari",
        producer="prod:bcd:campari",
        brand="brand:ttb-cola-registry:campari",
        aliases=["Campari Bitter", "Bitter Campari", "Campari Aperitivo"],
    ),
    Cluster(
        what="Modelo Negra -- filed as Negra Modelo, Negra Modelo Ale, Negra Modelo Draft (Dark "
             "Ale), and twice as Modelo Negra",
        canon="ttb:26131001000584",                # 2026, Crown Imports, the current label order
        fold=["ttb:24239001000702", "ttb:92062820", "ttb:07011000000101",
              "ttb:14041001000244", "ttb:08080000000309"],
    ),
    Cluster(
        what="Stella Artois -- the OFF row carries the barcode and the word `Beer`; the TTB row "
             "carries the name",
        canon="off:0018200261237",                 # keep the row the SKU points at
        fold=["ttb:99110000000117"],
        rename="Stella Artois",
    ),
    Cluster(
        what="Stranahan's Colorado Whiskey -- one filing truncated to `Stranahan's Colorado`, one "
             "as brand plus label plus brand, one misspelt Stronahan's",
        canon="ttb:11355001000129",
        fold=["ttb:05329003000035", "ttb:06031000000131"],
        rename="Stranahan's Colorado Whiskey",
    ),
]

# The Bermuda house. Nothing in the catalog names it: TTB names the importer of record, and
# OpenBreweryDB has no distilleries. Its rum sat under Tomatin, Winters, Mr. Boston, Burnett's,
# Evan Williams, Achaia Clauss and Tequila Ocho -- each true of some filing's paperwork, none
# of the bottle.
GOSLINGS = Producer(
    id="prod:bcd:goslings", name="Goslings", kind="distillery", country="Bermuda",
    city="Hamilton", aliases=["Gosling's", "Gosling Brothers Ltd."],
    website="https://www.goslingsrum.com",
).model_dump(mode="json")

# The Milanese house, as its label prints it under the wordmark: DAVIDE CAMPARI MILANO. OFF
# holds it three ways ("Campari", "Davide campari", "DCM S.p.A."), none with a product.
CAMPARI_HOUSE = Producer(
    id="prod:bcd:campari", name="Davide Campari-Milano", kind="distillery", country="Italy",
    city="Milan", aliases=["Davide Campari Milano", "Campari Group", "DCM S.p.A."],
    website="https://www.campari.com",
).model_dump(mode="json")

NEW_PRODUCERS = [GOSLINGS, CAMPARI_HOUSE]

# Miller's Milwaukee brewery, permit BR-WI-MIL-1, named "Redd's" by the permit-naming pass
# after one of its labels. OpenBreweryDB knows the building; folding that row in gives the
# permit its city and coordinates, as the Alchemist relink did for Stowe.
MILLER = "prod:ttb-cola-registry:permit:br-wi-mil-1"
MILLER_NAME = "Miller Brewing Company"
# OpenBreweryDB's "MillerCoors Brewing Co - Milwaukee"
MILLER_FOLD = ["prod:openbrewerydb:083550ee-a87a-40d1-bba1-7aeab8ace401"]

BACKUP = Path("data") / "merge_scan_dupes_backup.json"


@dataclass
class Plan:
    lines: list[str] = field(default_factory=list)
    touched: set[str] = field(default_factory=set)

    def say(self, s: str) -> None:
        self.lines.append(s)
        print(s)


def _name(store, gid: str) -> str:
    rec = store.get_gold(gid)
    if rec is None:
        return "<missing>"
    if rec.get("redirects_to"):
        return f"<redirect -> {rec['redirects_to']}>"
    return rec.get("name") or "<unnamed>"


def _with_alias(rec: dict, name: str | None) -> list[str]:
    """The row's aliases with `name` added -- the spelling a rename retires is still what
    some label prints."""
    return sorted(set(rec.get("aliases") or []) | ({name} if name else set()))


def _producer_of(store, rec: dict | None) -> str:
    pr = store.get_gold((rec or {}).get("producer_id") or "")
    return (pr or {}).get("name") or "?"


def plan(store) -> Plan:
    """Describe what the run would do against this store, touching nothing."""
    out = Plan()
    for c in CLUSTERS:
        canon = store.get_gold(c.canon)
        out.say(f"\n== {c.what}")
        if canon is None or canon.get("redirects_to"):
            out.say(f"   !! canonical {c.canon} is {_name(store, c.canon)}; cluster skipped")
            continue
        out.say(f"   keep  {c.canon}  {canon.get('name')!r}  ({_producer_of(store, canon)})")
        if c.rename and canon.get("name") != c.rename:
            out.say(f"   rename -> {c.rename!r}  (old name kept as an alias)")
        for a in c.fold:
            rec = store.get_gold(a)
            state = ("already a redirect" if rec and rec.get("redirects_to")
                     else "missing" if rec is None else f"({_producer_of(store, rec)})")
            out.say(f"   fold  {a}  {_name(store, a)!r}  {state}")
        if c.producer:
            out.say(f"   producer -> {c.producer} "
                    f"({'exists' if store.get_gold(c.producer) else 'to be created'})"
                    f"{', brand ' + c.brand + ' repointed too' if c.brand else ''}")
        if c.rebrand and canon.get("brand_id") != c.rebrand:
            out.say(f"   brand {canon.get('brand_id')} -> {c.rebrand} "
                    f"({_name(store, c.rebrand)!r})")
        if c.aliases is not None:
            out.say(f"   aliases -> {c.aliases}")
        out.touched.update([c.canon, *c.fold, *(x for x in (c.brand, c.rebrand) if x)])
    miller = store.get_gold(MILLER)
    out.say(f"\n== producer {MILLER}: {(miller or {}).get('name')!r} -> {MILLER_NAME!r}; fold "
            + ", ".join(f"{_name(store, f)!r}" for f in MILLER_FOLD))
    out.touched.update([MILLER, *MILLER_FOLD, *(p["id"] for p in NEW_PRODUCERS)])
    return out


def apply(store, touched: set[str]) -> None:
    # The first picture of every row is the one to keep: a row already in the backup stays
    # as it was before anything touched it, and a row a later cluster brings in is added.
    backup = json.loads(BACKUP.read_text()) if BACKUP.exists() else {}
    fresh = {i: store.get_gold(i) for i in sorted(touched) if i not in backup}
    if fresh:
        backup.update(fresh)
        BACKUP.write_text(json.dumps(backup, indent=1, ensure_ascii=False))
        print(f"\nbackup ({len(fresh)} rows added) -> {BACKUP}")

    # Producers first, so the products land under the right names.
    for house in NEW_PRODUCERS:
        if store.get_gold(house["id"]) is None:
            store.put_gold(house["id"], "producer", dict(house))
            print(f"created producer {house['id']} {house['name']!r}")
    miller = store.get_gold(MILLER)
    if miller is not None and miller.get("name") != MILLER_NAME:
        miller["aliases"] = _with_alias(miller, miller.get("name"))
        miller["name"] = MILLER_NAME
        miller["name_is_a_brand"] = True             # a name to show, not a label standing in
        store.put_gold(MILLER, "producer", miller)
        print(f"renamed {MILLER} -> {MILLER_NAME!r} (aliases {miller['aliases']})")
    pairs = {f: MILLER for f in MILLER_FOLD if store.get_gold(f) is not None}
    if pairs:
        t = time.time()
        print("merge_producers:", merge_producers(store, pairs), f"({time.time() - t:.0f}s)")

    renamed: list[str] = []
    for c in CLUSTERS:
        canon = store.get_gold(c.canon)
        if canon is None or canon.get("redirects_to"):
            continue
        if c.rename and canon.get("name") != c.rename:
            canon["aliases"] = _with_alias(canon, canon.get("name"))
            canon["name"] = c.rename
            store.put_gold(c.canon, "product", canon)
            renamed.append(c.canon)
            print(f"renamed {c.canon} -> {c.rename!r}")
        if c.producer and canon.get("producer_id") != c.producer:
            canon["producer_id"] = c.producer
            store.put_gold(c.canon, "product", canon)
            print(f"repointed {c.canon} -> producer {c.producer}")
        if c.rebrand and canon.get("brand_id") != c.rebrand and store.get_gold(c.rebrand):
            canon["brand_id"] = c.rebrand
            store.put_gold(c.canon, "product", canon)
            renamed.append(c.canon)                 # the qualified name changes with the brand
            print(f"rebranded {c.canon} -> {c.rebrand}")
        if c.producer and c.brand:
            brand = store.get_gold(c.brand)
            if brand is not None and brand.get("producer_id") != c.producer:
                brand["producer_id"] = c.producer
                store.put_gold(c.brand, "brand", brand)
                print(f"repointed brand {c.brand} -> producer {c.producer}")
        pairs = {a: c.canon for a in c.fold
                 if (rec := store.get_gold(a)) is not None and not rec.get("redirects_to")}
        if pairs:
            t = time.time()
            print(f"merge_products ({c.what.split(' -- ')[0]}):",
                  merge_products(store, pairs), f"({time.time() - t:.0f}s)")
        if c.aliases is not None:
            canon = store.get_gold(c.canon)
            if canon is not None and sorted(canon.get("aliases") or []) != sorted(c.aliases):
                canon["aliases"] = sorted(c.aliases)
                store.put_gold(c.canon, "product", canon)
                print(f"aliases {c.canon} -> {canon['aliases']}")
    if renamed:
        # `put_gold` leaves the match column alone; a renamed row would go on matching under
        # its old name.
        print("search names refreshed:", store.refresh_search_names(ids=sorted(set(renamed))))


def report(store) -> None:
    print("\n-- after --")
    for c in CLUSTERS:
        canon = store.get_gold(c.canon)
        if canon is None:
            continue
        print(f"{canon.get('name')!r} ({_producer_of(store, canon)})"
              f"  aliases={canon.get('aliases')}"
              f"  abv={((canon.get('spec') or {}).get('abv_pct') or {}).get('value')}")
        gone = [a for a in c.fold if (store.get_gold(a) or {}).get("redirects_to") == c.canon]
        print(f"   {len(gone)}/{len(c.fold)} folded rows now redirect here")
    miller = store.get_gold(MILLER) or {}
    print(f"{miller.get('name')!r}: city={miller.get('city')} region={miller.get('region')} "
          f"aliases={miller.get('aliases')}")


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    os.chdir(ROOT)
    _load_dotenv()
    store = open_store(root="./data")
    p = plan(store)
    if dry:
        print("\n(dry run: nothing written)")
        return 0
    apply(store, p.touched)
    report(store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
