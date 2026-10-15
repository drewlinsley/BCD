"""Fill a missing ABV from the open databases that publish one.

520,331 of 534,104 products do not say how strong they are, and the gap is one source: of
530,056 rows from the TTB COLA registry 9,725 carry an ABV (1.8%), while Open Food Facts is
at 91.9% (3,717 of 4,043). That is not a scraping failure to go back and fix. A COLA detail
page publishes sixteen fields -- TTB id, status, vendor code, serial, class/type code,
origin, brand name, fanciful name, application type, for sale in, total bottle capacity,
formula, approval date, qualifications, plant registry, contact -- and alcohol content is
not among them, nor is there a label image on the page to read one off. The strength is on
the physical can, and in other people's databases.

Two of those databases, both free, both citable:

  * Wikidata, where 18,564 named drinks state an alcohol by volume (P2665). Ask for labels
    in "mul" as well as a language: a brand name is language-agnostic, 18,274 of the items
    have no label in any of en/de/fr/es/it/nl, and the standard label service answers with
    the QID, which looks exactly like a nameless item.
  * the Craft Beers dataset: 2,410 canned US craft beers scraped from CraftCans.com in 2017
    for a teaching notebook, 2,348 of them with a strength -- dated, and the only free list
    that pairs a US craft can with its own.

A name is not a match. `match_typed_name` asks the catalog question -- could this row already
be this drink, misspellings included -- and three gates sit after it, each earned from a row
this pass got wrong:

  * the maker. EVERY word of the brewery's name must be in the row -- and because those words
    are also what the row is looked up BY, a row the brewery did not brew is usually never
    read at all. Without this the pass confirmed `J. Wakefield Brewing The End Of History`
    against BrewDog's famous 55% beer: `dedup._prep` drops a brand whose words all appear
    inside the product name as a mere echo, and TTB writes the brewery INTO the product name,
    so the brand gate stopped gating across most of the catalog. One shared word was then
    still enough -- `King's Reserve Bourbon Barrel Johan` was confirmed against Sun King
    Brewing on "king" alone. Set intersection is not agreement; containment is. `maker_agrees`
    stays after the lookup for the words the lookup leaves out: "Brouwerij 't IJ" is two
    letters of brewery, and an index key that short narrows nothing.
  * a plausible strength for the category, with its own band for a row whose NAME says
    non-alcoholic. Both halves earned: `Lowlander 0.00% Wit` was filled at 3.5% from
    Wikidata's alcoholic `Lowlander Wit` while the true 0.0% was held for being under the
    floor -- the one row where the number matters most, answered backwards.
  * what the row's own name states. A TTB name often carries a strength, and it got there off
    a filing: `CARLSBERG UNFILTERED 5.0 DEGRE ALCOOL` does not take 4.2% from anybody.

Several matching rows is not a match either: a name that fits three rows does not say which,
and picking one hangs a strength off the wrong can. Those are held and counted.

A fourth gate was measured and REFUSED: comparing the published strength to what the row's
style would predict. It sounds like the same kind of check and is not one. At a tolerance of
5 points it would hold 230 rows, and the 25 furthest from the prior are all CORRECT -- Dogfish
Head's Utopias Barrel-Aged World Wide Stout at 17.2% against a prior of 5.5%, a dozen pastry
stouts at 15% whose filed class reads "Flavored Malt Beverage". The prior is weak exactly
where TTB's class codes are not styles, which is where this pass does its work: a big gap is
the catalog being corrected, not a bad match. The outliers are in the dumped plan to read
instead (`--dump`).

What it writes: `product.spec.abv_pct` only, as `ExtractionMethod.OPEN_DATABASE` at 0.75 --
second-hand, but addressable, which is what separates it from a recall. It takes the place of
nothing, of a style prior's guess, of an LLM's recall, or of this pass's own earlier answer,
and never of anything read off a label, a filing, a maker's own page or a shop listing
(`_fillable`, after `bcd_enrich.profile._a_guess`). Every harvested row also lands in bronze
under its source id, keyed by its QID or dataset row, so a change to these rules is a
re-apply rather than a re-fetch.

What it came to, run on 2026-10-13: 20,912 published facts, 3,567 rows filled -- 3,105 from
Wikidata, 462 from the dataset, 3,486 of them rows that said nothing at all and 81 rows whose
only answer was a style guess or a model's recall. Of the rest, 15,917 facts name no row this
catalog holds under that name, 1,222 name several, 125 name another maker's drink, 27 were
held by the gates above and 54 were a second source agreeing with the first. 3,567
is 0.7% of the gap, and skewed towards the cans people actually scan, which is the half worth
having -- the other 519,000 are on their labels, where the camera reads them.

    .venv/bin/python scripts/fill_abv_from_open_data.py              # dry run: writes nothing
    .venv/bin/python scripts/fill_abv_from_open_data.py --verbose    # also every held bucket
    .venv/bin/python scripts/fill_abv_from_open_data.py --apply      # write it

The harvest is cached (`--cache`, default data/abv_open_data.json) so that the run that
writes is deciding on the same facts the dry run reported; `--refetch` goes back out. Writes
a backup of every gold row it touches to data/fill_abv_from_open_data_backup.json before the
first write, and nothing in a row but the ABV changes.

No index rebuild: no name changes, so the label index and its postings are untouched. No
enrichment pass either -- `bcd_enrich --abv` fills from a style prior and would now find
these rows already answered, which is the point. Restart the API afterwards; it memoizes the
rows it serves.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest", "services/enrich"):
    sys.path.insert(0, str(ROOT / p))

import httpx  # noqa: E402
import psycopg  # noqa: E402
from bcd_api.app import _load_dotenv  # noqa: E402
from bcd_enrich.style_prior import is_non_alcoholic  # noqa: E402
from bcd_ingest.dedup import _STYLE, _tokens, match_typed_name, producer_core  # noqa: E402
from bcd_ingest.pg_store import PostgresStore  # noqa: E402
from bcd_ingest.store import BronzeDoc, doc_id, open_store  # noqa: E402
from bcd_schema import ExtractionMethod, Product  # noqa: E402

CACHE = Path("data") / "abv_open_data.json"
BACKUP = Path("data") / "fill_abv_from_open_data_backup.json"
VERSION = "fill_abv_from_open_data/1"

WIKIDATA = "wikidata"
CRAFT_BEERS = "craft-beers-dataset"

_WDQS = "https://query.wikidata.org/sparql"
#: Every item with an alcohol by volume, its maker (P176) and what it is (P31). SAMPLE +
#: GROUP BY because an item can carry several labels and several makers, and one of each is
#: all this pass can use. "mul" is the language-agnostic label a brand name usually lives in.
_SPARQL = """
SELECT ?item ?abv (SAMPLE(?lab) AS ?label) (SAMPLE(?mk) AS ?maker) (SAMPLE(?ty) AS ?type)
WHERE {
  ?item wdt:P2665 ?abv .
  OPTIONAL { ?item rdfs:label ?lab .
             FILTER(LANG(?lab) IN ("en","mul","de","nl","fr","es","it","da","sv","cs","pl")) }
  OPTIONAL { ?item wdt:P176 ?m . ?m rdfs:label ?mk . FILTER(LANG(?mk) IN ("en","mul")) }
  OPTIONAL { ?item wdt:P31 ?t . ?t rdfs:label ?ty . FILTER(LANG(?ty) = "en") }
}
GROUP BY ?item ?abv
"""
_UA = "BCD-catalog/0.1 (https://github.com/drewlinsley/BCD) abv-fill"

_CRAFT = "https://raw.githubusercontent.com/nickhould/craft-beers-dataset/master/data/processed"

#: A strength this pass will believe for a row of this category. Deliberately wide -- the
#: gate is here to catch a wrong join or a bad source value, not to second-guess an odd
#: drink -- and deliberately applied, because a held fact can be recovered later and a wrong
#: one in the catalog is read off the can in somebody's hand.
_PLAUSIBLE: dict[str, tuple[float, float]] = {
    "beer": (0.5, 30.0),      # eisbocks reach the high twenties; 0.3% is a NA version
    "cider": (1.0, 15.0),
    "wine": (4.0, 24.0),      # fortified
    "sake": (4.0, 25.0),
    "mead": (3.0, 25.0),
    "rtd": (1.0, 25.0),
    "spirit": (10.0, 96.0),   # a liqueur is a spirit here, and so is Spirytus
    "other": (0.3, 96.0),
}

#: A row whose NAME says non-alcoholic has its own band, because the floors above would throw
#: away the truest facts in the harvest. `Lowlander 0.00% Wit` was held for being 0.0% while
#: Wikidata's alcoholic `Lowlander Wit` filled it at 3.5% -- the one row where the number
#: matters most, answered backwards. 1.2% is the generous legal ceiling (EU "low alcohol";
#: the US and German "non-alcoholic" line is 0.5%).
_NA_CEILING = 1.2

#: A strength the row's own NAME states. TTB names carry one often enough to matter --
#: "CARLSBERG UNFILTERED 5.0 DEGRE ALCOOL", "India Pale Ale 6.6%", "Lowlander 0.00% Wit" --
#: and it got there off a filing, which outranks anything second-hand. Only three rows in the
#: plan state one, and it settled all three: two contradictions held, one match corroborated.
#: The number must come FIRST and the unit after it, which is what keeps "100% MALT" and
#: "4X33CL" out (neither parses) without a list of words to exclude.
_STATED = re.compile(r"(?<![\d.,])(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:%|degre|degree|abv|alc|vol)",
                     re.IGNORECASE)
#: Half a point. The same slack the two-sources check uses: a label rounds, a database does not.
_STATED_TOLERANCE = 0.5


@dataclass(frozen=True)
class Fact:
    """One database's claim about one drink's strength."""

    name: str
    maker: str
    abv_pct: float
    source: str
    ref: str          # QID, or the dataset's own row id
    url: str
    kind: str = ""    # what the source says it is, for reading the held buckets

    def bronze(self) -> BronzeDoc:
        return BronzeDoc(
            id=doc_id(self.source, self.ref), source_id=self.source, natural_key=self.ref,
            fetched_at=datetime.now(UTC).isoformat(), url=self.url,
            payload={"name": self.name, "maker": self.maker, "abv_pct": self.abv_pct,
                     "kind": self.kind, "extractor_version": VERSION},
        )


# ---------------------------------------------------------------------------------------------
# the harvest
# ---------------------------------------------------------------------------------------------

def fetch_wikidata(client: httpx.Client) -> list[Fact]:
    r = client.get(_WDQS, params={"query": _SPARQL, "format": "json"},
                   headers={"User-Agent": _UA, "Accept": "application/sparql-results+json"},
                   timeout=300.0)
    r.raise_for_status()
    out: list[Fact] = []
    for b in r.json()["results"]["bindings"]:
        qid = str(b["item"]["value"]).rsplit("/", 1)[-1]
        label = str(b.get("label", {}).get("value") or "")
        maker = str(b.get("maker", {}).get("value") or "")
        if not label or label == qid:        # an item whose only "name" is its own id
            continue
        try:
            abv = round(float(b["abv"]["value"]), 1)
        except (KeyError, TypeError, ValueError):
            continue
        out.append(Fact(name=label, maker=maker, abv_pct=abv, source=WIKIDATA, ref=qid,
                        url=f"https://www.wikidata.org/wiki/{qid}",
                        kind=str(b.get("type", {}).get("value") or "")))
    return out


def fetch_craft_beers(client: httpx.Client) -> list[Fact]:
    """beers.csv joined to breweries.csv. ABV there is a fraction: 0.066 is 6.6%."""
    beers = client.get(f"{_CRAFT}/beers.csv", timeout=120.0)
    brews = client.get(f"{_CRAFT}/breweries.csv", timeout=120.0)
    beers.raise_for_status()
    brews.raise_for_status()
    makers = {row["id"]: (row["name"] or "").strip()
              for row in csv.DictReader(brews.text.splitlines())}
    out: list[Fact] = []
    for row in csv.DictReader(beers.text.splitlines()):
        name = (row.get("name") or "").strip()
        raw = (row.get("abv") or "").strip()
        if not name or not raw:
            continue
        try:
            abv = round(float(raw) * 100, 1)
        except ValueError:
            continue
        out.append(Fact(name=name, maker=makers.get(row.get("brewery_id", ""), ""),
                        abv_pct=abv, source=CRAFT_BEERS, ref=f"beers:{row.get('id') or ''}",
                        url=f"{_CRAFT}/beers.csv",
                        kind=(row.get("style") or "").strip()))
    return out


def harvest(cache: Path, refetch: bool) -> list[Fact]:
    if cache.exists() and not refetch:
        facts = [Fact(**f) for f in json.loads(cache.read_text())]
        print(f"harvest: {len(facts)} facts from {cache} (--refetch to go back out)")
        return facts
    with httpx.Client(follow_redirects=True) as client:
        t = time.time()
        wd = fetch_wikidata(client)
        print(f"harvest: wikidata {len(wd)} named drinks with an ABV ({time.time() - t:.0f}s)")
        cb = fetch_craft_beers(client)
        print(f"harvest: craft-beers {len(cb)} cans with an ABV")
    facts = wd + cb
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(f) for f in facts], indent=1, ensure_ascii=False))
    print(f"harvest: {len(facts)} facts -> {cache}")
    return facts


# ---------------------------------------------------------------------------------------------
# the catalog, and what could be what
# ---------------------------------------------------------------------------------------------

_CATALOG = """
SELECT id, record->>'name', record->>'brand_id', record->>'category',
       record->'spec'->'abv_pct'
FROM gold
WHERE entity_type = 'product' AND record->>'redirects_to' IS NULL
"""


def catalog(dsn: str) -> tuple[list[dict], dict[str, list[int]]]:
    """Every product once, plus a token -> rows index.

    The whole catalog, not only the rows missing an ABV: whether a name says WHICH row it is
    is a question about all of them, and a fact that fits two rows is held even when only one
    of the two could be filled.
    """
    rows: list[dict] = []
    with psycopg.connect(dsn) as db, db.cursor(name="abv_catalog") as cur:
        cur.itersize = 20_000
        cur.execute(_CATALOG)
        for pid, name, brand_id, category, abv in cur:
            brand = (brand_id or "").rsplit(":", 1)[-1].replace("-", " ")
            rows.append({"id": pid, "name": name or "", "brand": brand,
                         "category": category or "", "abv": abv})
    return rows, index_rows(rows)


def index_rows(rows: list[dict]) -> dict[str, list[int]]:
    """word -> the rows carrying it, by position. The brand's words count as the row's: TTB
    writes the brewery into the product name, but `off:` and `user:` rows do not."""
    postings: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        for t in _tokens(row["name"]) | _tokens(row["brand"]):
            postings[t].append(i)
    return postings


def _needed(fact: Fact) -> tuple[set[str], set[str]]:
    """The words that must appear in a row for it to be this drink: the name's own, and the
    maker's. Short, numeric and style words are left out of the lookup, not out of the gates
    -- an index key that does not narrow the search is a key that costs a long posting list,
    while a gate that is not applied is a wrong answer.
    """
    name = {t for t in _tokens(fact.name) if len(t) >= 3 and t not in _STYLE and not t.isdigit()}
    core = {t for t in _tokens(producer_core(fact.maker)) if len(t) >= 3 and t not in _STYLE}
    return name, core


def candidates(fact: Fact, rows: list[dict], postings: dict[str, list[int]]) -> list[dict]:
    """Rows that contain every word this drink is made of. The intersection of posting lists,
    smallest first.

    A typo in the SOURCE's spelling is invisible to this: the typo tolerance in
    `match_typed_name` runs after it, over the rows a shared exact word brought back. In this
    direction that is the right way round -- the catalog's spelling is the one on the label,
    and a source misspelling a beer costs us that one fact rather than a wrong row.
    """
    name, core = _needed(fact)
    if not name:                              # a name of nothing but style words
        return []
    lists = []
    for t in name | core:
        got = postings.get(t)
        if not got:
            return []                         # a word the catalog has never seen
        lists.append(got)
    lists.sort(key=len)
    keep = set(lists[0])
    for other in lists[1:]:
        keep &= set(other)
        if not keep:
            return []
    return [rows[i] for i in keep]


def maker_agrees(maker: str, row: dict) -> bool:
    """EVERY word of the maker's name is in the row, or they are not the same maker.

    "Sun King Brewing" shares "king" with `King's Reserve Bourbon Barrel Johan` and made no
    part of it. A maker name that folds to nothing -- punctuation, or a script the ASCII fold
    erases -- confirms nothing, rather than confirming everything.
    """
    core = _tokens(producer_core(maker)) - _STYLE
    if not core:
        return False
    return core <= (_tokens(row["name"]) | _tokens(row["brand"]))


def plausible(abv: float, category: str, name: str) -> bool:
    if is_non_alcoholic(name):      # the same reading the style prior makes of the same name
        return 0.0 <= abv <= _NA_CEILING
    lo, hi = _PLAUSIBLE.get(category or "other", _PLAUSIBLE["other"])
    return lo <= abv <= hi


def states_a_strength(name: str) -> list[float]:
    """Strengths the row's own name claims, which a second-hand source does not get to correct."""
    out = []
    for m in _STATED.finditer(name or ""):
        try:
            v = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        if 0.0 <= v <= 96.0:
            out.append(v)
    return out


_REPLACEABLE = {
    None,                                                      # no ABV at all
    ExtractionMethod.LLM_INFERRED_FROM_STYLE_PRIOR.value,      # a style's typical strength
    ExtractionMethod.LLM_RECALLED.value,                       # what a model remembered
    ExtractionMethod.OPEN_DATABASE.value,                      # this pass, earlier
}


def _fillable(abv: dict | None) -> str | None:
    """What this row's ABV is now, when an open database may replace it; None when it may not.

    After `bcd_enrich.profile._a_guess`: a second-hand fact may take the place of nothing, of
    a guess, or of another second-hand fact, and never of anything read off a label, a filing,
    the maker's own page or a shop's listing.
    """
    if not isinstance(abv, dict) or abv.get("value") is None:
        return "empty"
    method = (abv.get("provenance") or {}).get("method")
    if method in _REPLACEABLE:
        return {ExtractionMethod.LLM_INFERRED_FROM_STYLE_PRIOR.value: "a style guess",
                ExtractionMethod.LLM_RECALLED.value: "a recall",
                ExtractionMethod.OPEN_DATABASE.value: "ours"}.get(str(method), "empty")
    return None


@dataclass
class Fill:
    pid: str
    catalog_name: str
    category: str
    abv_pct: float
    was: str                  # what the row said before: empty, a style guess, a recall, ours
    said: float | None        # and the number it said, when it said one
    fact: Fact


def plan(facts: list[Fact], rows: list[dict],
         postings: dict[str, list[int]]) -> tuple[list[Fill], dict[str, list[str]]]:
    """Which rows get an ABV, and one line per fact that does not give one."""
    fills: dict[str, Fill] = {}
    held: dict[str, list[str]] = defaultdict(list)
    for fact in facts:
        seen = candidates(fact, rows, postings)
        if not seen:
            held["no row of that name"].append(f"{fact.maker} / {fact.name}")
            continue
        hits = set(match_typed_name(fact.name, fact.maker, seen))
        if not hits:
            held["no row of that name"].append(f"{fact.maker} / {fact.name}")
            continue
        by_id = {r["id"]: r for r in seen if r["id"] in hits}
        agreed = [r for r in by_id.values() if maker_agrees(fact.maker, r)]
        if not agreed:
            held["a different maker"].append(
                f"{fact.maker} / {fact.name} -> "
                f"{', '.join(sorted(r['name'] for r in by_id.values())[:3])}")
            continue
        if len(agreed) > 1:
            held["several rows it could be"].append(
                f"{fact.maker} / {fact.name} -> "
                f"{', '.join(sorted(r['name'] for r in agreed)[:4])}")
            continue
        row = agreed[0]
        if not plausible(fact.abv_pct, row["category"], row["name"]):
            held["an implausible strength"].append(
                f"{row['name']} <- {fact.abv_pct}% ({row['category'] or 'no category'})")
            continue
        stated = states_a_strength(row["name"])
        if stated and min(abs(v - fact.abv_pct) for v in stated) > _STATED_TOLERANCE:
            held["the row's own name says otherwise"].append(
                f"{row['name']} <- {fact.abv_pct}% from {fact.source}")
            continue
        was = _fillable(row["abv"])
        if was is None:
            held["already answered by a better source"].append(row["name"])
            continue
        if was == "ours" and abs(float((row["abv"] or {}).get("value") or 0) - fact.abv_pct) < 0.05:
            held["already written by this pass"].append(row["name"])
            continue
        prior = fills.get(row["id"])
        if prior is not None:
            if abs(prior.abv_pct - fact.abv_pct) >= 0.5:
                held["two sources that disagree"].append(
                    f"{row['name']}: {prior.fact.source} {prior.abv_pct}% vs "
                    f"{fact.source} {fact.abv_pct}%")
                fills.pop(row["id"])
            continue                          # first source wins a tie within half a point
        before = (row["abv"] or {}).get("value") if isinstance(row["abv"], dict) else None
        fills[row["id"]] = Fill(row["id"], row["name"], row["category"], fact.abv_pct, was,
                                before, fact)
    return list(fills.values()), held


# ---------------------------------------------------------------------------------------------
# saying what it found, and writing it
# ---------------------------------------------------------------------------------------------

def show(facts: list[Fact], fills: list[Fill], held: dict[str, list[str]],
         verbose: bool) -> None:
    print(f"\nfacts harvested ........ {len(facts)}")
    print(f"rows to fill ........... {len(fills)}")
    for source, n in Counter(f.fact.source for f in fills).most_common():
        print(f"    {n:6}  {source}")
    for was, n in Counter(f.was for f in fills).most_common():
        print(f"    replacing {was:24} {n}")
    print("\nheld:")
    for reason, lines in sorted(held.items(), key=lambda kv: -len(kv[1])):
        print(f"    {len(lines):6}  {reason}")
        for line in (lines if verbose else lines[:3]):
            print(f"            {line[:150]}")
    if over := [f for f in fills if f.was != "empty"]:
        print("\nreplacing an answer that was already there (the only writes that take "
              "something away):")
        for f in sorted(over, key=lambda f: f.catalog_name):
            print(f"    {f.catalog_name[:52]:54} {f.said:>5}% ({f.was}) -> {f.abv_pct}%")
    if not fills:
        return
    abvs = sorted(f.abv_pct for f in fills)
    print(f"\nstrengths: {abvs[0]}% to {abvs[-1]}%, median {abvs[len(abvs) // 2]}%")
    print("sample:")
    for f in sorted(fills, key=lambda f: f.catalog_name)[:15]:
        print(f"    {f.catalog_name[:52]:54} {f.abv_pct:5}%  <- {f.fact.source} "
              f"{f.fact.ref} ({f.fact.maker[:28]})")


def dump(fills: list[Fill], path: Path) -> None:
    """The plan, row by row, for reading before it is written."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["product_id", "catalog_name", "category", "abv_pct", "was", "said",
                    "source_name", "maker", "source", "ref", "source_kind"])
        for f in sorted(fills, key=lambda f: f.catalog_name):
            w.writerow([f.pid, f.catalog_name, f.category, f.abv_pct, f.was, f.said or "",
                        f.fact.name, f.fact.maker, f.fact.source, f.fact.ref, f.fact.kind])
    print(f"plan -> {path} ({len(fills)} rows)")


def _provenance(fact: Fact) -> dict:
    quote = (f"{fact.name} — alcohol by volume {fact.abv_pct}%"
             + (f", made by {fact.maker}" if fact.maker else "")
             + f" ({fact.source} {fact.ref})")
    return {"source_id": fact.source, "url": fact.url, "quote": quote,
            "method": ExtractionMethod.OPEN_DATABASE.value,
            # The ceiling for the method: second-hand, but it has an id you can look up.
            "confidence": 0.75,
            "extracted_at": datetime.now(UTC).isoformat(),
            "extractor_version": VERSION}


def apply(store: PostgresStore, facts: list[Fact], fills: list[Fill]) -> None:
    t = time.time()
    for fact in facts:                        # the receipts first: bronze is the cache
        store.put_bronze(fact.bronze())
    print(f"bronze: {len(facts)} source rows filed ({time.time() - t:.0f}s)")

    if not fills:
        print("nothing to write to gold")
        return
    # The first picture of every row is the one to keep.
    backup = json.loads(BACKUP.read_text()) if BACKUP.exists() else {}
    fresh = {f.pid: store.get_gold(f.pid) for f in fills if f.pid not in backup}
    if fresh:
        backup.update(fresh)
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        BACKUP.write_text(json.dumps(backup, indent=1, ensure_ascii=False))
        print(f"backup ({len(fresh)} rows added) -> {BACKUP}")

    t, batch, written, gone = time.time(), [], 0, 0
    for f in fills:
        rec = store.get_gold(f.pid)
        if not isinstance(rec, dict) or rec.get("redirects_to"):
            gone += 1                         # merged away since the plan was made
            continue
        spec = rec.get("spec") if isinstance(rec.get("spec"), dict) else {}
        if _fillable(spec.get("abv_pct")) is None:
            gone += 1                         # somebody better got here first
            continue
        spec["abv_pct"] = {"value": f.abv_pct, "provenance": _provenance(f.fact)}
        rec["spec"] = spec
        Product.model_validate(rec)            # never write a row the schema would refuse
        batch.append((f.pid, "product", rec))
        if len(batch) >= 500:
            store.put_gold_many(batch)
            written += len(batch)
            batch = []
    if batch:
        store.put_gold_many(batch)
        written += len(batch)
    print(f"gold: {written} rows now state a strength ({time.time() - t:.0f}s)"
          + (f"; {gone} skipped, the row changed under us" if gone else ""))

    print("\n-- after --")
    with psycopg.connect(store.dsn) as db, db.cursor() as cur:
        have, total = cur.execute(
            "SELECT count(*) FILTER (WHERE record->'spec'->'abv_pct'->>'value' IS NOT NULL),"
            " count(*) FROM gold WHERE entity_type='product'").fetchone()
    print(f"products stating an ABV: {have} of {total} ({100 * have / max(total, 1):.1f}%)")
    print("\nRestart the API to serve them (no index rebuild: no name changed).")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write it; otherwise a dry run")
    ap.add_argument("--refetch", action="store_true", help="go back to the sources")
    ap.add_argument("--cache", default=str(CACHE), help=f"harvest file (default {CACHE})")
    ap.add_argument("--verbose", action="store_true", help="every held fact, not three")
    ap.add_argument("--dump", help="write the plan to this CSV and read it before writing")
    args = ap.parse_args(argv)

    _load_dotenv()
    store = open_store(root="./data")
    if not isinstance(store, PostgresStore):
        print("this script reads the catalog in Postgres; set BCD_DATABASE_URL", file=sys.stderr)
        return 1
    facts = harvest(Path(args.cache), args.refetch)
    t = time.time()
    rows, postings = catalog(store.dsn)
    print(f"catalog: {len(rows)} products, {len(postings)} distinct words "
          f"({time.time() - t:.0f}s)")
    t = time.time()
    fills, held = plan(facts, rows, postings)
    show(facts, fills, held, args.verbose)
    print(f"(planned in {time.time() - t:.0f}s)")
    if args.dump:
        dump(fills, Path(args.dump))
    if not args.apply:
        print("\n(dry run: nothing written; add --apply to write)")
        return 0
    apply(store, facts, fills)
    return 0


if __name__ == "__main__":
    import os
    os.chdir(ROOT)
    raise SystemExit(main(sys.argv[1:]))
