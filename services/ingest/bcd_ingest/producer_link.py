"""Give producers a real location, from OpenBreweryDB.

The product detail screen leads with where a thing was made, but the producers behind our
catalog come from TTB permits ("The Alchemist LLC") and Open Food Facts brand strings —
neither carries an address. OpenBreweryDB carries city/state/country/lat-lon for every
brewery it lists. This links the two **by name** and copies the location across.

It deliberately does *not* repoint any product at a different producer. Merging producers
across sources is a much larger claim (same name, different company is common) and would
rewrite the entity graph; copying four location fields onto the record a product already
points at is reversible and cannot orphan anything.

Precision over recall throughout, because a wrong city is worse than no city:
  * names are reduced to their **distinctive** tokens — corporate and trade suffixes
    ("LLC", "Brewing Co") are exactly what two unrelated breweries have in common;
  * a key shared by two or more OpenBreweryDB rows is **dropped**, not guessed between
    ("Broken Spoke Brewing" exists in more than one state);
  * a name that reduces to nothing distinctive is skipped.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

#: Words that say what kind of business it is, not which one. Stripped before matching.
_TRADE = {
    "llc", "inc", "ltd", "limited", "co", "company", "corp", "corporation", "plc", "gmbh",
    "bv", "nv", "sa", "srl", "spa", "ag", "kg", "ab", "as", "oy", "aps", "pty", "llp",
    "brewing", "brewery", "breweries", "brewhouse", "brewers", "brewer", "brew", "brewco",
    "beer", "beers", "ale", "ales", "lager", "cerveceria", "cerveza", "brasserie",
    "braueri", "brauerei", "birra", "birrificio", "distillery", "distilling", "distillers",
    "distiller", "winery", "vineyards", "cidery", "meadery", "taproom", "brewpub",
    "the", "and", "of", "at", "on", "works", "craft", "family", "group", "holdings",
    "international", "usa", "us", "america", "american",
    # Placeholder producer names. "Unknown" is what Open Food Facts leaves behind when a
    # brand is missing; it must never key, or every unattributed drink inherits one address.
    "unknown", "unbranded", "various", "n", "a",
}

#: Words that name a *site* rather than a business. Only these may separate a producer name
#: from a longer OpenBreweryDB name in a containment match (see `_contained_match`).
_FACILITY = {
    "cannery", "barrelworks", "barrelhouse", "maison", "microbrouwerij", "brouwerij",
    "production", "facility", "plant", "taproom", "tasting", "room", "cellars", "cellar",
    "bay", "haus", "hof", "casa", "fabrica", "birreria", "quinta", "estate", "site",
}

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

#: A key needs this much signal before it can identify a business.
_MIN_KEY_CHARS = 4

#: Fields copied onto the linked producer. Location only — nothing about identity.
_LOCATION_FIELDS = ("city", "region", "country", "lat", "lon")

#: Source whose bronze evidence corroborates a match (see `_market_countries`).
_EVIDENCE_SOURCE = "openfoodfacts"

#: Coarse market regions. The corroboration below compares at *this* granularity, not by
#: country: a Belgian beer sold in the Netherlands is entirely normal, while a French-only
#: rhum matching a brewpub in Nevada is the collision we are trying to refuse. Anything not
#: listed abstains rather than rejecting.
_REGION = {
    "europe": {
        "albania", "andorra", "austria", "belgium", "bosnia", "bulgaria", "croatia",
        "czech republic", "de", "denmark", "england", "estonia", "finland", "france",
        "germany", "greece", "hungary", "iceland", "ireland", "isle of man", "italy",
        "latvia", "lithuania", "luxembourg", "malta", "netherlands", "norway", "poland",
        "portugal", "romania", "russia", "scotland", "serbia", "slovakia", "slovenia",
        "spain", "sweden", "switzerland", "ukraine", "united kingdom", "wales",
        # Overseas France: administratively European markets.
        "guadeloupe", "martinique", "réunion", "reunion", "saint martin", "french guiana",
        "french polynesia", "new caledonia",
    },
    "north_america": {"united states", "us", "usa", "canada", "mexico", "panama"},
    "south_america": {"argentina", "bolivia", "brazil", "chile", "colombia", "ecuador",
                      "peru", "uruguay"},
    "asia": {"china", "hong kong", "india", "israel", "japan", "kazakhstan", "singapore",
             "south korea", "thailand", "united arab emirates", "vietnam"},
    "africa": {"algeria", "burkina faso", "democratic republic of the congo", "egypt",
               "mauritius", "morocco", "nigeria", "south africa", "tunisia"},
    "oceania": {"australia", "new zealand"},
}
_COUNTRY_REGION = {c: region for region, cs in _REGION.items() for c in cs}


def _tokens(name: str) -> list[str]:
    d = unicodedata.normalize("NFKD", name or "")
    a = "".join(c for c in d if not unicodedata.combining(c)).casefold()
    return _WORD.findall(a)


def name_key(name: str) -> frozenset[str] | None:
    """The distinctive tokens of a business name, or None if nothing distinctive is left.

    "The Alchemist LLC" and "The Alchemist Brewery" both reduce to {"alchemist"}; "Brewing
    Company" reduces to nothing and is refused rather than matched against every brewery.
    """
    toks = {t for t in _tokens(name) if t not in _TRADE}
    if not toks or sum(len(t) for t in toks) < _MIN_KEY_CHARS:
        return None
    return frozenset(toks)


def _market_countries(store: Any) -> dict[str, set[str]]:
    """Product natural key -> the countries its source says it is sold in.

    Open Food Facts records where a product is *sold*, not where it was made, so this can
    never confirm an origin. It is used only to refuse one: a beer sold solely in France is
    not made by a brewery in Nevada. That is exactly the collision name matching produces —
    "Saint James" is a Martinique rhum house and also a brewpub in Reno.
    """
    out: dict[str, set[str]] = {}
    for doc in store.iter_bronze(_EVIDENCE_SOURCE):
        raw = (doc.payload or {}).get("countries") or ""
        if raw:
            out[doc.natural_key] = {c.strip().casefold() for c in raw.split(",") if c.strip()}
    return out


def _corroborated(match: dict, product_keys: list[str], markets: dict[str, set[str]]) -> bool:
    """Whether the match survives what we know about where the producer's drinks are sold.

    Compared by region, not country — Open Food Facts lists sales territories, and a brewery
    a border away is the normal case, not a contradiction. Silence is not evidence against:
    an unmapped or absent country lets the name match stand on its own.
    """
    region = _COUNTRY_REGION.get((match.get("country") or "").strip().casefold())
    if region is None:
        return True
    sold_in: set[str] = set()
    for key in product_keys:
        sold_in |= markets.get(key, set())
    regions = {_COUNTRY_REGION[c] for c in sold_in if c in _COUNTRY_REGION}
    return not regions or region in regions


def _has_location(rec: dict[str, Any]) -> bool:
    return any(rec.get(f) not in (None, "") for f in ("city", "region", "country"))


def build_index(store: Any, source: str = "openbrewerydb") -> dict[frozenset[str], dict]:
    """Name key -> the one OpenBreweryDB producer that owns it.

    A key claimed by two or more rows is dropped: those are different businesses that happen
    to share a name, and picking either would attach a plausible-looking wrong city.
    """
    seen: dict[frozenset[str], dict | None] = {}
    for rec in store.iter_gold("producer"):
        if source not in (rec.get("id") or "") or not _has_location(rec):
            continue
        key = name_key(rec.get("name") or "")
        if key is None:
            continue
        seen[key] = None if key in seen else rec  # second sighting poisons the key
    return {k: v for k, v in seen.items() if v is not None}


def _contained_match(key: frozenset[str], index: dict[frozenset[str], dict]) -> dict | None:
    """A producer name wholly *contained* in exactly one OpenBreweryDB name.

    "The Alchemist LLC" reduces to {alchemist}; the brewery behind Heady Topper is listed as
    "Alchemist Cannery" -> {alchemist, cannery}, so exact key equality never sees it. Requiring
    the containment to be **unique** is what keeps a common word from matching half the index —
    if two breweries both contain the key, we cannot tell which, and take neither.

    The extra words must also be *site* words. Containment on its own is far too generous:
    it matched the placeholder "Unknown" to Destination Unknown Beer Company, and Pinnacle
    (a vodka) to Groggs Pinnacle Brewing. In each the extra token was a different brand —
    the real identity — so only additions that name a site are allowed to differ.
    """
    hits = [(k, rec) for k, rec in index.items() if key < k and (k - key) <= _FACILITY]
    return hits[0][1] if len(hits) == 1 else None


def plan(store: Any, source: str = "openbrewerydb") -> list[tuple[dict, dict]]:
    """Every (producer_without_location, matching_obdb_producer) pair we would write."""
    index = build_index(store, source)
    markets = _market_countries(store)
    by_producer: dict[str, list[str]] = {}
    for prod in store.iter_gold("product"):
        pid = prod.get("producer_id")
        if pid:
            by_producer.setdefault(pid, []).append((prod.get("id") or "").split(":", 1)[-1])

    out: list[tuple[dict, dict]] = []
    for rec in store.iter_gold("producer"):
        if source in (rec.get("id") or "") or _has_location(rec):
            continue
        key = name_key(rec.get("name") or "")
        if key is None:
            continue
        match = index.get(key) or _contained_match(key, index)
        if match and _corroborated(match, by_producer.get(rec["id"], []), markets):
            out.append((rec, match))
    return out


def link(store: Any, source: str = "openbrewerydb", apply: bool = False) -> dict[str, int]:
    """Copy location onto every producer we can identify. Idempotent: a producer that already
    has a location is skipped, so re-running after a fresh OpenBreweryDB pull only fills gaps."""
    pairs = plan(store, source)
    if apply:
        for rec, match in pairs:
            updated = dict(rec)
            for field in _LOCATION_FIELDS:
                if match.get(field) not in (None, ""):
                    updated[field] = match[field]
            store.put_gold(updated["id"], "producer", updated)
    return {"matched": len(pairs), "written": len(pairs) if apply else 0}
