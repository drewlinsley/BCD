"""Ranking the catalog for a person: what we know about a drink counts, not only how well its
vector fits them.

The style floor gives every one of 534k products its style's centroid, so for someone who likes
IPAs every anonymous IPA in the registry scores the same 0.92 -- and the rows we actually know
something about (drinkers rated it; a profile of THIS product; a prior from its ingredient list)
sit a few points under that, because real beers are not centroids. Ranked on the score alone the
list is five registry rows no one has heard of, tied, in whatever order the index found them, and
the beers we know never appear. This ranks on three things, in order:

  1. the band the score falls in -- a match, a partial match, outside their usual -- which is the
     band the one-line reason is written in (`resolver.match_band`), so the list never puts
     "some citrus, which you like" above something it calls a match;
  2. the evidence behind the vector: rated by drinkers, then known (a profile of this product, or
     a prior from its ingredients), then a guess (the style's centroid, or a profile that only
     read the name);
  3. the score.

Rows carrying the same vector are the same recommendation -- a lineup profile stamped on a beer's
label variants across a maker's permits, or ten thousand registry IPAs on one centroid -- and
collapse to one entry, under the best-evidenced and then plainest-named row that names a product
at all: the registry files rows spelled "mezcal" and "Tequila blanco", and the plainest row of a
group is often one of them, which would stand for none of the rest. So a list of ten is ten
different drinks, and the floor appears once per style rather than filling the list.
Candidates come from two nearest-neighbour queries: the nearest *known* vectors, each with the
rows that carry it, which the floor's ties would otherwise keep out of any top-N (and which,
asked row by row, were seven beers in the nearest hundred rows); and the nearest rows overall,
which the ties fill and which stand for the styles.
"""

from __future__ import annotations

import re

from bcd_ingest.store import Store
from bcd_schema import Product, SensorySource, SensoryVector, TasteProfile

from .resolver import Resolver, match_band

RATED, KNOWN, GUESSED = 0, 1, 2
EVIDENCE = {RATED: "rated", KNOWN: "known", GUESSED: "guessed"}

# `bcd_enrich.profile` holds a style-only answer -- "the kind of beer this maker makes" -- under
# 0.45 and a known product's profile at or above it. Under the floor, a profile is a guess with a
# different pedigree and ranks with the guesses.
_KNOWN_FLOOR = 0.45


def evidence_tier(sensory: SensoryVector | None) -> int:
    """What stands behind the vector: drinkers, knowledge of the product, or a guess."""
    if sensory is None:
        return GUESSED
    if sensory.source in (SensorySource.REVIEW_CONSENSUS, SensorySource.RECONCILED):
        return RATED
    if sensory.source in (SensorySource.LLM_PROFILE, SensorySource.CHEMISTRY_PRIOR) and \
            sensory.confidence >= _KNOWN_FLOOR:
        return KNOWN
    return GUESSED


def rank_key(score: float, product: Product) -> tuple:
    """Sort key, ascending: the band first, then the evidence, then the score itself. Two rows
    alike on all three sort by the vector's confidence, then by name, so a run is repeatable."""
    sensory = product.sensory
    return (match_band(score), evidence_tier(sensory), -score,
            -(sensory.confidence if sensory else 0.0), product.name or "")


def _plain_name(name: str, maker: str | None) -> str:
    """The product's name without its maker's: "Rhinegeist Truth" and "Truth India Pale Ale"
    are one beer, and the shorter of "Truth" and "Truth India Pale Ale" is the plainer row."""
    out = name or ""
    if maker:
        out = re.sub(rf"\b{re.escape(maker)}\b", " ", out, flags=re.I)
        # The registry also files the maker without its trade suffix ("Brewing Co.").
        short = re.sub(r"\b(brewing|brewery|brewers|distillery|distilling|co\.?|company|inc\.?|"
                       r"llc|ltd\.?)\b", " ", maker, flags=re.I).strip()
        if short and short.lower() != maker.lower():
            out = re.sub(rf"\b{re.escape(short)}\b", " ", out, flags=re.I)
    return " ".join(out.split()) or (name or "")


# The words a label uses to say what kind of drink it is, and the filler between them. A name
# left with nothing else -- "mezcal", "Tequila blanco", "Gold Rum" -- names no product.
_KIND = re.compile(
    r"\b(tequila|mezcal|mescal|rum|ron|rhum|cacha[cç]a|vodka|gin|whisky|whiskey|scotch|bourbon"
    r"|rye|soju|shochu|baijiu|sake|agave|spirit|spirits|liqueur|beer|ale|lager|stout|ipa|cider"
    r"|blanco|plata|plato|silver|white|joven|reposado|a[ñn]ejo|cristalino|gold|dark|light|aged"
    r"|old|extra|spiced|flavou?red|single|malt|blend|blended|straight|premium|proof|pure|dry"
    r"|kentucky|tennessee|irish|japanese|canadian|american|london|caribbean|100|de|of|the|and"
    r"|y)\b", re.I)


def _legible(name: str) -> int:
    """How well a name can stand for every row that shares its vector, smallest first: a name a
    drinker could repeat, then a registry row whose two fields ran together ("4b ,plantation"),
    then one that says no more than its own kind ("mezcal") and so names none of them. The whole
    name is read, maker and all: "Odell Brewing Company IPA" names a beer, "IPA" does not."""
    if not re.sub(r"[^a-z0-9]", "", _KIND.sub(" ", name).lower()):
        return 2
    return 1 if " ," in name else 0


def rank_catalog(store: Store, resolver: Resolver, profile: TasteProfile | None, *,
                 limit: int = 10) -> list[dict]:
    """The `limit` products to recommend, best first, each with its maker, its score, its
    one-line reason, the cold-start flag and what the evidence behind it is."""
    if profile is not None and profile.sensory_ideal is not None:
        ideal = profile.sensory_ideal.to_array()
        # Over-fetch so the re-rank has room, and ask the known vectors separately: on the
        # live catalog the floor's ties fill any single top-N before the first known row
        # appears. The score is the cosine, so the nearest known vectors are the known
        # entries of the list in order; three deep leaves room for the floor's entries.
        # (Rated rows outrank known ones within a band, so once drinkers have rated more
        # than a list's worth of rows they will want asking for separately, the same way.)
        pool: dict[str, dict] = {}
        for members in store.nearest_known(ideal, limit=limit * 3):
            for rec in members:
                pool.setdefault(rec["id"], rec)
        for rec in store.nearest_by_sensory(ideal, limit=limit * 3):
            pool.setdefault(rec["id"], rec)
        candidates = list(pool.values())
    else:
        # No taste vector yet: the mild style-affinity prior, over the catalog.
        candidates = list(store.iter_gold("product"))

    scored: list[tuple[tuple, Product, float, str, bool]] = []
    for rec in candidates:
        product = Product.model_validate(rec)
        score, reason, cold = resolver.score(product, profile)
        scored.append((rank_key(score, product), product, score, reason, cold))

    # One entry per vector: the members are the same recommendation, so the best-evidenced
    # and then plainest-named stands for them, and the representatives are ranked.
    groups: dict[object, list[tuple[tuple, Product, float, str, bool]]] = {}
    for entry in scored:
        product = entry[1]
        key: object = tuple(product.sensory.to_array()) if product.sensory else product.id
        groups.setdefault(key, []).append(entry)
    makers: dict[str, str | None] = {}

    def _maker(product: Product) -> str | None:
        pid = product.producer_id or ""
        if pid not in makers:
            makers[pid] = (store.get_gold(pid) or {}).get("name") if pid else None
        return makers[pid]

    def _plainness(entry: tuple[tuple, Product, float, str, bool]) -> tuple:
        # the rank key up to (not including) the name, then whether the name can stand for the
        # others at all, then the plainness, then the name
        name = entry[1].name or ""
        plain = _plain_name(name, _maker(entry[1]))
        return (*entry[0][:-1], _legible(name), len(plain.split()), len(plain), name)

    chosen = [min(members, key=_plainness) if len(members) > 1 else members[0]
              for members in groups.values()]
    chosen.sort(key=lambda e: e[0])

    return [{"product_id": product.id, "name": product.name, "producer": _maker(product),
             "score": score, "reason": reason, "cold_start": cold,
             "evidence": EVIDENCE[evidence_tier(product.sensory)]}
            for _key, product, score, reason, cold in chosen[:limit]]
