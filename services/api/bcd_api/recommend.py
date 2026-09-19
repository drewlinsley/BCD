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

Candidates come from two nearest-neighbour queries: the nearest rows overall, which the floor's
ties fill, and the nearest *known* rows, which those ties would otherwise keep out of any top-N.
Rows carrying the same vector are the same recommendation -- a lineup profile stamped on a beer's
label variants across a maker's permits, or ten thousand registry IPAs on one centroid -- and
collapse to one entry, under the best-evidenced and then plainest-named row, so a list of ten is
ten different drinks, and the floor appears once per style rather than filling the list.
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


def rank_catalog(store: Store, resolver: Resolver, profile: TasteProfile | None, *,
                 limit: int = 10) -> list[dict]:
    """The `limit` products to recommend, best first, each with its maker, its score, its
    one-line reason, the cold-start flag and what the evidence behind it is."""
    if profile is not None and profile.sensory_ideal is not None:
        ideal = profile.sensory_ideal.to_array()
        # Over-fetch so the re-rank has room, and ask the known rows separately: on the live
        # catalog the floor's ties fill any single top-N before the first known row appears.
        # The known rows are asked for ten deep because they collapse hard -- a lineup profile
        # sits on every label variant of its beer (Enjoy By: 29 rows), and the hundred nearest
        # known rows for the demo profile are 25 beers.
        pool: dict[str, dict] = {}
        for rec in (*store.nearest_by_sensory(ideal, limit=limit * 10, known=True),
                    *store.nearest_by_sensory(ideal, limit=limit * 3)):
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
        # the rank key up to (not including) the name, then the plainness, then the name
        plain = _plain_name(entry[1].name or "", _maker(entry[1]))
        return (*entry[0][:-1], len(plain.split()), len(plain), entry[1].name or "")

    chosen = [min(members, key=_plainness) if len(members) > 1 else members[0]
              for members in groups.values()]
    chosen.sort(key=lambda e: e[0])

    return [{"product_id": product.id, "name": product.name, "producer": _maker(product),
             "score": score, "reason": reason, "cold_start": cold,
             "evidence": EVIDENCE[evidence_tier(product.sensory)]}
            for _key, product, score, reason, cold in chosen[:limit]]
