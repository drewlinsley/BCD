"""Entity resolution for near-duplicate catalog rows the exact-key merge can't reach.

The exact identity key (bcd_api.resolver._identity_key) collapses rows with identical brand+name.
It cannot link the SAME product described at different granularity across sources — a curated
`bcd-demo:Heady Topper` and the regulatory `ttb:The Alchemist Heady Topper`, or an OFF row named
bare `Pale Ale` (brand Sierra Nevada) and `ttb:Sierra Nevada Pale Ale`.

Naive substring matching is unsafe: "Pale Ale" is a substring of every pale ale, and "scotch
whisky" pulls Ardbeg, Cardhu and Bowmore into Lagavulin. What makes a containment merge SAFE is
three things together, so this module requires all of them:

  1. name coverage   — the alias's name tokens are all present in the canonical's name+brand,
                       and the canonical's name is strictly more specific (more name tokens);
  2. brand compatible — the two carry the same brand identity (one's brand words appear in the
                       other), or one has no independent brand (a placeholder, or a brand that just
                       echoes the name) — this is what rejects Ardbeg⊂Lagavulin and keeps
                       Sierra-Nevada "Pale Ale"⊂"Sierra Nevada Pale Ale";
  3. distinctive     — the tokens they actually share include a non-style word (not just
                       "ipa"/"scotch"/"whisky"), and the match is UNIQUE (an alias covered by two
                       different candidates is ambiguous and left alone).

Digits are not tokens here, so "Scotch Whisky 1824" and "Whiskey 40%" reduce to bare style and are
skipped. Returns (alias_id, canonical_id) structural pairs; the caller picks which row physically
survives (by provenance/richness) and repoints SKUs.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)  # runs of >=2 letters; digits are not tokens

# Style, varietal, packaging and stop words — shared alone they carry no product identity.
_STYLE = {
    "the", "and", "of", "for", "with", "by", "co", "cie", "vol", "abv", "cl", "ml", "oz",
    "beer", "beers", "bier", "biere", "bière", "cerveza", "cerveja", "birra", "pivo",
    "ale", "ales", "ipa", "apa", "lager", "lagers", "pils", "pilsner", "pilsener", "stout",
    "porter", "wheat", "weizen", "weiss", "weisse", "hefeweizen", "witbier", "wit", "blanche",
    "saison", "tripel", "dubbel", "abbaye", "abbey", "trappist", "kolsch", "kölsch", "helles",
    "dunkel", "bock", "radler", "shandy", "seltzer", "gose", "lambic", "kriek", "amber",
    "blonde", "blond", "brune", "brown", "red", "rouge", "pale", "dark", "gold", "golden",
    "light", "lite", "dry", "hazy", "juicy", "double", "triple", "imperial", "session",
    "extra", "strong", "special", "original", "premium", "classic", "traditional", "reserve",
    "fresh", "hop", "hops", "hopped", "craft", "brewing", "brewery", "brewers", "brasserie",
    "vodka", "gin", "rum", "rhum", "ron", "whisky", "whiskey", "bourbon", "scotch", "malt",
    "single", "blended", "grain", "kentucky", "straight", "tennessee", "irish", "rye", "years",
    "year", "old", "aged", "anos", "años", "ans", "tequila", "mezcal", "brandy", "cognac",
    "liqueur", "spiced", "coconut", "flavour", "flavored", "natural", "organic", "style",
    "new", "belgisch", "belgian", "german", "spirit", "spirits", "wine", "cider", "unknown",
}


def _tokens(s: str) -> set[str]:
    d = unicodedata.normalize("NFKD", s or "")
    a = "".join(c for c in d if not unicodedata.combining(c)).casefold()
    return set(_WORD.findall(a))


def _prep(p: dict) -> dict:
    nt = _tokens(p.get("name", ""))
    bt = _tokens(p.get("brand", ""))
    # A brand that is empty, a placeholder, or just echoes the product name carries no independent
    # identity (bcd-demo's "Heady Topper" brand on "Heady Topper"); drop it so it neither blocks a
    # real match nor stands in for one.
    ident = set() if (not bt or bt <= {"unknown"} or bt <= nt) else bt
    return {
        "id": p["id"], "src": p["id"].split(":", 1)[0],
        "name": p.get("name", ""), "brand": p.get("brand", ""),
        "nt": nt, "full": nt | bt, "brand_ident": ident,
    }


def _brand_compatible(a: dict, b: dict) -> bool:
    return (not a["brand_ident"] or not b["brand_ident"]
            or a["brand_ident"] <= b["full"] or b["brand_ident"] <= a["full"])


def find_substring_merges(products: Iterable[dict],
                          cross_source_only: bool = True) -> list[tuple[str, str]]:
    """Find safe (alias_id, canonical_id) merges where alias's name is a more-generic form of
    canonical's. `products` are dicts with id / name / brand. Only unique, brand-compatible,
    distinctively-shared containments qualify; when `cross_source_only`, the two must come from
    different id namespaces (bcd-demo / ttb / off)."""
    ps = [_prep(p) for p in products]
    out: list[tuple[str, str]] = []
    for sub in ps:
        if not sub["nt"]:
            continue
        cands = []
        for sup in ps:
            if sup["id"] == sub["id"] or len(sup["nt"]) <= len(sub["nt"]):
                continue  # canonical must have a strictly more specific name
            if cross_source_only and sub["src"] == sup["src"]:
                continue
            if not sub["nt"] <= sup["full"]:
                continue
            if not _brand_compatible(sub, sup):
                continue
            if not (sub["full"] & sup["full"]) - _STYLE:
                continue  # shared only style words -> coincidence, not identity
            cands.append(sup)
        if len(cands) == 1:  # a unique canonical; 2+ is ambiguous -> leave alone
            out.append((sub["id"], cands[0]["id"]))
    return out
