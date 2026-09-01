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

import difflib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)  # runs of >=2 letters; digits are not tokens
_DIGITS = re.compile(r"\d+")

# Two brand strings this close are the same brand mistyped or punctuated differently
# ("William Lawson's"/"william lawsons", "Leinenkugel's"/"Leinenliugels"). Measured on the
# real catalog the gap is wide: true variants score 0.83-0.96, while genuinely different
# brands that share a product name (Hoepfner/Gösser) top out at 0.43.
_BRAND_SIM = 0.80

# Age statements written in different languages. The only words allowed to be present on
# one side and absent on the other: "Lagavulin 16 ans" is "Lagavulin 16 Year Old".
_AGE_EQUIV = {"ans", "ano", "anos", "año", "años", "anni", "jahre", "jahren",
              "year", "years", "yr", "yrs", "old", "aged"}

# Style, varietal, packaging and stop words — shared alone they carry no product identity.
_STYLE = {
    "the", "and", "of", "for", "with", "by", "co", "cie", "vol", "abv", "cl", "ml", "oz",
    "beer", "beers", "bier", "biere", "bière", "cerveza", "cerveja", "birra", "pivo",
    "ale", "ales", "ipa", "apa", "lager", "lagers", "pils", "pilsner", "pilsener", "stout",
    "porter", "wheat", "weizen", "weiss", "weisse", "hefeweizen", "witbier", "wit",
    "blanche", "blanc",
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
    # regional/varietal category words: "Rhum blanc agricole" and "Blended Canadian
    # Whiskey" name a category that many distinct producers all sell.
    "agricole", "canadian", "japanese", "american", "mexican", "caribbean", "highland",
    "speyside", "islay", "anejo", "añejo", "reposado", "blanco", "silver", "ecosse",
    "vsop", "cerveses", "ouzo", "sake", "soju", "london",
    # packaging chrome: printed on the label, identifies nothing. A Heady Topper can says
    # "DRINK FROM THE CAN", which trigram-matched a product literally named "Life drink"
    # and outranked the real beer in the HUD.
    "drink", "drinks", "beverage", "bottle", "bottled", "canned", "brewed", "contents",
    "imported", "product",
}


def _tokens(s: str) -> set[str]:
    d = unicodedata.normalize("NFKD", s or "")
    a = "".join(c for c in d if not unicodedata.combining(c)).casefold()
    return set(_WORD.findall(a))


def _digits(s: str) -> frozenset[str]:
    """Digit runs in a name. These carry identity — an age statement, an ABV, a batch —
    so 8 vs 16 is a different bottle even when every word matches."""
    return frozenset(_DIGITS.findall(s or ""))


def _prep(p: dict) -> dict:
    nt = _tokens(p.get("name", ""))
    bt = _tokens(p.get("brand", ""))
    # A brand that is empty, a placeholder, or just echoes the product name carries no independent
    # identity (bcd-demo's "Heady Topper" brand on "Heady Topper"); drop it so it neither blocks a
    # real match nor stands in for one.
    ident = set() if (not bt or bt <= {"unknown"} or bt <= nt) else bt
    return {
        "id": p["id"], "src": p["id"].split(":", 1)[0], "dig": _digits(p.get("name", "")),
        "name": p.get("name", ""), "brand": p.get("brand", ""),
        "nt": nt, "bt": bt, "full": nt | bt, "brand_ident": ident,
    }


def _brand_compatible(a: dict, b: dict) -> bool:
    if (not a["brand_ident"] or not b["brand_ident"]
            or a["brand_ident"] <= b["full"] or b["brand_ident"] <= a["full"]):
        return True
    # ...or the same brand spelled differently. Compared as one string so word order
    # and a lost apostrophe do not read as two different companies.
    x, y = ("".join(sorted(v["brand_ident"])) for v in (a, b))
    return difflib.SequenceMatcher(None, x, y).ratio() >= _BRAND_SIM


def _covered(sub: dict, sup: dict) -> bool:
    """Is the alias name a less specific writing of the canonical's?

    Every identity-bearing token must appear in the canonical. Style/stop words on the
    alias side may be absent, which is what lets "Lagavulin 16 ans" reach "Lagavulin 16
    Year Old Single Malt Scotch Whisky" — the same bottle labelled in two languages,
    where requiring the French "ans" to appear in the English name blocks it forever.
    """
    if sub["nt"] <= sup["full"]:
        return True
    # Only an age/time word may go uncovered. Allowing any style word to vanish lets a
    # product collapse to its brand: "Sierra Nevada Pale Ale" would reach the unrelated
    # "Juicy Little Thing Hazy IPA" (same brewery), and "Bavarian Hefeweizen" would reach
    # "Bavarian Amber Lager". Style is identity when style is the difference.
    uncovered = sub["nt"] - sup["full"]
    if not uncovered <= _AGE_EQUIV:
        return False
    return bool((sub["nt"] - uncovered) - _STYLE)


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
            if not _covered(sub, sup):
                continue
            if sub["dig"] != sup["dig"]:
                continue  # different age/ABV/batch -> different bottle
            if not _brand_compatible(sub, sup):
                continue
            if not (sub["full"] & sup["full"]) - _STYLE:
                continue  # shared only style words -> coincidence, not identity
            cands.append(sup)
        if len(cands) == 1:  # a unique canonical; 2+ is ambiguous -> leave alone
            out.append((sub["id"], cands[0]["id"]))
    return out


def _richness(p: dict) -> tuple:
    """Which row of a duplicate cluster should survive.

    A brand echoing its own product name ("Coors" on "Coors Light") is the most reliable
    signal we have; preferring merely *having* a brand instead would let a mis-scraped one
    win, which is how "Coors Light" ends up filed under the jerky brand OLD TRAPPER.
    Failing that, take a real brand over a placeholder, then the fuller name. Ties break on
    id so the choice is stable across runs."""
    echoes = bool(p["bt"] & p["nt"])
    real = bool(p["bt"]) and p["bt"] != {"unknown"}
    return (echoes, real, len(p["name"]), p["id"])


def find_duplicate_merges(products: Iterable[dict],
                          cross_source_only: bool = False) -> list[tuple[str, str]]:
    """Find rows that are the SAME product written differently — word order, punctuation,
    accents, case: "Cerveza Heineken"/"Heineken Cerveza", "Fernet Branca"/"Fernet-Branca".

    Containment can't reach these (neither name is more specific than the other), so they
    survived the substring pass and show up as twins in a recommendation list.

    Identical name tokens are NOT sufficient on their own. Hoepfner and Gösser both sell a
    "Natur Radler" and they are different beers, so a compatible brand is required too —
    the same gate that lets a bare "Unknown" or a brand that merely echoes the name merge
    freely. Digits must match for the reason they always do here: age and ABV are identity.
    """
    ps = [_prep(p) for p in products]
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for p in ps:
        if p["nt"]:
            buckets[(frozenset(p["nt"]), p["dig"])].append(p)

    out: list[tuple[str, str]] = []
    for group in buckets.values():
        if len(group) < 2:
            continue
        # Grow clusters only where a row is compatible with EVERY existing member, so one
        # permissive row (brand "Unknown") cannot chain two rival brands together.
        # A name made only of style/category words ("Irish Whiskey", "Rhum blanc agricole")
        # names a class, not a product — two such rows are the same thing only if BOTH
        # carry a real, matching brand. Without this, every brand-less row gets adopted by
        # whichever same-named row happens to have a brand, which is a guess, not a merge.
        generic = not (group[0]["nt"] - _STYLE)

        def compatible(p: dict, q: dict, generic: bool = generic) -> bool:
            if generic and not (p["brand_ident"] and q["brand_ident"]):
                return False
            return _brand_compatible(p, q)

        clusters: list[list[dict]] = []
        for p in sorted(group, key=_richness, reverse=True):
            for c in clusters:
                if all(compatible(p, q) for q in c):
                    c.append(p)
                    break
            else:
                clusters.append([p])
        for c in clusters:
            if len(c) < 2:
                continue
            if cross_source_only and len({m["src"] for m in c}) == 1:
                continue
            canon = max(c, key=_richness)
            out.extend((m["id"], canon["id"]) for m in c if m["id"] != canon["id"])
    return out


def is_generic_token(token: str) -> bool:
    """Whether one word is a category/chrome word rather than something that identifies a
    product. Exposed so the resolver can ask the same question of a single OCR token that
    dedup asks of a whole name."""
    return bool(_tokens(token) and _tokens(token) <= _STYLE)


def is_generic_name(name: str) -> bool:
    """Whether a name is built entirely from category/style words and so identifies a class
    rather than a product ("Blended Canadian Whiskey", "Rhum blanc agricole", "London Dry
    Gin"). Many unrelated producers sell each of these, so such a name has to be anchored on
    its brand before it can be shown — otherwise the catalog reads as full of duplicates
    that are not duplicates. Shared with the connectors so "carries no identity" means one
    thing across ingest and resolution."""
    toks = _tokens(name)
    return bool(toks) and not (toks - _STYLE)


#: Brand strings that name no brand. Open Food Facts leaves these behind when the field is
#: missing, and prepending one to a product name adds only noise for the matcher to trip on.
_PLACEHOLDER_BRAND = {"unknown", "unbranded", "various", "none", "generic", "na"}


def search_name(name: str, brand: str | None) -> str:
    """The string a scanned label should be matched against: "<brand> <name>".

    A label reads "TITO'S HANDMADE VODKA", but the catalog splits it: a brand row called
    Tito's and a product row called "Handmade Vodka". Matched on its name alone that
    product can never account for the whole label, and the half it does show is a generic
    phrase that ties with every other handmade vodka on the shelf.

    The brand is added only when the name does not already carry it. Sharing a single
    token is enough to count as carrying it, which is what keeps the redundant cases out:
    brand "Bombay spirits" over "Bombay sapphire murcian lemon" would otherwise produce
    "Bombay spirits Bombay sapphire murcian lemon", and every extra "Bombay" dilutes the
    trigram score of the row that deserves to win.

    This is the exact inverse of the client's DisplayName, which strips a repeated brand
    so a screen reads cleanly. Names are stored for matching and shown for reading, and
    the two want opposite things.
    """
    name = (name or "").strip()
    brand = (brand or "").strip()
    if not name or not brand:
        return name
    btoks = _tokens(brand)
    if not btoks or btoks <= _PLACEHOLDER_BRAND:
        return name
    if btoks & _tokens(name):
        return name
    return f"{brand} {name}"
