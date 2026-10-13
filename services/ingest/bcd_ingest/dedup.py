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

#: Two words this close are the same word, one of them mistyped. Measured 2026-10-12 over
#: 40,000 catalog names (20,394 distinct identifying tokens): a single human typo -- a dropped
#: letter, two transposed, one doubled, one wrong key -- scores >= 0.80 against the original for
#: 99.8% of five-and-six letter words and 99.9% of longer ones, while two genuinely DIFFERENT
#: catalog words reach 0.80 in 0.05% and 0.02% of pairs respectively.
_TYPO_SIM = 0.80

#: And under five letters a word must match exactly, because the measurement says fuzzy matching
#: there is a losing trade: at 0.80 it catches half the typos in a four-letter word (50.8%) at
#: three to five times the false-pair rate (0.09%), and dropping the bar to 0.75 to catch 83.5%
#: takes false pairs to 0.28%. A short word is both more damaged by one wrong letter and more
#: likely to collide with an unrelated one -- "bock"/"back", "gose"/"rose".
_TYPO_MIN_LEN = 5

#: And a typo does not change how long a word is. One edit moves the length by at most one, so
#: two is already generous -- while `difflib` rewards one word being contained in another, which
#: is how "crusher" and "funcrusher" score 0.824 and read as the same word mistyped. They are
#: two words. A compound is not a slipped thumb.
_TYPO_MAX_LEN_GAP = 2

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
    # "india" belongs with these and was missing. "India Pale Ale" is the same construction
    # as "London Dry Gin" and "Blended Canadian Whiskey", both already here. Its absence made
    # INDIA the one identifying token of an otherwise wholly generic line, which put a probe
    # for it in the gate and let _token_supported admit every name containing the word -- 21,606
    # candidates for a line that names a style, not a beer.
    "india",
    # packaging chrome: printed on the label, identifies nothing. A Heady Topper can says
    # "DRINK FROM THE CAN", which trigram-matched a product literally named "Life drink"
    # and outranked the real beer in the HUD.
    "drink", "drinks", "beverage", "bottle", "bottled", "canned", "brewed", "contents",
    "imported", "product",
    # "from" belongs with the prepositions at the top, and was the one missing. A Heady
    # Topper can reads "DRINK FROM THE CAN": FROM was the line's only identifying token, so
    # it earned an index probe and then supported "From Away" and "There From Here" -- nine
    # wrong top answers in one session at the camera.
    "from",
    # The rest of the resolver's _PACKAGING vocabulary. It already refuses to match a line
    # made only of these, but the store's gate asks a different object the same question and
    # was still probing for them, so a catalog row named "Pint Cake" stayed reachable from
    # "1 PINT". Two vocabularies drifting apart is the bug; the words themselves are chrome
    # either way.
    "alc", "bottles", "can", "cans", "crowler", "draft", "draught", "growler", "keg",
    "liter", "liters", "litre", "litres", "milliliter", "milliliters", "net", "ounce",
    "ounces", "pack", "pint", "pints", "sixpack", "volume",
    # The ingredient line. "NO COLORS OR FLAVORS FROM ARTIFICIAL SOURCES" is printed on a
    # Miller High Life can, and "with natural and artificial flavors" on every flavored malt
    # beverage's TTB filing -- and in 2026-09-15's log those words, and only those, made two
    # lines of a Miller can agree on a stout whose registered name is its whole label.
    "flavor", "flavors", "flavours", "flavoured", "color", "colors", "colour", "colours",
    "colored", "coloured", "artificial", "artificially",
    # What an aperitivo shelf prints for what is in the bottle, in the languages it prints
    # it in. A producer registered as "Terrativo Aperitivo" was hypothesised off a Campari
    # label on the word APERITIVO alone (2026-09-15).
    "aperitivo", "aperitif", "amaro", "vermouth", "vermut", "bitter", "bitters", "liquore",
    "grappa", "sambuca", "limoncello",
}


def _tokens(s: str) -> set[str]:
    return set(_word_list(s))


def _word_list(s: str) -> list[str]:
    """`_tokens`, in the order they were written. For the one question a set cannot answer:
    whether a name reads as a phrase inside another name."""
    d = unicodedata.normalize("NFKD", s or "")
    a = "".join(c for c in d if not unicodedata.combining(c)).casefold()
    return _WORD.findall(a)


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


def same_word(a: str, b: str) -> bool:
    """Whether two tokens are one word, allowing for the way people type on a phone.

    For reading a name a DRINKER typed, which is the one place in this system where the input
    is known to contain typos: every other source is a filing or a scrape, where a wrong letter
    means a different product rather than a slipped thumb.
    """
    if a == b:
        return True
    if len(a) < _TYPO_MIN_LEN or len(b) < _TYPO_MIN_LEN:
        return False
    if abs(len(a) - len(b)) > _TYPO_MAX_LEN_GAP:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= _TYPO_SIM


def _covered_allowing_typos(sub: dict, sup: dict) -> bool:
    """`_covered`, for a typed name: a token counts as present if some token on the other side
    is the same word mistyped."""
    uncovered = {t for t in sub["nt"] if not any(same_word(t, s) for s in sup["full"])}
    if not uncovered:
        return True
    if not uncovered <= _AGE_EQUIV:
        return False
    return bool((sub["nt"] - uncovered) - _STYLE)


def _reads_as_a_phrase_in(want: list[str], phrase: list[str]) -> bool:
    """Whether `want` appears in `phrase` as consecutive words, typos allowed.

    What a set of tokens cannot see, and the thing that made containment too generous for
    common words. "Hazy Little Thing" and "Juicy Little Thing Hazy IPA" share every word of
    the former, so set containment called them the same drink -- but the words are not
    together and not in that order, and in the second one "Hazy" is a description while
    "Juicy" is the name. Measured on the live catalog, requiring the phrase drops a typed
    "Hazy Little Thing" from 5 candidate rows to 3, and the two it drops are the two that
    were not that beer.

    A product name is a phrase, and the words a person types off a label come in the order
    the label prints them. Extra words on either END are fine -- that is the brand in front
    ("Sierra Nevada Hazy Little Thing") and the style behind ("Hazy Little Thing IPA"), which
    is exactly what a less specific name is missing.
    """
    if not want:
        return True
    if len(want) > len(phrase):
        return False
    for i in range(len(phrase) - len(want) + 1):
        if all(same_word(w, phrase[i + j]) for j, w in enumerate(want)):
            return True
    return False


def match_typed_name(name: str, maker: str | None,
                     products: Iterable[dict]) -> list[str]:
    """Catalog ids a drink somebody TYPED could already be, misspellings included.

    The same three gates `find_substring_merges` applies, because the question is the same one
    -- is this row already the row somebody means -- with typo tolerance added and the strict
    "more specific" requirement dropped: a contribution matching a row exactly is the case this
    exists to catch, and "Hazy Little Thing" for "Sierra Nevada Hazy Little Thing" is the next.

    Returns every id that survives, and the CALLER must care how many there are. One is an
    answer; several mean the typed name does not say which, and that is not a merge to make
    automatically. A bare "Crusher" covers The Alchemist's, a different brewery's own bare
    "Crusher", and `Bone Crusher Stout` -- and quietly picking one would hang the wrong
    producer, the wrong flavour profile and the wrong town off the can in somebody's hand.

    Digits may be ADDED by the catalog but not by the typist: a typed "Lagavulin" can be the
    16-year-old, while a typed "Lagavulin 16" is not the row called plain "Lagavulin". Ages are
    identity, so a typed age that names three different bottles comes back as three.
    """
    typed = _prep({"id": "typed", "name": name, "brand": maker or ""})
    if not typed["nt"]:
        return []
    typed_seq = _word_list(name)
    out: list[str] = []
    for p in products:
        cand = _prep(p)
        if not typed["dig"] <= cand["dig"]:
            continue
        if not _brand_compatible(typed, cand):
            continue
        if not _covered_allowing_typos(typed, cand):
            continue
        # What they have in common has to include a word that identifies a product. Sharing
        # only "ipa" and "double" is sharing a category.
        shared = {t for t in typed["nt"] if any(same_word(t, s) for s in cand["full"])}
        if not (shared - _STYLE):
            continue
        # And it has to read as a phrase, not merely consist of the same words. Dropping the
        # words that appear nowhere first, which `_covered_allowing_typos` has already
        # certified are only age statements -- "Lagavulin 16 ans" still reaches "Lagavulin 16
        # Year Old" on "lagavulin", with the French word gone rather than demanded in place.
        absent = {t for t in typed["nt"] if not any(same_word(t, s) for s in cand["full"])}
        want = [t for t in typed_seq if t not in absent]
        phrase = _word_list(search_name(str(p.get("name") or ""), p.get("brand") or None))
        if not _reads_as_a_phrase_in(want, phrase):
            continue
        out.append(p["id"])
    return out


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


def carries_no_identity(name: str) -> bool:
    """Whether a name has no word that could pick this product off a shelf: nothing four
    letters or longer that isn't a category word.

    Deliberately the same question `_token_supported` asks of a candidate before it will
    accept agreement with a label, and the same one the gate asks of an OCR line -- kept
    here so ingest, resolution and the SQL gate cannot drift apart on what "carries no
    identity" means. Broader than is_generic_name, which requires *every* token to be a
    style word: "J&B" and "1664" carry no identity either, having no long word at all.
    """
    return not [t for t in _tokens(name) if len(t) >= 4 and not is_generic_token(t)]


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


# --- producers -------------------------------------------------------------------------
#
# TTB names a producer by the brand on each filing, so one company arrives once per spelling
# it has ever used: "Lagunitas", "Lagunitas Brewing Co", "The Lagunitas Brewing Company".
# Those are one producer, and a differing location does not make them two — a company cans
# in more than one place.

#: Stripped only from the END of a name, repeatedly ("Brewing Co., LLC"). A trade word is
#: noise where it trails the name and identity where it does not: stripping it anywhere
#: turns "Ale House Brewing Co" into "house", which then collects unrelated companies in
#: seven states.
_PRODUCER_SUFFIX = {
    "brewing", "brewery", "breweries", "brewers", "brewhouse", "brewpub", "brewer",
    "distilling", "distillery", "distilleries", "distillers", "distiller",
    "winery", "wineries", "vineyards", "vineyard", "cellars", "cellar",
    "company", "co", "corp", "corporation", "inc", "incorporated",
    "llc", "lc", "llp", "lp", "ltd", "limited", "plc", "pllc",
    "gmbh", "bv", "nv", "srl", "spa", "ag", "kg", "oy", "ab",
    "holdings", "group",
}

_BREW_WORDS = {"brewing", "brewery", "breweries", "brewers", "brewhouse", "brewpub"}
_DIST_WORDS = {"distilling", "distillery", "distilleries", "distillers"}


def _producer_tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ").replace("'", "").replace("`", "")
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def _is_suffix_token(tok: str) -> bool:
    """A trailing token that means incorporation — spelled correctly or not.

    Filers misspell their own suffix often enough to matter: "Brewing Comapny",
    "Companuy", "Destillery", "Disitllery", "Breewery". Each misspelling forks a company
    into its own producer. Only long tokens are fuzzy-matched, and only against this set,
    so a real word is not mistaken for a typo of one.
    """
    if tok in _PRODUCER_SUFFIX:
        return True
    if len(tok) < 6:
        return False
    return any(difflib.SequenceMatcher(None, tok, s).ratio() >= 0.85
               for s in _PRODUCER_SUFFIX if len(s) >= 6)


def producer_core(name: str) -> str:
    """The identifying part of a producer name: no leading "The", no trailing incorporation.

    Returns "" for a name that has nothing left — a punctuation-only name, or one written
    in a script the ASCII fold erases. Those must never be grouped: ten producers whose
    names are Greek or Korean all fold to the same empty string.
    """
    toks = _producer_tokens(name)
    while toks and toks[0] == "the":
        toks = toks[1:]
    while len(toks) > 1 and _is_suffix_token(toks[-1]):
        toks = toks[:-1]
    return " ".join(toks)


def producer_trade(name: str) -> str:
    """'brew', 'dist', or '' when the name does not say.

    A brewery and a distillery are different producers even under one name, so this
    partitions a cluster rather than describing it.
    """
    toks = set(_producer_tokens(name))
    brew, dist = bool(toks & _BREW_WORDS), bool(toks & _DIST_WORDS)
    if brew == dist:          # neither, or confusingly both
        return ""
    return "brew" if brew else "dist"


def _producer_canonical(rows: list[dict]) -> dict:
    """The row the others fold into: the plainest spelling of the name.

    Shortest wins, because that is the bare brand a label actually carries — "Lagunitas"
    over "The Lagunitas Brewing Company". Ties break on id so the choice is stable.
    """
    return min(rows, key=lambda r: (len(r.get("name") or ""), r.get("id") or ""))


def find_producer_merges(producers: Iterable[dict]) -> list[tuple[str, str]]:
    """(alias_id, canonical_id) for producer rows that name one company."""
    by_core: dict[str, list[dict]] = defaultdict(list)
    for p in producers:
        core = producer_core(p.get("name", ""))
        if core:
            by_core[core].append(p)

    out: list[tuple[str, str]] = []
    for rows in by_core.values():
        by_trade: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_trade[producer_trade(r.get("name", ""))].append(r)
        known = [k for k in by_trade if k]

        if len(known) == 1:
            # Only one trade under this name, so a row that does not say which it is can
            # only be that one.
            groups = [by_trade[known[0]] + by_trade.get("", [])]
        elif not known:
            groups = [by_trade[""]]
        else:
            # Both a brewery and a distillery answer to this name. They are different
            # companies, and a row saying neither cannot be assigned to either, so it is
            # left alone rather than guessed at.
            groups = [by_trade[k] for k in known]

        for group in groups:
            if len(group) < 2:
                continue
            canon = _producer_canonical(group)
            out.extend((r["id"], canon["id"]) for r in group if r["id"] != canon["id"])
    return out
