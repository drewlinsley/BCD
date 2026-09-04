"""Scan resolution + personal scoring — the server half of the latency-critical path.

Kept deliberately simple and dependency-light here (LIKE match + a transparent
chemistry-based cold-start scorer) so it runs on the laptop store. In production the
match step is Postgres trigram + pgvector ANN, and scoring blends the learned
ingredient->sensory model with the user's TasteProfile. The *shape* is what the iOS
client codes against and what we optimize behind.
"""

from __future__ import annotations

import re
import unicodedata

from bcd_ingest.dedup import is_generic_token, search_name
from bcd_ingest.store import Store, _cosine
from bcd_schema import (
    SENSORY_AXES,
    Brand,
    Producer,
    Product,
    ResolvedProduct,
    ScanResolveRequest,
    ScanResolveResponse,
    ScoredCandidate,
    SensoryVector,
    TasteProfile,
)

# A text detection resolves to a product only if its name-match clears this floor. Tuned
# against the real catalog: real beers (Heineken 1.0, Krombacher 0.69, a "GUINNESS DRAUGHT
# 440ML" line 0.56) clear it; OCR chrome ("12 FL OZ" 0.38, "BREWED AND BOTTLED BY" 0.33) does
# not — so noise resolves to nothing instead of a confident wrong beer.
_MIN_MATCH = 0.5
# A very short catalog name ("J&B", "1664") is low-information and trigram-matches garbled OCR
# far too easily, so it must clear a near-exact bar instead of the normal floor. Observed live: a
# mangled Heady-Topper-can frame matched the scotch "J&B" at exactly 0.5.
_SHORT_MIN_MATCH = 0.8
_SHORT_NAME_LEN = 5
# Cap overlays per frame so a busy shelf can't bury the HUD (the client caps + anchors too).
_MAX_CANDIDATES = 8

# Score bands for the overlay's one-line 'why'. Above _STRONG_MATCH we claim a match;
# below _MILD_MATCH we say so plainly rather than dressing up a miss.
_STRONG_MATCH = 0.8
_MILD_MATCH = 0.6

# A detection is a *product identity* only if it carries a real word — a run of >=3 letters (a
# brand or name token). A bare number is label chrome, not a name: "15" off a 15th-anniversary
# can, "40" for proof, "500" for mL, "5" for %ABV. Trigram-matching those resolves to whatever
# junk shares the digits (a can's "15" once matched a spirit literally named "15"), so they must
# resolve to nothing. The one numeric exception is a 4+-digit run that IS the identity ("1664").
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_LONGNUM_RE = re.compile(r"^[0-9]{4,}$")


def _is_identity_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _WORD_RE.search(t):
        return True
    return bool(_LONGNUM_RE.match(t.replace(" ", "")))


# Even a line that clears the score floor can be a *coincidental* trigram window rather than a
# real name hit. OCR mangles the ubiquitous Surgeon-General warning ("...impairs your ability to
# drive A CAR OR operate machinery") into a fragment like "BACAR OR", which word-similarity-matches
# "Bacardi" at 0.625 — HIGHER than a legitimately-embedded "GUINNESS DRAUGHT 440ML" line scores
# (0.56). Measured against the real catalog, no similarity threshold separates the two. What does
# separate them is token agreement: a genuine match carries an OCR token that *is* a name word
# ("GUINNESS" == "Guinness"), while the garble only has a truncation ("BACAR" vs "Bacardi" ~ 0.56,
# "OR" vs "Bacardi" = 0). So a text match must also carry a real name token (>=4 letters) that
# closely matches some OCR token — otherwise the "brand" is only a coincidental sub-window.
_TOKEN_SUPPORT_MIN = 0.85
_MIN_NAME_TOKEN_LEN = 4
_TOKEN_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)  # runs of >=2 unicode letters


def _norm_token(s: str) -> str:
    """Casefold and strip diacritics so 'Bière' and 'BIERE' compare equal."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _tokens(s: str) -> list[str]:
    return [_norm_token(t) for t in _TOKEN_RE.findall(s or "")]


def _trigrams(token: str) -> set[str]:
    # pg_trgm-style padding: two leading spaces + one trailing, then 3-grams. Mirrors the
    # store's similarity() closely enough to calibrate one threshold across both.
    padded = f"  {token} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def _trigram_sim(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


#: How much of the line a candidate must account for before a frame with nothing to
#: corroborate against is allowed to certify itself. Measured on real frames: the fragments
#: top out at 0.667 ("Chemist" against "CHEMIST VER") and the whole-label reads start at
#: 0.862, so 0.7 sits in the gap rather than on either population.
_ACCOUNTS_FOR_LINE = 0.7


def _flatten(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _accounts_for_the_line(name: str, line: str) -> bool:
    """Whether the candidate is the whole of what was read, or only a piece of it.

    Containment cannot tell the difference: `word_similarity` is 1.0 for ANY name wholly
    inside the line, so a catalog row named "Mist" scores 1.0 against "ACHE MIST-VERM" --
    which is THE ALCHEMIST VERMONT with the wordmark split mid-word by the recognizer. The
    can produced "ALCHE MIST VERM" and "ACHE MISTVERN" too, and every one of those pieces is
    a real product name someone has registered.

    Plain similarity is the measure that penalises what the name leaves out, which is exactly
    the question here. It also, correctly, denies the exemption to "Draught Stout" read off
    "GUINNESS DRAUGHT STOUT": that row is not the whole label either, and the frame should
    say so rather than certify itself.
    """
    return _trigram_sim(_flatten(name), _flatten(line)) >= _ACCOUNTS_FOR_LINE


def _identifying_tokens(name: str) -> list[str]:
    """Name tokens that could actually pick this product off a shelf: long enough to be a real
    word, and not a category or packaging word every other label carries too."""
    return [t for t in _tokens(name)
            if len(t) >= _MIN_NAME_TOKEN_LEN and not is_generic_token(t)]


def _token_supported(query: str, name: str) -> bool:
    """True if an *identifying* name token closely matches some OCR token — evidence the brand
    word is actually present in the line, not a coincidental trigram window.

    Agreement on a category word is not evidence. A Heady Topper can carries "AMERICAN DOUBLE
    IPA" and "DRINK FROM THE CAN"; matching on "double" and "drink" pulled in an unrelated hazy
    IPA and a product named "Life drink", and both outranked the real beer because all three sat
    within 0.012 of each other just above the floor. A name with nothing identifying in it
    ("J&B", "1664", "Hazy Double IPA") has nothing to anchor on and defers to the raised floor
    the caller applies instead."""
    name_tokens = [t for t in _tokens(name) if len(t) >= _MIN_NAME_TOKEN_LEN]
    if not name_tokens:
        return True
    identifying = _identifying_tokens(name)
    if not identifying:
        # Real words, but every one of them is a category word: "Ipa Ipa", "Irish Whiskey".
        # Such a name can only be what the label names if the label is equally generic. When
        # the line does carry something specific the row cannot account for, the agreement is
        # a coincidence — "DOGFISH HEAD 60 MINUTE IPA" resolved to a row literally named "Ipa
        # Ipa", because containment scores it 1.0 and so the raised floor this used to defer
        # to never bit.
        return not _identifying_tokens(query)
    q_tokens = _tokens(query)
    return any(_trigram_sim(nt, qt) >= _TOKEN_SUPPORT_MIN
               for nt in identifying for qt in q_tokens)


# ---- frame-level corroboration ----
# A real label corroborates itself. A can carries its brewery *and* its beer, so the product
# it names is named by more than one line: "THE ALCHEMIST" and "HEADY TOPPER" agree. Label
# chrome does not corroborate — "PINT" names exactly one catalog row ("Pint Cake") and nothing
# else on the can agrees with it.
#
# Scoring each line in isolation cannot see that difference. Word-similarity asks how well a
# line matches *part of* a name, not how much of the name it accounts for, so "PINT" is a
# flawless hit inside "Pint Cake" and scores 1.00 — identical to the real beer. At 4.7k
# products that was harmless because no row was named "Pint Cake"; at 363k every common word
# stamped on a can (PINT, CAN, DRINK, DOUBLE) is a perfect word-match for *something*, four
# candidates tie at 1.00, and the right one ranks by luck. Counting how many distinct lines
# name a candidate is what separates the beer from the chrome.
_UNCORROBORATED = 0.75
# Below this many identity-bearing lines there is no corroborating evidence to be had, so a
# lone hit is not evidence of weakness — a barcode or a single clean brand line must stay
# confident. The penalty applies only where other lines *could* have agreed and none did.
_MIN_FRAME_FOR_PENALTY = 2


# Packaging and measure words. The shared dedup vocabulary already knows the *style* words a
# label carries ("american", "double", "ipa", "stout", "drink"); it does not know the words that
# describe the container, because a container word is a perfectly good part of a catalog *name*
# and dedup must not fold "Proper Pint" into "Proper". Here the question is different — whether
# an OCR line is worth matching at all — so the resolver keeps its own list rather than widening
# dedup's and changing how the catalog merges.
#
# This is where "PINT" was getting in. It is not a beer, it is the size of the can, and matching
# it cost 1.5s to return "Pint Cake".
_PACKAGING = {
    "pint", "pints", "can", "cans", "canned", "bottle", "bottled", "bottles",
    "draft", "draught", "keg", "growler", "crowler", "ounce", "ounces",
    "milliliter", "milliliters", "litre", "litres", "liter", "liters",
    "pack", "sixpack", "contents", "volume", "net", "vol", "alc",
}


def _worth_matching(text: str) -> bool:
    """Whether an OCR line could name a product at all — asked *before* the trigram query.

    A line built only from container words names a size, not a drink, so matching it can only
    produce a coincidence — "PINT" cost 1.5s to return "Pint Cake". Filtering before the query
    is what makes it cheap: a short common word is also the most expensive thing to match,
    because it matches tens of thousands of rows.

    Deliberately narrower than "has nothing identifying in it". A line of pure *category* words
    must still be matched, because a catalog name can be pure category too: "FML Hazy Double
    IPA" is a real product and a clean read of it has to resolve. Those lines are already
    handled — `_token_supported` lets a generic line match a generic name and nothing else — so
    widening this to category words costs recall and buys nothing the frame does not already
    fix by ranking."""
    meaningful = [t for t in _tokens(text) if len(t) >= _MIN_NAME_TOKEN_LEN]
    if not meaningful:
        return True                      # nothing to judge; leave it to the existing guards
    return any(t not in _PACKAGING for t in meaningful)


# ---- the producer path ----
# When the camera cannot read the product name, it can often still read the maker. A Heady
# Topper can's wordmark is a wavy psychedelic script: across 38 live frames Vision returned the
# beer's own name 3 times out of 100 lines — once as Cyrillic, "АДУ ТОРИ" — while the small rim
# print, "ALCHEMIST-VER…", came through 23 times. The brewery is the readable half of that
# label, and a brewery with two products is a far narrower answer than 363k rows.
_PRODUCER_MATCH_MIN = 0.6
# Past this many products in the hinted category, the label has identified a *maker* and not a
# drink. Offering a guess then would be inventing one.
_PRODUCER_MAX_PRODUCTS = 4
# A producer hit is indirect evidence, so it must never outrank a product the label actually
# names — only the nothing it is competing against.
_PRODUCER_EVIDENCE = 0.6
# A candidate whose category *contradicts* the label's own fine print. Not merely unsupported —
# the frame says one thing and the row says another, which is evidence against, not absence of
# evidence. "A CHEMIST VER" off this can matched a distillery's `Chemist` at 1.00 while the same
# frame read "ALE"; 17 of that maker's 19 products are spirits. 'other' is never a contradiction,
# because it means the catalog does not know, not that it disagrees.
_CATEGORY_CONTRADICTS = 0.5

# The category words a label prints in its fine print. This is the one part of a stylized can
# the OCR reads reliably — "ALE / ALC. 8% BY VOL / 1 PINT" came through on 25 of those 100
# lines, unfailingly, while the brand did not. It cannot name a product, but it names a
# category, and that is exactly what separates the right maker from the wrong one here:
# Alchemist makes 1 beer, while Chemist makes 17 spirits and Cocktail Chemist 7.
_CATEGORY_WORDS = {
    "beer": {"ale", "lager", "stout", "porter", "pilsner", "pilsener", "ipa", "beer",
             "bock", "saison", "gose", "witbier", "weisse", "hefeweizen", "kolsch",
             "brew", "brewed", "malt", "pale", "amber", "dunkel", "tripel", "dubbel"},
    "spirit": {"vodka", "gin", "whiskey", "whisky", "rum", "tequila", "bourbon", "brandy",
               "cognac", "mezcal", "liqueur", "scotch", "rye", "absinthe", "schnapps",
               "distilled", "proof"},
    "wine": {"wine", "vino", "chardonnay", "merlot", "cabernet", "riesling", "rose",
             "prosecco", "champagne", "sauvignon", "pinot", "syrah", "zinfandel"},
}


def _category_hint(detections) -> str | None:
    """The category the label's own fine print names, or None if it says nothing or disagrees.

    Counted across the whole frame rather than taken from the first match, because a single
    word is easy to misread and a can carries several ("ALE", "ALC", "1 PINT"). A tie means the
    label is ambiguous and the hint is withheld — a wrong category filter is worse than none.
    """
    votes: dict[str, int] = {}
    for det in detections:
        for tok in _tokens(det.text):
            for cat, words in _CATEGORY_WORDS.items():
                if tok in words:
                    votes[cat] = votes.get(cat, 0) + 1
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _candidate_vocabulary(resolved: ResolvedProduct) -> list[str]:
    """The identifying words that would name this product on a label: its own name, its brand
    and its producer. The producer is what carries the signal — "THE ALCHEMIST" is the line
    that tells the real Heady Topper apart from a one-word coincidence."""
    seen: dict[str, None] = {}
    for part in (resolved.product.name, resolved.brand.name, resolved.producer.name):
        for t in _identifying_tokens(part or ""):
            seen[t] = None
    return list(seen)


# Two lines that read the same printed phrase are one piece of evidence, not two.
_LINE_REREAD = 0.7


def _independent_lines(line_tokens: list[list[str]]) -> list[list[str]]:
    """Collapse detections that are re-reads of one another.

    Corroboration is supposed to mean separate parts of the frame agreeing. A four-pack prints
    its brand once per can, so a single phrase arrives as three detections -- and whatever they
    happened to share got certified by its own echo. "LITTLE" read twice off a Little Willow
    pack proved six unrelated products with `little` in the name, and "DRINK FROM THE CAN!"
    read three times proved one called `Now & Then`, off the word THEN.

    Longest reading first, so the fullest version of a repeated phrase is the one kept: a line
    whose identifying tokens are nearly all already accounted for is an echo of it. Matching is
    fuzzy because each re-read is garbled differently -- "LITTLE WILLOW BREWING COMPANT" and
    "LITTLE / KEWING" are the same text off two cans.
    """
    kept: list[list[str]] = []
    for toks in sorted(line_tokens, key=len, reverse=True):
        sig = [t for t in toks if len(t) >= _MIN_NAME_TOKEN_LEN]
        if not sig:
            continue
        echo = any(
            sum(any(_trigram_sim(t, k) >= _TOKEN_SUPPORT_MIN for k in seen) for t in sig)
            / len(sig) >= _LINE_REREAD
            for seen in kept
        )
        if not echo:
            kept.append(sig)
    return kept


def _frame_support(vocab: list[str], line_tokens: list[list[str]], *,
                   category: str | None = None, hint: str | None = None) -> int:
    """How many distinct pieces of the frame agree with this candidate.

    Mostly that means detections naming it, using the same token-agreement test the per-line
    guard uses — so the answer is directly comparable and one threshold calibrates both.

    The label's category counts as one more, because that is what it is: another line of the
    frame agreeing. It is also the *reliably read* one. A stylized can whose brand OCRs as
    Cyrillic still prints "ALE / ALC. 8% BY VOL" in plain type, and that line is what separates
    `The Alchemist Heady Topper` (beer) from `Alchemist Amer` (other) when both come off the
    same brewery and the name match alone favours the wrong one.
    """
    n = 0
    if vocab:
        n = sum(
            any(_trigram_sim(v, t) >= _TOKEN_SUPPORT_MIN for v in vocab for t in toks)
            for toks in line_tokens
        )
    if hint and category and category == hint:
        n += 1
    return n


# A name too short to contain an identifying token has nothing for `_token_supported` to
# anchor on, so that guard waves it through and only the raised floor stands between it and any
# fragment that starts with the same letters. That is not enough, because the floor is measured
# with `word_similarity`, which asks whether the name appears *inside* the line — and a 3-letter
# name appears inside almost anything. A catalog row literally named `Ver` matched "VERMIKI",
# "VERMIL" and "VERM" off a Vermont can, all at ~1.0.
#
# Plain similarity is the right question for these, because it is the one measure that penalises
# what the name leaves out: 'ver' scores 1.00 against "VER" and 0.33 against "VERMONT".
_SHORT_NAME_SIM = 0.8


def _short_name_supported(query: str, name: str) -> bool:
    """Whether a very short name was actually *read*, rather than merely contained.

    A name with no letter tokens at all ("1664", "J&B") cannot be tested this way and defers to
    the raised floor, exactly as before — those are real products and must stay reachable."""
    name_toks = _tokens(name)
    if not name_toks:
        return True
    return any(_trigram_sim(nt, qt) >= _SHORT_NAME_SIM
               for nt in name_toks for qt in _tokens(query))


def _upc_variants(upc: str) -> list[str]:
    """A barcode's equivalent GTIN forms. A UPC-A (12 digits) and its EAN-13 form differ only by a
    leading zero and identify the *same* item, but a scanner and the catalog may store different
    forms — so a lookup tries both. EAN-8 and other lengths are used as-is."""
    u = (upc or "").strip()
    out = [u]
    if u.isdigit():
        if len(u) == 12:
            out.append("0" + u)             # UPC-A -> EAN-13
        elif len(u) == 13 and u.startswith("0"):
            out.append(u[1:])               # EAN-13 -> UPC-A
    return list(dict.fromkeys(out))         # de-dup, preserve order


_KEY_STRIP_RE = re.compile(r"[^a-z0-9]+")


def _norm_key(s: str) -> str:
    """Lowercase, strip accents, keep alphanumerics — digits included. Unlike `_tokens` (which
    keeps only letters, for word matching), the identity key must preserve numbers: "0.0%", "12
    ans", "Select 55" are real product distinctions, not noise. "Jupiler 0,0%" -> "jupiler 0 0"."""
    decomposed = unicodedata.normalize("NFKD", s or "")
    ascii_ = "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
    return _KEY_STRIP_RE.sub(" ", ascii_).strip()


def _identity_key(name: str, brand: str, pid: str) -> str:
    """One key per *real* product, so duplicate catalog records collapse into a single overlay.

    The catalog carries the same beer under multiple rows — different UPCs of one product (Lagunitas
    IPA ×2, Heineken ×7), or an OFF pull plus a TTB record. Keyed on the raw id they'd each draw
    their own overlay; keyed on normalized brand+name they merge. Brand is part of the key on
    purpose: OFF often names a product only by its class ("Blended Scotch Whisky" for eight
    different distilleries), and those must stay distinct — their brands (Johnnie Walker vs Queen
    Margot) are what separate them. A placeholder "Unknown" brand (or one that just echoes the name)
    carries no identity, so it drops out and the name alone keys — which is what lets the two
    Unknown-branded "Lagunitas IPA" rows collapse. Digits are kept, so an alcohol-free "0.0%" or an
    age-stated "12 ans" stays a distinct product from its sibling. A row with no alphanumeric name
    falls back to its id, so unnamed (e.g. non-Latin) rows never merge into one another."""
    n = _norm_key(name)
    if not n:
        return f"id:{pid}"
    b = _norm_key(brand)
    if not b or b == "unknown" or b == n:
        return n
    return f"{b}\x1f{n}"


class Resolver:
    def __init__(self, store: Store) -> None:
        self.store = store

    # Matching is delegated to the store: token-overlap on the SQLite dev store,
    # real pg_trgm trigram similarity on Postgres — same signature either way.
    def _resolve_by_upc(self, upc: str) -> dict | None:
        for key in _upc_variants(upc):
            sku = self.store.get_gold(f"sku:{key}")
            if sku:
                return self.store.get_gold(sku["product_id"])
        return None

    def _hydrate(self, product_rec: dict) -> ResolvedProduct | None:
        producer = self.store.get_gold(product_rec.get("producer_id", ""))
        brand = self.store.get_gold(product_rec.get("brand_id", ""))
        if producer is None:
            producer = Producer(id="unknown", name="Unknown").model_dump(mode="json")
        if brand is None:
            brand = Brand(id="unknown", producer_id=producer["id"],
                          name=product_rec.get("name", "")).model_dump(mode="json")
        return ResolvedProduct(
            product=Product.model_validate(product_rec),
            producer=Producer.model_validate(producer),
            brand=Brand.model_validate(brand),
        )

    def _by_producer(self, lines: list[tuple[int, str]],
                     hint: str | None) -> list[tuple[int, dict, float]]:
        """Products inferred from the maker the label names, when it never named a drink.

        Reached only for a frame nothing corroborates, and discounted, so this competes with
        the nothing it would otherwise return — never with a product the label actually says.

        The category hint is what makes it safe. "A CHEMIST VER" off a Heady Topper can is a
        better trigram match for a distillery named `Chemist` (1.00) than for `Alchemist`
        (0.60), because OCR dropped the "AL" and word-similarity cannot recover a lost prefix.
        No name threshold separates those. What does is the fine print the same frame read
        perfectly: the can says ALE, `Chemist` makes 17 spirits, and `Alchemist` makes the beer.
        """
        match = getattr(self.store, "match_producers", None)
        products_of = getattr(self.store, "products_of", None)
        if match is None or products_of is None:
            return []                       # a store without the producer path; not an error
        out: list[tuple[int, dict, float]] = []
        seen: set[str] = set()
        for i, text in lines:
            # A line with nothing identifying in it cannot name a maker either. Without this
            # the two generic halves agree with each other: _token_supported accepts a
            # styleless name against a styleless line, so "DRINK FROM" -- now that both words
            # are known chrome -- reached a producer literally registered as "drink drink!"
            # and offered its beer at 0.60.
            if not _identifying_tokens(text):
                continue
            for prod, sc in match(text):
                pid = prod.get("id") or ""
                if sc < _PRODUCER_MATCH_MIN or pid in seen:
                    continue
                # Same coincidental-window guards the product path uses: the producer's name
                # must actually be a word in the line, not a trigram accident — and a very
                # short one must have been read rather than merely contained. A producer
                # literally named `Ver` was reached from "VERMIKI" and "VERMONT" here after
                # the product path had already been taught not to.
                pname = prod.get("name") or ""
                if not _token_supported(text, pname):
                    continue
                if len(pname) < _SHORT_NAME_LEN and not _short_name_supported(text, pname):
                    continue
                seen.add(pid)
                items = products_of(pid)
                if hint:
                    items = [p for p in items if (p.get("category") or "") == hint]
                if not items or len(items) > _PRODUCER_MAX_PRODUCTS:
                    # A maker with a whole shelf has been identified; a drink has not. Saying
                    # which one would be inventing it.
                    continue
                for rec in items:
                    out.append((i, rec, round(sc * _PRODUCER_EVIDENCE, 3)))
        return out

    def _match_lines(self, texts: list[str]) -> list[list[tuple[dict, float]]]:
        """A frame's name matches, concurrently where the store can. The fallback keeps any
        store that only implements the single-line `match_products` working unchanged."""
        many = getattr(self.store, "match_products_many", None)
        if many is not None:
            return many(texts)
        return [self.store.match_products(t) for t in texts]

    def _qualified_name(self, rec: dict) -> str:
        """"<brand> <name>" for a product row, matching what `search_name` stores.

        The catalog splits a label in two, so half of it is invisible to any check that
        reads only `name`. Resolved through the store rather than a join because there are
        at most a handful of candidates per detection.
        """
        brand_id = rec.get("brand_id")
        brand = self.store.get_gold(brand_id) if brand_id else None
        return search_name(rec.get("name") or "", (brand or {}).get("name"))

    # ---- scoring ----
    def score(self, product: Product, profile: TasteProfile | None) -> tuple[float, str, bool]:
        """Predicted 0-1 enjoyment + a one-line reason + cold_start flag.

        Cold start = we scored it from chemistry/style alone, no reviews needed. That is
        the differentiator, so we flag and surface it.
        """
        sensory = product.sensory
        cold_start = sensory is not None and sensory.source.value in (
            "chemistry_prior", "style_prior"
        )
        if profile is None or profile.sensory_ideal is None or sensory is None:
            # No personalization yet: fall back to a mild style-affinity prior.
            style = (product.style.value if product.style else "") or ""
            aff = (profile.style_affinities.get(style, 0.0) if profile else 0.0)
            return (0.5 + 0.5 * aff, "based on style", cold_start)

        sim = _cosine(sensory.to_array(), profile.sensory_ideal.to_array())
        score = max(0.0, min(1.0, 0.5 + 0.5 * sim))
        return (round(score, 3), _match_reason(score, sensory, profile.sensory_ideal), cold_start)

    def resolve(self, req: ScanResolveRequest,
                profile: TasteProfile | None = None) -> ScanResolveResponse:
        """Resolve a whole frame, not a list of independent lines.

        A label is one object photographed once, so its lines are evidence about the *same*
        product and are strongest read together. Two passes: per line, every candidate that
        clears the existing guards; then per candidate, what the rest of the frame says about
        it. The guards are unchanged — they decide what may be evidence at all — and the frame
        decides which evidence wins.
        """
        line_tokens = [_tokens(d.text) for d in req.detections]
        # What corroboration is allowed to count: the frame's distinct readings, not its echoes.
        independent = _independent_lines(line_tokens)
        identity_lines = sum(1 for d in req.detections if _is_identity_text(d.text))
        hint = _category_hint(req.detections)

        # ---- pass 1: every candidate any line supports, not just that line's best ----
        # Keeping only the top hit per line is what let chrome crowd out the beer: the real
        # product could be a line's second candidate and never be considered at all.
        hits: list[tuple[int, dict, float]] = []          # (line, record, that line's score)
        by_upc: set[str] = set()                         # records a barcode identified outright
        resolved_lines: set[int] = set()
        to_match: list[int] = []                          # lines worth a name query
        for i, det in enumerate(req.detections):
            if det.kind == "barcode":
                rec = self._resolve_by_upc(det.text)
                if rec is not None:
                    hits.append((i, rec, 1.0))
                    resolved_lines.add(i)
                    by_upc.add(rec.get("id") or "")
                continue
            if not _is_identity_text(det.text) or not _worth_matching(det.text):
                # A bare number/fragment is label chrome, not a name — and so is a line made
                # only of category and packaging words. Skipped rather than trigram-matched.
                continue
            to_match.append(i)

        # Ask for the frame's lines together. Each one costs a GIN scan sized by how common
        # its trigrams are, so in series a six-line can spends ~2s against a 700ms HUD tick;
        # the Postgres store runs them concurrently and the frame costs about its slowest
        # line instead of their sum.
        for i, found in zip(to_match, self._match_lines(
                [req.detections[i].text for i in to_match]), strict=True):
            det = req.detections[i]
            for rec, sc in found:
                # Judge the evidence on the brand-qualified name, because that is what the
                # label actually says. A row named "Irish Whiskey" is anonymous on its own;
                # as "Jameson Irish Whiskey" it is the product the line names.
                name = self._qualified_name(rec)
                # Low-information either way: too short to be distinctive, or built only from
                # category words. Both trigram-match label chrome far too easily, so they must
                # clear a near-exact bar rather than the normal floor.
                too_short = len(name) < _SHORT_NAME_LEN
                low_info = too_short or not _identifying_tokens(name)
                floor = _SHORT_MIN_MATCH if low_info else _MIN_MATCH
                if sc < floor or not _token_supported(det.text, name):
                    continue
                # A very short name additionally has to have been read, not just contained.
                if too_short and not _short_name_supported(det.text, name):
                    continue
                hits.append((i, rec, sc))
                resolved_lines.add(i)
        # Distinct lines backing each record, read straight off the hits — enough to tell
        # whether the frame agreed on anything, without paying to hydrate first.
        backing: dict[str, set[int]] = {}
        for i, rec, _ in hits:
            backing.setdefault(rec.get("id") or "", set()).add(i)
        if not any(len(v) >= _MIN_FRAME_FOR_PENALTY for v in backing.values()):
            # Nothing the frame corroborates: the label has not named a product to us. Ask who
            # made it before giving up — on a stylized can the maker is the readable half.
            hits += self._by_producer([(i, req.detections[i].text) for i in to_match], hint)
            resolved_lines.update(i for i, _, _ in hits)

        unresolved = [i for i in range(len(req.detections)) if i not in resolved_lines]

        # ---- pass 2: ask the whole frame about each distinct candidate ----
        best_hit: dict[str, tuple[int, float, dict]] = {}   # record id -> best (line, score)
        qualified_by_id: dict[str, str] = {}
        for i, rec, sc in hits:
            rid = rec.get("id") or ""
            prev = best_hit.get(rid)
            if prev is None or sc > prev[1]:
                best_hit[rid] = (i, sc, rec)
            qualified_by_id.setdefault(rid, self._qualified_name(rec))

        def _is_whole_label(rid: str, name: str, raw_score: float, line_i: int) -> bool:
            """True when the line this candidate matched is its name and essentially nothing
            else -- proof on its own, needing no second line to agree.

            Shared by the penalty and the corroboration test below because they are the same
            judgement, and when they were written separately they contradicted each other: the
            penalty pushed the score under the very bar the proof required.
            """
            return raw_score >= _STRONG_MATCH and _accounts_for_the_line(
                qualified_by_id.get(rid, name), req.detections[line_i].text)

        scored: list[tuple[int, ScoredCandidate]] = []
        named_by_id: dict[str, int] = {}
        whole_label: dict[str, bool] = {}
        for line_i, sc, rec in best_hit.values():
            resolved = self._hydrate(rec)
            if resolved is None:
                continue
            cat = resolved.product.category.value if resolved.product.category else None
            vocab = _candidate_vocabulary(resolved)
            support = _frame_support(vocab, line_tokens, category=cat, hint=hint)
            # The same count without the category's point. Agreeing on the category is real
            # evidence for *ranking* -- it is what separates `The Alchemist Heady Topper` from
            # `Alchemist Amer` -- but it cannot certify a frame, because on a can that prints
            # "ALE" every beer in the catalog earns it. Counting it here let a row named "Ache"
            # reach the corroboration bar off one mis-segmented fragment, and a certified frame
            # is precisely the one the client does not ask the model about.
            named = _frame_support(vocab, independent)
            named_by_id[resolved.product.id] = named
            # Report a score the frame actually justifies. One line naming a candidate while
            # several others sit there disagreeing is weaker evidence than the same number in
            # a frame that had nothing to corroborate with, and the overlay should say so.
            score = sc
            if hint and cat and cat != hint and cat in _CATEGORY_WORDS:
                score = sc * _CATEGORY_CONTRADICTS
            # `named`, not `support`, for the same reason corroboration uses it: the
            # category's point is not one of the frame's lines agreeing that this is the
            # product. A fragment match that only the category backs showed 1.00 in the HUD,
            # which is the number a user reads as certainty.
            # Whether the line *is* this product's name, judged on the raw similarity before
            # any markdown. Corroboration and confidence are different questions: the reported
            # score still says "one line, others disagreeing" -- a coincidence like `Chemist`
            # off "CHEMIST" beside two lines naming the Alchemist beer must still rank below it
            # -- while proof asks only whether some line is wholly this label, which is what a
            # shelf gives every product on it.
            whole_label[resolved.product.id] = _is_whole_label(
                resolved.product.id, resolved.product.name, sc, line_i)
            if named <= 1 and identity_lines >= _MIN_FRAME_FOR_PENALTY:
                score *= _UNCORROBORATED
            score = round(score, 3)
            personal, reason, cold = (
                (*self.score(resolved.product, profile),) if req.include_score
                else (None, None, False)
            )
            scored.append((
                support,
                ScoredCandidate(
                    detection_index=line_i,
                    resolved=resolved,
                    match_score=score,
                    personal_score=personal,
                    reason=reason,
                    cold_start=cold,
                ),
            ))

        # Collapse to one overlay per *real* product and cap the frame — the server-side
        # backstop against the crowding (and the duplicate-catalog-record double overlays) the
        # HUD showed. Keyed on canonical brand+name, not the raw id, so two rows for the same
        # beer merge. Corroboration outranks similarity: two independent lines naming a product
        # is stronger evidence than one perfect match on a fragment, which is exactly the
        # comparison a tie at 1.00 cannot make. On a tie the richer record (has ABV / sensory)
        # represents it, so the surviving overlay carries the most complete data — and that
        # also picks the better-linked of two duplicate rows.
        def _rank(entry: tuple[int, ScoredCandidate]) -> tuple:
            support, c = entry
            p = c.resolved.product
            return (support, c.match_score, bool(p.spec and p.spec.abv_pct), bool(p.sensory))

        best: dict[str, tuple[int, ScoredCandidate]] = {}
        for entry in scored:
            c = entry[1]
            key = _identity_key(c.resolved.product.name, c.resolved.brand.name,
                                c.resolved.product.id)
            if key not in best or _rank(entry) > _rank(best[key]):
                best[key] = entry
        ranked = sorted(best.values(), key=_rank, reverse=True)
        # One line names one product. Identity-keying collapses duplicate rows for the same
        # beer, but not a brand-level row sitting beside a product one -- `Lagunitas` next to
        # `Lagunitas IPA`, three Blue Moon variants next to each other. Each accounts for the
        # same line and so each proves itself against it, and a shelf of three beers drew five
        # overlays. Best candidate per line represents that line; the rest are readings of text
        # already spoken for.
        per_line: dict[int, tuple[int, ScoredCandidate]] = {}
        for entry in ranked:
            per_line.setdefault(entry[1].detection_index, entry)
        ranked = sorted(per_line.values(), key=_rank, reverse=True)[:_MAX_CANDIDATES]
        def _is_proven(c: ScoredCandidate) -> bool:
            # A barcode is an identifier, not a reading of one. Nothing in the frame needs to
            # agree with it, and a scan that succeeded must not be sent to the model to be
            # second-guessed -- nor capped below, since two barcodes legitimately name two
            # products.
            return (
                c.resolved.product.id in by_upc
                or named_by_id.get(c.resolved.product.id, 0) >= _MIN_FRAME_FOR_PENALTY
                # A line this candidate accounts for *entirely* proves it on its own, however
                # many other labels share the frame. This used to require the frame to hold
                # fewer than two identity lines -- which is to say, it only worked on a single
                # label photographed alone, and switched itself off on the one input the HUD
                # exists for. A shelf gives every product one line naming it and no second line
                # to agree, so nothing could corroborate, and the unproven-frame cap then threw
                # away all but one: three beers in view returned a single guess the client
                # withheld, and the shelf showed nothing at all.
                or whole_label.get(c.resolved.product.id, False)
            )

        proven = [e for e in ranked if _is_proven(e[1])]
        corroborated = bool(proven)
        # A frame nothing corroborates has no evidence to rank a list with, so offering one
        # implies a differentiation we cannot make. Measured over 78 such frames from a real
        # can: the right answer was first once, deeper never, and absent 77 times -- while the
        # frames carried two, three and five candidates each. They were not competing readings
        # of the label, they were the same wrong guess spelled five ways ("Chemist", "Chemist
        # 151", "Chemist Spirits", "Chemist Bierbrand"). One guess is as much as this frame has
        # earned the right to say, and the client is about to ask the model anyway.
        if corroborated:
            # Corroboration is a property of a candidate, but it was only ever applied to the
            # frame -- so the unproven candidates rode in on the proven one's coat-tails. Three
            # four-packs in view is a frame that legitimately corroborates *something*, and that
            # opened the gate for every junk match beside it: reported from the camera as "a
            # number of answers stacked on top of each other", with the right answer behind
            # them. A shelf of real products still returns all of them -- each proves itself.
            ranked = proven
        else:
            ranked = ranked[:1]
        return ScanResolveResponse(
            candidates=[c for _, c in ranked],
            unresolved_indices=unresolved,
            # Agreement across the frame, or — where there was no second line to agree with
            # — a strong read of the only line there was. Mirrors the penalty above: a lone
            # clean "BOMBAY SAPPHIRE LONDON DRY GIN" is not weak evidence, it is the whole
            # label, and asking the model about it would spend a second to confirm a 1.00.
            corroborated=corroborated,
        )



def _top_axis(sv: SensoryVector) -> str | None:
    if not sv.axes:
        return None
    return max(sv.axes.items(), key=lambda kv: kv[1])[0].replace("_", " ")


def _match_reason(score: float, sensory: SensoryVector, ideal: SensoryVector) -> str:
    """Explain the score honestly.

    The axis we name is the one that actually drove the agreement — high on the product
    *and* high in the profile — not the product's loudest note. Naming the loudest note
    made a poor match still read "matches your smoky peat preference", which is the
    overlay telling the user something the score itself contradicts.
    """
    shared = _agreeing_axis(sensory, ideal)
    if score >= _STRONG_MATCH:
        return f"matches your {shared} preference" if shared else "matches your taste profile"
    if score >= _MILD_MATCH:
        return f"some {shared}, which you like" if shared else "a partial match"
    loud = _top_axis(sensory)
    return f"outside your usual — mostly {loud}" if loud else "outside your usual"


def _agreeing_axis(sensory: SensoryVector, ideal: SensoryVector) -> str | None:
    """The axis contributing most to the match: argmax of product·profile, per axis."""
    a, b = sensory.to_array(), ideal.to_array()
    weight, axis = max((a[i] * b[i], SENSORY_AXES[i]) for i in range(len(SENSORY_AXES)))
    return axis.replace("_", " ") if weight > 0 else None
