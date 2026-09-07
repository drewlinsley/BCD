"""Label-text matching — turning OCR evidence on one object into a confident (or honestly
unconfident) product identification.

Why this exists: the first HUD fired an overlay on whatever product best matched *each*
OCR fragment. Stylized label type (Heady Topper, hazy-IPA cans, Dogfish Head) OCRs into
fragments like "Chemist", "hop chemist", "Mist" — and each one found *some* product to
match. The fix has three parts, all here:

  1. **Score the object, not the fragment.** All text seen on one can is one query.
  2. **Score against the whole identity** — product name, brand, producer, aliases — with
     generic label words ("IPA", "ALE", "16 FL OZ", "BREWING CO") down-weighted so they
     can't carry a match on their own.
  3. **A verdict with a floor and a margin.** `resolved` only when the best candidate
     clears an absolute score AND beats the runner-up by a margin; otherwise `ambiguous`
     (hand the shortlist to the client's fine stage) or `unresolved` (show nothing).

Everything is backend-independent: the store does cheap candidate *retrieval* (trigram on
Postgres, token overlap on SQLite); this module does the *decision* so both backends
behave identically at the HUD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

# ---- thresholds -------------------------------------------------------------------

RESOLVE_MIN_SCORE = 0.6  # absolute floor for a 'resolved' verdict
RESOLVE_MIN_MARGIN = 0.15  # best must beat the runner-up by this much
AMBIGUOUS_MIN_SCORE = 0.2  # below this nothing is even worth a second look
FUZZY_TOKEN_MIN = 0.8  # SequenceMatcher ratio for two tokens to count as the same word
FUZZY_TOKEN_MIN_LEN = 4  # never fuzzy-match very short tokens ("ipa" vs "ipl")

# Weights inside a name: a token that appears on half the cans in the fridge should not
# be able to identify a product by itself. Kept small, not zero, so "Hazy Little Thing"
# still benefits from "hazy" once "little thing" is on the table.
GENERIC_WEIGHT = 0.25
NAME_WEIGHT = 0.7  # share of the score carried by the product name / aliases
IDENTITY_WEIGHT = 0.3  # share carried by brand / producer
# A product whose name is *only* generic tokens ("IPA") can't be resolved from its name
# alone — cap the name contribution unless the producer corroborates.
GENERIC_ONLY_NAME_CAP = 0.3

# Words that appear on labels regardless of which product it is. Category nouns, style
# words, packaging, legal boilerplate. Lowercase, letters/digits only.
GENERIC_TOKENS: frozenset[str] = frozenset({
    # category / style
    "ale", "beer", "lager", "ipa", "dipa", "tipa", "neipa", "pale", "india", "double",
    "triple", "imperial", "hazy", "juicy", "session", "stout", "porter", "pilsner", "pils",
    "sour", "gose", "saison", "wheat", "weisse", "weizen", "hefeweizen", "kolsch", "amber",
    "brown", "blonde", "golden", "red", "black", "white", "dark", "light", "lite", "dry",
    "hopped", "hop", "hops", "hoppy", "craft", "cider", "hard", "seltzer", "wine",
    "whiskey", "whisky", "bourbon", "rye", "scotch", "vodka", "gin", "rum", "tequila",
    "mezcal", "brandy", "liqueur", "spirit", "spirits", "malt", "barley", "beverage",
    "reserve", "original", "classic", "special", "limited", "edition", "release", "batch",
    "series", "select", "premium", "extra", "small", "single", "cask", "barrel", "aged",
    "straight", "kentucky", "tennessee", "american", "west", "coast", "east", "new",
    "england", "style", "fresh", "cold",
    # producer boilerplate
    "brewing", "brewery", "brewers", "brewed", "brew", "brewhouse", "distillery",
    "distilling", "distillers", "distilled", "winery", "cidery", "company", "co", "inc",
    "llc", "ltd", "bros", "brothers", "and", "the", "of", "by", "for", "with", "from",
    "est", "since", "handcrafted", "handmade", "family", "farm", "farmhouse", "house",
    "works", "project", "collective", "supply", "trading",
    # packaging / legal
    "fl", "oz", "ml", "cl", "abv", "alc", "vol", "alcohol", "volume", "proof", "ibu",
    "contains", "sulfites", "government", "warning", "surgeon", "general", "pregnancy",
    "drink", "responsibly", "please", "recycle", "refund", "deposit", "bottled", "canned",
    "packaged", "produced", "product", "made", "usa", "ca", "cash", "me", "vt", "ny", "mi",
    "or", "ia", "ct", "ma", "hi", "keep", "refrigerated", "enjoy", "net", "contents",
    "pint", "quart", "liter", "litre", "can", "cans", "bottle", "bottles", "pack",
    "twelve", "sixteen", "ounces", "ounce",
})

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


# ---- text normalization -------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokens, order-preserving, deduped, 2+ chars. Kept permissive on
    length so 'IPA'-style tokens survive to be *down-weighted* rather than lost."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN_RE.split((text or "").lower()):
        if len(tok) < 2 or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def object_query(texts: list[str], barcode: str | None = None) -> str:
    """Join every fragment on one object into a single normalized query string."""
    words: list[str] = []
    seen: set[str] = set()
    for t in texts:
        for tok in tokenize(t):
            if tok not in seen:
                seen.add(tok)
                words.append(tok)
    if barcode:
        words.append(barcode)
    return " ".join(words)


def token_weight(tok: str) -> float:
    return GENERIC_WEIGHT if tok in GENERIC_TOKENS or tok.isdigit() else 1.0


def token_similarity(a: str, b: str) -> float:
    """1.0 on an exact match; a fuzzy ratio for OCR-mangled words ('toppfr' ~ 'topper');
    0 for anything short or dissimilar. OCR substitutions are usually 1-2 characters, so
    a high ratio still catches them ('chemist' ~ 'alchemist' at 0.875 — a real read of
    that label) while keeping unrelated short fragments out ('mist' vs 'alchemist' is
    0.6, rejected)."""
    if a == b:
        return 1.0
    if len(a) < FUZZY_TOKEN_MIN_LEN or len(b) < FUZZY_TOKEN_MIN_LEN:
        return 0.0
    r = SequenceMatcher(None, a, b).ratio()
    return r if r >= FUZZY_TOKEN_MIN else 0.0


def coverage(query_tokens: list[str], target_tokens: list[str]) -> float:
    """Weighted fraction of the *target* name's tokens that the query accounts for.
    Biased toward covering the name (so extra label words in the query cost nothing) and
    weighted so generic tokens contribute little."""
    if not target_tokens:
        return 0.0
    total = 0.0
    hit = 0.0
    for t in target_tokens:
        w = token_weight(t)
        total += w
        best = max((token_similarity(q, t) for q in query_tokens), default=0.0)
        hit += w * best
    return hit / total if total else 0.0


def has_specific_token(tokens: list[str]) -> bool:
    return any(token_weight(t) == 1.0 for t in tokens)


# ---- scoring one candidate ------------------------------------------------------------


@dataclass
class Identity:
    """The name-ish fields of a hydrated product, as the scorer wants them."""

    product_name: str
    aliases: list[str] = field(default_factory=list)
    brand_name: str | None = None
    producer_name: str | None = None
    producer_aliases: list[str] = field(default_factory=list)

    @classmethod
    def from_records(cls, product: dict[str, Any], brand: dict[str, Any] | None,
                     producer: dict[str, Any] | None) -> Identity:
        return cls(
            product_name=product.get("name", "") or "",
            aliases=list(product.get("aliases") or []),
            brand_name=(brand or {}).get("name"),
            producer_name=(producer or {}).get("name"),
            producer_aliases=list((producer or {}).get("aliases") or []),
        )


def score_identity(query_tokens: list[str], ident: Identity) -> float:
    """0-1 confidence that the query text is this product's label."""
    name_tokens = tokenize(ident.product_name)
    name_cov = coverage(query_tokens, name_tokens)
    for alias in ident.aliases:
        name_cov = max(name_cov, coverage(query_tokens, tokenize(alias)))

    who: list[float] = []
    if ident.brand_name and ident.brand_name.lower() != ident.product_name.lower():
        who.append(coverage(query_tokens, tokenize(ident.brand_name)))
    if ident.producer_name:
        who.append(coverage(query_tokens, tokenize(ident.producer_name)))
    for alias in ident.producer_aliases:
        who.append(coverage(query_tokens, tokenize(alias)))
    who_cov = max(who, default=0.0)

    if not has_specific_token(name_tokens) and who_cov < 0.5:
        name_cov = min(name_cov, GENERIC_ONLY_NAME_CAP)

    return round(NAME_WEIGHT * name_cov + IDENTITY_WEIGHT * who_cov, 3)


# ---- the verdict -----------------------------------------------------------------------


def verdict(scores: list[float], min_score: float | None = None) -> str:
    """'resolved' | 'ambiguous' | 'unresolved' from a best-first list of scores."""
    floor = RESOLVE_MIN_SCORE if min_score is None else min_score
    if not scores:
        return "unresolved"
    best = scores[0]
    runner = scores[1] if len(scores) > 1 else 0.0
    if best >= floor and (best - runner) >= RESOLVE_MIN_MARGIN:
        return "resolved"
    if best >= AMBIGUOUS_MIN_SCORE:
        return "ambiguous"
    return "unresolved"
