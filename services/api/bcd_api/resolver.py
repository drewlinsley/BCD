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

from bcd_ingest.store import Store, _cosine
from bcd_schema import (
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


def _token_supported(query: str, name: str) -> bool:
    """True if a real name token (>=4 letters) closely matches some OCR token — evidence the
    brand word is actually present in the line, not a coincidental trigram window. A name with
    no such token ("J&B", "1664") has nothing to anchor on and defers to the score floors."""
    name_tokens = [t for t in _tokens(name) if len(t) >= _MIN_NAME_TOKEN_LEN]
    if not name_tokens:
        return True
    q_tokens = _tokens(query)
    return any(_trigram_sim(nt, qt) >= _TOKEN_SUPPORT_MIN
               for nt in name_tokens for qt in q_tokens)


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


def _identity_key(name: str, brand: str, pid: str) -> str:
    """One key per *real* product, so duplicate catalog records collapse into a single overlay.

    The catalog carries the same beer under multiple rows — different UPCs of one product (Lagunitas
    IPA ×2, Heineken ×7), or an OFF pull plus a TTB record. Keyed on the raw id they'd each draw
    their own overlay; keyed on normalized brand+name they merge. Brand is part of the key on
    purpose: OFF often names a product only by its class ("Blended Scotch Whisky" for eight
    different distilleries), and those must stay distinct — their brands (Johnnie Walker vs Queen
    Margot) are what separate them. A placeholder "Unknown" brand (or one that just echoes the name)
    carries no identity, so it drops out and the name alone keys — which is what lets the two
    Unknown-branded "Lagunitas IPA" rows collapse. A row with no usable name has nothing to
    canonicalize on and falls back to its id, so unnamed rows never merge into one another."""
    n = " ".join(_tokens(name))
    if not n:
        return f"id:{pid}"
    b = " ".join(_tokens(brand))
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
        top = _top_axis(sensory)
        reason = f"matches your {top} preference" if top else "matches your taste profile"
        return (round(score, 3), reason, cold_start)

    def resolve(self, req: ScanResolveRequest,
                profile: TasteProfile | None = None) -> ScanResolveResponse:
        candidates: list[ScoredCandidate] = []
        unresolved: list[int] = []
        for i, det in enumerate(req.detections):
            product_rec = None
            match_score = 0.0
            if det.kind == "barcode":
                product_rec = self._resolve_by_upc(det.text)
                match_score = 1.0 if product_rec else 0.0
            elif _is_identity_text(det.text):
                # Text path only for lines that could name a product; a bare number/fragment
                # (and a failed barcode's digits) is skipped rather than trigram-matched.
                # Take the best candidate that clears BOTH the score floor and token support.
                # The floor rejects OCR chrome ("12 FL OZ"); token support rejects a coincidental
                # sub-window (a garbled warning line that outscores the real product) — iterating
                # rather than checking only matches[0] means such a coincidence is skipped, not
                # allowed to block a genuine lower-ranked hit. Short names use a near-exact floor.
                for rec, sc in self.store.match_products(det.text):
                    name = rec.get("name") or ""
                    floor = _SHORT_MIN_MATCH if len(name) < _SHORT_NAME_LEN else _MIN_MATCH
                    if sc >= floor and _token_supported(det.text, name):
                        product_rec, match_score = rec, sc
                        break
            if product_rec is None:
                unresolved.append(i)
                continue
            resolved = self._hydrate(product_rec)
            if resolved is None:
                unresolved.append(i)
                continue
            personal, reason, cold = (
                (*self.score(resolved.product, profile),) if req.include_score
                else (None, None, False)
            )
            candidates.append(
                ScoredCandidate(
                    detection_index=i,
                    resolved=resolved,
                    match_score=match_score,
                    personal_score=personal,
                    reason=reason,
                    cold_start=cold,
                )
            )
        # Collapse to one overlay per *real* product and cap the frame — the server-side backstop
        # against the crowding (and the duplicate-catalog-record double overlays) the HUD showed.
        # Keyed on canonical brand+name, not the raw id, so two rows for the same beer merge; the
        # strongest match wins, and on a tie the richer record (has ABV / sensory) represents it so
        # the surviving overlay carries the most complete data.
        def _rank(c: ScoredCandidate) -> tuple:
            p = c.resolved.product
            return (c.match_score, bool(p.spec and p.spec.abv_pct), bool(p.sensory))

        best: dict[str, ScoredCandidate] = {}
        for c in candidates:
            key = _identity_key(c.resolved.product.name, c.resolved.brand.name,
                                c.resolved.product.id)
            if key not in best or _rank(c) > _rank(best[key]):
                best[key] = c
        candidates = sorted(
            best.values(), key=lambda c: c.match_score, reverse=True
        )[:_MAX_CANDIDATES]
        return ScanResolveResponse(candidates=candidates, unresolved_indices=unresolved)


def _top_axis(sv: SensoryVector) -> str | None:
    if not sv.axes:
        return None
    return max(sv.axes.items(), key=lambda kv: kv[1])[0].replace("_", " ")
