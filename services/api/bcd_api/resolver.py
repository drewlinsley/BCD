"""Scan resolution + personal scoring — the server half of the latency-critical path.

Kept deliberately simple and dependency-light here (LIKE match + a transparent
chemistry-based cold-start scorer) so it runs on the laptop store. In production the
match step is Postgres trigram + pgvector ANN, and scoring blends the learned
ingredient->sensory model with the user's TasteProfile. The *shape* is what the iOS
client codes against and what we optimize behind.
"""

from __future__ import annotations

import re

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


class Resolver:
    def __init__(self, store: Store) -> None:
        self.store = store

    # Matching is delegated to the store: token-overlap on the SQLite dev store,
    # real pg_trgm trigram similarity on Postgres — same signature either way.
    def _resolve_by_upc(self, upc: str) -> dict | None:
        sku = self.store.get_gold(f"sku:{upc}")
        if not sku:
            return None
        return self.store.get_gold(sku["product_id"])

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
                matches = self.store.match_products(det.text)
                # Confidence floor: below it a line is OCR chrome that still trigram-matches
                # *something* — leave it unresolved rather than show a wrong beer. Short catalog
                # names need a higher, near-exact floor (see _SHORT_MIN_MATCH).
                if matches:
                    rec, sc = matches[0]
                    floor = (_SHORT_MIN_MATCH
                             if len((rec.get("name") or "")) < _SHORT_NAME_LEN else _MIN_MATCH)
                    if sc >= floor:
                        product_rec, match_score = rec, sc
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
        # Collapse to one overlay per product (strongest match wins) and cap the frame —
        # the server-side backstop against the crowding the HUD was showing.
        best: dict[str, ScoredCandidate] = {}
        for c in candidates:
            pid = c.resolved.product.id
            if pid not in best or c.match_score > best[pid].match_score:
                best[pid] = c
        candidates = sorted(
            best.values(), key=lambda c: c.match_score, reverse=True
        )[:_MAX_CANDIDATES]
        return ScanResolveResponse(candidates=candidates, unresolved_indices=unresolved)


def _top_axis(sv: SensoryVector) -> str | None:
    if not sv.axes:
        return None
    return max(sv.axes.items(), key=lambda kv: kv[1])[0].replace("_", " ")
