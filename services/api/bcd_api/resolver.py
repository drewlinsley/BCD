"""Scan resolution + personal scoring — the server half of the latency-critical path.

Kept deliberately simple and dependency-light here (store-side candidate retrieval + a
backend-independent identity scorer + a transparent chemistry-based cold-start scorer)
so it runs on the laptop store. In production the retrieval step is Postgres trigram +
pgvector ANN, and scoring blends the learned ingredient->sensory model with the user's
TasteProfile. The *shape* is what the iOS client codes against and what we optimize behind.

Two inputs, one decision procedure:

  * `objects`    — the coarse-to-fine path. Each object carries every OCR line seen on one
                   can/bottle over several frames (plus a barcode if one was read). We
                   retrieve candidates for the whole object, score each against the full
                   identity (name + brand + producer), and return a verdict —
                   resolved / ambiguous / unresolved — with the shortlist attached so the
                   client's fine stage can finish the job when we can't.
  * `detections` — the original per-line path. Still served, now through the same scorer
                   and the same confidence floor, so a lone fragment can no longer fire.
"""

from __future__ import annotations

from bcd_ingest.store import Store, _cosine
from bcd_schema import (
    Brand,
    DetectedObject,
    ObjectResolution,
    Producer,
    Product,
    ResolvedProduct,
    ScanResolveRequest,
    ScanResolveResponse,
    ScoredCandidate,
    SensoryVector,
    TasteProfile,
)

from .matching import (
    GENERIC_TOKENS,
    Identity,
    object_query,
    score_identity,
    tokenize,
    verdict,
)

RETRIEVE_PER_OBJECT = 10  # candidates pulled for the whole-object query
RETRIEVE_PER_FRAGMENT = 5  # ... and per individual OCR line, unioned in
RETRIEVE_PRODUCERS = 3  # producer/brand name hits to expand into their products
RETRIEVE_PER_OWNER = 25  # ... and how many products per hit
SHORTLIST = 5  # candidates returned on an ambiguous verdict


class Resolver:
    def __init__(self, store: Store) -> None:
        self.store = store

    # ---- lookups ----
    # Matching is delegated to the store: token-overlap on the SQLite dev store,
    # real pg_trgm trigram similarity on Postgres — same signature either way.
    def _resolve_by_upc(self, upc: str) -> dict | None:
        sku = self.store.get_gold(f"sku:{upc}")
        if not sku:
            return None
        return self.store.get_gold(sku["product_id"])

    def _lookup_identity(self, product_rec: dict) -> tuple[dict, dict]:
        producer = self.store.get_gold(product_rec.get("producer_id", ""))
        brand = self.store.get_gold(product_rec.get("brand_id", ""))
        if producer is None:
            producer = Producer(id="unknown", name="Unknown").model_dump(mode="json")
        if brand is None:
            brand = Brand(id="unknown", producer_id=producer["id"],
                          name=product_rec.get("name", "")).model_dump(mode="json")
        return producer, brand

    def _hydrate(self, product_rec: dict) -> ResolvedProduct | None:
        producer, brand = self._lookup_identity(product_rec)
        return ResolvedProduct(
            product=Product.model_validate(product_rec),
            producer=Producer.model_validate(producer),
            brand=Brand.model_validate(brand),
        )

    def _retrieve(self, texts: list[str]) -> list[dict]:
        """Candidate generation: the store's cheap fuzzy match on the whole-object query
        and on each fragment, unioned. Retrieval is deliberately generous — the scorer
        below is what says no."""
        seen: set[str] = set()
        out: list[dict] = []

        def take(rec: dict) -> None:
            rid = rec.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                out.append(rec)

        queries = [object_query(texts)] + [t for t in texts if t.strip()]
        for i, q in enumerate(queries):
            limit = RETRIEVE_PER_OBJECT if i == 0 else RETRIEVE_PER_FRAGMENT
            for rec, _ in self.store.match_products(q, limit=limit):
                take(rec)
            # The producer's name is often the most legible thing on a can. Pull that
            # producer's (or brand's) products in as candidates too; the scorer decides
            # whether the beer's own name is there to pick one.
            for owner, _ in self.store.match_producers(q, limit=RETRIEVE_PRODUCERS):
                for rec in self.store.products_by_producer(owner["id"], limit=RETRIEVE_PER_OWNER):
                    take(rec)
        return out

    def _rank(self, texts: list[str]) -> list[tuple[dict, float]]:
        """Retrieve, then score every candidate against its full identity. Best-first."""
        query_tokens = tokenize(object_query(texts))
        if not query_tokens:
            return []
        ranked: list[tuple[dict, float]] = []
        for rec in self._retrieve(texts):
            producer, brand = self._lookup_identity(rec)
            ident = Identity.from_records(rec, brand, producer)
            ranked.append((rec, score_identity(query_tokens, ident)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

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

    def _candidate(self, product_rec: dict, match_score: float, profile: TasteProfile | None,
                   include_score: bool, *, detection_index: int | None = None,
                   object_id: str | None = None) -> ScoredCandidate | None:
        resolved = self._hydrate(product_rec)
        if resolved is None:
            return None
        personal, reason, cold = (
            (*self.score(resolved.product, profile),) if include_score
            else (None, None, False)
        )
        return ScoredCandidate(
            detection_index=detection_index,
            object_id=object_id,
            resolved=resolved,
            match_score=match_score,
            personal_score=personal,
            reason=reason,
            cold_start=cold,
        )

    # ---- resolution ----
    def resolve_object(self, obj: DetectedObject, profile: TasteProfile | None = None,
                       include_score: bool = True,
                       min_score: float | None = None) -> ObjectResolution:
        query = object_query(obj.texts, obj.barcode)

        # A barcode is ground truth: resolve on it and skip the text entirely.
        if obj.barcode:
            rec = self._resolve_by_upc(obj.barcode)
            if rec is not None:
                cand = self._candidate(rec, 1.0, profile, include_score, object_id=obj.id)
                if cand is not None:
                    return ObjectResolution(object_id=obj.id, status="resolved",
                                            query=query, candidates=[cand])

        ranked = self._rank(obj.texts)
        status = verdict([s for _, s in ranked], min_score)
        if status == "unresolved":
            return ObjectResolution(object_id=obj.id, status="unresolved", query=query)

        keep = ranked[:1] if status == "resolved" else ranked[:SHORTLIST]
        cands = [
            c for rec, s in keep
            if (c := self._candidate(rec, s, profile, include_score, object_id=obj.id))
        ]
        if not cands:
            return ObjectResolution(object_id=obj.id, status="unresolved", query=query)
        return ObjectResolution(object_id=obj.id, status=status, query=query, candidates=cands)

    def resolve(self, req: ScanResolveRequest,
                profile: TasteProfile | None = None) -> ScanResolveResponse:
        candidates: list[ScoredCandidate] = []
        unresolved: list[int] = []
        objects: list[ObjectResolution] = []

        # Per-object path.
        for obj in req.objects:
            res = self.resolve_object(obj, profile, req.include_score, req.min_match_score)
            objects.append(res)
            if res.status == "resolved":
                candidates.append(res.candidates[0])

        # Per-line path: same scorer, same floor, no margin test (there is no shortlist
        # to hand back on this path, so 'ambiguous' collapses to 'unresolved').
        for i, det in enumerate(req.detections):
            product_rec = None
            match_score = 0.0
            if det.kind == "barcode":
                product_rec = self._resolve_by_upc(det.text)
                match_score = 1.0 if product_rec else 0.0
            if product_rec is None:
                ranked = self._rank([det.text])
                if ranked and verdict([ranked[0][1]], req.min_match_score) == "resolved":
                    product_rec, match_score = ranked[0]
            if product_rec is None:
                unresolved.append(i)
                continue
            cand = self._candidate(product_rec, match_score, profile, req.include_score,
                                   detection_index=i)
            if cand is None:
                unresolved.append(i)
                continue
            candidates.append(cand)

        return ScanResolveResponse(candidates=candidates, unresolved_indices=unresolved,
                                   objects=objects)

    # ---- lexicon ----
    def lexicon(self, limit: int = 5000) -> list[str]:
        """Names the on-device recognizer should know: product, brand and producer names
        plus their aliases and their distinctive tokens. Generic label words are left out
        — the recognizer already knows 'IPA'; what it doesn't know is 'Alchemist'."""
        words: list[str] = []
        seen: set[str] = set()

        def add(s: str | None) -> None:
            s = (s or "").strip()
            if not s:
                return
            key = s.lower()
            if key not in seen:
                seen.add(key)
                words.append(s)
            for tok in s.split():
                clean = "".join(ch for ch in tok if ch.isalnum())
                low = clean.lower()
                if len(low) >= 4 and low not in GENERIC_TOKENS and low not in seen:
                    seen.add(low)
                    words.append(clean)

        for kind in ("product", "brand", "producer"):
            for rec in self.store.iter_gold(kind):
                add(rec.get("name"))
                for alias in rec.get("aliases") or []:
                    add(alias)
                if len(words) >= limit:
                    return words[:limit]
        return words[:limit]


def _top_axis(sv: SensoryVector) -> str | None:
    if not sv.axes:
        return None
    return max(sv.axes.items(), key=lambda kv: kv[1])[0].replace("_", " ")
