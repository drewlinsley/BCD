"""Resolver — matching + cold-start scoring against a seeded store."""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.resolver import Resolver
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    SKU,
    Brand,
    Category,
    DetectedObject,
    DetectedText,
    ExtractionMethod,
    Producer,
    Product,
    ProductSpec,
    Provenance,
    ScanResolveRequest,
    SensorySource,
    SensoryVector,
    Sourced,
    TasteProfile,
)


@pytest.fixture()
def store():
    d = tempfile.mkdtemp()
    s = MedallionStore(root=d)
    prov = Provenance(source_id="ttb", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)
    s.put_gold("prod:x", "producer", Producer(id="prod:x", name="Alchemist").model_dump(mode="json"))
    s.put_gold("brand:x", "brand", Brand(id="brand:x", producer_id="prod:x", name="Heady").model_dump(mode="json"))
    p = Product(
        id="ttb:1", brand_id="brand:x", producer_id="prod:x",
        category=Category.BEER, name="Heady Topper",
        spec=ProductSpec(abv_pct=Sourced[float](value=8.0, provenance=prov)),
        sensory=SensoryVector(source=SensorySource.CHEMISTRY_PRIOR, confidence=0.6,
                              axes={"tropical": 0.9, "citrus": 0.8, "bitterness": 0.6}),
    )
    s.put_gold("ttb:1", "product", p.model_dump(mode="json"))
    s.put_gold("sku:854416001019", "sku",
               SKU(id="sku:854416001019", product_id="ttb:1", container="can",
                   upc="854416001019").model_dump(mode="json"))
    yield s
    s.close()


def test_resolves_by_barcode(store):
    r = Resolver(store)
    req = ScanResolveRequest(detections=[DetectedText(text="854416001019", kind="barcode")])
    resp = r.resolve(req)
    assert len(resp.candidates) == 1
    assert resp.candidates[0].resolved.product.name == "Heady Topper"
    assert resp.candidates[0].match_score == 1.0


def test_resolves_by_ocr_text(store):
    r = Resolver(store)
    req = ScanResolveRequest(detections=[DetectedText(text="HEADY TOPPER 16oz", kind="text")])
    resp = r.resolve(req)
    assert resp.candidates and resp.candidates[0].resolved.product.name == "Heady Topper"


def test_cold_start_scoring_needs_no_reviews(store):
    r = Resolver(store)
    profile = TasteProfile(
        user_id="u", sensory_ideal=SensoryVector(
            source=SensorySource.RECONCILED, axes={"tropical": 1.0, "citrus": 0.9}),
    )
    req = ScanResolveRequest(detections=[DetectedText(text="854416001019", kind="barcode")],
                             include_score=True)
    resp = r.resolve(req, profile=profile)
    cand = resp.candidates[0]
    assert cand.cold_start is True  # scored from chemistry_prior sensory, zero reviews
    assert cand.personal_score is not None and cand.personal_score > 0.7  # tropical match


def test_unresolved_text_reported(store):
    r = Resolver(store)
    req = ScanResolveRequest(detections=[DetectedText(text="zzzzz nonexistent", kind="text")])
    resp = r.resolve(req)
    assert resp.unresolved_indices == [0]
    assert not resp.candidates


# ---- object-level (coarse-to-fine) path ----


@pytest.fixture()
def alchemist_store(store):
    """Add a sibling product from the same producer so producer-only evidence is
    genuinely ambiguous, plus an unrelated product with a generic name."""
    prov = Provenance(source_id="ttb", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)
    store.put_gold("brand:fb", "brand",
                   Brand(id="brand:fb", producer_id="prod:x", name="Focal Banger")
                   .model_dump(mode="json"))
    store.put_gold("ttb:2", "product", Product(
        id="ttb:2", brand_id="brand:fb", producer_id="prod:x",
        category=Category.BEER, name="Focal Banger",
        spec=ProductSpec(abv_pct=Sourced[float](value=7.0, provenance=prov)),
    ).model_dump(mode="json"))
    store.put_gold("prod:d", "producer",
                   Producer(id="prod:d", name="Dogfish Head Craft Brewery").model_dump(mode="json"))
    store.put_gold("brand:60", "brand",
                   Brand(id="brand:60", producer_id="prod:d", name="60 Minute")
                   .model_dump(mode="json"))
    store.put_gold("ttb:3", "product", Product(
        id="ttb:3", brand_id="brand:60", producer_id="prod:d",
        category=Category.BEER, name="60 Minute IPA",
    ).model_dump(mode="json"))
    return store


def _obj(*texts: str, barcode: str | None = None, oid: str = "obj-1") -> DetectedObject:
    return DetectedObject(id=oid, texts=list(texts), barcode=barcode,
                          x=0.1, y=0.2, w=0.3, h=0.5, frames_seen=4)


def test_object_with_full_label_resolves(alchemist_store):
    r = Resolver(alchemist_store)
    resp = r.resolve(ScanResolveRequest(objects=[_obj("THE ALCHEMIST", "HEADY", "TOPPER",
                                                      "16 FL OZ")]))
    assert len(resp.objects) == 1
    res = resp.objects[0]
    assert res.status == "resolved"
    assert res.candidates[0].resolved.product.name == "Heady Topper"
    assert res.candidates[0].object_id == "obj-1"
    # The confident answer is also in the flat list, so the legacy HUD path still works.
    assert [c.resolved.product.name for c in resp.candidates] == ["Heady Topper"]


def test_object_with_garbage_fragments_does_not_fire(alchemist_store):
    # The reported failure: "Chemist", "hop chemist", "Mist" from a Heady Topper can.
    r = Resolver(alchemist_store)
    resp = r.resolve(ScanResolveRequest(objects=[_obj("Chemist", "hop chemist", "Mist")]))
    res = resp.objects[0]
    assert res.status != "resolved"
    assert resp.candidates == []  # nothing reaches the overlay list
    if res.status == "ambiguous":
        # The shortlist is the *right* producer's beers, for the fine stage to pick from.
        names = {c.resolved.product.name for c in res.candidates}
        assert names <= {"Heady Topper", "Focal Banger"}
        assert all(c.match_score < 0.6 for c in res.candidates)


def test_producer_only_evidence_returns_sibling_shortlist(alchemist_store):
    r = Resolver(alchemist_store)
    resp = r.resolve(ScanResolveRequest(objects=[_obj("THE ALCHEMIST", "WATERBURY VT")]))
    res = resp.objects[0]
    assert res.status == "ambiguous"
    assert {c.resolved.product.name for c in res.candidates} == {"Heady Topper", "Focal Banger"}


def test_object_ocr_typo_resolves(alchemist_store):
    r = Resolver(alchemist_store)
    resp = r.resolve(ScanResolveRequest(objects=[_obj("HEADY", "TOPPFR")]))
    assert resp.objects[0].status == "resolved"
    assert resp.objects[0].candidates[0].resolved.product.name == "Heady Topper"


def test_generic_style_word_alone_is_unresolved(alchemist_store):
    r = Resolver(alchemist_store)
    resp = r.resolve(ScanResolveRequest(objects=[_obj("IPA", "12 FL OZ")]))
    assert resp.objects[0].status == "unresolved"
    assert resp.objects[0].candidates == []


def test_object_barcode_wins_over_text(alchemist_store):
    r = Resolver(alchemist_store)
    resp = r.resolve(ScanResolveRequest(objects=[_obj("Chemist", barcode="854416001019")]))
    res = resp.objects[0]
    assert res.status == "resolved" and res.candidates[0].match_score == 1.0
    assert res.candidates[0].resolved.product.name == "Heady Topper"


def test_client_can_raise_the_floor(alchemist_store):
    r = Resolver(alchemist_store)
    # Name fully read, but nothing corroborates the producer: 0.7 by default.
    assert r.resolve(ScanResolveRequest(objects=[_obj("FOCAL", "BANGER")])) \
        .objects[0].status == "resolved"
    req = ScanResolveRequest(objects=[_obj("FOCAL", "BANGER")], min_match_score=0.8)
    resp = r.resolve(req)
    assert resp.objects[0].status == "ambiguous"  # a HUD that wants certainty gets a shortlist
    assert resp.candidates == []


def test_legacy_fragment_no_longer_fires(alchemist_store):
    # Per-line path: a lone fragment used to resolve to the best token-overlap hit.
    r = Resolver(alchemist_store)
    resp = r.resolve(ScanResolveRequest(detections=[DetectedText(text="Chemist", kind="text")]))
    assert resp.unresolved_indices == [0]
    assert resp.candidates == []


def test_lexicon_carries_names_not_generic_words(alchemist_store):
    words = Resolver(alchemist_store).lexicon()
    assert "Heady Topper" in words and "Alchemist" in words and "Dogfish" in words
    assert "IPA" not in words and "Brewery" not in words
