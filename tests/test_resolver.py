"""Resolver — matching + cold-start scoring against a seeded store."""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.resolver import Resolver, _identity_key, _token_supported, _upc_variants
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    SKU,
    Brand,
    Category,
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


def test_resolves_barcode_across_gtin_forms(store):
    # The fixture seeds the SKU as a 12-digit UPC-A; a scanner returning the 13-digit EAN-13 form
    # (a leading zero) is the same GTIN and must still resolve the product.
    r = Resolver(store)
    req = ScanResolveRequest(detections=[DetectedText(text="0854416001019", kind="barcode")])
    resp = r.resolve(req)
    assert resp.candidates and resp.candidates[0].resolved.product.name == "Heady Topper"
    assert resp.candidates[0].match_score == 1.0


def test_upc_variants_covers_gtin_forms():
    assert "0854416001019" in _upc_variants("854416001019")   # UPC-A -> EAN-13
    assert "854416001019" in _upc_variants("0854416001019")   # EAN-13 -> UPC-A
    assert _upc_variants("75032814") == ["75032814"]          # EAN-8 left as-is
    assert _upc_variants("not-a-barcode") == ["not-a-barcode"]


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


# ---- token-support gate (garbled-fragment false positives) ----

@pytest.mark.parametrize("query, name", [
    ("BACARDI", "Bacardi"),                                   # exact brand scan
    ("BACARDI SUPERIOR RUM", "Bacardi"),                      # brand embedded in a line
    ("GUINNESS DRAUGHT 440ML EXTRA STOUT", "Bière Brune Draught 4,2% GUINNESS"),
    ("HEADY TOPPER 16oz", "Heady Topper"),
    ("KROMBACHER", "Krombacher Pils"),
])
def test_token_support_keeps_real_hits(query, name):
    assert _token_supported(query, name) is True


@pytest.mark.parametrize("query, name", [
    ("BACAR OR", "Bacardi"),                          # OCR of "...drive a car or..." warning
    ("DRIVE A CAR OR OPERATE MACHINERY", "Malibu"),   # clean warning line
    ("ACCORDING TO THE SURGEON GENERAL", "Gentiane"),
])
def test_token_support_rejects_coincidental_windows(query, name):
    assert _token_supported(query, name) is False


def test_token_support_defers_for_short_names():
    # "J&B" has no >=4-letter token to anchor on, so the gate abstains and the length-aware
    # score floor stays in charge (a garble must clear the near-exact 0.8 bar instead).
    assert _token_supported("anything at all", "J&B") is True


class _FakeMatchStore:
    """A store that returns pre-scored matches, so the resolver's token-support gate can be
    tested against the exact scores the pg_trgm backend produces — the SQLite dev store
    tokenizes and never manufactures the coincidental "BACAR OR" -> "Bacardi" hit at all."""

    db_path = ":fake:"

    def __init__(self, matches):
        self._matches = matches  # list[(product_rec, score)]

    def match_products(self, text, limit=3):
        return self._matches[:limit]

    def get_gold(self, gid):
        return None  # producer/brand unknown -> resolver fills placeholders


def _product(name, pid):
    return Product(id=pid, brand_id="b", producer_id="pr",
                   category=Category.SPIRIT, name=name).model_dump(mode="json")


def test_rejects_high_score_without_token_support():
    # The garbled warning fragment word-matches "Bacardi" at 0.625 (above the 0.5 floor) but no
    # OCR token actually *is* "Bacardi" -> it must resolve to nothing, not a wrong bottle.
    store = _FakeMatchStore([(_product("Bacardi", "p:bac"), 0.625)])
    r = Resolver(store)
    req = ScanResolveRequest(detections=[DetectedText(text="BACAR OR", kind="text")])
    resp = r.resolve(req)
    assert resp.unresolved_indices == [0]
    assert not resp.candidates


def test_skips_unsupported_leader_for_supported_runnerup():
    # A coincidental match can out-score the real product; the gate skips it and takes the next
    # candidate that a token actually supports, instead of blocking on matches[0].
    store = _FakeMatchStore([
        (_product("Bacardi", "p:bac"), 0.625),            # coincidence, unsupported
        (_product("Guinness Draught", "p:gui"), 0.55),    # real, "GUINNESS" present
    ])
    r = Resolver(store)
    req = ScanResolveRequest(detections=[DetectedText(text="GUINNESS DRAUGHT BACAR", kind="text")])
    resp = r.resolve(req)
    assert resp.candidates and resp.candidates[0].resolved.product.name == "Guinness Draught"
    assert resp.candidates[0].match_score == 0.55


# ---- duplicate-record collapse (one overlay per real product) ----

@pytest.mark.parametrize("a, b", [
    # Same beer under two UPCs, brand is the "Unknown" placeholder -> name alone must merge them.
    (("Lagunitas IPA", "Unknown", "off:1"), ("Lagunitas IPA", "Unknown", "off:2")),
    # Brand just echoes the name (Heineken/Heineken) -> the echo drops out, still merges.
    (("Heineken", "Heineken", "off:a"), ("Heineken", "Heineken", "off:b")),
    # Cross-source (a TTB row + an OFF row) with a shared brand+name is the same product.
    (("Sierra Nevada Pale Ale", "Sierra Nevada", "ttb:1"),
     ("Sierra Nevada Pale Ale", "Sierra Nevada", "off:9")),
    # A digit-only name is a real identity (Kronenbourg 1664); the two rows are one beer.
    (("1664", "1664", "off:a"), ("1664", "1664", "off:b")),
    # Case / punctuation / accents don't distinguish a product.
    (("BUD LIGHT", "Bud Light", "off:a"), ("Bud light", "Unknown", "off:b")),
])
def test_identity_key_merges_same_beer(a, b):
    assert _identity_key(*a) == _identity_key(*b)


@pytest.mark.parametrize("a, b", [
    # A generic class name OFF reuses across distilleries -> the distinct brands keep them apart.
    (("Blended Scotch Whisky", "Johnnie Walker", "off:1"),
     ("Blended Scotch Whisky", "Queen Margot", "off:2")),
    # Unnamed rows have nothing to canonicalize on -> id fallback, never merged into each other.
    (("", "", "off:e1"), ("", "", "off:e2")),
    # A line extension is a different product, not a duplicate.
    (("Heineken", "Heineken", "off:a"), ("Heineken Light", "Heineken", "off:b")),
    # Alcohol-free variants are NOT the same product as the full-strength sibling — the digit in
    # "0.0%" must survive normalization (it's dropped if the key runs through word-only _tokens).
    (("Jupiler", "Jupiler", "off:a"), ("Jupiler 0,0%", "Jupiler", "off:b")),
    (("Carlsberg", "Carlsberg", "off:a"), ("Carlsberg 0%", "Carlsberg", "off:b")),
    # Different age statements are different whiskies.
    (("Aberlour 10 ans", "Aberlour", "off:a"), ("Aberlour 12 ans", "Aberlour", "off:b")),
])
def test_identity_key_separates_distinct_products(a, b):
    assert _identity_key(*a) != _identity_key(*b)


def _seed_product(store, pid, name, brand_name, upc, brand_id):
    if store.get_gold("prod:seed") is None:
        store.put_gold("prod:seed", "producer",
                       Producer(id="prod:seed", name="Seed Co").model_dump(mode="json"))
    store.put_gold(brand_id, "brand",
                   Brand(id=brand_id, producer_id="prod:seed", name=brand_name)
                   .model_dump(mode="json"))
    store.put_gold(pid, "product",
                   Product(id=pid, brand_id=brand_id, producer_id="prod:seed",
                           category=Category.BEER, name=name).model_dump(mode="json"))
    store.put_gold(f"sku:{upc}", "sku",
                   SKU(id=f"sku:{upc}", product_id=pid, container="can", upc=upc)
                   .model_dump(mode="json"))


def test_collapses_duplicate_records_of_same_beer(store):
    # Two catalog rows for the identical beer (two real UPCs) must draw ONE overlay, not two.
    _seed_product(store, "off:1", "Lagunitas IPA", "Unknown", "111", "brand:unk")
    _seed_product(store, "off:2", "Lagunitas IPA", "Unknown", "222", "brand:unk")
    r = Resolver(store)
    req = ScanResolveRequest(detections=[
        DetectedText(text="111", kind="barcode"),
        DetectedText(text="222", kind="barcode"),
    ])
    resp = r.resolve(req)
    assert len(resp.candidates) == 1
    assert resp.candidates[0].resolved.product.name == "Lagunitas IPA"


def test_does_not_collapse_distinct_products_with_generic_names(store):
    # Two different scotches OFF named only "Blended Scotch Whisky" stay as two overlays — the
    # brand is the identity, so the name-only collision must not merge them.
    _seed_product(store, "off:s1", "Blended Scotch Whisky", "Johnnie Walker", "301", "brand:jw")
    _seed_product(store, "off:s2", "Blended Scotch Whisky", "Queen Margot", "302", "brand:qm")
    r = Resolver(store)
    req = ScanResolveRequest(detections=[
        DetectedText(text="301", kind="barcode"),
        DetectedText(text="302", kind="barcode"),
    ])
    resp = r.resolve(req)
    assert len(resp.candidates) == 2
