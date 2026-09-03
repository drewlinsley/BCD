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


# ---- label chrome must not outrank the real beer ------------------------------------
# Live failure on a real Heady Topper can: the can's own printing resolved to two wrong
# products that both outranked the right one, all three inside a 0.012 band above the floor.

@pytest.mark.parametrize("query, name", [
    ("DRINK FROM THE CAN", "Life drink"),              # the phrase printed on the can
    ("DRINK RESPONSIBLY", "Black Spiced Spirit Drink"),
    ("AMERICAN DOUBLE IPA", "Hazy Double IPA Thing"),
])
def test_token_support_rejects_category_word_agreement(query, name):
    # Agreeing on "drink" or "double" is not evidence of identity — every other label on the
    # shelf carries those words too. Only a token that identifies something counts.
    assert _token_supported(query, name) is False


def test_wholly_generic_name_needs_a_near_exact_read():
    # "FML Hazy Double IPA" has no identifying token at all (hazy/double/ipa are category
    # words, "FML" is under the length floor), so it behaves like a short name: a partial
    # chrome line must not claim it...
    store = _FakeMatchStore([(_product("FML Hazy Double IPA", "p:fml"), 0.55)])
    req = ScanResolveRequest(detections=[DetectedText(text="AMERICAN DOUBLE IPA", kind="text")])
    assert Resolver(store).resolve(req).unresolved_indices == [0]

    # ...but a clean read of the name itself still resolves, so recall is not lost.
    clean = _FakeMatchStore([(_product("FML Hazy Double IPA", "p:fml"), 0.95)])
    resp = Resolver(clean).resolve(
        ScanResolveRequest(detections=[DetectedText(text="FML HAZY DOUBLE IPA", kind="text")]))
    assert resp.candidates[0].resolved.product.name == "FML Hazy Double IPA"


class _PerTextMatchStore(_FakeMatchStore):
    """Matches keyed by the detection text, so one frame's several OCR lines can each be
    scored the way the live pg_trgm backend scored them."""

    def __init__(self, by_text):
        self._by_text = by_text

    def match_products(self, text, limit=3):
        return self._by_text.get(text, [])[:limit]


def test_can_chrome_loses_to_the_brand_line():
    frame = {
        "THE ALCHEMIST": [(_product("The Alchemist Heady Topper", "p:ht"), 0.538)],
        "AMERICAN DOUBLE IPA": [(_product("FML Hazy Double IPA", "p:fml"), 0.550)],
        "DRINK FROM THE CAN": [(_product("Life drink", "p:life"), 0.545)],
    }
    req = ScanResolveRequest(detections=[
        DetectedText(text=t, kind="text") for t in frame
    ])
    resp = Resolver(_PerTextMatchStore(frame)).resolve(req)
    names = [c.resolved.product.name for c in resp.candidates]
    assert names == ["The Alchemist Heady Topper"]  # the only line that identifies anything
    assert sorted(resp.unresolved_indices) == [1, 2]


def test_a_longer_catalog_name_is_not_penalised_for_being_specific():
    """"BOMBAY SAPPHIRE" must not resolve to "Gin Bombay".

    Scoring only how much of the *catalog name* the OCR line covers rewards stubby
    entries: every extra word in the correct answer dilutes it while a short wrong one
    keeps a high score. Real regression — the label reads BOMBAY SAPPHIRE and the app
    said Gin Bombay.
    """
    store = MedallionStore(root=tempfile.mkdtemp())
    for pid, name in (
        ("off:1", "Gin Bombay"),
        ("off:2", "Bombay Sapphire London Dry Gin"),
        ("off:3", "Bombay London Dry Gin"),
    ):
        store.put_gold(pid, "product", {"id": pid, "name": name})

    ranked = store.match_products("BOMBAY SAPPHIRE", limit=3)
    assert ranked[0][0]["name"] == "Bombay Sapphire London Dry Gin"
    assert ranked[0][1] == 1.0


def test_a_short_name_inside_a_noisy_line_still_wins():
    """The other direction has to keep working: here the OCR line is the noisy one.

    The rival is a real rival — another brewery's stout — rather than a brandless
    "Draught Stout", which is not a row the catalog would contain and which ties on every
    metric by construction.
    """
    store = MedallionStore(root=tempfile.mkdtemp())
    store.put_gold("brand:g", "brand", {"id": "brand:g", "name": "Guinness"})
    store.put_gold("brand:s", "brand", {"id": "brand:s", "name": "Samuel Smith"})
    store.put_gold("off:1", "product",
                   {"id": "off:1", "name": "Guinness", "brand_id": "brand:g"})
    store.put_gold("off:2", "product",
                   {"id": "off:2", "name": "Extra Stout", "brand_id": "brand:s"})

    ranked = store.match_products("GUINNESS DRAUGHT 440ML EXTRA STOUT", limit=2)
    assert ranked[0][0]["name"] == "Guinness"


def test_the_brand_half_of_a_split_row_is_matchable():
    """A label names brand and product together; the catalog stores them apart.

    "Handmade Vodka" under brand Tito's must win "TITOS HANDMADE VODKA" over a rival whose
    name alone is just as contained in the label.
    """
    store = MedallionStore(root=tempfile.mkdtemp())
    store.put_gold("brand:t", "brand", {"id": "brand:t", "name": "Tito's"})
    store.put_gold("brand:o", "brand", {"id": "brand:o", "name": "Other Distillery"})
    store.put_gold("off:1", "product",
                   {"id": "off:1", "name": "Handmade Vodka", "brand_id": "brand:t"})
    store.put_gold("off:2", "product",
                   {"id": "off:2", "name": "Handmade Vodka", "brand_id": "brand:o"})

    ranked = store.match_products("TITOS HANDMADE VODKA", limit=2)
    assert ranked[0][0]["id"] == "off:1"


def test_a_placeholder_brand_is_not_glued_onto_the_name():
    """OFF writes "Unknown" when it has no brand; prepending it is pure noise."""
    from bcd_ingest.dedup import search_name

    assert search_name("Bombay Sapphire Gin", "Unknown") == "Bombay Sapphire Gin"
    assert search_name("Bombay sapphire murcian lemon", "Bombay spirits") == (
        "Bombay sapphire murcian lemon"
    )
    assert search_name("Handmade Vodka", "Tito's") == "Tito's Handmade Vodka"


def test_an_all_category_name_needs_an_equally_generic_label():
    """A row named only with category words cannot claim a label that names something.

    "DOGFISH HEAD 60 MINUTE IPA" resolved to a product literally called "Ipa Ipa": every
    token in that name is a category word, so token support abstained, and the raised
    floor it deferred to never bit because containment scores a wholly-contained name 1.0.
    Dogfish Head is not in the catalog at all — unresolved is the correct answer.
    """
    # "Irish Whiskey" is anonymous read alone, and would be refused — which is why the
    # resolver judges the brand-qualified name instead. Qualified, it is evidence.
    assert not _token_supported("JAMESON IRISH WHISKEY", "Irish Whiskey")
    assert _token_supported("JAMESON IRISH WHISKEY", "Jameson Irish Whiskey")
    # A generic line against a generic name is still allowed to match.
    assert _token_supported("IRISH WHISKEY", "Irish Whiskey")
    # A name carrying one real word of its own never depended on the brand.
    assert _token_supported("TITOS HANDMADE VODKA", "Handmade Vodka")


class _FrameStore(_PerTextMatchStore):
    """Per-line matches *plus* the producer rows behind them — frame corroboration reads a
    candidate's producer name, which the plain fake leaves as a placeholder."""

    def __init__(self, by_text, gold=None):
        super().__init__(by_text)
        self._gold = gold or {}

    def get_gold(self, gid):
        return self._gold.get(gid)


def _prod_of(name, pid, producer_id):
    return Product(id=pid, brand_id="b", producer_id=producer_id,
                   category=Category.BEER, name=name).model_dump(mode="json")


def _producer(pid, name):
    return Producer(id=pid, name=name).model_dump(mode="json")


def test_the_frame_promotes_the_beer_two_lines_name():
    """The reported bug: a can of Heady Topper answered "Chemist".

    Every line is a *perfect* word match for something — "CHEMIST" (a bad read of ALCHEMIST) is
    1.0 against a row literally named "Chemist", exactly as "HEADY TOPPER" is 1.0 against the
    real beer. Per-line scoring has nothing left to break that tie. Read as one frame the
    difference is plain: two lines name the Alchemist beer, one line names the other thing.
    """
    frame = {
        "THE ALCHEMIST": [(_prod_of("The Alchemist Heady Topper", "p:ht", "pr:alch"), 1.0)],
        "HEADY TOPPER": [(_prod_of("The Alchemist Heady Topper", "p:ht", "pr:alch"), 1.0)],
        "CHEMIST": [(_prod_of("Chemist", "p:chem", "pr:chem"), 1.0)],
    }
    gold = {"pr:alch": _producer("pr:alch", "Alchemist"),
            "pr:chem": _producer("pr:chem", "Chemist")}
    req = ScanResolveRequest(detections=[DetectedText(text=t, kind="text") for t in frame])
    resp = Resolver(_FrameStore(frame, gold)).resolve(req)

    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"
    # ...and the coincidence is not merely second, it is reported less confidently, so the
    # overlay stops claiming a certainty the frame does not support.
    assert resp.candidates[0].match_score > resp.candidates[1].match_score


def test_a_lone_line_keeps_its_confidence():
    """The corroboration penalty must not fire when there was nothing to corroborate with.

    A barcode, or a single clean brand line, is one piece of evidence because that is all the
    frame holds — not because the rest of the frame disagreed."""
    frame = {"HEADY TOPPER": [(_prod_of("Heady Topper", "p:ht", "pr:alch"), 1.0)]}
    gold = {"pr:alch": _producer("pr:alch", "Alchemist")}
    resp = Resolver(_FrameStore(frame, gold)).resolve(
        ScanResolveRequest(detections=[DetectedText(text="HEADY TOPPER", kind="text")]))
    assert resp.candidates[0].match_score == 1.0


def test_a_pure_packaging_line_is_never_matched():
    """"PINT" is the size of the can, not a drink. It resolved to a product named "Pint Cake" —
    at 1.0, and at 1.5s, because a short common word matches tens of thousands of rows."""
    frame = {"PINT": [(_prod_of("Pint Cake", "p:cake", "pr:cake"), 1.0)]}
    resp = Resolver(_FrameStore(frame)).resolve(
        ScanResolveRequest(detections=[DetectedText(text="PINT", kind="text")]))
    assert resp.candidates == []
    assert resp.unresolved_indices == [0]


def test_a_category_line_is_still_matched():
    """The packaging filter is narrower than "nothing identifying in it" on purpose: a catalog
    name can be pure category, so a clean read of one must still resolve."""
    frame = {"FML HAZY DOUBLE IPA": [(_prod_of("FML Hazy Double IPA", "p:fml", "pr:fml"), 0.95)]}
    resp = Resolver(_FrameStore(frame)).resolve(
        ScanResolveRequest(detections=[DetectedText(text="FML HAZY DOUBLE IPA", kind="text")]))
    assert resp.candidates[0].resolved.product.name == "FML Hazy Double IPA"


def test_a_second_candidate_on_one_line_can_still_win_the_frame():
    """Keeping only each line's best hit is what let chrome crowd out the beer: the right
    product can be a line's *second* candidate, and per-line scoring would never look at it."""
    frame = {
        "ALCHEMIST": [(_prod_of("Axis Alchemist", "p:axis", "pr:axis"), 1.0),
                      (_prod_of("The Alchemist Heady Topper", "p:ht", "pr:alch"), 0.62)],
        "HEADY TOPPER": [(_prod_of("The Alchemist Heady Topper", "p:ht", "pr:alch"), 1.0)],
    }
    gold = {"pr:axis": _producer("pr:axis", "Axis"),
            "pr:alch": _producer("pr:alch", "Alchemist")}
    req = ScanResolveRequest(detections=[DetectedText(text=t, kind="text") for t in frame])
    resp = Resolver(_FrameStore(frame, gold)).resolve(req)
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"
