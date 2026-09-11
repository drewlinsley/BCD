"""Resolver — matching + cold-start scoring against a seeded store."""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.resolver import (
    Resolver,
    _accounts_for_sighting,
    _identity_key,
    _token_supported,
    _upc_variants,
    _frame_support,
)
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
    # ...and the coincidence is not beside it at all. It used to ride along in second place at
    # a marked-down score; since a one-word line stopped proving a row on its own
    # (`_is_whole_label`), "CHEMIST" has nothing to prove `Chemist` with, and a corroborated
    # frame carries only what it proved.
    assert [c.resolved.product.name for c in resp.candidates] == ["The Alchemist Heady Topper"]


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


def test_the_frame_is_matched_in_one_batched_call():
    """The resolver hands the store the whole frame, so a store that can run the lines
    concurrently gets the chance to. Results must stay aligned with the lines that produced
    them, or a match gets attributed to the wrong overlay."""
    frame = {
        "THE ALCHEMIST": [(_prod_of("The Alchemist Heady Topper", "p:ht", "pr:alch"), 1.0)],
        "HEADY TOPPER": [(_prod_of("The Alchemist Heady Topper", "p:ht", "pr:alch"), 1.0)],
    }
    calls: list[list[str]] = []

    class _BatchStore(_FrameStore):
        def match_products_many(self, texts, limit=3):
            calls.append(list(texts))
            return [self.match_products(t, limit) for t in texts]

    store = _BatchStore(frame, {"pr:alch": _producer("pr:alch", "Alchemist")})
    req = ScanResolveRequest(detections=[DetectedText(text=t, kind="text") for t in frame])
    resp = Resolver(store).resolve(req)

    assert calls == [["THE ALCHEMIST", "HEADY TOPPER"]]   # one call, not one per line
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"


def test_a_store_without_batched_matching_still_resolves():
    """The dev store and any older Store implementation only have `match_products`; the
    resolver must not require the batch entry point."""
    frame = {"HEADY TOPPER": [(_prod_of("Heady Topper", "p:ht", "pr:alch"), 1.0)]}
    store = _FrameStore(frame, {"pr:alch": _producer("pr:alch", "Alchemist")})
    assert not hasattr(store, "match_products_many")
    resp = Resolver(store).resolve(
        ScanResolveRequest(detections=[DetectedText(text="HEADY TOPPER", kind="text")]))
    assert resp.candidates[0].resolved.product.name == "Heady Topper"


def test_lines_not_worth_matching_are_never_sent_to_the_store():
    """The packaging filter has to run *before* the query — skipping the work is most of the
    point, since a short common word is the most expensive thing to match."""
    asked: list[list[str]] = []

    class _BatchStore(_FrameStore):
        def match_products_many(self, texts, limit=3):
            asked.append(list(texts))
            return [self.match_products(t, limit) for t in texts]

    frame = {"HEADY TOPPER": [(_prod_of("Heady Topper", "p:ht", "pr:alch"), 1.0)]}
    req = ScanResolveRequest(detections=[
        DetectedText(text=t, kind="text")
        for t in ("HEADY TOPPER", "PINT", "12 FL OZ", "CANS")
    ])
    Resolver(_BatchStore(frame, {"pr:alch": _producer("pr:alch", "Alchemist")})).resolve(req)
    assert asked == [["HEADY TOPPER"]]


class _MakerStore(_FrameStore):
    """A store with the producer path wired: name-matched producers and their catalogs."""

    def __init__(self, by_text, gold=None, producers=None, catalog=None):
        super().__init__(by_text, gold)
        self._producers = producers or {}     # line -> [(producer_rec, score)]
        self._catalog = catalog or {}         # producer id -> [product_rec]

    def match_producers(self, text, limit=3):
        return self._producers.get(text, [])[:limit]

    def products_of(self, producer_id, limit=8):
        return self._catalog.get(producer_id, [])[:limit]


def _beer(name, pid, producer_id):
    return Product(id=pid, brand_id="b", producer_id=producer_id,
                   category=Category.BEER, name=name).model_dump(mode="json")


def test_the_maker_answers_when_the_product_name_is_unreadable():
    """The real failure this exists for: a Heady Topper can's wordmark OCR'd as Cyrillic, so
    the beer's own name never reached the resolver — but "ALCHEMIST-VER" did, 23 times."""
    line = "ALCHEMIST-VER"
    store = _MakerStore(
        by_text={},                                   # no product matches the garbled line
        gold={"pr:alch": _producer("pr:alch", "Alchemist")},
        producers={line: [(_producer("pr:alch", "Alchemist"), 1.0)]},
        catalog={"pr:alch": [_beer("The Alchemist Heady Topper", "p:ht", "pr:alch")]},
    )
    resp = Resolver(store).resolve(ScanResolveRequest(detections=[
        DetectedText(text=line, kind="text"),
        DetectedText(text="ALE\nALC. 8% BY VOL\n1 PINT", kind="text"),
    ]))
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"
    # Indirect evidence: it must be reported less confidently than a label that named the beer.
    assert resp.candidates[0].match_score < 1.0


def test_a_maker_with_a_whole_shelf_is_not_guessed_from():
    """Knowing who made it is not knowing what it is. Past a handful of products the honest
    answer is nothing, not a coin flip between eight of them."""
    line = "BIG BREWERY CO"
    store = _MakerStore(
        by_text={},
        producers={line: [(_producer("pr:big", "Big Brewery"), 1.0)]},
        catalog={"pr:big": [_beer(f"Beer {n}", f"p:{n}", "pr:big") for n in range(8)]},
    )
    resp = Resolver(store).resolve(
        ScanResolveRequest(detections=[DetectedText(text=line, kind="text")]))
    assert resp.candidates == []


def test_the_label_category_separates_two_beers_from_one_maker():
    """Both are that brewery's, and the garbled line names neither. The fine print does: the
    can says ALE, and only one of them is a beer."""
    line = "ALCHEMIST-VER"
    other = Product(id="p:amer", brand_id="b", producer_id="pr:alch",
                    category=Category.OTHER, name="Alchemist Amer").model_dump(mode="json")
    store = _MakerStore(
        by_text={line: [(other, 0.72)]},              # the sibling the *product* path finds
        gold={"pr:alch": _producer("pr:alch", "Alchemist")},
        producers={line: [(_producer("pr:alch", "Alchemist"), 1.0)]},
        catalog={"pr:alch": [_beer("The Alchemist Heady Topper", "p:ht", "pr:alch"), other]},
    )
    resp = Resolver(store).resolve(ScanResolveRequest(detections=[
        DetectedText(text=line, kind="text"),
        DetectedText(text="ALE\nALC. 8% BY VOL\n1 PINT", kind="text"),
    ]))
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"


def test_a_contradicting_category_is_marked_down():
    """The frame says ALE and the row is a spirit — evidence against, not merely absent."""
    spirit = Product(id="p:gin", brand_id="b", producer_id="pr:c",
                     category=Category.SPIRIT, name="Chemist Gin").model_dump(mode="json")
    frame = {"CHEMIST GIN": [(spirit, 1.0)]}
    plain = Resolver(_FrameStore(frame)).resolve(
        ScanResolveRequest(detections=[DetectedText(text="CHEMIST GIN", kind="text")]))
    contradicted = Resolver(_FrameStore(frame)).resolve(ScanResolveRequest(detections=[
        DetectedText(text="CHEMIST GIN", kind="text"),
        DetectedText(text="ALE\nALC. 8% BY VOL", kind="text"),
    ]))
    assert contradicted.candidates[0].match_score < plain.candidates[0].match_score


def test_an_ambiguous_label_withholds_the_category_hint():
    from bcd_api.resolver import _category_hint
    say = lambda *t: [DetectedText(text=x, kind="text") for x in t]  # noqa: E731
    assert _category_hint(say("ALE", "ALC 8% BY VOL")) == "beer"
    assert _category_hint(say("LONDON DRY GIN")) == "spirit"
    assert _category_hint(say("NOTHING CATEGORICAL HERE")) is None
    # A malt-whisky label says both; a wrong filter is worse than none.
    assert _category_hint(say("ALE", "WHISKY")) is None


def test_a_three_letter_name_must_be_read_not_merely_contained():
    """A catalog row literally named `Ver` matched "VERMIKI", "VERMIL" and "VERM" off a Vermont
    can. Such a name has no token for the support guard to anchor on, so that guard passes it
    unconditionally and only the raised floor stands — and the floor is measured with
    word_similarity, which asks whether the name appears *inside* the line. A 3-letter name
    appears inside almost anything."""
    ver = Product(id="p:ver", brand_id="b", producer_id="pr:ver",
                  category=Category.OTHER, name="Ver").model_dump(mode="json")
    for garble in ("VERM", "VERMIL", "VERMIKI", "VERMONT"):
        resp = Resolver(_FrameStore({garble: [(ver, 1.0)]})).resolve(
            ScanResolveRequest(detections=[DetectedText(text=garble, kind="text")]))
        assert resp.candidates == [], garble

    # ...but a clean read of that same short name still resolves, so recall is not lost.
    resp = Resolver(_FrameStore({"VER": [(ver, 1.0)]})).resolve(
        ScanResolveRequest(detections=[DetectedText(text="VER", kind="text")]))
    assert resp.candidates[0].resolved.product.name == "Ver"


def test_a_digit_name_is_still_reachable():
    """"1664" has no letter tokens at all, so the new test cannot judge it and must defer to
    the raised floor rather than making a real beer unreachable."""
    beer = Product(id="p:1664", brand_id="b", producer_id="pr:k",
                   category=Category.BEER, name="1664").model_dump(mode="json")
    resp = Resolver(_FrameStore({"1664": [(beer, 1.0)]})).resolve(
        ScanResolveRequest(detections=[DetectedText(text="1664", kind="text")]))
    assert resp.candidates[0].resolved.product.name == "1664"


def test_the_category_line_cannot_certify_a_fragment():
    """The reported bug: a Heady Topper can answered "Ache" and "Mist", at 1.00.

    The recognizer split the wordmark mid-word -- THE ALCHEMIST VERMONT came off the can as
    "ACHE MIST-VERM" and "ALCHE MIST VERM" -- and both halves are real registered product
    names. Containment scores them 1.0, and the short-name guard cannot object because the
    token genuinely was read.

    What made it worse than a bad guess is that the frame *certified* itself: one line named
    the fragment and the "ALE / ALC. 8% BY VOL" line agreed on the category, which reached the
    corroboration bar. A certified frame is exactly the one the client does not ask the model
    about -- so the answer that would have been right never got asked for.
    """
    frame = {"ACHE MIST-VERM": [(_prod_of("Ache", "p:ache", "pr:ache"), 1.0)]}
    gold = {"pr:ache": _producer("pr:ache", "Ache")}
    req = ScanResolveRequest(detections=[
        DetectedText(text="ACHE MIST-VERM", kind="text"),
        DetectedText(text="ALE\nALC. 8% BY VOL\n1 PINT", kind="text"),
        DetectedText(text="DRINK FROM", kind="text"),
    ])
    resp = Resolver(_FrameStore(frame, gold)).resolve(req)

    assert resp.candidates, "still offered, just not trusted"
    assert resp.candidates[0].resolved.product.name == "Ache"
    assert not resp.corroborated, "the category is not one of the frame's lines naming it"
    assert resp.candidates[0].match_score < 1.0, "and the overlay must not read as certainty"


def test_a_whole_label_read_on_one_line_still_certifies_itself():
    """The exemption this must not break: one clean line that IS the label. Asking the model
    to confirm a 1.00 spends a second to learn nothing."""
    frame = {"BLUE MOON BELGIAN WHITE": [
        (_prod_of("Blue Moon Belgian White", "p:bm", "pr:bm"), 1.0)]}
    gold = {"pr:bm": _producer("pr:bm", "Blue Moon")}
    req = ScanResolveRequest(detections=[
        DetectedText(text="BLUE MOON BELGIAN WHITE", kind="text")])
    resp = Resolver(_FrameStore(frame, gold)).resolve(req)

    assert resp.corroborated
    assert resp.candidates[0].match_score == 1.0


def test_accounts_for_the_line_separates_a_fragment_from_a_whole_label():
    from bcd_api.resolver import _accounts_for_the_line

    # measured off the real can: every one of these is a piece of THE ALCHEMIST VERMONT
    assert not _accounts_for_the_line("Mist", "ACHE MIST-VERM")
    assert not _accounts_for_the_line("Ache", "ACHE MISTVERN")
    assert not _accounts_for_the_line("Chemist", "CHEMIST VER")
    # the whole label, however it was cased or punctuated
    assert _accounts_for_the_line("Bombay Sapphire London Dry Gin",
                                  "BOMBAY SAPPHIRE LONDON DRY GIN")
    assert _accounts_for_the_line("Heady Topper", "**Heady Topper**")
    # leaves out the brand the label shows -- correctly not "the whole label"
    assert not _accounts_for_the_line("Draught Stout", "GUINNESS DRAUGHT STOUT")


def test_a_line_of_pure_chrome_names_no_maker():
    """"DRINK FROM" is what a Heady Topper can prints, not a brand.

    Both words are known chrome, so the line carries no identity -- and _token_supported
    accepts a styleless *name* against a styleless *line*, which is right for a product
    ("Irish Whiskey" read off a label that says only that) and wrong here: the two generic
    halves simply agree with each other. A producer registered as "drink drink!" was reached
    this way and its beer offered at 0.60.
    """
    producers = {"DRINK FROM": [(_producer("pr:dd", "drink drink!"), 1.0)]}
    catalog = {"pr:dd": [_prod_of("Trotinette", "p:tro", "pr:dd")]}
    gold = {"pr:dd": _producer("pr:dd", "drink drink!")}
    req = ScanResolveRequest(detections=[DetectedText(text="DRINK FROM", kind="text")])
    resp = Resolver(_MakerStore({}, gold, producers, catalog)).resolve(req)

    assert resp.candidates == []
    assert not resp.corroborated


def test_the_maker_path_still_answers_a_line_that_names_one():
    """The guard must not close the door the producer path exists to open."""
    producers = {"ALCHEMIST VER": [(_producer("pr:alch", "The Alchemist LLC"), 0.9)]}
    catalog = {"pr:alch": [_prod_of("The Alchemist Heady Topper", "p:ht", "pr:alch")]}
    gold = {"pr:alch": _producer("pr:alch", "The Alchemist LLC")}
    req = ScanResolveRequest(detections=[DetectedText(text="ALCHEMIST VER", kind="text")])
    resp = Resolver(_MakerStore({}, gold, producers, catalog)).resolve(req)

    assert resp.candidates, "a line that does name a maker still reaches its catalog"
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"


def test_a_shelf_returns_every_beer_on_it():
    """The HUD's actual job, and the case the suite never covered: several labels in one frame.

    Corroboration was reachable two ways — two lines naming one product, or a strong read of a
    line in a frame holding fewer than two identity lines. A shelf is neither. Every bottle gets
    exactly one line naming it and no second line to agree with it, and the frame has many
    identity lines, so *nothing* could corroborate; the unproven-frame cap then kept a single
    guess and the client withheld even that. Three beers in view, an empty screen.
    """
    frame = {
        "SIERRA NEVADA PALE ALE": [(_prod_of("Sierra Nevada Pale Ale", "p:sn", "pr:sn"), 1.0)],
        "LAGUNITAS IPA": [(_prod_of("Lagunitas IPA", "p:lag", "pr:lag"), 1.0)],
        "GUINNESS DRAUGHT STOUT": [(_prod_of("Guinness Draught Stout", "p:gui", "pr:gui"), 1.0)],
    }
    req = ScanResolveRequest(detections=[DetectedText(text=t, kind="text") for t in frame])
    resp = Resolver(_FrameStore(frame)).resolve(req)

    assert resp.corroborated
    assert {c.resolved.product.name for c in resp.candidates} == {
        "Sierra Nevada Pale Ale", "Lagunitas IPA", "Guinness Draught Stout"}


def test_one_line_yields_one_overlay():
    """Two catalog rows for the same beer — a brand-level row beside the product one — both
    account for the same line, so on a shelf each proved itself against it and one bottle drew
    two overlays. Three beers drew five, five drew eight."""
    frame = {
        "LAGUNITAS IPA": [(_prod_of("Lagunitas IPA", "p:lag", "pr:lag"), 1.0),
                          (_prod_of("Lagunitas", "p:lagb", "pr:lag"), 1.0)],
    }
    req = ScanResolveRequest(detections=[DetectedText(text="LAGUNITAS IPA", kind="text")])
    resp = Resolver(_FrameStore(frame)).resolve(req)

    assert len(resp.candidates) == 1


def test_a_four_pack_does_not_corroborate_itself():
    """A four-pack prints its brand once per can, so one phrase arrives as several detections.
    Counting each as independent agreement certified whatever they happened to share: "LITTLE"
    read twice off a Little Willow pack proved six unrelated products with `little` in the name,
    and "DRINK FROM THE CAN!" read three times proved one called `Now & Then` off the word THEN.
    """
    echo = _prod_of("Now & Then", "p:nt", "pr:nt")
    frame = {
        "ECAN! DRINK FROM THEN": [(echo, 0.9)],
        "AN! DRINK FROM THEN": [(echo, 0.9)],
        "FROM THE CAN! DRIN": [(echo, 0.9)],
    }
    req = ScanResolveRequest(detections=[DetectedText(text=t, kind="text") for t in frame])
    resp = Resolver(_FrameStore(frame)).resolve(req)

    assert not resp.corroborated, "three readings of one slogan are one piece of evidence"


# --- naming a label from the picture ------------------------------------------------

def test_a_clean_reading_of_the_whole_label_is_accounted_for():
    # A model reading the picture returns the label the way the label is printed: maker and
    # drink together. Similarity reads that as a poor match (0.43) -- which is why this rule
    # asks a different question.
    assert _accounts_for_sighting("Heady Topper", "The Alchemist", "The Alchemist Heady Topper")
    assert _accounts_for_sighting("Heady Topper", "The Alchemist", "Heady Topper")
    assert _accounts_for_sighting("Pale Ale", "Sierra Nevada", "Sierra Nevada Brewing Co. Pale Ale")


def test_a_row_that_is_a_piece_of_the_name_is_a_different_beer():
    # `Banger` scores the same 0.43 against "Focal Banger" that the *correct* row above scores
    # against its own label, so no threshold separates them. The leftover word does.
    assert not _accounts_for_sighting("Banger", "The Alchemist", "Focal Banger")
    assert not _accounts_for_sighting("Mist", "Whoever", "The Alchemist Vermont Ale")
    assert not _accounts_for_sighting("Heady Topper", "The Alchemist", "Heady Topper Vermont")


def test_what_every_label_prints_is_not_a_leftover():
    assert _accounts_for_sighting("Heady Topper", "The Alchemist",
                                  "The Alchemist Heady Topper Double IPA 16 oz can")


def test_a_reading_that_names_no_product_is_not_accounted_for_by_any():
    # Otherwise a sighting of "IPA" would be answered by whichever IPA sorted first.
    assert not _accounts_for_sighting("Heady Topper", "The Alchemist", "IPA")
    assert not _accounts_for_sighting("Heady Topper", "The Alchemist", "Focal Banger")


def test_a_reading_is_answered_by_the_row_it_names_not_a_piece_of_it(store):
    # The store seeds `Heady Topper` and a decoy literally named `Banger`. Containment scores
    # any name wholly inside the reading at 1.00, which is how a row named `Lawson's` came back
    # for "Lawson's Sip of Sunshine" and `Green` for "Other Half Green City".
    r = Resolver(store)
    assert r.resolve_reading("The Alchemist Heady Topper").resolved.product.name == "Heady Topper"
    assert r.resolve_reading("Focal Banger") is None
    assert r.resolve_reading("Pliny The Elder") is None


def test_a_reading_carries_a_score_and_its_place_in_the_frame(store):
    r = Resolver(store)
    cand = r.resolve_reading("Heady Topper", index=2)
    assert cand.detection_index == 2
    assert 0 < cand.match_score <= 1.0
    assert cand.personal_score is not None       # scored for the caller, like every other path


# --- objects: one can, one verdict ------------------------------------------------------
#
# The rows below are the ones measured against real iPhone frames in review of the first
# object resolver: junk catalog rows (`Top's`, `Ache`, `Theo P.`, `Banger`) that a
# coverage-of-the-target score let win by saying less. Each row here is a device reading
# from that table, and the verdict is what the HUD is allowed to do with it.


@pytest.fixture()
def shelf(store):
    def producer(pid, name):
        store.put_gold(pid, "producer", Producer(id=pid, name=name).model_dump(mode="json"))

    def product(pid, name, producer_id, brand=None):
        bid = f"brand:{pid}"
        store.put_gold(bid, "brand", Brand(id=bid, producer_id=producer_id,
                                           name=brand or name).model_dump(mode="json"))
        store.put_gold(pid, "product", Product(id=pid, name=name, producer_id=producer_id,
                                               brand_id=bid, category=Category.BEER
                                               ).model_dump(mode="json"))

    producer("prod:x2", "The Alchemist LLC")
    product("ttb:focal", "Focal Banger", "prod:x2")
    producer("prod:tops", "Top's Brewing")
    product("ttb:tops", "Top's", "prod:tops")
    producer("prod:ache", "Ache Brewing")
    product("ttb:ache", "Ache", "prod:ache")
    producer("prod:theo", "Theo P. Brewing")
    product("ttb:theo", "Theo P.", "prod:theo")
    producer("prod:banger", "Banger")
    product("ttb:banger", "Banger", "prod:banger")
    producer("prod:chem", "Chemist Spirits")
    product("ttb:chem", "Chemist", "prod:chem")
    return store


def _verdict(store, texts, barcode=None, **kw):
    r = Resolver(store)
    obj = DetectedObject(id="o1", texts=texts, barcode=barcode)
    resp = r.resolve(ScanResolveRequest(objects=[obj], **kw))
    assert len(resp.objects) == 1
    return resp, resp.objects[0]


def _names(res):
    return [c.resolved.product.name for c in res.candidates]


@pytest.mark.parametrize("reading, junk", [
    ("FADY TOP", "Top's"),
    ("ACHE MIST-VERM", "Ache"),
    ("NK FROM THEO BANGE", "Theo P."),
])
def test_a_garbled_fragment_never_fires_a_short_junk_row(shelf, reading, junk):
    resp, res = _verdict(shelf, [reading])
    assert res.status == "unresolved", (reading, _names(res))
    assert junk not in _names(res)
    assert resp.candidates == [] and not resp.corroborated


def test_a_clean_full_label_read_resolves_to_the_row_that_accounts_for_it(shelf):
    # `Banger` matches the one word it has at 1.0; `Focal Banger` matches at 1.0 too. The
    # leftover-word rule decides: "focal" and "alchemist" are unexplained by `Banger`.
    resp, res = _verdict(shelf, ["FOCAL BANGER THE ALCHEMIST INDIA PALE ALE"])
    assert res.status == "resolved"
    assert _names(res) == ["Focal Banger"]
    assert res.candidates[0].object_id == "o1"
    assert res.candidates[0].detection_index == -1
    assert resp.corroborated and [c.resolved.product.name for c in resp.candidates] == ["Focal Banger"]


def test_two_lines_naming_one_beer_resolve_it(shelf):
    _, res = _verdict(shelf, ["HEADY TOPPER", "THE ALCHEMIST", "STOWE VERMONT",
                              "DRINK FROM THE CAN"])
    assert res.status == "resolved" and _names(res) == ["Heady Topper"]


def test_a_barcode_on_the_object_resolves_it_outright(shelf):
    _, res = _verdict(shelf, ["ANYTHING AT ALL"], barcode="854416001019")
    assert res.status == "resolved" and _names(res) == ["Heady Topper"]
    assert res.candidates[0].match_score == 1.0


def test_a_partial_read_is_ambiguous_not_wrong(shelf):
    # The brewery read cleanly, the beer's name half-read: the frame proves nothing, but
    # `Focal Banger` explains "focal" and "alchemist" — worth the fine stage, not an overlay.
    _, res = _verdict(shelf, ["FOCAL BAN", "THE ALCHEMIST", "INDIA PALE ALE"])
    assert res.status in ("ambiguous", "resolved")
    assert "Focal Banger" in _names(res)
    assert "Banger" not in _names(res) and "Chemist" not in _names(res)


def test_label_chrome_alone_resolves_nothing(shelf):
    _, res = _verdict(shelf, ["DRINK FROM THE CAN", "16 FL OZ", "INDIA PALE ALE"])
    assert res.status == "unresolved" and res.candidates == []


def test_the_client_can_raise_the_floor(shelf):
    _, res = _verdict(shelf, ["FOCAL BANGER THE ALCHEMIST INDIA PALE ALE"], min_match_score=1.01)
    assert res.status != "resolved"


def test_objects_and_lines_share_one_response(shelf):
    r = Resolver(shelf)
    resp = r.resolve(ScanResolveRequest(
        detections=[DetectedText(text="HEADY TOPPER"), DetectedText(text="THE ALCHEMIST")],
        objects=[DetectedObject(id="o2", texts=["FOCAL BANGER THE ALCHEMIST INDIA PALE ALE"])]))
    by_kind = {(c.detection_index, c.object_id): c.resolved.product.name for c in resp.candidates}
    assert by_kind[(0, None)] == "Heady Topper" or by_kind.get((1, None)) == "Heady Topper"
    assert by_kind[(-1, "o2")] == "Focal Banger"
    assert resp.objects[0].status == "resolved"


def test_lexicon_carries_names_not_generic_words(shelf):
    words = Resolver(shelf).lexicon(limit=100)
    assert "alchemist" in words and "topper" in words and "focal" in words
    assert "ipa" not in words and "brewing" not in words


def test_duplicate_rows_do_not_flicker_between_frames():
    """Two catalog rows for one bottle must not alternate on screen.

    Bombay Sapphire was filed twice — "Bombay Sapphire Dry Gin" and "Bombay Sapphire London
    Dry Gin" — and both account for the same reading with the same score, so the winner came
    down to whichever the store happened to return first. pg_trgm does not promise an order
    between equal scores, so consecutive frames of a *motionless* bottle named different rows
    and the HUD flickered. The answer has to be a function of the reading alone.
    """
    a = (_product("Bombay Sapphire Dry Gin", "off:a"), 1.0)
    b = (_product("Bombay Sapphire London Dry Gin", "off:b"), 1.0)
    reading = DetectedText(text="BOMBAY SAPPHIRE LONDON DRY GIN", kind="text")

    seen = set()
    for matches in ([a, b], [b, a]):          # the only difference is retrieval order
        r = Resolver(_FakeMatchStore(list(matches)))
        resp = r.resolve(ScanResolveRequest(detections=[reading]))
        assert resp.candidates, "a clean label reading should still resolve"
        seen.add(resp.candidates[0].resolved.product.id)

    assert len(seen) == 1, f"retrieval order changed the answer: {seen}"


def test_tie_prefers_the_row_the_reading_accounts_for():
    """A tie must not be broken by name length.

    Bombay Sapphire East is filed as "East Vapour Infused London Dry Gin" under the maker
    "Bombay Sapphire", so the maker supplies "bombay" and "sapphire" and a frame reading
    BOMBAY / SAPPHIRE / LONDON DRY GIN is explained by BOTH rows. The row whose own name the
    camera actually read has to win: "east", "vapour" and "infused" are nowhere in the frame.

    Scope, honestly: against the live catalog this frame DID resolve to East when the tie was
    broken on name length, and it does not here -- the seeded store gives the plain row more
    frame support, so the tie-break never decides it. So this pins the outcome, not the
    tie-break itself; the tie-break is covered by the flicker test above and was verified
    end-to-end against the 534k-row catalog.
    """
    plain = _prod_of("Bombay Sapphire London Dry Gin", "off:plain", "pr:bs")
    east = _prod_of("East Vapour Infused London Dry Gin", "off:east", "pr:bs")
    gold = {"pr:bs": _producer("pr:bs", "Bombay Sapphire")}
    lines = ["BOMBAY", "SAPPHIRE", "LONDON DRY GIN"]

    for order in ([plain, east], [east, plain]):
        store = _FrameStore({t: [(rec, 1.0) for rec in order] for t in lines}, gold)
        r = Resolver(store)
        resp = r.resolve(ScanResolveRequest(
            detections=[DetectedText(text=t, kind="text") for t in lines]))
        assert resp.candidates, "the frame names a gin the catalog holds"
        assert resp.candidates[0].resolved.product.id == "off:plain", (
            f"picked {resp.candidates[0].resolved.product.name!r} for a frame that never "
            "read 'east', 'vapour' or 'infused'"
        )


def _prod_with_aliases(name, pid, producer_id, aliases):
    return Product(id=pid, brand_id="b", producer_id=producer_id, category=Category.SPIRIT,
                   name=name, aliases=aliases).model_dump(mode="json")


@pytest.mark.parametrize("line,row", [
    ("CHEMIST", "Chemist"),      # the tail of THE ALCHEMIST, read exactly, on a Heady Topper can
    ("DeadEye", "Deadeye"),      # the on-device model's tidying of a garbled HEADY
])
def test_a_single_word_does_not_prove_a_label(line, row):
    """An exact one-word read must not certify a one-word row.

    Both of these were drawn over a can of Heady Topper on 2026-09-10: `Chemist` (a distillery)
    and `Deadeye` (a rum), each a perfect 1.00 against the only line in its frame, each seven
    letters -- past the character floor that was raised for `Bale` and `Mist`, and past the
    similarity bar that was raised for "CHEMIST-VE". No threshold separates an exact read of a
    word from an exact read of a word. A label is a phrase; one word has nothing beside it to
    agree, so it may match but it may not prove.
    """
    frame = {line: [(_prod_of(row, "p:x", "pr:x"), 1.0)]}
    gold = {"pr:x": _producer("pr:x", row)}
    resp = Resolver(_FrameStore(frame, gold)).resolve(
        ScanResolveRequest(detections=[DetectedText(text=line, kind="text")]))
    assert not resp.corroborated, f"{line!r} alone certified {row!r}"


def test_two_words_still_prove_a_label():
    """The control for the test above: "STONE IPA" is the least substantial label the
    recogniser is meant to know, and it is two words, so it still proves itself."""
    frame = {"STONE IPA": [(_prod_of("Stone IPA", "p:stone", "pr:stone"), 1.0)]}
    gold = {"pr:stone": _producer("pr:stone", "Stone Brewing")}
    resp = Resolver(_FrameStore(frame, gold)).resolve(
        ScanResolveRequest(detections=[DetectedText(text="STONE IPA", kind="text")]))
    assert resp.corroborated
    assert resp.candidates[0].resolved.product.name == "Stone IPA"


def test_aliases_are_words_the_label_prints():
    """A merged row's aliases are names the bottle carries, and the frame may support it
    through them.

    The Bombay Sapphire label prints VAPOUR INFUSED. Those words are in the *name* of `East
    Vapour Infused London Dry Gin` -- a different gin by the same house -- and only in an
    *alias* of the plain gin's row, so a frame reading BOMBAY / SAPPHIRE / INFUSED INFUSE gave
    its third line to East alone and East won, on a bottle that printed EAST nowhere. Replayed
    from the scan log (scans9, 2026-09-08 04:53:43).
    """
    plain = _prod_with_aliases("Bombay Sapphire London Dry Gin", "off:plain", "pr:bs",
                               ["Bombay Sapphire Vapour Infused London Dry Gin"])
    east = _prod_with_aliases("East Vapour Infused London Dry Gin", "off:east", "pr:bs", [])
    gold = {"pr:bs": _producer("pr:bs", "Bombay Sapphire")}
    lines = ["BOMBAY", "SAPPHIRE", "INFUSED INFUSE"]
    for order in ([plain, east], [east, plain]):
        frame = {t: [(rec, 0.6) for rec in order] for t in lines}
        resp = Resolver(_FrameStore(frame, gold)).resolve(
            ScanResolveRequest(detections=[DetectedText(text=t, kind="text") for t in lines]))
        assert resp.candidates, "the frame names a gin the catalog holds"
        assert resp.candidates[0].resolved.product.id == "off:plain", (
            f"picked {resp.candidates[0].resolved.product.name!r} for a frame that never read EAST")


def test_a_one_word_reading_cannot_account_for_an_object():
    """The leftover-word rule asks whether the row explains everything read, and a row explains
    one word for free.

    On a can of Heady Topper (2026-09-10) "DRINK FROM THE CAN!" misread as DRINK FRONT. Once the
    chrome is stripped that reading is the single word FRONT, `Front Flips` accounted for it in
    full, and the object resolved to a beer from Maine at 0.545. A single word is not a label on
    the object path any more than on the line path: it may match, it may not certify.
    """
    frame = {"THE CAN! DRINK FRONT": [(_prod_of("Front Flips", "p:ff", "pr:ml"), 0.545)]}
    gold = {"pr:ml": _producer("pr:ml", "Mast Landing Brewing Company")}
    verdict = Resolver(_FrameStore(frame, gold)).resolve_object(
        DetectedObject(id="o1", texts=["THE CAN! DRINK FRONT"]))
    assert verdict.status != "resolved", f"one word FRONT certified {verdict.candidates[0].resolved.product.name!r}"


def test_two_reads_of_one_phrase_agree_once():
    """Two lines that name a candidate through the same words are one printed phrase read twice.

    "ITHE CAN! DRINKER" and "SITHE CAN! DRINKER" are the same fine print with THE garbled two
    ways. The re-read check scored them 0.5 on that difference and kept both, and they then
    certified `Day Drinker` by agreeing with each other about DRINKER (2026-09-10). Agreement is
    counted by the candidate's words a line carries, so differently garbled reads of one word
    land on the same key -- while a line reading a *new* word of the name is still new evidence.
    """
    assert _frame_support(["day", "drinker"],
                          [["ithe", "can", "drinker"], ["sithe", "can", "drinker"]]) == 1
    # ...and the control: two lines carrying different words of the name are two.
    assert _frame_support(["alchemist", "heady", "topper"],
                          [["heady", "topper"], ["the", "alchemist"]]) == 2

    frame = {t: [(_prod_of("Day Drinker", "p:dd", "pr:fs"), 0.727)]
             for t in ("ITHE CAN! DRINKER", "SITHE CAN! DRINKER")}
    gold = {"pr:fs": _producer("pr:fs", "Feisty Spirits")}
    verdict = Resolver(_FrameStore(frame, gold)).resolve_object(
        DetectedObject(id="o1", texts=list(frame)))
    assert verdict.status != "resolved", "its own echo certified `Day Drinker`"
