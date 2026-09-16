"""Resolver — matching + cold-start scoring against a seeded store."""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.resolver import (
    Resolver,
    _accounts_for_sighting,
    _affix_read,
    _frame_support,
    _identifying_tokens,
    _identity_key,
    _is_business_name,
    _latin,
    _reads_the_name,
    _same_read,
    _token_supported,
    _tokens,
    _unread,
    _upc_variants,
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
    ("CAMPAR\nDavide Campani\nMILANO", "Campari"),           # the wordmark's last letter, lost
    ("LCHEMIST VERMONT", "The Alchemist"),                   # ...or its first
])
def test_token_support_keeps_real_hits(query, name):
    assert _token_supported(query, name) is True


@pytest.mark.parametrize("query, name", [
    ("BACAR OR", "Bacardi"),                          # OCR of "...drive a car or..." warning
    ("DRIVE A CAR OR OPERATE MACHINERY", "Malibu"),   # clean warning line
    ("ACCORDING TO THE SURGEON GENERAL", "Gentiane"),
    ("HARI CAMPA", "Campari"),                        # two letters gone is a fragment
    ("CAMP", "Campari"),
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


def test_an_objects_verdict_does_not_carry_the_frames_guess_with_it():
    """A response with a resolved object is corroborated, and the client draws a corroborated
    response whole. The frame's single unproven guess -- kept only for a model that will now
    not be asked -- went up beside the verdict: `Vermont Pale Lager` off the word VERMONT,
    next to the Heady Topper the tracked object had settled on (2026-09-11)."""
    alch = _producer("pr:alch", "The Alchemist")
    hf = _producer("pr:hf", "Hill Farmstead Brewery")
    lager = _prod_of("Vermont Pale Lager", "p:vpl", "pr:hf")
    store = _MakerStore(
        by_text={"HEMIST-VERMONT": [(lager, 0.6)]},          # the live lines' one guess
        gold={"pr:alch": alch, "pr:hf": hf},
        producers={"chemist": [(alch, 0.5)], "CHEMIST-VERMONT": [(alch, 0.5)]},
        catalog={"pr:alch": [_beer(n, pid, "pr:alch") for n, pid in _ALCHEMIST]},
    )
    resp = Resolver(store).resolve(ScanResolveRequest(
        detections=[DetectedText(text=t, kind="text") for t in ("CYTOPPER", "HEMIST-VERMONT")],
        objects=[DetectedObject(id="o1", texts=["CHEMIST-VERMONT", "MY-TOPPER", "ALE\nALC. 8% BY VOL"])]))
    assert resp.corroborated
    assert [c.resolved.product.name for c in resp.candidates] == ["The Alchemist Heady Topper"], (
        f"the frame's guess rode along: {[c.resolved.product.name for c in resp.candidates]}")


def test_objects_and_lines_share_one_response(shelf):
    r = Resolver(shelf)
    resp = r.resolve(ScanResolveRequest(
        detections=[DetectedText(text="HEADY TOPPER"), DetectedText(text="THE ALCHEMIST")],
        objects=[DetectedObject(id="o2", texts=["FOCAL BANGER THE ALCHEMIST INDIA PALE ALE"])]))
    by_kind = {(c.detection_index, c.object_id): c.resolved.product.name for c in resp.candidates}
    assert by_kind[(0, None)] == "Heady Topper" or by_kind.get((1, None)) == "Heady Topper"
    assert by_kind[(-1, "o2")] == "Focal Banger"
    assert resp.objects[0].status == "resolved"


def _put_beer(store, pid, name, producer_id, producer_name):
    store.put_gold(producer_id, "producer",
                   Producer(id=producer_id, name=producer_name).model_dump(mode="json"))
    bid = f"brand:{pid}"
    store.put_gold(bid, "brand", Brand(id=bid, producer_id=producer_id, name=name
                                       ).model_dump(mode="json"))
    store.put_gold(pid, "product", Product(id=pid, name=name, producer_id=producer_id,
                                           brand_id=bid, category=Category.BEER
                                           ).model_dump(mode="json"))


@pytest.mark.parametrize("a, b, same", [
    ("fong", "long", True),          # one letter substituted
    ("files", "fires", True),
    ("topplng", "toppling", True),   # one letter dropped
    ("five", "files", False),        # two edits is another word
    ("dud", "dude", False),          # too short to reconcile
    ("heady", "hazy", False),
])
def test_same_read_is_one_letter_of_garble(a, b, same):
    assert _same_read(a, b) is same


def test_two_garbles_of_one_wordmark_are_one_line(shelf):
    """A tracked can accumulates every read of its wordmark, and a script wordmark reads
    differently every tick. Long Live's arrived as "Fong files" and "Long fiRes" -- each a
    letter off the other -- and, as two independent lines, they agreed on `Long-fong`, a
    spirit whose two words each happened to be one of the garbles (2026-09-15)."""
    _put_beer(shelf, "ttb:fong", "Long-fong", "prod:mkl", "Mei Kuei Lu Chiew")
    _, res = _verdict(shelf, ["DUDE", "Fong files", "Long fiRes", "WIDESCREEN"])
    assert res.status == "unresolved", _names(res)
    assert "Long-fong" not in _names(res)


def test_a_name_longer_than_the_reading_does_not_account_for_it(shelf):
    """The leftover-word rule asks whether the row explains everything read. A row that
    says more than the label passes it for free: "Long five" was accounted for in full by
    `Long Distance High Five`, DISTANCE and HIGH read nowhere (2026-09-15). Nor may such a
    row be shortlisted alone -- a model asked to pick among one picks it."""
    _put_beer(shelf, "ttb:ldhf", "Long Distance High Five", "prod:bench", "Benchtop Brewing")
    _, res = _verdict(shelf, ["Long five"])
    assert res.status == "unresolved", (res.status, _names(res))


def test_one_unread_word_still_earns_the_shortlist(shelf):
    # "FADY TOPPE" reads TOPPER and loses HEADY: the garble the shortlist exists for.
    _, res = _verdict(shelf, ["FADY TOPPE", "THE ALCHEMIST"])
    assert res.status in ("ambiguous", "resolved") and "Heady Topper" in _names(res)


# The Campari frame as the camera gave it, and the store's numbers for it: `Campari` is a
# 0.67 against CAMPAR (the wordmark's last letter is never read), which is the one hit the
# per-line match has for the line.
CAMPARI_LINES = ["CAMPAR\nDavide Campani\nMILANO", "Aperiti\nRasalo", "RAMAZIO"]


def _campari_store(house_id, house_name, extra=()):
    campari = Product(id="off:campari", brand_id="b:campari", producer_id=house_id,
                      category=Category.SPIRIT, name="Campari").model_dump(mode="json")
    gold = {house_id: _producer(house_id, house_name),
            "b:campari": Brand(id="b:campari", producer_id=house_id, name="Campari"
                               ).model_dump(mode="json")}
    by_text = {CAMPARI_LINES[0]: [(campari, 0.67)]}
    for rec, score, line in extra:
        gold[rec["id"]] = rec
        by_text.setdefault(line, []).append((rec, score))
    return _FrameStore(by_text=by_text, gold=gold)


def test_a_one_word_label_is_proven_by_its_houses_line():
    """CAMPARI is the whole of the name on the bottle, and under it the house: DAVIDE
    CAMPARI MILANO. The camera read the wordmark as CAMPAR on thirty frames of thirty, the
    house as "Davide Campani MILANO", and the bottle drew nothing (2026-09-15): one word is
    not a label, and a word read twice is one piece of evidence. The house's phrase is the
    second word the name does not have."""
    store = _campari_store("pr:dcm", "Davide Campari-Milano")
    resp = Resolver(store).resolve(ScanResolveRequest(
        detections=[DetectedText(text=t) for t in CAMPARI_LINES]))
    assert resp.corroborated
    assert [c.resolved.product.name for c in resp.candidates] == ["Campari"]
    assert resp.candidates[0].match_score == 0.67, "corroborated, so not marked down"
    _, res = _verdict(store, CAMPARI_LINES)
    assert res.status == "resolved" and _names(res) == ["Campari"]


def test_the_house_has_to_be_the_labels_own():
    # The same bottle, the row filed under the importer's other brand -- what TTB actually
    # holds. "Cutty Sark" is on no line of a Campari bottle, and one word stays one word.
    store = _campari_store("pr:cutty", "Cutty Sark")
    resp = Resolver(store).resolve(ScanResolveRequest(
        detections=[DetectedText(text=t) for t in CAMPARI_LINES]))
    assert not resp.corroborated
    _, res = _verdict(store, CAMPARI_LINES)
    assert res.status != "resolved"


def test_a_house_of_one_word_proves_nothing(shelf):
    # `Lagunitas` under "Lagunitas Brewing Company": the suffix stripped, the house is the
    # same one word as the label, and reading it is still reading one word.
    _put_beer(shelf, "ttb:lag", "Lagunitas", "prod:lag", "Lagunitas Brewing Company")
    _, res = _verdict(shelf, ["LAGUNITAS", "LAGUNITAS BREWING COMPANY", "PETALUMA CALIFORNIA"])
    assert res.status != "resolved"


def test_the_house_proven_brand_row_yields_to_the_proven_bottle():
    # BOMBAY SAPPHIRE read whole proves the brand row `Bombay` by its house's line, and the
    # gin by its own words; one bottle, one name, and the gin is the bottle.
    house = _producer("pr:bs", "Bombay Sapphire")
    bombay = Product(id="ttb:bombay", brand_id="b", producer_id="pr:bs",
                     category=Category.SPIRIT, name="Bombay").model_dump(mode="json")
    gin = Product(id="off:gin", brand_id="b", producer_id="pr:bs", category=Category.SPIRIT,
                  name="Bombay Sapphire London Dry Gin").model_dump(mode="json")
    store = _FrameStore(
        by_text={"BOMBAY SAPPHIRE": [(gin, 0.6), (bombay, 1.0)],
                 "LONDON DRY GIN": [(gin, 0.55)],
                 "BOMBAY SAPPHIRE LONDON DRY GIN": [(gin, 1.0), (bombay, 0.3)]},
        gold={"pr:bs": house})
    resp = Resolver(store).resolve(ScanResolveRequest(detections=[
        DetectedText(text="BOMBAY SAPPHIRE"), DetectedText(text="LONDON DRY GIN"),
        DetectedText(text="BOMBAY SAPPHIRE LONDON DRY GIN")]))
    names = [c.resolved.product.name for c in resp.candidates]
    assert "Bombay Sapphire London Dry Gin" in names and "Bombay" not in names, names


def test_a_verdict_owns_only_the_lines_that_are_its_labels():
    """The tracker follows a screen region: across a pan one object gathered CAMPARI, then
    BLACK SEAL BERMUDA BLACK RUM, settled on Campari, and owned the rum's line -- so the
    frame's own proof of the Gosling's was dropped (2026-09-16)."""
    rum = Product(id="ttb:seal", brand_id="b:seal", producer_id="pr:gos", category=Category.SPIRIT,
                  name="Goslings Black Seal").model_dump(mode="json")
    store = _campari_store("pr:dcm", "Davide Campari-Milano", extra=[
        (rum, 0.9, "BLACK SEAL BERMUDA BLACK RUM"), (rum, 0.7, "Goslings")])
    store._gold["pr:gos"] = _producer("pr:gos", "Goslings")
    store._gold["b:seal"] = Brand(id="b:seal", producer_id="pr:gos", name="Goslings Black Seal"
                                  ).model_dump(mode="json")
    resp = Resolver(store).resolve(ScanResolveRequest(
        detections=[DetectedText(text="BLACK SEAL BERMUDA BLACK RUM"), DetectedText(text="Goslings")],
        objects=[DetectedObject(id="drift", texts=[CAMPARI_LINES[0], "BLACK SEAL BERMUDA BLACK RUM"])]))
    assert resp.objects[0].status == "resolved" and _names(resp.objects[0]) == ["Campari"]
    names = [c.resolved.product.name for c in resp.candidates]
    assert "Goslings Black Seal" in names, names


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


# ---- the maker's beers, told apart by the shape of the wordmark ----

_ALCHEMIST = [("The Alchemist Heady Topper", "p:ht"), ("Focal Banger", "p:fb"),
              ("Beelzebub", "p:bz"), ("Petit Mutant", "p:pm"), ("Holy Cow", "p:hc"),
              ("Alena", "p:al"), ("Luscious", "p:lu"), ("Rapture", "p:ra"), ("Skadoosh", "p:sk"),
              ("Just Say Gay", "p:jsg"), ("Beautiful Neon Light", "p:bnl"), ("Broken Spoke", "p:bs")]


def _alchemist_store(*lines):
    """The maker read cleanly, twelve of its beers on file, and nothing the product path can
    match -- the wordmark is garble."""
    maker = _producer("pr:alch", "The Alchemist")
    return _MakerStore(
        by_text={},
        gold={"pr:alch": maker},
        producers={line: [(maker, 1.0)] for line in lines},
        catalog={"pr:alch": [_beer(n, pid, "pr:alch") for n, pid in _ALCHEMIST]},
    )


def _frame(*texts):
    return ScanResolveRequest(detections=[DetectedText(text=t, kind="text") for t in texts])


@pytest.mark.parametrize("wordmark,expect", [
    ("FADY TOPPE", "The Alchemist Heady Topper"),     # scans5 2026-09-04, 23 frames like it
    ("ADY TOPP", "The Alchemist Heady Topper"),
    ("ROY TOPP / FADY TOPP", "The Alchemist Heady Topper"),
    ("FOCAL BAN", "Focal Banger"),                    # the other can, same maker
])
def test_the_wordmark_shape_picks_the_makers_beer(wordmark, expect):
    """A stylized can reads its maker in plain type and its own name as garble. Against the
    catalog the garble is noise; against the maker's dozen beers it is not. Measured over 245
    logged frames: 100 picks, 82 Heady and 18 Focal Banger, none wrong."""
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", wordmark, "ALE\nALC. 8% BY VOL\n1 PINT"))
    assert resp.corroborated, f"{wordmark!r} under a read maker should resolve"
    assert resp.candidates[0].resolved.product.name == expect
    # Indirect evidence still: scored below a label that named the beer outright.
    assert resp.candidates[0].match_score < 1.0


def test_the_fine_print_does_not_pick_a_beer():
    """Every can prints ALE / ALC. 8% BY VOL, and "ale" scores 0.43 against a beer called
    `Alena`. Chrome is excluded from the scoring on both sides; the maker alone names nothing."""
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", "ALE\nALC. 8% BY VOL\n1 PINT"))
    assert not resp.corroborated
    assert all(c.resolved.product.name != "Alena" for c in resp.candidates)


def test_the_makers_own_garbled_name_does_not_pick_a_beer():
    """"alcher" and "ALCHEMIS" are garbles of the maker, and they resemble `Alena` too. The
    maker's tokens -- and anything that reads as a garble of them -- are not the beer's."""
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", "w alchemistbeer.c", "Wo alcher"))
    assert not resp.corroborated
    assert all(c.resolved.product.name != "Alena" for c in resp.candidates)


def test_one_token_against_a_two_word_name_is_not_a_read():
    """"FOCAILS" (scans5 03:12:59) resembles "focal banger" at 0.24 -- one token against a
    phrase. "ecan", off DRINK FROM THE CAN, resembled `pecan cream` at 0.21 the same way and
    was picked. Two words agreeing on two words, or a near read: one token is the 0.40 floor
    whichever side it is on."""
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", "FOCAILS"))
    assert not resp.corroborated


@pytest.mark.parametrize("wordmark", ["DY TOPP", "DY TOPT", "BADY TO"])
def test_a_short_fragment_counts_toward_the_shape(wordmark):
    """The wordmark arrived as "DY TOPP" five times in one session (2026-09-11): HEADY TOPPER
    with the first letters of each word lost. Dropping the two-letter DY as carrying nothing
    left TOPP alone against a two-word name, which is rightly no read -- so a can with its maker
    in plain type and its name in that shape drew nothing for a minute. DY is not a name, but
    it is half the shape of the line it sits in, and "dy topp" resembles `heady topper` at 0.31
    and nothing else the maker brews."""
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", wordmark, "ALE\nALC. 8% BY VOL\n1 PINT"))
    assert resp.corroborated, f"{wordmark!r} under a read maker should resolve"
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"


@pytest.mark.parametrize("fragment", ["DY", "AL", "ALE\nAL"])
def test_a_short_fragment_is_not_a_window_by_itself(fragment):
    """"AL" off "ALC." scored 0.29 against `Alena` once; a fragment may make a shape with a word
    beside it, never alone."""
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", fragment))
    assert not resp.corroborated


@pytest.mark.parametrize("wordmark", ["ЯДУ ТОРР", "АДУ ТОРО", "ГАДУ ТОРРА", "ПОУ ТОРРЕ"])
def test_a_cyrillic_read_of_a_latin_wordmark_is_read_in_latin(wordmark):
    """The recognizer picks a script per line by what the letterforms resemble, and asking it
    for en-US does not stop it: 11.5% of one session's frames came back non-Latin with the pin,
    9-16% without. On a can of Heady Topper the wordmark arrived as "ЯДУ ТОРР" -- the right
    shapes in the wrong alphabet. Each Cyrillic letter is drawn like the Latin one it was read
    for, so mapped back it is "RDY TOPP", a read the maker pick uses like any other."""
    assert _tokens(_latin("ЯДУ ТОРР")) == ["rdy", "topp"]
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", wordmark, "ALE\nALC. 8% BY VOL\n1 PINT"))
    assert resp.corroborated, f"{wordmark!r} under a read maker should resolve"
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"


def test_a_cyrillic_read_reaches_the_object_path_in_latin():
    """The same alphabet at the other door: an object's texts are the camera's too."""
    store = _alchemist_store("THE ALCHEMIST")
    verdict = Resolver(store).resolve_object(
        DetectedObject(id="o1", texts=["THE ALCHEMIST", "ЯДУ ТОРР", "ALE\nALC. 8% BY VOL\n1 PINT"]))
    assert verdict.status == "resolved"
    assert verdict.candidates[0].resolved.product.name == "The Alchemist Heady Topper"


@pytest.mark.parametrize("reading,row", [
    ("MIST", "Sno Mist"),                       # 04:15:34, shortlisted alone at 1.00
    ("THE CAN! DRINK FRONT", "Front Flips"),    # 04:15:27, shortlisted alone at 0.41
])
def test_one_word_does_not_make_a_shortlist(reading, row):
    """A shortlist is a choice. "MIST" put `Sno Mist` on one by itself, "DRINK FRONT" put
    `Front Flips` on one by itself, and the model asked to pick among one picked it -- so the
    beer `_accounts_for_object` had just learned not to certify off one word was drawn anyway,
    by the other door (2026-09-11). One word may match; it may not shortlist."""
    frame = {reading: [(_prod_of(row, "p:x", "pr:x"), 1.0)]}
    gold = {"pr:x": _producer("pr:x", "Some Brewing")}
    verdict = Resolver(_FrameStore(frame, gold)).resolve_object(
        DetectedObject(id="o1", texts=[reading]))
    assert verdict.status == "unresolved", f"{reading!r} gave the model {verdict.candidates[0].resolved.product.name!r} to rubber-stamp"


def test_a_proven_pick_keeps_its_line_from_a_coincidence():
    """"DY TOPP" is a 0.62 against `Snipes Mountain Lefty Topp's`, with the word TOPP to back
    it -- and the same line, by its shape, is what picked Heady Topper from the maker's beers.
    One candidate represents each line, and ranked on resemblance the coincidence took the
    line and the proven beer was dropped as a second reading of it: the maker pick fired and
    the screen stayed blank (2026-09-11, five frames). Proof outranks resemblance."""
    maker = _producer("pr:alch", "The Alchemist")
    lefty = _prod_of("Snipes Mountain Lefty Topp's", "p:lefty", "pr:snipes")
    store = _MakerStore(
        by_text={"DY TOPP": [(lefty, 0.62)]},
        gold={"pr:alch": maker, "pr:snipes": _producer("pr:snipes", "Snipes Mountain")},
        # The maker line as the can actually reads: a hypothesis, not a read, so no line of
        # the frame supports the beer by its letters -- the shape is all it has.
        producers={"ACHEMIST-VERM": [(maker, 0.44)]},
        catalog={"pr:alch": [_beer(n, pid, "pr:alch") for n, pid in _ALCHEMIST]},
    )
    resp = Resolver(store).resolve(_frame("DY TOPP", "CAN! DRINK FROMTHO", "ACHEMIST-VERM"))
    assert resp.corroborated, f"drew {[c.resolved.product.name for c in resp.candidates]}"
    assert [c.resolved.product.name for c in resp.candidates] == ["The Alchemist Heady Topper"]


def test_the_town_on_the_maker_line_does_not_hide_the_maker():
    """"CHEMIST-VERMONT" is the maker's name with its town after it. Matched as a line it
    resembles seven Vermont producers better than it resembles `The Alchemist`, which never
    made the hypotheses (2026-09-11, a whole scan blank). Each word of the line nominates
    makers on its own: "chemist" reaches the brewery, "vermont" the town's, and the wordmark
    contest sorts them out."""
    alch = _producer("pr:alch", "The Alchemist")
    vt = _producer("pr:vt", "Vermont Beer Makers")
    store = _MakerStore(
        by_text={}, gold={"pr:alch": alch, "pr:vt": vt},
        # The line's whole-text matches, then each word's, as the index answers them.
        producers={"CHEMIST-VERMONT": [(vt, 0.67)], "chemist": [(alch, 0.5)], "vermont": [(vt, 1.0)]},
        catalog={"pr:alch": [_beer(n, pid, "pr:alch") for n, pid in _ALCHEMIST],
                 "pr:vt": [_beer("Vermont Pale Lager", "p:vpl", "pr:vt"),
                           _beer("Green Mountain Amber", "p:gma", "pr:vt")]},
    )
    resp = Resolver(store).resolve(_frame("ADY TOPPE", "CHEMIST-VERMONT", "ALE\nALC. 8% BY VOL\n1 PINT"))
    assert resp.corroborated, f"drew {[c.resolved.product.name for c in resp.candidates]}"
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"


def test_a_maker_named_after_a_beer_cannot_veto_the_beer():
    """The catalog holds producers named after beers -- a permit filed as `Topper's`. The
    wordmark line DY TOPPER therefore "reads" a maker, and excluding a read maker's words
    from every window let that hypothesis, which found no beer, veto The Alchemist's, which
    found one (2026-09-11). A maker is read as a phrase, two or more of its words in the
    frame; on one word it is a hypothesis, and a hypothesis excludes nothing."""
    alch = _producer("pr:alch", "The Alchemist")
    tops = _producer("pr:tops", "Topper's")
    store = _MakerStore(
        by_text={}, gold={"pr:alch": alch, "pr:tops": tops},
        producers={"CHEMIST-VERMONT": [(alch, 0.5)], "DY TOPPER": [(tops, 0.78)],
                   "topper": [(tops, 1.0)], "chemist": [(alch, 0.5)]},
        catalog={"pr:alch": [_beer(n, pid, "pr:alch") for n, pid in _ALCHEMIST],
                 "pr:tops": [_beer("Topper's Lager", "p:tl", "pr:tops"),
                             _beer("Topper's Stout", "p:ts", "pr:tops")]},
    )
    resp = Resolver(store).resolve(_frame("DY TOPPER", "CHEMIST-VERMONT", "ALE\nALC. 8% BY VOL\n1 PINT"))
    assert resp.corroborated, f"drew {[c.resolved.product.name for c in resp.candidates]}"
    assert [c.resolved.product.name for c in resp.candidates] == ["The Alchemist Heady Topper"]
    # ...and the control the rule was written for: the maker read on the label's own line
    # is still no evidence for a sibling filed under a stray producer.
    bs = _producer("pr:bs", "Bombay Sapphire")
    stray = _producer("pr:stray", "Bombay spirits")
    store = _MakerStore(
        by_text={}, gold={"pr:bs": bs, "pr:stray": stray},
        producers={"BOMBAY SAPPHIRE": [(bs, 1.0), (stray, 0.6)], "bombay": [(bs, 0.7), (stray, 0.7)],
                   "sapphire": [(bs, 0.7)]},
        catalog={"pr:bs": [_prod_of("Bombay Sapphire London Dry Gin", "off:plain", "pr:bs"),
                           _prod_of("Bombay Bramble", "off:bramble", "pr:bs")],
                 "pr:stray": [_prod_of("Bombay Sapphire Murcian Lemon", "off:lemon", "pr:stray")]},
    )
    resp = Resolver(store).resolve(_frame("BOMBAY SAPPHIRE", "SAPPHIRE SANTED"))
    assert all(c.resolved.product.id != "off:lemon" for c in resp.candidates), "the maker's own word named the stray row"


@pytest.mark.parametrize("wordmark", ["OYTOPPER", "CYTOPPER", "ATOPPER", "TOPPER"])
def test_a_wordmark_read_as_one_word_is_compared_as_one_word(wordmark):
    """Stacked type reads as one word when the leading is tight: HEADY over TOPPER came in as
    "OYTOPPER", "CYTOPPER", "ATOPPER" on nine frames of one scan (2026-09-11), and one token
    against a two-word name is refused. A token that carries one of the name's words whole is
    compared to the name written as one word -- the whole word is the corroboration a second
    token would have been."""
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", wordmark, "ALE\nALC. 8% BY VOL\n1 PINT"))
    assert resp.corroborated, f"{wordmark!r} under a read maker should resolve"
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"


@pytest.mark.parametrize("wordmark", ["LYTOPER", "YTOPPFR", "FOCAILS"])
def test_a_merged_read_needs_a_whole_word_inside_it(wordmark):
    """"LYTOPER" and "YTOPPFR" are the same wordmark with a letter wrong, and "FOCAILS" is the
    garble that started the two-word rule: none carries a word of the name whole, and none is
    a read."""
    store = _alchemist_store("THE ALCHEMIST")
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", wordmark))
    assert not resp.corroborated


def test_a_merged_read_of_a_word_two_siblings_share_picks_neither():
    """"WILDCHILDX" carries WILD and CHILD whole -- and `Wild Child Peche` and `Wild Child
    Positive` both have them. The margin rule holds for merged reads as for any other."""
    maker = _producer("pr:alch", "The Alchemist")
    store = _MakerStore(
        by_text={}, gold={"pr:alch": maker},
        producers={"THE ALCHEMIST": [(maker, 1.0)]},
        catalog={"pr:alch": [_beer("Wild Child Peche", "p:wcp", "pr:alch"),
                             _beer("Wild Child Positive", "p:wcpo", "pr:alch"),
                             _beer("Focal Banger", "p:fb", "pr:alch")]},
    )
    resp = Resolver(store).resolve(_frame("THE ALCHEMIST", "WILDCHILDX", "ALE\nALC. 8% BY VOL\n1 PINT"))
    assert not resp.corroborated, f"drew {[c.resolved.product.name for c in resp.candidates]}"


# ---- the spirits shelf, 2026-09-14: five wrong draws, five doors ----


def test_a_producers_trade_suffix_identifies_nothing():
    """Nearly every can prints BREWING COMPANY. A Miller High Life can agreed with `Pariah
    Brewing Company` on COMPANY, and with COLORS off "NO COLORS OR FLAVORS FROM ARTIFICIAL
    SOURCES" that was two lines naming a beer called `Colors`."""
    assert _identifying_tokens("Pariah Brewing Company") == ["pariah"]
    # COLORS itself has since become chrome (the ingredient line, see `_STYLE`), so the row
    # here is one named for the slogan's last word, which is not.
    assert _identifying_tokens("Colors") == []
    assert _identifying_tokens("Sources") == ["sources"]
    frame = {"• COLORS OR FLAVORS FROM\nARTIFICIAL\nSOURCES": [(_prod_of("Sources", "p:sources", "pr:pariah"), 1.0)]}
    gold = {"pr:pariah": _producer("pr:pariah", "Pariah Brewing Company")}
    resp = Resolver(_FrameStore(frame, gold)).resolve(_frame(
        "BREWING COMPANY MIL\nPREMIUM\nMiller.\nBREWED\nHIGH LIFE\nEST 1903",
        "• COLORS OR FLAVORS FROM\nARTIFICIAL\nSOURCES"))
    assert not resp.corroborated, "COMPANY corroborated `Sources`"


def test_a_phrase_read_with_a_word_lost_is_the_same_phrase():
    """A Campari label read "MILANO BITTER" once and "MILANO TER" once, and {milano, bitter}
    beside {milano} made two lines naming `Gran Milano Bitter` -- another maker's amaro."""
    assert _frame_support(["gran", "milano", "bitter"],
                          [["since", "campar", "milano", "bitter"],
                           ["since", "campar", "milano", "ter"]]) == 1
    # ...while two lines each carrying a word the other lacks are still two.
    assert _frame_support(["gran", "milano", "bitter"],
                          [["milano", "amaro"], ["gran", "bitter"]]) == 2


def test_a_line_that_is_a_piece_of_another_line_is_that_line_read_short():
    """Among a tracked object's reads of BLACK SEAL 80 PROOF BERMUDA BLACK RUM was "BLACK
    SEA" -- the first line with its last letter lost, two words, a 1.00 against a spirit
    called `Black Sea`. The whole label is the fuller read."""
    sea = _prod_of("Black Sea", "p:sea", "pr:sea")
    frame = {"BLACK SEA": [(sea, 1.0)],
             "BLACK SEAL\n80 PROOF\nBERMUDA BLACK RUM": [(sea, 0.4)]}
    gold = {"pr:sea": _producer("pr:sea", "Black Sea")}
    verdict = Resolver(_FrameStore(frame, gold)).resolve_object(
        DetectedObject(id="o1", texts=["BLACK SEAL\n80 PROOF\nBERMUDA BLACK RUM", "BLACK SEA"]))
    assert verdict.status != "resolved", "a truncated read certified `Black Sea`"


def test_a_maker_hypothesis_rests_on_a_word_of_the_makers_name_with_an_end_lost():
    """"LOURE", a garble of FLAVORS off a can's fine print, resembled `Money Lure` at 0.38 and
    nominated that brewery; the next read of the same fine print then named `Colorado
    Fisherman` by shape. The resemblance a hypothesis may rest on is the recognizer's own
    failure -- letters lost at one end -- and nothing else."""
    assert _affix_read("CHEMIST-VER", "The Alchemist")          # end kept, start lost
    assert _affix_read("ACHEMIST-VERM", "The Alchemist")
    assert _affix_read("CAMPAR", "Campari")                     # start kept, end lost
    assert _affix_read("VERMONT BEER MAKERS", "Vermont Beer Makers")
    assert not _affix_read("NO COLDEN LOURE FROM", "Money Lure")
    assert not _affix_read("MIST", "Alchemist")                 # too short to be an end
    lure = _producer("pr:lure", "Money Lure")
    store = _MakerStore(
        by_text={}, gold={"pr:lure": lure},
        producers={"NO COLDEN LOURE FROM": [(lure, 0.38)], "loure": [(lure, 0.38)]},
        catalog={"pr:lure": [_beer("Colorado Fisherman", "p:cf", "pr:lure"),
                             _beer("Ripped Lip", "p:rl", "pr:lure"),
                             _beer("Captain Quint", "p:cq", "pr:lure")]},
    )
    resp = Resolver(store).resolve(_frame("NO COLDEN LOURE FROM", "NO COLORS OR FLAVORS FROM",
                                          "ARTIFICIAL"))
    assert not resp.corroborated, f"drew {[c.resolved.product.name for c in resp.candidates]}"


@pytest.mark.parametrize("line", [
    "BREWING COMPANY MIL\nPREMIUM\nMiller.\nBREWED\nHIGH LIFE\nEST 1903\nThe Champagne of Bette\nDID DUNCES\n1.355 LITE",
    "Miller\nHIGH LIFE\n8570 1903",
])
def test_a_line_that_prints_the_whole_name_proves_it(line):
    """A label is one block of type to the recognizer, and against a line like Miller's no
    name is ever most of the text -- but every word of `Miller High Life` is in it, in order,
    with one flourish (BREWED) between. Until this door the can drew nothing, or drew what
    an echo of HIGH happened to corroborate."""
    assert _reads_the_name("Miller High Life", line)
    miller = _prod_of("Miller High Life", "p:mhl", "pr:miller")
    frame = {line: [(miller, 0.5)]}
    gold = {"pr:miller": _producer("pr:miller", "Miller Brewing Company")}
    resp = Resolver(_FrameStore(frame, gold)).resolve(_frame(line, "RIDGE FARM\n1937"))
    assert resp.corroborated
    assert resp.candidates[0].resolved.product.name == "Miller High Life"


@pytest.mark.parametrize("name,line", [
    ("Sierra Nevada 6 & Out", "SIERRA NEVADA PALE ALE"),          # the 6 and the OUT are unread
    ("Stella Artois 0.0%", "<Stella Artois> <Stella Artois>"),    # so is the 0.0
    ("Miller High Life Ice", "Miller.\nBREWED\nHIGH LIFE"),        # ICE is three letters and unread
    ("Must Have", "the hop we have worked so\neu MUST pour it into a glas"),        # out of order
    ("Keep Pouring", "Pouring it in a glass ... When it is young Keep it cold"),   # not together
    ("Colors", "• COLORS OR FLAVORS FROM"),                       # one word
])
def test_the_whole_name_means_every_word_in_order_and_together(name, line):
    assert not _reads_the_name(name, line)


def test_a_paragraph_names_nothing():
    """Two common words will always end up near each other in a back-label essay."""
    essay = " ".join(["word"] * 40) + " keep pouring " + " ".join(["more"] * 5)
    assert not _reads_the_name("Keep Pouring", essay)
    assert _reads_the_name("Keep Pouring", "KEEP POURING IPA 6.5% ALC BY VOL")


def test_the_row_that_explains_more_of_the_line_represents_it():
    """`High Life`, a permit filed under an importer, and `Miller High Life` both print whole
    on a Miller can and both are proven; the one that also explains MILLER is in view. And
    "Miller High Life High Life" -- brand plus label, filed as one -- explains the same words
    with two to spare, and the one with nothing to spare is the one the line printed."""
    line = "BREWING COMPANY MIL\nPREMIUM\nMiller.\nBREWED\nHIGH LIFE\nEST 1903"
    frame = {line: [(_prod_of("High Life", "p:hl", "pr:winters"), 1.0),
                    (_prod_of("Miller High Life High Life", "p:mhlhl", "pr:redds"), 1.0),
                    (_prod_of("Miller High Life", "p:mhl", "pr:miller"), 1.0)]}
    gold = {"pr:winters": _producer("pr:winters", "Winters"),
            "pr:redds": _producer("pr:redds", "Redd's"),
            "pr:miller": _producer("pr:miller", "Miller Brewing Company")}
    resp = Resolver(_FrameStore(frame, gold)).resolve(_frame(line))
    assert resp.corroborated
    assert [c.resolved.product.id for c in resp.candidates] == ["p:mhl"]


def test_a_beer_named_after_its_maker_cannot_lose_a_shape_contest():
    """"Bombay Sapphire London Dry Gin" is the maker plus a style: nothing of its own for a
    wordmark to resemble. A frame that read BOMBAY / SAPPHIRE and some garble is consistent
    with it, so the shape may not hand the frame to a sibling with a longer name -- which is
    exactly how `East Vapour Infused London Dry Gin` displaced it in four logged frames."""
    maker = _producer("pr:bs", "Bombay Sapphire")
    store = _MakerStore(
        by_text={}, gold={"pr:bs": maker},
        producers={"BOMBAY SAPPHIRE": [(maker, 1.0)]},
        catalog={"pr:bs": [_prod_of("Bombay Sapphire London Dry Gin", "off:plain", "pr:bs"),
                           _prod_of("East Vapour Infused London Dry Gin", "off:east", "pr:bs"),
                           _prod_of("Bombay Sapphire Murcian Lemon", "off:lemon", "pr:bs"),
                           _prod_of("Bombay Sapphire Gin & Light Tonic", "off:tonic", "pr:bs"),
                           _prod_of("Bombay Bramble", "off:bramble", "pr:bs")]},
    )
    resp = Resolver(store).resolve(_frame("BOMBAY SAPPHIRE", "INEUSED INFUSE"))
    assert all(c.resolved.product.id != "off:east" for c in resp.candidates if resp.corroborated)


def test_a_line_a_product_accounts_for_is_not_a_wordmark():
    """On a shelf, BLUE MOON BELGIAN WHITE is Blue Moon's line. The word WHITE in it scored
    `Guinness White Ale`'s own name at 1.00 and the maker path drew it beside the Guinness
    that was actually there. A line some product accounts for in full is that product's."""
    guinness = _producer("pr:g", "Guinness")
    bm = _prod_of("Blue Moon Belgian White", "p:bm", "pr:bm")
    store = _MakerStore(
        by_text={"BLUE MOON BELGIAN WHITE": [(bm, 1.0)]},
        gold={"pr:g": guinness, "pr:bm": _producer("pr:bm", "Blue Moon")},
        producers={"GUINNESS DRAUGHT STOUT": [(guinness, 1.0)]},
        catalog={"pr:g": [_beer("Guinness White Ale", "p:gw", "pr:g"), _beer("Guinness Draught", "p:gd", "pr:g"),
                          _beer("Guinness Extra Stout", "p:ge", "pr:g"), _beer("Guinness Foreign Extra", "p:gf", "pr:g"),
                          _beer("Guinness Over The Moon Milk Stout", "p:gm", "pr:g")]},
    )
    resp = Resolver(store).resolve(_frame("BLUE MOON BELGIAN WHITE", "GUINNESS DRAUGHT STOUT"))
    assert all(c.resolved.product.name != "Guinness White Ale" for c in resp.candidates)


def test_a_garbled_maker_is_read_by_the_wordmark_that_agrees_with_it():
    """The maker line is garbled too: THE ALCHEMIST reaches the resolver as "CHEMIST-VER" sixty
    times for every four clean reads, and the producer guards rightly refuse that. The pick
    starts from a maker merely resembled and keeps it only when the wordmark names one of
    its beers by a margin -- two weak reads that agree are one strong read."""
    maker = _producer("pr:alch", "The Alchemist")
    store = _MakerStore(
        by_text={}, gold={"pr:alch": maker},
        producers={"ELCHEMIST-VE": [(_producer("pr:chem", "Chemist"), 1.0), (maker, 0.44)]},
        catalog={"pr:alch": [_beer(n, pid, "pr:alch") for n, pid in _ALCHEMIST],
                 "pr:chem": [Product(id=f"p:c{i}", brand_id="b", producer_id="pr:chem", category=Category.SPIRIT,
                                     name=n).model_dump(mode="json")
                             for i, n in enumerate(("Chemist Gin", "Chemist Single Malt Pecan Cream Liqueur",
                                                    "Chemist Forager's Gin", "Chemist Eau De Vie", "Chemist 151"))]},
    )
    resp = Resolver(store).resolve(_frame("ELCHEMIST-VE", "FADY TOPPE", "ALE\nALC. 8% BY VOL"))
    assert resp.corroborated
    assert resp.candidates[0].resolved.product.name == "The Alchemist Heady Topper"
    # ...and the resemblance alone, with no wordmark agreeing, reads nothing.
    resp = Resolver(store).resolve(_frame("ELCHEMIST-VE", "ECAN! DRINK FROM THE!"))
    assert not resp.corroborated


def test_a_one_word_beer_needs_a_near_read():
    """A lone garbled word resembles many things. `Beelzebub` off "BELZBU" (0.5) is a read;
    off a passing resemblance it is not."""
    store = _alchemist_store("THE ALCHEMIST")
    near = Resolver(store).resolve(_frame("THE ALCHEMIST", "BEELZEBU"))
    assert near.corroborated and near.candidates[0].resolved.product.name == "Beelzebub"
    far = Resolver(store).resolve(_frame("THE ALCHEMIST", "BEEZE"))
    assert not far.corroborated


def test_two_beers_that_look_alike_pick_neither():
    """The margin: when the garble fits two of the maker's beers about equally, the honest
    answer is the maker, not a coin flip.

    Five beers, not three, so the maker is past `_PRODUCER_MAX_PRODUCTS` and the pick is the
    only route in. (With three, the older two-lines-agree proof certifies *both* look-alikes
    off TWIN BREWING + SUMMER and draws one by id -- a weakness of that rule, not of this one,
    and not what this test is about.)
    """
    maker = _producer("pr:tw", "Twin Brewing")
    store = _MakerStore(
        by_text={}, gold={"pr:tw": maker},
        producers={"TWIN BREWING": [(maker, 1.0)]},
        catalog={"pr:tw": [_beer("Summer Haze", "p:a", "pr:tw"), _beer("Summer Daze", "p:b", "pr:tw"),
                           _beer("Winter Warmer", "p:c", "pr:tw"), _beer("Autumn Amber", "p:d", "pr:tw"),
                           _beer("Spring Bock", "p:e", "pr:tw")]},
    )
    resp = Resolver(store).resolve(_frame("TWIN BREWING", "SUMMER"))
    assert not resp.corroborated
    # ...while a read that reaches the distinguishing word does decide it.
    resp = Resolver(store).resolve(_frame("TWIN BREWING", "SUMMER HAZ"))
    assert resp.corroborated and resp.candidates[0].resolved.product.name == "Summer Haze"


# ---- one bottle, one name, 2026-09-15 ----


def _prod_branded(name, pid, producer_id, brand_id):
    return Product(id=pid, brand_id=brand_id, producer_id=producer_id,
                   category=Category.BEER, name=name).model_dump(mode="json")


def _brand(bid, name, producer_id):
    return Brand(id=bid, producer_id=producer_id, name=name).model_dump(mode="json")


def test_a_settled_objects_verdict_speaks_for_its_own_lines():
    """The frame path is the shelf's -- every product that proves itself is drawn -- and run
    over one bottle's lines it draws the bottle's siblings. A tracked bottle of Bombay
    Sapphire settled on the gin while the same tick's lines, the object's own, proved `East
    Vapour Infused London Dry Gin` off INFUSED: three names on one bottle (2026-09-15). A line
    the object owns has been answered. A line it does not own -- the next bottle over -- has
    not."""
    label = "SAPPHIRE\nDistilled\nLONDON\nDRY GIN\nINFUSED"
    maker_line = "BOMBAy"
    # The catalog's plain gin carries the label's own words as an alias (a merge left them).
    plain = _prod_with_aliases("Bombay Sapphire London Dry Gin", "off:plain", "pr:bs",
                               ["Bombay Sapphire Vapour Infused London Dry Gin"])
    east = _prod_of("East Vapour Infused London Dry Gin", "off:east", "pr:bs")
    rum = _prod_of("Goslings Black Seal Bermuda Black Rum", "off:rum", "pr:gos")
    frame = {label: [(east, 0.75)],                          # the store's top three miss the plain gin
             maker_line: [(east, 0.4)],
             "BOMBAY SAPPHIRE LONDON DRY GIN": [(plain, 0.95)],   # a read the object accumulated
             "GOSLINGS BLACK SEAL BERMUDA BLACK RUM": [(rum, 0.95)]}
    gold = {"pr:bs": _producer("pr:bs", "Bombay Sapphire"), "pr:gos": _producer("pr:gos", "Goslings")}
    r = Resolver(_FrameStore(frame, gold))
    alone = r.resolve(_frame(label, maker_line))
    assert alone.corroborated and [c.resolved.product.id for c in alone.candidates] == ["off:east"], (
        "the premise: on the frame path INFUSED beside BOMBAY proves the sibling")
    resp = r.resolve(ScanResolveRequest(
        detections=[DetectedText(text=t, kind="text")
                    for t in (label, maker_line, "GOSLINGS BLACK SEAL BERMUDA BLACK RUM")],
        objects=[DetectedObject(id="o1", texts=[label, maker_line, "BOMBAY SAPPHIRE LONDON DRY GIN"])]))
    assert resp.objects[0].status == "resolved"
    assert resp.objects[0].candidates[0].resolved.product.id == "off:plain"
    drawn = [c.resolved.product.id for c in resp.candidates]
    assert "off:east" not in drawn, f"the sibling was drawn beside the verdict: {drawn}"
    assert "off:plain" in drawn and "off:rum" in drawn, f"the verdict or the next bottle went missing: {drawn}"


@pytest.mark.parametrize("name,line", [
    ("West Coast Wheat", "DINO BREAK\nWEST COAST STYLE DOUBLE INDIA PALE ALE WITH SIMCOE"),
    ("Toppling Goliath Brewing Co. Zz Hop", "TOPPLING GOLIATH BREWING CO."),
    ("Bombay Sapphire London Dry Gin", "Bombay Sapphire"),   # the brand alone is every sibling's line
])
def test_the_whole_name_includes_its_category_words(name, line):
    """What was left of `West Coast Wheat` without its category word was "west coast", which
    a double IPA's can printed as WEST COAST STYLE. The words a name shares with its category
    are part of the name, and a line that prints the name prints them."""
    assert not _reads_the_name(name, line)
    assert _reads_the_name("West Coast Wheat", "WEST COAST WHEAT ALE 5.2%")
    assert _reads_the_name("Sierra Nevada Pale Ale", "SIERRA NEVADA\nPALE ALE")


def test_lines_agreeing_only_on_the_makers_words_have_named_the_maker():
    """A can of Long Live Beerworks read LIVE on one line and LONG FIRES on another, and those
    two named `Long Live Beerwoks Hola Fantasma` -- the brewery's beer that the store's top
    three for "JONG LIVE" happened to hold, HOLA and FANTASMA read nowhere. Evidence every
    sibling shares equally is one piece of evidence, for the maker."""
    hola = _prod_of("Long Live Beerwoks Hola Fantasma", "p:hola", "pr:ll")
    gold = {"pr:ll": _producer("pr:ll", "Long Live Beerworks")}
    frame = {"JONG LIVE": [(hola, 0.58)], "HOLA FANTASMA": [(hola, 0.9)]}
    r = Resolver(_FrameStore(frame, gold))
    resp = r.resolve(_frame("LIVE", "Long fires", "JONG LIVE", "WIDESCREEN"))
    assert not resp.corroborated, f"drew {[c.resolved.product.name for c in resp.candidates]}"
    # ...and a second line that reads a word of the beer's own is the second piece.
    resp = r.resolve(_frame("JONG LIVE", "HOLA FANTASMA"))
    assert resp.corroborated and resp.candidates[0].resolved.product.id == "p:hola"


def test_a_business_name_is_not_a_product():
    """A permit filed under the brewery's name with no beer on it: `Toppling Goliath Brewing
    Co.` was drawn beside `Toppling Goliath Brewing Co. Dino Break` off the brewery line of a
    Dino Break can. A maker's name without a suffix may be a flagship's, and stays."""
    tg = _producer("pr:tg", "Toppling Goliath Brewing Co.")
    gold = {"pr:tg": tg, "b:tg": _brand("b:tg", "Toppling Goliath Brewing Co.", "pr:tg"),
            "pr:campari": _producer("pr:campari", "Campari"), "b:campari": _brand("b:campari", "Campari", "pr:campari")}
    house = _prod_branded("Toppling Goliath Brewing Co.", "p:tg", "pr:tg", "b:tg")
    dino = _prod_branded("Toppling Goliath Brewing Co. Dino Break", "p:dino", "pr:tg", "b:tg")
    campari = _prod_branded("Campari", "p:campari", "pr:campari", "b:campari")
    r = Resolver(_FrameStore({}, gold))
    assert _is_business_name(r._hydrate(house))
    assert not _is_business_name(r._hydrate(dino))
    assert not _is_business_name(r._hydrate(campari))
    # A row the catalog holds no brand for is not its own maker.
    assert not _is_business_name(r._hydrate(_prod_of("Pariah Brewing Company", "p:pariah", "pr:none")))
    brewery_line = "TOPPLING GOLIATH BREWING CO."
    beer_line = "DINO BREAK\nWEST COAST STYLE DOUBLE INDIA PALE ALE WITH SIMCOE, AMARILLO & CENTENNIAL HOPS"
    frame = {brewery_line: [(house, 1.0), (dino, 0.8)], beer_line: [(dino, 0.69)]}
    resp = Resolver(_FrameStore(frame, gold)).resolve(_frame("BEAGLEPUSS", beer_line, brewery_line))
    assert resp.corroborated
    assert [c.resolved.product.id for c in resp.candidates] == ["p:dino"]
    resp = Resolver(_FrameStore(frame, gold)).resolve(_frame("AGEPUSS", "SESAME", brewery_line))
    assert not resp.corroborated, "the brewery alone named a product"


def test_a_maker_hypothesis_rests_on_a_word_that_identifies_the_maker():
    """"Aperi" starts the way APERITIVO starts, and a producer registered as `Terrativo
    Aperitivo` was hypothesised off a bottle of Campari on the word for what is in it."""
    assert not _affix_read("Aperi\nRascal", "Terrativo Aperitivo")
    assert _affix_read("TERRAT", "Terrativo Aperitivo")
    assert _identifying_tokens("Terrativo Aperitivo") == ["terrativo"]


def test_the_ingredient_line_identifies_nothing():
    """"NO COLORS OR FLAVORS FROM ARTIFICIAL SOURCES" on a Miller can, read twice, agreed with
    a stout whose registered name is its whole label -- on those words and no other."""
    dump = ("Great Falls Brewing Co Peanut Butter Happy Camper S'Mores Stout Malt Beverage With "
            "Natural And Artificial Flavors And Artificial Color Artificially Colored With Titanium Oxide")
    ident = _identifying_tokens(dump)
    assert not {"flavors", "artificial", "color", "colored", "artificially", "natural"} & set(ident)
    assert "titanium" in ident
    assert _frame_support(ident, [["colors", "flavors", "from", "artificia", "our"],
                                  ["ho", "color", "ca", "hung", "hom"]]) == 0


def test_a_row_whose_brand_is_its_whole_label_needs_every_word_read():
    """164,000 rows are filed with no fanciful name, so their brand is their whole label and
    nothing sets the beer's words apart from the brewery's. LONG on one line and LIVE on
    another proved `Long Live Local Honey Brown Lager`, a Pennsylvania beer, off a Long Live
    Beerworks can -- and GOSLINGS beside BLACK SEAL proved `Goslings Gold Seal`. Two lines
    make such a row's case only when every word of it was read."""
    gold = {"pr:hbc": _producer("pr:hbc", "Hit By Car"),
            "b:honey": _brand("b:honey", "Long Live Local Honey Brown Lager", "pr:hbc"),
            "pr:gos": _producer("pr:gos", "Goslings"),
            "b:gold": _brand("b:gold", "Goslings Gold Seal", "pr:gos"),
            "b:black": _brand("b:black", "Goslings Black Seal", "pr:gos")}
    honey = _prod_branded("Long Live Local Honey Brown Lager", "p:honey", "pr:hbc", "b:honey")
    gold_seal = _prod_branded("Goslings Gold Seal", "p:gold", "pr:gos", "b:gold")
    black_seal = _prod_branded("Goslings Black Seal", "p:black", "pr:gos", "b:black")
    frame = {"JONG LIVE": [(honey, 0.58)], "Goslings\nSince 1806": [(gold_seal, 0.6), (black_seal, 0.6)]}
    r = Resolver(_FrameStore(frame, gold))
    resp = r.resolve(_frame("LIVE", "Long fires", "JONG LIVE", "WIDESCREEN"))
    assert not resp.corroborated, f"drew {[c.resolved.product.name for c in resp.candidates]}"
    resp = r.resolve(_frame("BLACK SEAL\n80 PROOF\nBERMUDA BLACK RUM", "Goslings\nSince 1806"))
    assert resp.corroborated
    assert [c.resolved.product.id for c in resp.candidates] == ["p:black"], (
        f"drew {[c.resolved.product.name for c in resp.candidates]}")
    # ...and the case it must not touch: the flagship whose every word the can prints.
    miller = _prod_branded("Miller High Life", "p:mhl", "pr:mhl", "b:mhl")
    gold.update({"pr:mhl": _producer("pr:mhl", "Miller High Life"),
                 "b:mhl": _brand("b:mhl", "Miller High Life", "pr:mhl")})
    resp = Resolver(_FrameStore({"Miller": [(miller, 0.5)], "HIGH LIFE": [(miller, 0.6)]}, gold)).resolve(
        _frame("Miller", "HIGH LIFE", "RIDGE FARM\n1937"))
    assert resp.corroborated and resp.candidates[0].resolved.product.id == "p:mhl"


class _FrameMatchStore(_FrameStore):
    """A store that also answers `match_frame`: what the lines name between them."""

    def __init__(self, by_text, gold=None, by_frame=None):
        super().__init__(by_text, gold)
        self._by_frame = by_frame or []       # [(product_rec, line index, that line's score)]

    def match_frame(self, lines, limit=8):
        return self._by_frame[:limit]


def test_a_candidate_the_frame_names_between_its_lines_is_held_to_every_word():
    """`Goslings Black Seal` reaches the frame through `match_frame` on the BLACK SEAL line
    at 0.55 and is proven by GOSLINGS on the other; a frame-level hit with a word the frame
    never read -- `Taft's Paint The Town Hoppy` off TOWN and HOPPY -- is not a candidate at
    all, whatever the two lines agree on."""
    gos = _producer("pr:gos", "Goslings")
    rum = _prod_of("Goslings Black Seal", "p:black", "pr:gos")
    taft = _prod_of("Taft's Paint The Town Hoppy Double IPA", "p:taft", "pr:taft")
    gold = {"pr:gos": gos, "pr:taft": _producer("pr:taft", "Taft's Brewing Company")}
    lines = ["BLACK SEAL\n80 PROOF\nBERMUDA BLACK RUM", "Goslings\nSince 1806"]
    store = _FrameMatchStore({}, gold, by_frame=[(rum, 0, 0.55)])
    resp = Resolver(store).resolve(_frame(*lines))
    assert resp.corroborated and [c.resolved.product.id for c in resp.candidates] == ["p:black"]
    lines = ["HOPPY\nIPA\n6.5%", "• BE HOPPY", "TOWN"]
    store = _FrameMatchStore({}, gold, by_frame=[(taft, 2, 0.6)])
    resp = Resolver(store).resolve(_frame(*lines))
    assert not resp.corroborated, f"drew {[c.resolved.product.name for c in resp.candidates]}"


_PROV = Provenance(source_id="ttb", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)


def test_a_row_of_another_kind_than_the_label_names_is_not_a_candidate():
    """"BLACK SEAL / 80 PROOF / BERMUDA BLACK RUM" prints the whole name of a 1984 London dry
    gin called `Black Seal`, and that proved it on every frame of a bottle of Gosling's. The
    label said RUM. A can that says nothing about its kind, and a row with no class, are
    unknown, not wrong."""
    gin = Product(id="p:gin", brand_id="b", producer_id="pr:w", category=Category.SPIRIT,
                  name="Black Seal", style=Sourced[str](value="London Dry Gin", provenance=_PROV)
                  ).model_dump(mode="json")
    rum = Product(id="p:rum", brand_id="b", producer_id="pr:g", category=Category.SPIRIT,
                  name="Goslings Black Seal",
                  style=Sourced[str](value="Other Rum Gold Usb", provenance=_PROV)).model_dump(mode="json")
    gold = {"pr:w": _producer("pr:w", "Winters"), "pr:g": _producer("pr:g", "Goslings")}
    line = "BLACK SEAL\n80 PROOF\nBERMUDA BLACK RUM"
    r = Resolver(_FrameStore({line: [(gin, 1.0), (rum, 0.55)]}, gold))
    resp = r.resolve(_frame(line, "Goslings\nSince 1806"))
    assert [c.resolved.product.id for c in resp.candidates] == ["p:rum"], (
        [c.resolved.product.name for c in resp.candidates])
    # The same gin off a label that names no kind is still the whole label.
    resp = Resolver(_FrameStore({"BLACK SEAL": [(gin, 1.0)]}, gold)).resolve(_frame("BLACK SEAL", "80 PROOF"))
    assert resp.corroborated and resp.candidates[0].resolved.product.id == "p:gin"
    # ...and across categories the label's word is a slogan or a garble as often as a fact:
    # "The Champagne of Beers" does not make a Miller can a wine.
    miller = Product(id="p:mhl", brand_id="b", producer_id="pr:m", category=Category.BEER,
                     name="Miller High Life", style=Sourced[str](value="Lager", provenance=_PROV)
                     ).model_dump(mode="json")
    gold["pr:m"] = _producer("pr:m", "Miller Brewing Company")
    line = "BREWING COMPANY MIL\nPREMIUM\nMiller.\nBREWED\nHIGH LIFE\nEST 1903\nThe Champagne of Bett"
    resp = Resolver(_FrameStore({line: [(miller, 0.5)]}, gold)).resolve(_frame(line, "RIDGE FARM\n1937"))
    assert resp.corroborated and resp.candidates[0].resolved.product.id == "p:mhl"


def test_a_name_word_with_its_first_letters_lost_is_still_read():
    """TOPPLING arrives as PLING and PPLING off the brewery line of a Dino Break can. A
    candidate the frame names between its lines is held to every word, and the tolerance
    for that is the recognizer's own failure: letters lost at one end of a word long enough
    to survive it. GOLD is four letters and unread when the frame read BLACK."""
    assert _unread(["toppling", "goliath", "dino", "break"], {"ppling", "goliath", "dino", "break"}) == []
    assert _unread(["toppling"], {"pling"}) == []
    assert _unread(["goslings", "gold", "seal"], {"goslings", "black", "seal"}) == ["gold"]
    assert _unread(["taft", "paint", "town", "hoppy"], {"town", "hoppy"}) == ["taft", "paint"]
