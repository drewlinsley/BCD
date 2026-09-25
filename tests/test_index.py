"""The label index — retrieval from memory, scored the way the store scores."""

from __future__ import annotations

import os
import tempfile

import pytest
from bcd_api.index import (
    IndexedStore,
    LabelIndex,
    _in_sorted,
    identifying_tokens,
    match_score,
    similarity,
    word_similarity,
)
from bcd_api.resolver import Resolver
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    Brand,
    Category,
    DetectedObject,
    DetectedText,
    ExtractionMethod,
    Producer,
    Product,
    Provenance,
    ScanResolveRequest,
    Sourced,
)


def _seed(store: MedallionStore) -> None:
    def producer(pid: str, name: str) -> None:
        store.put_gold(pid, "producer", Producer(id=pid, name=name).model_dump(mode="json"))

    def product(pid: str, name: str, producer_id: str, brand: str | None = None) -> None:
        bid = "brand:none"          # a brand row that does not exist: hydrates as Unknown
        if brand:
            bid = f"brand:{pid}"
            store.put_gold(bid, "brand", Brand(id=bid, producer_id=producer_id,
                                               name=brand).model_dump(mode="json"))
        store.put_gold(pid, "product", Product(id=pid, name=name, producer_id=producer_id,
                                               brand_id=bid, category=Category.BEER
                                               ).model_dump(mode="json"))

    producer("prod:alch", "The Alchemist LLC")
    producer("prod:dfh", "Dogfish Head Craft Brewery")
    producer("prod:tops", "Top's Brewing")
    producer("prod:ache", "Ache Brewing")
    producer("prod:theo", "Theo P. Brewing")
    producer("prod:banger", "Banger")
    producer("prod:titos", "Fifth Generation Inc")
    product("p:heady", "Heady Topper", "prod:alch", brand="Heady")
    product("p:focal", "Focal Banger", "prod:alch")
    product("p:60", "60 Minute IPA", "prod:dfh", brand="Dogfish Head")
    product("p:tops", "Top's", "prod:tops")
    product("p:ache", "Ache", "prod:ache")
    product("p:theo", "Theo P.", "prod:theo")
    product("p:banger", "Banger", "prod:banger")
    product("p:titos", "Handmade Vodka", "prod:titos", brand="Tito's")
    product("p:ipa", "IPA", "prod:dfh")
    product("p:1664", "1664", "prod:dfh")


@pytest.fixture()
def store():
    d = tempfile.mkdtemp()
    s = MedallionStore(root=d)
    _seed(s)
    yield s
    s.close()


@pytest.fixture()
def index(store):
    return LabelIndex.build(store)


# --- tokens and scores ---------------------------------------------------------------


def test_identifying_tokens_drop_category_words_and_keep_long_numbers():
    assert identifying_tokens("FOCAL BANGER THE ALCHEMIST INDIA PALE ALE") == \
        ["focal", "banger", "alchemist"]
    assert identifying_tokens("INDIA PALE ALE 16 FL OZ") == []
    assert identifying_tokens("Kronenbourg 1664") == ["kronenbourg", "1664"]


def test_scores_mirror_pg_trgm_directions():
    # `word_similarity` finds a short name inside a noisy line and a short line inside a
    # long name; plain `similarity` penalises what a row leaves out — the same three terms
    # the Postgres store ranks by, with the same outcomes it was measured on.
    assert word_similarity("Guinness", "GUINNESS DRAUGHT 440ML EXTRA STOUT") == 1.0
    assert word_similarity("BOMBAY SAPPHIRE", "Bombay Sapphire London Dry Gin") == 1.0
    assert similarity("Gin Bombay", "BOMBAY SAPPHIRE") < 0.7
    assert match_score("Heady Topper", "Heady Topper", "HEADY TOPPER") == 1.0
    assert match_score("Focal Banger", "Focal Banger", "FADY TOP") < 0.3


# --- retrieval -------------------------------------------------------------------------


def test_finds_a_product_by_its_name(index):
    ids = [pid for pid, _ in index.match_products("HEADY TOPPER", limit=3)]
    assert ids[0] == "p:heady"


def test_finds_a_product_through_its_brand(index):
    # The catalog splits "Tito's Handmade Vodka" into brand + name; the index posts the
    # brand-qualified name so the label as printed reaches the row.
    ids = [pid for pid, _ in index.match_products("TITOS HANDMADE VODKA", limit=3)]
    assert ids[0] == "p:titos"


def test_reaches_a_misread_word_one_edit_away(index):
    ids = [pid for pid, _ in index.match_products("HEADY TOPPFR", limit=3)]
    assert "p:heady" in ids


def test_reaches_a_truncated_word_by_prefix(index):
    # "FOCAL BAN" is what a Focal Banger can actually OCRs to; "ALCHEMIS" is the rim print
    # cut off. Products post their brand-qualified name, producers their own — the same
    # split the store has, so the producer path is the one that reaches the maker.
    ids = [pid for pid, _ in index.match_products("HEADY TOPP", limit=5)]
    assert ids[0] == "p:heady"
    assert [pid for pid, _ in index.match_producers("THE ALCHEMIS", limit=3)][0] == "prod:alch"


def test_reaches_a_word_missing_its_first_letters_by_suffix(index):
    """The mirror of the prefix case. A stylized wordmark loses its first letters first -- the
    initial is the letter drawn largest and strangest -- and on a can of Heady Topper the
    recognizer read THE ALCHEMIST as "CHEMIST-VER" sixty times for every four "ALCHEMIST".
    "chemist" is no edit of "alchemist" and no prefix of it, so the maker's row was never a
    candidate at all, and the maker pick had nothing to try."""
    ids = [pid for pid, _ in index.match_producers("CHEMIST-VER", limit=3)]
    assert "prod:alch" in ids
    # ...and a token too short to be a safe suffix reaches nothing that way: "MIST" is not
    # evidence of ALCHEMIST.
    assert "prod:alch" not in [pid for pid, _ in index.match_producers("MIST", limit=3)]


def test_a_generic_line_reaches_only_generic_names(index):
    ids = [pid for pid, _ in index.match_products("IPA", limit=5)]
    assert ids == ["p:ipa"]


def test_a_digit_name_is_reachable(index):
    ids = [pid for pid, _ in index.match_products("1664", limit=3)]
    assert ids == ["p:1664"]


def test_the_row_explaining_more_of_the_line_ranks_first(index):
    # Both `Banger` and `Focal Banger` sit wholly inside the line and tie at 1.0 on
    # containment; the tiebreak is which accounts for more of what was read.
    ids = [pid for pid, _ in index.match_products(
        "FOCAL BANGER THE ALCHEMIST INDIA PALE ALE", limit=3)]
    assert ids[0] == "p:focal"


def test_producers_and_their_products(index):
    ids = [pid for pid, _ in index.match_producers("THE ALCHEMIST", limit=3)]
    assert ids[0] == "prod:alch"
    assert set(index.products_of("prod:alch")) == {"p:heady", "p:focal"}


def test_lexicon_is_identifying_vocabulary_only(index):
    words = index.lexicon(limit=50)
    assert "alchemist" in words and "topper" in words and "dogfish" in words
    assert "ipa" not in words and "brewing" not in words and "the" not in words


# --- the store front ------------------------------------------------------------------


def test_indexed_store_answers_matching_and_passes_the_rest_through(store, index):
    front = IndexedStore(store, index)
    (rec, sim), *_ = front.match_products("HEADY TOPPER")
    assert rec["id"] == "p:heady" and sim == 1.0
    assert front.get_gold("p:heady")["name"] == "Heady Topper"
    assert front.counts() == store.counts()
    assert [r["id"] for r in front.products_of("prod:alch")] == ["p:focal", "p:heady"]


def test_resolver_behaves_the_same_on_the_index(store, index):
    plain = Resolver(store)
    fast = Resolver(IndexedStore(store, index))
    req = ScanResolveRequest(detections=[DetectedText(text="HEADY TOPPER"),
                                         DetectedText(text="THE ALCHEMIST")])
    a, b = plain.resolve(req), fast.resolve(req)
    assert a.corroborated and b.corroborated
    assert [c.resolved.product.id for c in a.candidates] == \
        [c.resolved.product.id for c in b.candidates] == ["p:heady"]
    obj = DetectedObject(id="o", texts=["FOCAL BANGER THE ALCHEMIST INDIA PALE ALE"])
    assert fast.resolve(ScanResolveRequest(objects=[obj])).objects[0].status == "resolved"


# --- persistence ----------------------------------------------------------------------


def test_index_is_cached_and_invalidated_by_the_catalog(store, index):
    path = os.path.join(tempfile.mkdtemp(), "label_index.pkl")
    index.save(path)
    again = LabelIndex.load(path, LabelIndex.signature_of(store))
    assert again is not None and again.ids == index.ids
    store.put_gold("p:new", "product", Product(id="p:new", name="Sip of Sunshine",
                                               producer_id="prod:alch", brand_id="brand:none",
                                               category=Category.BEER).model_dump(mode="json"))
    assert LabelIndex.load(path, LabelIndex.signature_of(store)) is None
    rebuilt = LabelIndex.for_store(store, path)
    assert "p:new" in rebuilt.ids
    assert LabelIndex.load(path, LabelIndex.signature_of(store)) is not None


# --- what the lines name between them ------------------------------------------------


def _goslings_shelf() -> MedallionStore:
    """The catalog as it stood around a bottle of Gosling's Black Seal on 2026-09-15: the
    rum filed once under its own brand, beside a 1984 gin called `Black Seal`, two other
    Bermuda rums, a stout, the brand row, a row named `1806`, and every shorter Goslings."""
    s = MedallionStore(root=tempfile.mkdtemp())

    def producer(pid: str, name: str) -> None:
        s.put_gold(pid, "producer", Producer(id=pid, name=name).model_dump(mode="json"))

    def product(pid: str, name: str, producer_id: str, brand: str, category=Category.SPIRIT) -> None:
        bid = f"brand:{pid}"
        s.put_gold(bid, "brand", Brand(id=bid, producer_id=producer_id, name=brand).model_dump(mode="json"))
        s.put_gold(pid, "product", Product(id=pid, name=name, producer_id=producer_id, brand_id=bid,
                                           category=category).model_dump(mode="json"))

    producer("prod:gos", "Goslings")
    producer("prod:winters", "Winters")
    producer("prod:other", "Bermuda Brand")
    producer("prod:abbey", "Abbey Ale")
    product("p:black", "Goslings Black Seal", "prod:gos", "Goslings Black Seal")
    product("p:gin", "Black Seal", "prod:winters", "Black Seal")
    product("p:brand", "Bermuda Brand Black Rum", "prod:other", "Bermuda Brand")
    product("p:stout", "Black Seal Stout", "prod:abbey", "Black Seal Stout", Category.BEER)
    product("p:bgold", "Bermuda Gold", "prod:other", "Bermuda Gold")
    product("p:gos", "Goslings", "prod:gos", "Goslings")
    product("p:1806", "1806", "prod:other", "1806")
    product("p:gold", "Goslings Gold Seal", "prod:gos", "Goslings Gold Seal")
    product("p:papa", "Goslings Papa Seal", "prod:gos", "Goslings Papa Seal")
    product("p:light", "Goslings Light Rum", "prod:gos", "Goslings Light Rum")
    return s


GOSLINGS_LINES = ["BLACK SEAL\n80 PROOF\nBERMUDA BLACK RUM", "Goslings\nSince 1806"]


def test_a_name_printed_across_two_lines_is_found_by_the_frame():
    """Per line, `Goslings Black Seal` is behind a gin, a stout and two other Bermuda rums on
    the first line and behind the brand row, `1806` and every shorter Goslings on the second;
    a top three per line never holds it. Against the frame's tokens together it is first,
    reported with the line it reads best on and that line's score."""
    index = LabelIndex.build(_goslings_shelf())
    per_line = [[pid for pid, _ in index.match_products(t, limit=3)] for t in GOSLINGS_LINES]
    assert all("p:black" not in ids for ids in per_line), per_line
    (pid, at, sim), *_ = index.match_frame(GOSLINGS_LINES, limit=8)
    assert pid == "p:black" and at == 0 and 0.5 <= sim < 1.0


def test_the_frame_names_the_rum_and_only_the_rum():
    """End to end on the index: the frame path finds the rum between its two lines, `Goslings
    Gold Seal` (GOLD read nowhere) is shadowed by it, and the gin called `Black Seal`, which
    the first line prints whole, gives way to the row that also explains GOSLINGS."""
    store = _goslings_shelf()
    resp = Resolver(IndexedStore(store, LabelIndex.build(store))).resolve(
        ScanResolveRequest(detections=[DetectedText(text=t) for t in GOSLINGS_LINES]))
    assert resp.corroborated
    assert [c.resolved.product.id for c in resp.candidates] == ["p:black"], (
        [c.resolved.product.name for c in resp.candidates])
    # ...and a frame-level candidate is held to every word: the two lines alone do not
    # surface `Goslings Gold Seal` or `Bermuda Brand Black Rum` as proven.
    obj = DetectedObject(id="o", texts=GOSLINGS_LINES)
    verdict = Resolver(IndexedStore(store, LabelIndex.build(store))).resolve(
        ScanResolveRequest(objects=[obj])).objects[0]
    assert verdict.status == "resolved" and verdict.candidates[0].resolved.product.id == "p:black"


def _campari_shelf() -> MedallionStore:
    """A catalog shaped like the one around a bottle of Campari on 2026-09-16: the wordmark
    read as CAMPAR reaches six Campari rows by one edit at 0.7 of the weight, and the house
    line's words -- DAVIDE, CARPET for the garbled CAMPARI -- each match thirty rows exactly,
    every one of which outweighs the row the line actually names."""
    s = MedallionStore(root=tempfile.mkdtemp())
    s.put_gold("prod:x", "producer", Producer(id="prod:x", name="Some House").model_dump(mode="json"))

    def product(pid: str, name: str) -> None:
        s.put_gold(pid, "product", Product(id=pid, name=name, producer_id="prod:x",
                                           brand_id="brand:none", category=Category.SPIRIT
                                           ).model_dump(mode="json"))

    for i in range(4934):
        product(f"p:filler{i}", f"Filler Row {i} Zx{i}")
    product("p:campari", "Campari")
    for i, tail in enumerate(["Negroni", "Cask Tales", "Cask Tales Rum", "Cask Tales Bourbon",
                              "Cask Tales Tequila"]):
        product(f"p:campari{i}", f"Campari {tail}")
    for i in range(30):
        product(f"p:davide{i}", f"Davide Q{i}ver")
        product(f"p:carpet{i}", f"W{i}ing Carpet")
    return s


def test_a_garbled_word_is_heard_over_the_lines_ordinary_ones():
    """"CAMPAR Davide Carpet MIL A": the wordmark with its last letter lost, then the house's
    line garbled. `Campari` is a one-edit match at 0.7 of the weight; DAVIDE and CARPET are
    exact and each match thirty rows at full weight, and the top 48 by total evidence held
    only those -- the row the bottle names was cut before it was scored (2026-09-16)."""
    index = LabelIndex.build(_campari_shelf())
    ids = [pid for pid, _ in index.match_products("CAMPAR\nDavide Carpet\nMIL\nA", limit=3)]
    assert "p:campari" in ids, ids


def test_an_exact_word_earns_no_extra_seats():
    # The guarantee is for garbles. An exact word is already heard at full weight, and
    # letting its group in too put one-word rows -- a 1.0 by containment each -- on every
    # word of a long label, ahead of the label's own row.
    index = LabelIndex.build(_campari_shelf())
    acc, reached = index._token_evidence("CAMPAR Davide Carpet", index.product_post, len(index.ids))
    assert len(reached) == 1, "only CAMPAR, the garbled one, gets a group"


def test_word_similarity_searches_extents_the_length_of_the_name_plus_slack():
    """A name of k words is matched by k words of the line plus the stray word or two
    grouped between them; a longer extent only adds trigrams the name lacks and scores
    lower. Capping the search there is what makes a forty-word sticker affordable."""
    line = "BREWING COMPANY MILWAUKEE PREMIUM Miller BREWED HIGH LIFE ESTD 1903 The Champagne of Beers"
    # the capped search finds the same best extent an exhaustive one does
    ta = _trigrams_of("Miller High Life")
    exhaustive = max(_jaccard_of(ta, w) for w in _all_windows(line))
    assert abs(word_similarity("Miller High Life", line) - exhaustive) < 1e-9
    assert word_similarity("Miller High Life", line) > 0.5


def _trigrams_of(s):
    from bcd_api.index import _trigrams
    return _trigrams(s)


def _jaccard_of(a, b):
    from bcd_api.index import _jaccard
    return _jaccard(a, b)


def _all_windows(s):
    from bcd_api.index import _windows
    return _windows(s, 0)


def _bombay_shelf() -> MedallionStore:
    """The catalog around a bottle of Bombay Sapphire on 2026-09-17: the canon, merged by
    hand and carrying the label's own words as an alias; `East`, the same house's other gin,
    named for those words; and the decoys the lines otherwise reach."""
    s = MedallionStore(root=tempfile.mkdtemp())

    def producer(pid: str, name: str) -> None:
        s.put_gold(pid, "producer", Producer(id=pid, name=name).model_dump(mode="json"))

    def product(pid: str, name: str, producer_id: str, brand: str, aliases=()) -> None:
        bid = f"brand:{pid}"
        s.put_gold(bid, "brand", Brand(id=bid, producer_id=producer_id, name=brand).model_dump(mode="json"))
        s.put_gold(pid, "product", Product(id=pid, name=name, producer_id=producer_id, brand_id=bid,
                                           category=Category.SPIRIT,
                                           aliases=list(aliases)).model_dump(mode="json"))

    producer("prod:bs", "Bombay Sapphire")
    producer("prod:whims", "Whims Brewing")
    producer("prod:bombay", "Bombay")
    product("p:plain", "Bombay Sapphire London Dry Gin", "prod:bs", "Bombay Sapphire",
            aliases=["Bombay Sapphire Vapour Infused London Dry Gin", "Bombay Sapphire Dry Gin"])
    product("p:east", "East Vapour Infused London Dry Gin", "prod:bs", "Bombay Sapphire")
    product("p:whims", "Sapphire", "prod:whims", "Whims Brewing")
    product("p:bombay", "Bombay", "prod:bombay", "Bombay")
    product("p:bomber", "Bombay Bomber", "prod:bombay", "Bombay")
    # `East`'s brand row is the house's own, as in the catalog
    s.put_gold("brand:p:east", "brand",
               Brand(id="brand:p:east", producer_id="prod:bs", name="Bombay Sapphire").model_dump(mode="json"))
    return s


BOMBAY_LABEL = "SAPPHIRE\nDistilled Distile\nLONDON\nDRY GIN\nVapour\nINFUSED INFUSE\nà la vapour"


def test_a_rows_aliases_are_names_it_answers_to():
    """The canon carries "Bombay Sapphire Vapour Infused London Dry Gin" as an alias, and
    that is what the label prints. Indexed under its name alone it never surfaced for the
    line that read VAPOUR INFUSED, and `East Vapour Infused` had the line to itself."""
    index = LabelIndex.build(_bombay_shelf())
    hits = dict(index.match_products(BOMBAY_LABEL, limit=3))
    assert "p:plain" in hits, hits
    assert hits["p:plain"] >= 0.5


def test_the_plain_gin_is_not_east_until_bombay_is_read():
    """On every tick the wordmark came in as SOMBAA, VAPOUR INFUSED on two lines proved
    `East` and the bottle was drawn as the other gin (2026-09-17). East claims a word the
    frame never read that the plain gin does not; it yields, and the plain gin waits for
    BOMBAY -- read, it is the bottle."""
    store = _bombay_shelf()
    resolver = Resolver(IndexedStore(store, LabelIndex.build(store)))
    lines = [BOMBAY_LABEL, "SOMBAA", "INFUSED\nINFUSE\nà la vapeur", "750 ml e\n40% alc./vol"]
    resp = resolver.resolve(ScanResolveRequest(objects=[DetectedObject(id="o1", texts=lines)]))
    names = [c.resolved.product.id for c in resp.objects[0].candidates]
    assert resp.objects[0].status != "resolved", names
    assert "p:east" not in names, names
    resp = resolver.resolve(ScanResolveRequest(objects=[DetectedObject(id="o1", texts=[*lines, "BOMBAY"])]))
    assert resp.objects[0].status == "resolved"
    assert [c.resolved.product.id for c in resp.objects[0].candidates] == ["p:plain"]


# --- the label's word for the drink, as a tie-break -----------------------------------


_KIND_PROV = Provenance(source_id="ttb", method=ExtractionMethod.REGULATORY_FILING,
                        confidence=1.0)


def _whale_shelf() -> MedallionStore:
    """A catalog shaped like the one around a bottle of Gray Whale Gin on 2026-09-23.

    "WHALE" is one identifying word and hundreds of rows carry it; GIN, the other word on
    the label, is a category word and reaches none of them. So the rows tie -- level on the
    one token they share -- and which of them fills the last of the `CANDIDATES` places was
    the order of a dict. The gin scores 1.00 against that line and was never scored at all.
    """
    s = MedallionStore(root=tempfile.mkdtemp())
    s.put_gold("prod:x", "producer",
               Producer(id="prod:x", name="Some House").model_dump(mode="json"))

    def product(pid: str, name: str, cat: Category, style: str | None) -> None:
        s.put_gold(pid, "product", Product(
            id=pid, name=name, producer_id="prod:x", brand_id="brand:none", category=cat,
            style=None if style is None else Sourced[str](value=style, provenance=_KIND_PROV),
        ).model_dump(mode="json"))

    # every one of these carries "whale" and nothing else the label says
    for i in range(400):
        product(f"p:whale{i}", f"Whale Q{i}ver", Category.BEER, "Ale")
    product("p:whale", "Whale", Category.BEER, "Ale")
    product("p:gin", "Gray Whale Gin", Category.SPIRIT, "Contemporary gin")
    return s


def test_the_labels_word_for_the_drink_settles_a_tie_between_level_rows():
    """"WHALE GIN" off a bottle of Gray Whale Gin. Four hundred rows tie on `whale`, and the
    gin is the only one the registry filed as a gin: the word the label prints for what is in
    the bottle is the whole of the difference between them (2026-09-23)."""
    index = LabelIndex.build(_whale_shelf())
    ids = [pid for pid, _ in index.match_products("WHALE GIN", limit=3)]
    assert "p:gin" in ids, ids


def test_a_category_word_still_reaches_nothing_on_its_own():
    """It sorts the rows a name word found; it does not find any. Otherwise GIN alone would
    reach every gin in the catalog, which is the reason `identifying_tokens` drops it."""
    index = LabelIndex.build(_whale_shelf())
    assert index.match_products("GIN", limit=3) == []


def test_a_class_a_quarter_of_the_catalog_shares_sorts_nothing():
    """ALE is filed against 154,000 of the real catalog's rows. Reading it off "MIST-VERMONT
    ALE" -- a Heady Topper can, ALCHEMIST VERMONT with the wordmark broken -- drew
    `U.s.s. Vermont Ale`, which shares that word with every other ale there is (2026-09-23).
    Here every filler row is an ale, so the word divides nothing and must not order anything.
    """
    index = LabelIndex.build(_whale_shelf())
    assert index._kind_lists("WHALE ALE") == []
    assert index._kind_lists("WHALE GIN"), "a class 1 row in 400 carries is still evidence"


def test_a_category_word_in_a_rows_name_does_not_promote_it():
    """`Cerveza Negra` is a lager. Its name carries CERVEZA, and a can of Modelo Negra prints
    CERVEZA too, so counting the word off the *name* handed it four frames of the Modelo
    whose wordmark had been read as "odelo" (measured over 1,817 logged frames, and
    reverted). A row's name is already the string the trigram stage scores the line against;
    only what the registry filed it as is new information, and that is all this reads."""
    s = _whale_shelf()
    s.put_gold("p:named", "product", Product(
        id="p:named", name="Whale Gin Blossom", producer_id="prod:x", brand_id="brand:none",
        category=Category.BEER, style=Sourced[str](value="Ale", provenance=_KIND_PROV),
    ).model_dump(mode="json"))
    index = LabelIndex.build(s)
    pos = {pid: i for i, pid in enumerate(index.ids)}
    rows = index._kind_lists("WHALE GIN")
    assert rows, "the gin's own class is still read"
    assert not _in_sorted(rows, pos["p:named"]), "an ale is not a gin"
    assert _in_sorted(rows, pos["p:gin"])


def test_the_tie_break_cannot_reorder_rows_that_are_not_tied():
    """It is the second half of the sort key, so it settles equals and nothing else. A row
    the line names two words of outranks one it names a word of however well the label and
    the registry agree about the class -- which is what keeps a row that says less than the
    label from winning by saying it."""
    index = LabelIndex.build(_whale_shelf())
    pos = {pid: i for i, pid in enumerate(index.ids)}
    # "Q7VER" is `Whale Q7ver`'s own word: that row is named by two of the line's words and
    # the gin by one, and only the gin is filed as what the line says the drink is.
    acc, _ = index._token_evidence("WHALE Q7VER GIN", index.product_post, len(index.ids))
    assert acc[pos["p:whale7"]] > acc[pos["p:gin"]]
    assert [pid for pid, _ in index.match_products("WHALE Q7VER GIN", limit=1)] == ["p:whale7"]
