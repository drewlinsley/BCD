"""The label index — retrieval from memory, scored the way the store scores."""

from __future__ import annotations

import os
import tempfile

import pytest
from bcd_api.index import (
    IndexedStore,
    LabelIndex,
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
    Producer,
    Product,
    ScanResolveRequest,
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
