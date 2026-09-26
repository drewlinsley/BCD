"""The recommender's order: a match before a partial one, what drinkers rated before what we
know before what we guess, and then the score -- on a catalog where the style floor's ties
would otherwise fill the whole list."""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.recommend import (
    GUESSED,
    KNOWN,
    RATED,
    _legible,
    evidence_tier,
    rank_catalog,
    rank_key,
    similar_profile,
)
from bcd_api.resolver import Resolver
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    Category,
    ExtractionMethod,
    Product,
    Provenance,
    SensorySource,
    SensoryVector,
    Sourced,
    TasteProfile,
)

PROV = Provenance(source_id="test", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)
IDEAL = {"citrus": 0.8, "tropical": 0.9, "piney_resinous": 0.6, "bitterness": 0.6,
         "body_fullness": 0.5}


def _sv(source, confidence, **axes):
    return SensoryVector(source=source, confidence=confidence, axes=axes)


def _product(pid, name, producer, sensory, style="IPA"):
    return Product(id=pid, brand_id=f"brand:{producer}", producer_id=f"prod:{producer}",
                   category=Category.BEER, name=name,
                   style=Sourced[str](value=style, provenance=PROV),
                   sensory=sensory).model_dump(mode="json")


@pytest.fixture()
def store():
    s = MedallionStore(root=tempfile.mkdtemp())
    for pid, name in (("anon", "Anonymous Brewing"), ("alch", "The Alchemist"),
                      ("rg", "Rhinegeist"), ("hp", "Harpoon"), ("law", "Lawson's Finest"),
                      ("st", "Stout House")):
        s.put_gold(f"prod:{pid}", "producer", {"id": f"prod:{pid}", "name": name})
    # The floor: three registry IPAs no one has heard of, all on the centroid -- which is the
    # user's ideal itself, so nothing can out-score them.
    for i in range(3):
        s.put_gold(f"floor{i}", "product", _product(
            f"floor{i}", f"Registry IPA {i}", "anon", _sv(SensorySource.STYLE_PRIOR, 0.35, **IDEAL)))
    # A real profile, a little off the centroid, as real beers are.
    s.put_gold("heady", "product", _product("heady", "The Alchemist Heady Topper", "alch",
               _sv(SensorySource.LLM_PROFILE, 0.9, citrus=0.75, tropical=0.9, piney_resinous=0.6,
                   bitterness=0.6, body_fullness=0.5, stone_fruit=0.15)))
    # One lineup profile stamped on two rows of one beer.
    truth = _sv(SensorySource.LLM_PROFILE, 0.75, citrus=0.75, tropical=0.6, piney_resinous=0.5,
                bitterness=0.7, body_fullness=0.4, grassy=0.3)
    s.put_gold("truth1", "product", _product("truth1", "Truth India Pale Ale", "rg", truth))
    s.put_gold("truth2", "product", _product("truth2", "Rhinegeist Truth", "rg", truth))
    # A maker-level guess: "the kind of beer Harpoon makes", nearly on the ideal.
    s.put_gold("harpoon", "product", _product("harpoon", "Harpoon", "hp",
               _sv(SensorySource.LLM_PROFILE, 0.35, **{**IDEAL, "malty_bready": 0.2}), style="Ale"))
    # Drinkers rated these: one a match, one only a partial one.
    s.put_gold("sip", "product", _product("sip", "Sip of Sunshine", "law",
               _sv(SensorySource.REVIEW_CONSENSUS, 0.7, citrus=0.6, tropical=0.8,
                   piney_resinous=0.4, bitterness=0.5, body_fullness=0.6, stone_fruit=0.4)))
    s.put_gold("stout", "product", _product("stout", "Rated Stout", "st",
               _sv(SensorySource.REVIEW_CONSENSUS, 0.8, roasted_coffee_choc=0.8,
                   body_fullness=0.7, bitterness=0.5, sweet=0.3, tropical=0.2), style="Stout"))
    yield s
    s.close()


PROFILE = TasteProfile(user_id="u", sensory_ideal=_sv(SensorySource.RECONCILED, 0.6, **IDEAL))


def test_evidence_is_read_off_the_vectors_source_and_confidence():
    assert evidence_tier(None) == GUESSED
    assert evidence_tier(_sv(SensorySource.STYLE_PRIOR, 0.35, citrus=1)) == GUESSED
    assert evidence_tier(_sv(SensorySource.LLM_PROFILE, 0.35, citrus=1)) == GUESSED  # style-only
    assert evidence_tier(_sv(SensorySource.LLM_PROFILE, 0.45, citrus=1)) == KNOWN
    assert evidence_tier(_sv(SensorySource.CHEMISTRY_PRIOR, 0.6, citrus=1)) == KNOWN
    assert evidence_tier(_sv(SensorySource.REVIEW_CONSENSUS, 0.5, citrus=1)) == RATED
    assert evidence_tier(_sv(SensorySource.RECONCILED, 0.5, citrus=1)) == RATED


def test_a_match_outranks_a_partial_match_whatever_stands_behind_it():
    rated = Product.model_validate(_product("r", "R", "x", _sv(SensorySource.RECONCILED, 0.9, citrus=1)))
    guess = Product.model_validate(_product("g", "G", "x", _sv(SensorySource.STYLE_PRIOR, 0.3, citrus=1)))
    assert rank_key(0.85, guess) < rank_key(0.79, rated)  # a match beats a partial match
    assert rank_key(0.85, rated) < rank_key(0.92, guess)  # ...and within the band, evidence
    assert rank_key(0.92, guess) < rank_key(0.85, guess)  # ...and then the score


def test_what_we_know_ranks_above_the_floors_ties(store):
    got = rank_catalog(store, Resolver(store), PROFILE, limit=10)
    names = [r["name"] for r in got]
    # Drinkers first, then the beers we know, then the guesses -- the centroid rows the
    # score alone would have put first, and the maker-level profile that is a guess by
    # another name -- and the rated stout last, because it is only a partial match.
    assert names[:3] == ["Sip of Sunshine", "The Alchemist Heady Topper", "Rhinegeist Truth"]
    assert [r["evidence"] for r in got[:3]] == ["rated", "known", "known"]
    guesses = got[3:-1]
    assert {r["evidence"] for r in guesses} == {"guessed"}
    assert guesses[0]["score"] > got[0]["score"]  # the centroid did out-score every known beer
    assert got[-1]["name"] == "Rated Stout" and got[-1]["reason"].startswith("some")
    # Two rows of one beer are one entry, under the plainer name; the three registry rows on
    # the centroid are one entry -- the floor shows once per style, not once per row.
    assert "Truth India Pale Ale" not in names
    assert sum(n.startswith("Registry IPA") for n in names) == 1
    assert [r["name"] for r in guesses] == ["Registry IPA 0", "Harpoon"]
    assert all(set(r) == {"product_id", "name", "producer", "score", "reason", "cold_start",
                          "evidence"} for r in got)
    assert got[0]["producer"] == "Lawson's Finest"


def test_the_known_vectors_are_asked_for_separately(store):
    """On the live catalog the floor's ties fill any single top-N; the known rows have to be
    fetched on their own or they never reach the ranker -- and fetched by vector, because a
    lineup profile sits on every label variant of its beer and row by row the nearest
    hundred were seven beers."""
    ideal = PROFILE.sensory_ideal.to_array()
    assert all(r["sensory"]["source"] == "style_prior"
               for r in store.nearest_by_sensory(ideal, limit=3))
    groups = store.nearest_known(ideal, limit=3)
    assert [[r["name"] for r in g] for g in groups] == [
        ["The Alchemist Heady Topper"], ["Harpoon"],
        ["Rhinegeist Truth", "Truth India Pale Ale"]]  # one vector, shortest name first
    assert all(r["sensory"]["source"] != "style_prior" for g in groups for r in g)


def test_a_name_that_says_only_its_kind_cannot_stand_for_the_rest():
    """One profile sits on hundreds of registry rows, and the plainest of them is spelled
    "mezcal" -- which names none of them, and a filer's two fields ran together in the next.
    The entry goes out under the plainest row that names a product."""
    s = MedallionStore(root=tempfile.mkdtemp())
    s.put_gold("prod:anon", "producer", {"id": "prod:anon", "name": "Anonymous"})
    vec = _sv(SensorySource.LLM_PROFILE, 0.6, **IDEAL)
    for pid, name in (("a", "mezcal"), ("b", "4b ,oaxaca"), ("c", "Madre Mezcal"),
                      ("d", "Bruja Mezcal Artesanal")):
        s.put_gold(pid, "product", _product(pid, name, "anon", vec))
    assert [r["name"] for r in rank_catalog(s, Resolver(s), PROFILE, limit=5)] == ["Madre Mezcal"]
    s.close()


def test_no_taste_vector_yet_still_ranks_the_catalog(store):
    got = rank_catalog(store, Resolver(store), TasteProfile(user_id="new"), limit=3)
    assert len(got) == 3 and all(r["reason"] == "based on style" for r in got)


# --- what you have already judged is not a suggestion ---------------------------------


def test_a_drink_you_have_rated_is_not_recommended(store):
    """The list is what to drink next. "For you" led with a beer whose own row on the same
    screen carried the face for `spat it out`, because nothing ever took a rated product
    out of the ranking (2026-09-24)."""
    all_names = [r["name"] for r in rank_catalog(store, Resolver(store), PROFILE, limit=10)]
    assert "Sip of Sunshine" in all_names
    got = rank_catalog(store, Resolver(store), PROFILE, limit=10, exclude={"sip"})
    names = [r["name"] for r in got]
    assert "Sip of Sunshine" not in names
    # ...and the rest of the order is untouched: this removes an entry, it does not re-rank.
    assert names == [n for n in all_names if n != "Sip of Sunshine"]


def test_rating_one_row_of_a_beer_retires_the_beer(store):
    """`Truth India Pale Ale` and `Rhinegeist Truth` are two registry rows of one beer on one
    vector, and this module's own rule is that such rows are the same recommendation. Taking
    out only the row the user rated would put its twin in the list under another name --
    which is the same suggestion, and would read as the app ignoring the verdict."""
    got = rank_catalog(store, Resolver(store), PROFILE, limit=10, exclude={"truth1"})
    names = [r["name"] for r in got]
    assert "Rhinegeist Truth" not in names and "Truth India Pale Ale" not in names


def test_excluding_nothing_changes_nothing(store):
    before = rank_catalog(store, Resolver(store), PROFILE, limit=10)
    assert rank_catalog(store, Resolver(store), PROFILE, limit=10, exclude=set()) == before


# ---- what else tastes like this -------------------------------------------------------------

def test_a_row_carrying_its_styles_centroid_has_nothing_to_be_similar_to(store):
    """507,341 of 534,103 products are in this state. Their vector is their style's average, so
    the nearest rows are every other row of that style, all at distance zero -- a style filter
    wearing a similarity list. Saying so is the honest answer."""
    floor = Product.model_validate(store.get_gold("floor0"))
    assert similar_profile(store, floor) == {"basis": "style_only", "results": []}


def test_a_profiled_row_gets_its_neighbours_nearest_first(store):
    """On the live catalog this is what earns the feature: Lagavulin 16 returns Bunnahabhain
    Moine and Ardbeg An Oa, Campari returns Aperol and Calisaya."""
    heady = Product.model_validate(store.get_gold("heady"))
    out = similar_profile(store, heady)
    assert out["basis"] == "profile"
    names = [r["name"] for r in out["results"]]
    assert "The Alchemist Heady Topper" not in names          # never itself
    assert "Sip of Sunshine" in names                          # the nearest real profile
    assert names.index("Sip of Sunshine") < names.index("Rated Stout")


def test_one_row_per_maker(store):
    """Guinness Draught's nearest are Guinness Dublin and Guinness 0.0%: the same drink
    answering a question about what else to try."""
    heady = Product.model_validate(store.get_gold("heady"))
    makers = [r["producer"] for r in similar_profile(store, heady)["results"]]
    assert len(makers) == len(set(makers))


def test_the_name_shown_for_a_shared_vector_is_the_one_a_drinker_could_repeat(store):
    """Two rows of one beer share a vector, so one stands for both -- and it has to be the
    plainer name. The same rule `rank_catalog` uses, so the two lists cannot disagree about
    what a vector is called."""
    heady = Product.model_validate(store.get_gold("heady"))
    hits = {r["name"]: r for r in similar_profile(store, heady)["results"]}
    assert "Rhinegeist Truth" in hits
    assert "Truth India Pale Ale" not in hits
    assert hits["Rhinegeist Truth"]["also"] == 1               # says one more shares it


def test_two_characters_is_not_a_name_a_vector_can_go_by():
    """The rest of the sort prefers the shortest name -- "Truth" over "Truth India Pale Ale" --
    which hands the slot to whatever registry stub is shortest of all. A vodka vector shared by
    ten rows was standing as "PA" on the live catalog, and an Italian bitter as "Aper@it"."""
    assert _legible("PA") == 2
    assert _legible("Aper@it") == 0          # seven letters; odd, but it names something
    assert _legible("Truth") == 0
    assert _legible("mezcal") == 2
