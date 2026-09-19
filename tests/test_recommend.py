"""The recommender's order: a match before a partial one, what drinkers rated before what we
know before what we guess, and then the score -- on a catalog where the style floor's ties
would otherwise fill the whole list."""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.recommend import GUESSED, KNOWN, RATED, evidence_tier, rank_catalog, rank_key
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


def test_the_known_rows_are_asked_for_separately(store):
    """On the live catalog the floor's ties fill any single top-N; the known rows have to be
    fetched on their own or they never reach the ranker."""
    ideal = PROFILE.sensory_ideal.to_array()
    assert all(r["sensory"]["source"] == "style_prior"
               for r in store.nearest_by_sensory(ideal, limit=3))
    known = store.nearest_by_sensory(ideal, limit=3, known=True)
    assert known and all(r["sensory"]["source"] != "style_prior" for r in known)


def test_no_taste_vector_yet_still_ranks_the_catalog(store):
    got = rank_catalog(store, Resolver(store), TasteProfile(user_id="new"), limit=3)
    assert len(got) == 3 and all(r["reason"] == "based on style" for r in got)
