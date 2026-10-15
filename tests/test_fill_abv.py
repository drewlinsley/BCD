"""The gates on an ABV that came from somebody else's database.

`scripts/fill_abv_from_open_data.py` matches a published strength to a catalog row. Every
test here is a row the pass got wrong at some point and now refuses, or a true fact it was
throwing away.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

fill = pytest.importorskip("fill_abv_from_open_data")


def _fact(name, maker, abv, source="wikidata", ref="Q1"):
    return fill.Fact(name=name, maker=maker, abv_pct=abv, source=source, ref=ref,
                     url=f"https://www.wikidata.org/wiki/{ref}")


def _row(pid, name, brand="", category="beer", abv=None):
    return {"id": pid, "name": name, "brand": brand, "category": category, "abv": abv}


def _plan(facts, rows):
    return fill.plan(facts, rows, fill.index_rows(rows))


def _sourced(value, method, source="ttb-cola-registry"):
    return {"value": value, "provenance": {"source_id": source, "method": method,
                                           "confidence": 0.9, "url": None, "quote": None}}


# ---------------------------------------------------------------- the maker has to agree

def test_every_word_of_the_brewery_must_be_in_the_row():
    """"Sun King Brewing" shares "king" with `King's Reserve Bourbon Barrel Johan` and made
    no part of it. Set intersection is not agreement."""
    assert not fill.maker_agrees("Sun King Brewing", _row("a", "King's Reserve Bourbon Johan"))
    assert fill.maker_agrees("Sun King Brewing", _row("b", "Sun King Wee Mac Scottish Ale"))


def test_a_maker_name_that_folds_to_nothing_confirms_nothing():
    """Ten producers written in a script the ASCII fold erases all become the same empty
    string, and an empty set is a subset of everything."""
    assert not fill.maker_agrees("—", _row("a", "Anything At All"))
    assert not fill.maker_agrees("", _row("a", "Anything At All"))


def test_a_brand_written_into_the_name_does_not_switch_the_maker_check_off():
    """The first run of this pass confirmed BrewDog's 55% beer against a J. Wakefield row.
    `dedup._prep` discards a brand whose words all appear inside the product name as an echo,
    and TTB writes the brewery into the name, so the brand gate stopped gating. Here the
    brewery's own word is what the row is looked up by, so the wrong row is never even read."""
    rows = [_row("ttb:1", "J. Wakefield Brewing The End Of History", brand="j-wakefield-brewing")]
    fills, held = _plan([_fact("The End of History", "BrewDog", 55.0)], rows)
    assert fills == []
    assert "no row of that name" in held


def test_a_two_letter_brewery_is_still_checked():
    """The lookup leaves short words out -- an index key that does not narrow the search costs
    a long posting list -- so the maker gate is what notices that "Brouwerij 't IJ" did not
    brew De Molen's Struis. A gate skipped where the index cannot help is a wrong answer."""
    rows = [_row("ttb:1", "Brouwerij De Molen Struis", brand="brouwerij-de-molen")]
    postings = fill.index_rows(rows)
    fact = _fact("Struis", "Brouwerij 't IJ", 9.0)
    assert [r["id"] for r in fill.candidates(fact, rows, postings)] == ["ttb:1"]
    assert not fill.maker_agrees(fact.maker, rows[0])
    fills, held = _plan([fact], rows)
    assert fills == []
    assert "a different maker" in held


# ---------------------------------------------------------------- what the row already says

def test_a_row_that_states_its_own_strength_is_not_corrected():
    """"CARLSBERG UNFILTERED 5.0 DEGRE ALCOOL" got that number off a filing. A database
    saying 4.2% does not get to overrule the label the row is named after."""
    rows = [_row("ttb:1", "Carlsberg 4X33CL CARLSBERG UNFILTERED 5.0 DEGRE ALCOOL",
                 brand="carlsberg")]
    fills, held = _plan([_fact("Carlsberg Unfiltered", "Carlsberg Group", 4.2)], rows)
    assert fills == []
    assert "the row's own name says otherwise" in held


def test_a_stated_strength_that_agrees_corroborates_the_match():
    rows = [_row("ttb:1", "Kane Brewing Company Black Tie Optional India Pale Ale 6.6%",
                 brand="kane-brewing-company")]
    fills, _ = _plan([_fact("Black Tie Optional", "Kane Brewing Company", 6.4)], rows)
    assert [(f.pid, f.abv_pct) for f in fills] == [("ttb:1", 6.4)]


def test_a_volume_or_a_proportion_is_not_a_strength():
    """The number comes first and the unit after it, which is what keeps a case count and an
    all-malt claim from reading as an ABV."""
    assert fill.states_a_strength("4X33CL CARLSBERG") == []
    assert fill.states_a_strength("100% MALT LAGER") == []
    assert fill.states_a_strength("750ML Bottle") == []
    assert fill.states_a_strength("India Pale Ale 6.6%") == [6.6]
    assert fill.states_a_strength("UNFILTERED 5.0 DEGRE ALCOOL") == [5.0]


# ---------------------------------------------------------------- non-alcoholic rows

def test_a_non_alcoholic_row_does_not_get_an_alcoholic_number():
    """`Lowlander 0.00% Wit` was filled at 3.5% from Wikidata's alcoholic `Lowlander Wit`,
    while the true 0.0% was held for being below the floor: the one row where the number
    matters most, answered backwards."""
    rows = [_row("ttb:1", "Lowlander 0.00% Wit", brand="lowlander")]
    fills, held = _plan([_fact("Lowlander Wit", "Lowlander", 3.5)], rows)
    assert fills == []
    assert "an implausible strength" in held

    fills, _ = _plan([_fact("Lowlander 0.00% Wit", "Lowlander", 0.0)], rows)
    assert [(f.pid, f.abv_pct) for f in fills] == [("ttb:1", 0.0)]


def test_a_strength_outside_the_category_is_refused():
    assert fill.plausible(7.2, "beer", "Heady Topper")
    assert not fill.plausible(0.3, "beer", "Omnipollo Konx")       # a NA version, or wrong
    assert not fill.plausible(55.0, "beer", "Tactical Penguin")    # past the eisbock ceiling
    assert fill.plausible(40.0, "spirit", "Lagavulin 16")
    assert not fill.plausible(4.0, "spirit", "Lagavulin 16")


# ---------------------------------------------------------------- which row, and whose answer

def test_a_name_that_fits_two_rows_fills_neither():
    """A name that does not say which row it is cannot hang a strength off either."""
    rows = [_row("ttb:1", "Amstel Light", brand="amstel"),
            _row("ttb:2", "Amstel Light (110 Calories)", brand="amstel")]
    fills, held = _plan([_fact("Amstel Light", "Amstel", 3.5)], rows)
    assert fills == []
    assert "several rows it could be" in held


def test_a_better_source_is_left_alone():
    """A filing, a label read, the maker's own word or a shop listing all outrank this."""
    for method in ("regulatory_filing", "label_ocr", "stated_by_producer", "retailer_listing"):
        assert fill._fillable(_sourced(8.0, method)) is None
    rows = [_row("ttb:1", "The Alchemist Heady Topper", brand="the-alchemist",
                 abv=_sourced(8.0, "label_ocr"))]
    fills, held = _plan([_fact("Heady Topper", "The Alchemist", 9.0)], rows)
    assert fills == []
    assert "already answered by a better source" in held


def test_a_guess_and_a_recall_are_replaced():
    """A style's typical strength and what a model remembered are both second-hand with no
    id to look up; an open database row has one. `Little Creatures Pale Ale` was carrying a
    40% style guess."""
    assert fill._fillable(None) == "empty"
    assert fill._fillable(_sourced(40.0, "llm_inferred_from_style_prior")) == "a style guess"
    assert fill._fillable(_sourced(6.0, "llm_recalled")) == "a recall"
    rows = [_row("ttb:1", "Little Creatures Pale Ale", brand="little-creatures",
                 abv=_sourced(40.0, "llm_inferred_from_style_prior"))]
    fills, _ = _plan([_fact("Little Creatures Pale Ale", "Little Creatures Brewery", 5.2)], rows)
    assert [(f.was, f.said, f.abv_pct) for f in fills] == [("a style guess", 40.0, 5.2)]


def test_two_sources_that_disagree_fill_nothing():
    """Two Wikidata items for Fuller's London Pride, 4.7% and 4.1%. Neither is evidence."""
    rows = [_row("ttb:1", "Fullers London Pride", brand="fullers")]
    facts = [_fact("London Pride", "Fuller's Brewery", 4.7, ref="Q1"),
             _fact("London Pride", "Fuller's Brewery", 4.1, ref="Q2")]
    fills, held = _plan(facts, rows)
    assert fills == []
    assert "two sources that disagree" in held


def test_two_sources_within_half_a_point_agree():
    rows = [_row("ttb:1", "Fullers London Pride", brand="fullers")]
    facts = [_fact("London Pride", "Fuller's Brewery", 4.7, ref="Q1"),
             _fact("London Pride", "Fuller's Brewery", 4.4, ref="Q2")]
    fills, _ = _plan(facts, rows)
    assert [(f.abv_pct, f.fact.ref) for f in fills] == [(4.7, "Q1")]


def test_a_name_of_nothing_but_style_words_identifies_no_drink():
    rows = [_row("ttb:1", "Sierra Nevada Pale Ale", brand="sierra-nevada")]
    fills, held = _plan([_fact("Pale Ale", "Sierra Nevada Brewing", 5.6)], rows)
    assert fills == []
    assert "no row of that name" in held


def test_a_row_is_only_looked_at_when_it_holds_every_word():
    rows = [_row("ttb:1", "Mortalis Brewing Company Hydra Blue Milk", brand="mortalis-brewing"),
            _row("ttb:2", "Mortalis Brewing Company Hydra Sangria", brand="mortalis-brewing")]
    postings = fill.index_rows(rows)
    seen = fill.candidates(_fact("Hydra | Blue Milk", "Mortalis Brewing Company", 7.0),
                           rows, postings)
    assert [r["id"] for r in seen] == ["ttb:1"]


# ---------------------------------------------------------------- the receipt

def test_a_fact_keeps_the_id_you_can_look_it_up_by():
    doc = _fact("Heady Topper", "The Alchemist", 8.0, ref="Q5").bronze()
    assert doc.source_id == "wikidata"
    assert doc.natural_key == "Q5"
    assert doc.url == "https://www.wikidata.org/wiki/Q5"
    assert doc.payload["abv_pct"] == 8.0
    # Same question, same id: a re-fetch upserts rather than piling up.
    assert doc.id == _fact("Heady Topper", "The Alchemist", 8.1, ref="Q5").bronze().id


def test_the_provenance_says_where_and_how_sure():
    prov = fill._provenance(_fact("Heady Topper", "The Alchemist", 8.0, ref="Q5"))
    assert prov["method"] == "open_database"
    assert prov["confidence"] == 0.75
    assert prov["url"] == "https://www.wikidata.org/wiki/Q5"
    assert "8.0%" in prov["quote"] and "Q5" in prov["quote"]
