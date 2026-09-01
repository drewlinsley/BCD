"""Substring / cross-source entity resolution — high precision, no false merges.

Cases are the real catalog pairs the audit surfaced: two true cross-source dupes among twelve
containments, the rest coincidental style-word overlaps that must NOT merge.
"""
from __future__ import annotations

from bcd_ingest.dedup import find_duplicate_merges, find_substring_merges

# The real cross-source neighbourhood: two true dupes + the Lagavulin/Buffalo-Trace style-word
# fan-in + the demo-IPA false hits.
CATALOG = [
    {"id": "bcd-demo:heady", "name": "Heady Topper", "brand": "Heady Topper"},
    {"id": "ttb:heady", "name": "The Alchemist Heady Topper", "brand": "The Alchemist"},
    {"id": "off:sn-pale", "name": "Pale Ale", "brand": "Sierra Nevada"},
    {"id": "ttb:sn-pale", "name": "Sierra Nevada Pale Ale", "brand": "Sierra Nevada"},
    {"id": "ttb:lagavulin", "name": "Lagavulin 16 Year Old Single Malt Scotch Whisky",
     "brand": "Lagavulin"},
    {"id": "off:ardbeg", "name": "10 y. o. whisky 46%", "brand": "ARDBEG"},
    {"id": "off:cardhu", "name": "Scotch Whisky 1824", "brand": "Cardhu"},
    {"id": "off:bowmore", "name": "Single Malt Scotch Whisky", "brand": "Bowmore"},
    {"id": "ttb:buffalo", "name": "Buffalo Trace Kentucky Straight Bourbon Whiskey",
     "brand": "Buffalo Trace"},
    {"id": "off:fourroses", "name": "Whiskey 40%", "brand": "FOUR ROSES"},
    {"id": "bcd-demo:wcr", "name": "West Coast Reference IPA", "brand": "West Coast Reference IPA"},
    {"id": "off:420", "name": "420 IPA", "brand": "Sweetwater Brewing Company"},
    {"id": "off:hefe", "name": "Hefeweizen", "brand": "Unknown"},
    {"id": "bcd-demo:bavhefe", "name": "Bavarian Hefeweizen", "brand": "Bavarian Hefeweizen"},
]


def test_finds_exactly_the_two_true_cross_source_dupes():
    merges = set(find_substring_merges(CATALOG))
    assert merges == {
        ("bcd-demo:heady", "ttb:heady"),   # brand echoes name -> compatible
        ("off:sn-pale", "ttb:sn-pale"),    # bare "Pale Ale" rescued by matching brand
    }


def test_rejects_style_word_fan_in():
    # Ardbeg / Cardhu / Bowmore / Four Roses share only "scotch/whisky/bourbon" with the TTB
    # bottles — a different distillery is a different product.
    merges = find_substring_merges(CATALOG)
    aliased = {a for a, _ in merges}
    for wrong in ("off:ardbeg", "off:cardhu", "off:bowmore", "off:fourroses"):
        assert wrong not in aliased


def test_rejects_brand_mismatch_and_bare_style():
    merges = find_substring_merges(CATALOG)
    aliased = {a for a, _ in merges}
    assert "off:420" not in aliased      # Sweetwater != West Coast Reference IPA
    assert "off:hefe" not in aliased     # bare "Hefeweizen", no brand -> nothing to anchor


def test_ambiguous_multiple_supersets_are_left_alone():
    # A bare brand covered by several distinct products has no unique canonical -> skip, never
    # collapse "Absolut" into whichever flavour sorts first.
    cat = [
        {"id": "off:absolut", "name": "Absolut", "brand": "Absolut"},
        {"id": "off:absolut-vodka", "name": "Absolut Vodka", "brand": "Absolut"},
        {"id": "off:absolut-razz", "name": "Absolut Raspberri", "brand": "Absolut"},
    ]
    assert find_substring_merges(cat, cross_source_only=False) == []


def test_cross_source_only_flag():
    # Same-source containment is out of scope by default; opening the flag still requires the guards.
    cat = [
        {"id": "off:a", "name": "Guinness", "brand": "Guinness"},
        {"id": "off:b", "name": "Guinness Draught Stout", "brand": "Guinness"},
    ]
    assert find_substring_merges(cat, cross_source_only=True) == []
    assert find_substring_merges(cat, cross_source_only=False) == [("off:a", "off:b")]


# ---- equivalence pass: same product, written differently ------------------------------
# Real rows from the OFF catalog. The traps are deliberate: identical names that are NOT the
# same beer, and generic names shared by rival producers.
EQUIV = [
    {"id": "off:hein-a", "name": "Heineken Cerveza", "brand": "Heineken"},
    {"id": "off:hein-b", "name": "Cerveza Heineken", "brand": "Heineken"},
    # same name, genuinely different breweries -> must stay apart
    {"id": "off:radler-a", "name": "Natur Radler", "brand": "Hoepfner"},
    {"id": "off:radler-b", "name": "Natur Radler", "brand": "Gösser"},
    # bare category + a brand-less row: adopting it would be a guess
    {"id": "off:irish-a", "name": "Irish whiskey", "brand": "Unknown"},
    {"id": "off:irish-b", "name": "Irish Whiskey", "brand": "Bushmills"},
    # bare category, but both carry the same brand punctuated differently -> a real dupe
    {"id": "off:lawson-a", "name": "Blended Scotch Whisky", "brand": "William Lawson's"},
    {"id": "off:lawson-b", "name": "blended scotch whisky", "brand": "william lawsons"},
    # rival producers of the same rhum category
    {"id": "off:rhum-a", "name": "Rhum blanc agricole", "brand": "Rhum J.M"},
    {"id": "off:rhum-b", "name": "Rhum blanc agricole", "brand": "HSE"},
    # a mis-scraped brand must not win the survivor slot
    {"id": "off:coors-a", "name": "Coors Light", "brand": "Coors"},
    {"id": "off:coors-b", "name": "Coors Light", "brand": "OLD TRAPPER"},
]


def _pairs(catalog=EQUIV):
    return set(find_duplicate_merges(catalog))


def test_merges_word_order_variants():
    assert ("off:hein-a", "off:hein-b") in _pairs() or ("off:hein-b", "off:hein-a") in _pairs()


def test_same_name_different_brewery_is_not_a_duplicate():
    """Hoepfner and Gösser both sell a "Natur Radler" and they are different beers."""
    merged = {a for a, _ in _pairs()} | {c for _, c in _pairs()}
    assert "off:radler-a" not in merged
    assert "off:radler-b" not in merged


def test_bare_category_name_does_not_adopt_a_brandless_row():
    """"Irish Whiskey" names a class; the unbranded row could be anyone's."""
    assert not {p for p in _pairs() if "irish" in p[0]}


def test_bare_category_merges_when_both_brands_are_the_same():
    assert ("off:lawson-a", "off:lawson-b") in _pairs() or \
           ("off:lawson-b", "off:lawson-a") in _pairs()


def test_rival_producers_of_a_category_stay_apart():
    assert not {p for p in _pairs() if "rhum" in p[0]}


def test_survivor_is_the_row_whose_brand_matches_its_name():
    """Preferring merely *having* a brand files Coors Light under a jerky company."""
    coors = [p for p in _pairs() if "coors" in p[0]]
    assert coors == [("off:coors-b", "off:coors-a")]


# ---- containment: the relaxation must not let a product collapse to its brand ---------

def test_age_statement_in_another_language_still_merges():
    cat = [
        {"id": "off:lag", "name": "Lagavulin 16 ans", "brand": "Isle of Islay"},
        {"id": "ttb:lag", "name": "Lagavulin 16 Year Old Single Malt Scotch Whisky",
         "brand": "Lagavulin"},
    ]
    assert set(find_substring_merges(cat)) == {("off:lag", "ttb:lag")}


def test_different_age_statements_are_different_bottles():
    cat = [
        {"id": "off:lag8", "name": "Lagavulin 8", "brand": "Lagavulin"},
        {"id": "ttb:lag16", "name": "Lagavulin 16 Year Old Single Malt Scotch", "brand": "Lagavulin"},
    ]
    assert find_substring_merges(cat) == []


def test_style_words_are_identity_not_noise():
    """Both traps the relaxation originally fell into: a product must not reduce to the
    brand/region word it shares with an unrelated sibling."""
    cat = [
        {"id": "ttb:snpale", "name": "Sierra Nevada Pale Ale", "brand": "Sierra Nevada"},
        {"id": "off:juicy", "name": "Juicy Little Thing Hazy IPA", "brand": "Sierra Nevada"},
        {"id": "bcd-demo:bavhefe", "name": "Bavarian Hefeweizen", "brand": "Bavarian Hefeweizen"},
        {"id": "off:bavamber", "name": "Bavarian Amber Lager", "brand": "Red Oak"},
    ]
    assert find_substring_merges(cat) == []
