"""Substring / cross-source entity resolution — high precision, no false merges.

Cases are the real catalog pairs the audit surfaced: two true cross-source dupes among twelve
containments, the rest coincidental style-word overlaps that must NOT merge.
"""
from __future__ import annotations

from bcd_ingest.dedup import find_substring_merges

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
