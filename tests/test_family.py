"""Drink families — the shelf a drinker browses, between Category and style."""

from __future__ import annotations

import pytest
from bcd_schema.family import FAMILIES, Family, family_of, styles_in


def test_the_three_spellings_of_bourbon_are_one_shelf():
    """The catalog files 8,098 rows as `Straight Bourbon`, 4,590 as `Bourbon` and 216 as
    `Kentucky straight bourbon`. A drinker looking for bourbon wants all of them."""
    assert {family_of(s) for s in ("Bourbon", "Straight Bourbon", "Kentucky straight bourbon")} \
        == {Family.BOURBON}


def test_a_regulators_class_name_names_no_family():
    """`Ale` is 96,751 rows and `Flavored Malt Beverage` is 60,765 — TTB classes, which say a
    filing was for a malt beverage and nothing about what is in the can. Shelving them would
    put a hundred thousand unknown beers under a name none of them earned."""
    for style in ("Ale", "Beer", "Flavored Malt Beverage", "Specialty", "Malt Beverage"):
        assert family_of(style) is None, style


def test_no_family_is_not_some_other_family():
    assert family_of(None) is None
    assert family_of("") is None
    assert family_of("a style nobody has ever filed") is None


def test_the_style_decides_not_the_category():
    """620 Tequila rows, 363 Bourbon rows and 512 Amaro rows sit under category `other`. A
    drinker looking for tequila should find those, so the table never consults the category."""
    assert family_of("Tequila") is Family.AGAVE
    assert family_of("Amaro") is Family.AMARO


def test_case_and_padding_do_not_make_a_new_style():
    """The catalog holds both `Flavored Vodka` (4,700) and `Flavored vodka` (437)."""
    assert family_of("Flavored vodka") is family_of("  FLAVORED VODKA  ") is Family.VODKA


def test_scotch_and_irish_and_bourbon_are_not_filed_under_whiskey():
    """`Whiskey` is the remainder, not the parent. Someone asking for scotch is not asking for
    Canadian blends, which is the whole reason this sits below Category."""
    assert family_of("Single Malt Scotch") is Family.SCOTCH
    assert family_of("Irish Whiskey") is Family.IRISH
    assert family_of("Rye Whiskey") is Family.RYE
    assert family_of("Blended Whiskey") is Family.WHISKEY


def test_a_style_belongs_to_exactly_one_family():
    """Enforced when the table is built — this asserts the guard is really there, since a style
    claimed twice would silently take whichever claim ran last."""
    seen: dict[str, Family] = {}
    for family in FAMILIES:
        for style in styles_in(family):
            assert style not in seen, f"{style} in {family} and {seen[style]}"
            seen[style] = family


def test_every_family_can_name_its_styles():
    for family in FAMILIES:
        styles = styles_in(family)
        assert styles, family
        assert all(family_of(s) is family for s in styles)


def test_every_family_is_listed_in_the_browsing_order():
    """`FAMILIES` is what an answer iterates. A member missing from it is a shelf that exists
    and is never shown."""
    assert set(FAMILIES) == set(Family)


def test_every_family_says_its_name_out_loud():
    for family in FAMILIES:
        assert family.label and family.label[0].isupper()


@pytest.mark.parametrize("style,expected", [
    ("New England IPA", Family.IPA),
    ("West Coast IPA", Family.IPA),
    ("Pilsner", Family.LAGER),
    ("Helles", Family.LAGER),
    ("Oatmeal Stout", Family.STOUT),
    ("Saison", Family.BELGIAN),
    ("Mezcal", Family.AGAVE),
    ("Cognac", Family.BRANDY),
    ("London Dry Gin", Family.GIN),
    ("Triple Sec", Family.LIQUEUR),
])
def test_the_catalogs_biggest_styles_land_where_a_drinker_would_look(style, expected):
    assert family_of(style) is expected
