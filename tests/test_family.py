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


# ---- aisles ---------------------------------------------------------------------------------

def test_every_shelf_is_in_exactly_one_aisle():
    """A shelf missing from `families_in` is one Discover can never reach, since the screen
    walks aisles and not families."""
    from bcd_schema.family import GROUPS, families_in
    seen = [f for g in GROUPS for f in families_in(g)]
    assert sorted(seen, key=FAMILIES.index) == list(FAMILIES)
    assert len(seen) == len(set(seen))


def test_beer_holds_the_beer_shelves():
    from bcd_schema.family import Group, families_in
    assert set(families_in(Group.BEER)) == {
        Family.IPA, Family.PALE_ALE, Family.LAGER, Family.STOUT,
        Family.PORTER, Family.WHEAT, Family.SOUR, Family.BELGIAN}


def test_spirits_holds_the_spirit_shelves():
    from bcd_schema.family import Group, families_in
    assert Family.GIN in families_in(Group.SPIRITS)
    assert Family.BOURBON in families_in(Group.SPIRITS)
    assert Family.IPA not in families_in(Group.SPIRITS)


def test_sake_and_cider_are_their_own_aisles():
    """Neither is a beer and neither is a spirit, and a drinker looking for sake is looking for
    sake -- not for the drawer of things that fitted nowhere."""
    from bcd_schema.family import Group, families_in
    assert Family.SAKE.group is Group.SAKE
    assert Family.CIDER.group is Group.CIDER
    assert families_in(Group.SAKE) == (Family.SAKE,)
    assert families_in(Group.CIDER) == (Family.CIDER,)


def test_an_aisle_keeps_the_browsing_order_of_its_shelves():
    from bcd_schema.family import Group, families_in
    beer = families_in(Group.BEER)
    assert beer[0] is Family.IPA
    assert list(beer) == sorted(beer, key=FAMILIES.index)


# ---- a style wearing a qualifier ------------------------------------------------------------

@pytest.mark.parametrize("style,category,expected", [
    ("Contemporary gin", "spirit", Family.GIN),
    ("Peated Scotch", "spirit", Family.SCOTCH),
    ("Blended Bourbon", "spirit", Family.BOURBON),
    ("Cognac VS", "spirit", Family.BRANDY),
    ("Tennessee whiskey", "spirit", Family.WHISKEY),
    ("Vodka 90-99 Proof", "spirit", Family.VODKA),
    ("Hazy IPA", "beer", Family.IPA),
    ("Barrel-aged imperial stout", "beer", Family.STOUT),
    ("Pale lager", "beer", Family.LAGER),
    ("Imperial porter", "beer", Family.PORTER),
])
def test_a_qualifier_does_not_hide_the_style(style, category, expected):
    """5,712 rows across 436 styles name a style the table knows, wearing one it does not.
    Measured on the live catalog 2026-09-30, every one of them on no shelf at all — and the
    failure hides itself, because the shelf a drinker rated on simply stays dark."""
    assert family_of(style, category) is expected


@pytest.mark.parametrize("style,expected", [
    ("Apricot Brandy", Family.BRANDY),
    ("Blackberry Flavored Brandy", Family.BRANDY),
    ("Cinnamon whisky liqueur", Family.LIQUEUR),
    ("Gin Liqueurs", Family.LIQUEUR),
])
def test_the_last_style_word_decides(style, expected):
    """English puts the head noun last. An apricot brandy is a brandy and a whisky liqueur is
    a liqueur, and reading the FIRST style word would file both under the ingredient."""
    assert family_of(style) is expected


@pytest.mark.parametrize("style,expected", [
    ("Fruited sour ale", Family.SOUR),            # `sour ale` over the `ale` marker
    ("Blended straight bourbon", Family.BOURBON),
    ("Dry Irish stout", Family.STOUT),
    ("Fruit wheat ale", Family.WHEAT),            # `wheat ale` over `ale`
])
def test_the_longest_phrase_ending_last_wins(style, expected):
    assert family_of(style) is expected


def test_a_bare_class_name_is_still_no_family():
    """The guard that makes the whole fallback safe: a phrase spanning the WHOLE style is
    ignored, so `Ale` stays shelfless while `Scotch ale` reaches the ale shelf. Without it the
    marker would put 98,440 unknown beers on a shelf in one line."""
    for style in ("Ale", "ale", "  ALE  ", "Beer", "Malt Beverage", "Specialty", "Liqueur"):
        assert family_of(style, "beer") is None, style
    assert family_of("Scotch ale", "beer") is Family.PALE_ALE


@pytest.mark.parametrize("style", [
    "Ginger Beer",          # gin
    "Butterscotch sauce",   # scotch
    "Original recipe",      # gin
    "Palest shade",         # pale ale
])
def test_a_style_word_inside_another_word_is_not_that_style(style):
    """The cask-finish pass learned this the expensive way: `gin` lives inside `ginger` and
    `original`, `oro` inside `oloroso`. A guess is exactly where that would happen again."""
    assert family_of(style) is None


def test_the_rows_own_category_can_reject_a_guess_but_never_make_one():
    """A beer whose style trails off into a cask is still a beer. The exact table stays
    category-blind on purpose — 620 Tequila rows are filed under category `other` — but a
    guess does not get that benefit."""
    assert family_of("Stout aged in bourbon") is Family.BOURBON          # last word, no category
    assert family_of("Stout aged in bourbon", "beer") is Family.STOUT    # ...the aisle says no
    assert family_of("Tequila", "other") is Family.AGAVE                 # exact: unguarded


# ---- which spellings fill a shelf -----------------------------------------------------------

def test_shelf_styles_reads_the_spellings_the_catalog_really_holds():
    """`styles_in` names what somebody wrote in the table; the shelf query has to match what is
    in the catalog. Otherwise the Scotch shelf calls itself yours the moment you rate a
    `Peated Scotch`, and then does not contain one."""
    from bcd_schema.family import shelf_styles
    found = shelf_styles([("spirit", "Peated Scotch"), ("spirit", "Single Malt Scotch"),
                          ("beer", "Hazy IPA"), ("beer", "Ale"), (None, "")])
    assert found[Family.SCOTCH] == ["peated scotch", "single malt scotch"]
    assert found[Family.IPA] == ["hazy ipa"]
    assert Family.PALE_ALE not in found


def test_shelf_styles_drops_a_spelling_two_aisles_both_claim():
    """The query filters on the style alone, so one spelling cannot be a beer on one shelf and
    a whisky on another. Dropping it leaves a shelf short; keeping it puts the wrong drink on
    somebody's shelf, which is worse."""
    from bcd_schema.family import shelf_styles
    found = shelf_styles([("beer", "Scotch ale"), ("spirit", "Scotch ale")])
    assert found == {}


# ---- the label spells it out ---------------------------------------------------------------

@pytest.mark.parametrize("style", [
    "India Pale Ale",
    "American India Pale Ale",
    "Double India Pale Ale",
    "AMERICAN DOUBLE INDIA PALE ALE",     # read verbatim off a can of Crusher
    "Imperial India Pale Ale",
    "Session India Pale Ale",
    "New England India Pale Ale",
])
def test_a_longhand_ipa_is_an_ipa_not_a_pale_ale(style):
    """A catalog writes `Double IPA`; a CAN prints `AMERICAN DOUBLE INDIA PALE ALE`. Until these
    were claimed the rightmost match in that line was `pale ale`, so every spelled-out IPA on a
    label came back a Pale Ale -- wrong shelf, and wrong answer for anything comparing what a
    label says against what the catalog holds."""
    assert family_of(style) is Family.IPA


@pytest.mark.parametrize("style,family", [
    ("Pale Ale", Family.PALE_ALE),
    ("American Pale Ale", Family.PALE_ALE),
    ("English Pale Ale", Family.PALE_ALE),
    ("India Pale Lager", Family.LAGER),      # the head noun is still the lager
])
def test_claiming_the_longhand_did_not_swallow_the_pale_ales(style, family):
    """The new claims end where `pale ale` ends and are longer, so they win there -- which is
    exactly what must NOT happen to a plain pale ale."""
    assert family_of(style) is family
