"""Enrichment — a published hop bill yields a real sensory estimate with zero reviews.
This is the cold-start moat; if it breaks, day-zero scoring breaks."""

from __future__ import annotations

from bcd_enrich import sensory_from_recipe
from bcd_schema import (
    ExtractionMethod,
    IngredientRole,
    Provenance,
    RecipeGraph,
    RecipeIngredient,
    SensorySource,
)


def _ing(name, role):
    return RecipeIngredient(
        role=role, entity_kind="hop", raw_name=name,
        provenance=Provenance(source_id="producer",
                              method=ExtractionMethod.STATED_BY_PRODUCER, confidence=1.0),
    )


def test_citra_mosaic_neipa_reads_tropical_citrusy():
    recipe = RecipeGraph(ingredients=[
        _ing("Citra", IngredientRole.DRY_HOP),
        _ing("Mosaic", IngredientRole.DRY_HOP),
        _ing("Pilsner malt", IngredientRole.BASE_MALT),
    ])
    sv = sensory_from_recipe(recipe)
    assert sv.source == SensorySource.CHEMISTRY_PRIOR
    # The differentiator: a real estimate, no reviews involved.
    assert sv.axes["tropical"] > 0.6
    assert sv.axes["citrus"] > 0.5
    assert sv.confidence > 0.3


def test_roasty_stout_reads_coffee_chocolate():
    recipe = RecipeGraph(ingredients=[
        _ing("Roasted barley", IngredientRole.BASE_MALT),
        _ing("Chocolate malt", IngredientRole.SPECIALTY_MALT),
    ])
    sv = sensory_from_recipe(recipe)
    assert sv.axes.get("roasted_coffee_choc", 0) > 0.3


def test_empty_recipe_low_confidence():
    sv = sensory_from_recipe(RecipeGraph())
    assert sv.confidence <= 0.35  # nothing to go on


# ---- the registry's class codes ----------------------------------------------------------------

from bcd_enrich.style_prior import (  # noqa: E402
    class_style,
    detect_style,
    normalize_class,
    readable_style,
    sensory_from_style,
)


def test_a_bottling_suffix_is_not_part_of_the_class():
    """TTB says "Scotch Whisky Fb" for foreign-bottled Scotch and "Tequila Usb" for US-bottled
    tequila; the suffix is paperwork, not style."""
    assert normalize_class("Scotch Whisky Fb") == normalize_class("Scotch Whisky Usb") == "scotch whisky"
    assert normalize_class("Bitters - Beverage*") == "bitters - beverage"


def test_the_registrys_classes_name_styles():
    for code, style in [
        ("Straight Bourbon Whisky", "bourbon"), ("Single Malt Scotch Whisky", "scotch"),
        ("Other Rum Gold Usb", "aged_rum"), ("Virgin Islands Rum (white)", "white_rum"),
        ("Tequila Fb", "tequila"), ("Mezcal", "mezcal"), ("Cognac (brandy) Fb", "brandy"),
        ("Vodka - Other Flavored", "flavored_vodka"), ("Vodka 80-89 Proof", "vodka"),
        ("Sake - Imported", "sake"), ("Cereal Beverages - Near Beer (non Alcoholic)", "na_beer"),
        ("Malt Beverages Specialities - Flavored", "flavored_malt"), ("Malt Liquor", "malt_liquor"),
        ("Coffee (cafe) Liqueur", "coffee_liqueur"), ("Creme De Menthe Green", "mint_liqueur"),
        ("Other (herbs & Seeds)", "herbal_liqueur"), ("Bitters - Beverage*", "bitters"),
        ("Stout", "stout"), ("Ale", "ale"), ("Beer", "beer"),
    ]:
        assert class_style(code) == style, code
    assert class_style(None) is None and class_style("") is None


def test_a_specific_class_decides_and_a_generic_one_defers_to_the_name():
    """`Absolut Citron` filed as a flavored vodka is a flavored vodka whatever the brand keyword
    makes of it; a double IPA filed as "Ale" is a double IPA; a Campari filed under the
    registry's specialties bucket is what its name says."""
    assert detect_style("Absolut Citron", "spirit", "Vodka - Other Flavored") == "flavored_vodka"
    assert detect_style("Here There Be Dragons Hazy Double India Pale Ale", "beer", "Ale") == "neipa"
    assert detect_style("Dino Break Double India Pale Ale", "beer", "Ale") == "dipa"
    assert detect_style("Campari", "other", "Other Specialties & Proprietaries") == "amaro"
    assert detect_style("Ramazzotti Aperitivo Rosato", "other", "Other Specialties & Proprietaries") == "amaro"
    assert detect_style("Twisted Tea Original", "beer", "Malt Beverages Specialities - Flavored") == "flavored_malt"
    assert detect_style("Blue Moon Belgian White", "beer", "Malt Beverages Specialities - Flavored") == "wheat"
    assert detect_style("Goslings Black Seal", "spirit", "Other Rum Gold Usb") == "aged_rum"
    assert detect_style("Junmai Hachitsuru", "sake", "Sake - Imported") == "sake"
    assert detect_style("Anything", "other", None) == "spirit"


def test_the_class_reads_as_a_person_would_say_it():
    """"Other Rum Gold Usb" on a bottle of Gosling's was printed as the style; reported from
    the camera as "should be ID'd as rum" (2026-09-16)."""
    assert readable_style("Other Rum Gold Usb") == "Gold Rum"
    assert readable_style("Straight Bourbon Whisky") == "Straight Bourbon"
    assert readable_style("Cognac (brandy) Fb") == "Cognac"
    assert readable_style("Whisky Specialties") == "Whisky"
    assert readable_style("Vodka 80-89 Proof Fb") == "Vodka"
    assert readable_style("Other Specialties & Proprietaries") == "Specialty"
    # a generic class prints the style the name detects; a specific one keeps its own words
    assert readable_style("Ale", detected="ipa") == "IPA"
    assert readable_style("Ale", detected="ale") == "Ale"
    assert readable_style("Malt Beverages Specialities - Flavored", detected="lager") == "Lager"
    assert readable_style("Straight Bourbon Whisky", detected="bourbon") == "Straight Bourbon"
    assert readable_style("Made Up Class 2099") == "Made Up Class 2099"


def test_every_registry_row_gets_a_vector_and_near_beer_no_warmth():
    sv = sensory_from_style("Some Distillery Reserve", "spirit", style_hint="Straight Bourbon Whisky")
    assert sv is not None and sv.axes["vanilla_oak"] > 0.5
    na = sensory_from_style("Some Brewery Free", "beer",
                            style_hint="Cereal Beverages - Near Beer (non Alcoholic)")
    assert na is not None and na.axes["alcohol_warmth"] == 0.0
    assert sensory_from_style("Junmai", "sake", style_hint="Sake - Imported") is not None
    assert sensory_from_style("Munch-n-pump Strawberry", "other",
                              style_hint="Other Specialties & Proprietaries") is not None
