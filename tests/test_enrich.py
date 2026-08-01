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
