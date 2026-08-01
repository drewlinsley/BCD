"""Schema invariants — the data model's guarantees, pinned."""

from __future__ import annotations

import pytest
from bcd_schema import (
    SENSORY_AXES,
    ExtractionMethod,
    IngredientRole,
    Provenance,
    RecipeGraph,
    RecipeIngredient,
    SensorySource,
    SensoryVector,
    Sourced,
)


def test_confidence_clamped_to_method_ceiling():
    # An inferred fact can never masquerade as a stated one.
    p = Provenance(
        source_id="x",
        method=ExtractionMethod.LLM_INFERRED_FROM_STYLE_PRIOR,
        confidence=0.99,
    )
    assert p.confidence == 0.4


def test_regulatory_filing_keeps_full_confidence():
    p = Provenance(source_id="ttb", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)
    assert p.confidence == 1.0


def test_sourced_generic_carries_value_and_provenance():
    abv = Sourced[float](
        value=8.0,
        provenance=Provenance(source_id="ttb", method=ExtractionMethod.REGULATORY_FILING,
                              confidence=1.0),
    )
    assert abv.value == 8.0
    assert abv.provenance.source_id == "ttb"


def test_sensory_vector_roundtrips_through_array():
    sv = SensoryVector(source=SensorySource.CHEMISTRY_PRIOR,
                       axes={"citrus": 0.8, "tropical": 0.9})
    arr = sv.to_array()
    assert len(arr) == len(SENSORY_AXES)
    rt = SensoryVector.from_array(arr, SensorySource.CHEMISTRY_PRIOR)
    assert rt.axes["tropical"] == pytest.approx(0.9)
    assert rt.axes["citrus"] == pytest.approx(0.8)


def test_recipe_completeness_signal():
    empty = RecipeGraph()
    assert empty.completeness == 0.0
    prov = Provenance(source_id="x", method=ExtractionMethod.STATED_BY_PRODUCER, confidence=1.0)
    rg = RecipeGraph(ingredients=[
        RecipeIngredient(role=IngredientRole.BASE_MALT, entity_kind="malt",
                         raw_name="Pilsner", provenance=prov),
        RecipeIngredient(role=IngredientRole.AROMA_HOP, entity_kind="hop",
                         raw_name="Citra", provenance=prov),
        RecipeIngredient(role=IngredientRole.YEAST, entity_kind="yeast",
                         raw_name="US-05", provenance=prov),
    ])
    assert rg.completeness == pytest.approx(0.75)  # malt+hop+yeast, no process steps
