"""SensoryVector — the bridge between 'what's in it' and 'what it tastes like'.

Learned two ways and reconciled:
  1. bottom-up from chemistry (hop oils, malt color, yeast esters) — needs NO reviews,
     so a brand-new release with a published hop bill gets a real score on day zero.
  2. top-down from the SNAP aspect-rated review corpora.

Axes follow the ASBC/EBC/MBAA beer flavor wheel (Meilgaard) plus spirit-relevant axes.
All values 0-1. This is the object we embed for similarity search and personalization.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SensorySource(str, Enum):
    CHEMISTRY_PRIOR = "chemistry_prior"  # derived from RecipeGraph + ingredient DB
    REVIEW_CONSENSUS = "review_consensus"  # extracted from reviews
    RECONCILED = "reconciled"  # weighted blend of the two
    STYLE_PRIOR = "style_prior"  # fallback: BJCP style centroid


# The canonical descriptor axes. Kept flat and stable — codegen and embeddings depend
# on this order, so append new axes at the END, never reorder.
SENSORY_AXES: tuple[str, ...] = (
    # aroma/flavor families
    "citrus",
    "tropical",
    "stone_fruit",
    "berry",
    "floral",
    "herbal",
    "piney_resinous",
    "grassy",
    "spicy_phenolic",
    "malty_bready",
    "caramel_toffee",
    "roasted_coffee_choc",
    "nutty",
    "vanilla_oak",
    "smoky_peat",
    "honey",
    "banana_ester",
    "funk_brett",
    "sour_tart",
    "sweet",
    # structure
    "bitterness",
    "body_fullness",
    "carbonation",
    "alcohol_warmth",
    "dryness_finish",
)


class SensoryVector(BaseModel):
    """Named axes 0-1. Serializes to a dense vector via `to_array` for pgvector."""

    source: SensorySource
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    axes: dict[str, float] = Field(default_factory=dict)

    def to_array(self) -> list[float]:
        """Dense vector in SENSORY_AXES order. Missing axes -> 0.0."""
        return [float(self.axes.get(a, 0.0)) for a in SENSORY_AXES]

    @classmethod
    def from_array(cls, arr: list[float], source: SensorySource,
                   confidence: float = 0.5) -> SensoryVector:
        return cls(
            source=source,
            confidence=confidence,
            axes={a: v for a, v in zip(SENSORY_AXES, arr, strict=False)},
        )
