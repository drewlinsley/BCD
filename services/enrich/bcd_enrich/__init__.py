"""Enrichment — turns a RecipeGraph into a SensoryVector WITHOUT reviews.

This is the cold-start moat in code. The production version reconciles a learned
ingredient->sensory model (trained on the SNAP aspect-rated corpora) with the chemistry
priors below. Here we ship the transparent chemistry half: it already produces a real,
defensible sensory estimate for a brand-new release from its published hop/malt bill.
"""

from __future__ import annotations

from bcd_schema import (
    IngredientRole,
    RecipeGraph,
    SensorySource,
    SensoryVector,
)

# Hop canonical_id -> aroma axis contributions (0-1), from oil composition + descriptors.
# A stand-in for the ingredient DB join; real code looks these up from the Hop entities.
_HOP_AROMA: dict[str, dict[str, float]] = {
    "citra": {"citrus": 0.9, "tropical": 0.9, "stone_fruit": 0.4},
    "mosaic": {"tropical": 0.8, "berry": 0.6, "citrus": 0.6, "piney_resinous": 0.3},
    "simcoe": {"piney_resinous": 0.8, "citrus": 0.5, "stone_fruit": 0.4},
    "cascade": {"citrus": 0.7, "floral": 0.5, "grassy": 0.4},
    "galaxy": {"tropical": 0.95, "citrus": 0.6, "stone_fruit": 0.5},
    "saaz": {"herbal": 0.7, "floral": 0.6, "grassy": 0.4},
    "centennial": {"citrus": 0.8, "floral": 0.4, "piney_resinous": 0.4},
}

_ROAST_MALT_KEYWORDS = {
    "roasted_coffee_choc": ("roast", "chocolate", "black", "coffee"),
    "caramel_toffee": ("crystal", "caramel", "cara", "toffee"),
    "malty_bready": ("pilsner", "pale", "vienna", "munich", "wheat", "maris"),
    "nutty": ("biscuit", "victory", "brown"),
}


def sensory_from_recipe(recipe: RecipeGraph, *, style_hint: str | None = None) -> SensoryVector:
    """Bottom-up chemistry prior. Confidence scales with recipe completeness."""
    axes: dict[str, float] = {}

    def bump(axis: str, amount: float) -> None:
        axes[axis] = min(1.0, axes.get(axis, 0.0) + amount)

    hop_count = 0
    for ing in recipe.ingredients:
        name = ing.raw_name.lower()
        role = ing.role
        if "HOP" in role.name or role in (IngredientRole.AROMA_HOP, IngredientRole.DRY_HOP):
            hop_count += 1
            key = next((h for h in _HOP_AROMA if h in name), None)
            if key:
                for axis, v in _HOP_AROMA[key].items():
                    # dry/aroma hops drive perceived aroma harder than bittering
                    aroma = role in (IngredientRole.DRY_HOP, IngredientRole.AROMA_HOP)
                    bump(axis, v * (0.9 if aroma else 0.4))
            if role == IngredientRole.BITTERING_HOP:
                bump("bitterness", 0.3)
        elif "MALT" in role.name or role == IngredientRole.BASE_MALT:
            for axis, kws in _ROAST_MALT_KEYWORDS.items():
                if any(k in name for k in kws):
                    bump(axis, 0.5)
        elif role == IngredientRole.YEAST:
            if "hefe" in name or "wit" in name or "weizen" in name:
                bump("banana_ester", 0.6)
                bump("spicy_phenolic", 0.5)
            if "brett" in name or "sour" in name:
                bump("funk_brett", 0.7)
                bump("sour_tart", 0.5)
        elif role == IngredientRole.FRUIT:
            bump("berry", 0.5)
            bump("sweet", 0.3)

    # Hop-forward beers read as more bitter and dry.
    if hop_count >= 2:
        bump("bitterness", 0.3)
        bump("dryness_finish", 0.3)

    confidence = 0.3 + 0.5 * recipe.completeness
    return SensoryVector(source=SensorySource.CHEMISTRY_PRIOR,
                         confidence=round(confidence, 2), axes=axes)


__all__ = ["sensory_from_recipe"]
