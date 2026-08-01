"""Canonical ingredient entities — the indexed 'parts' the recipe graph points at.

The product thesis: decompose every drink into indexed parts, attach provenance to
each, then prompt models with that structure. These classes are those parts. They are
shared, deduplicated entities (one 'Citra' row, referenced by thousands of beers), each
carrying the chemistry that lets us predict flavor with zero reviews.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IngredientKind(str, Enum):
    HOP = "hop"
    MALT = "malt"
    YEAST = "yeast"
    ADJUNCT = "adjunct"  # sugar, corn, rice, oats, lactose...
    FRUIT = "fruit"
    SPICE = "spice"  # incl. botanicals for gin
    WATER_PROFILE = "water_profile"
    BARREL = "barrel"
    GRAIN = "grain"  # for spirits mash bills (corn, rye, wheat, barley)
    OTHER = "other"


class HopOilFractions(BaseModel):
    """Fractions of total oil, 0-1. The single strongest predictor of hop aroma."""

    myrcene: float | None = None  # resinous, green, citrus
    humulene: float | None = None  # noble, woody, herbal
    caryophyllene: float | None = None  # peppery, spicy
    farnesene: float | None = None  # floral, green
    linalool: float | None = None  # floral, citrus — key aroma marker
    geraniol: float | None = None  # rose, geranium — biotransforms to citronellol
    b_pinene: float | None = None  # pine


class Hop(BaseModel):
    kind: IngredientKind = IngredientKind.HOP
    canonical_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    breeder: str | None = None
    purpose: str | None = None  # bittering | aroma | dual
    alpha_acid_pct: float | None = None
    beta_acid_pct: float | None = None
    cohumulone_pct: float | None = None  # of alpha; lower = smoother bitterness
    total_oil_ml_per_100g: float | None = None
    oils: HopOilFractions | None = None
    thiol_precursors: bool | None = None  # 3MH/4MMP potential — tropical, "juicy"
    descriptors: list[str] = Field(default_factory=list)
    substitutes: list[str] = Field(default_factory=list)  # canonical_ids


class Malt(BaseModel):
    kind: IngredientKind = IngredientKind.MALT
    canonical_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    maltster: str | None = None
    grain: str | None = None  # barley, wheat, rye, oats
    malt_type: str | None = None  # base | crystal | roasted | specialty
    color_lovibond: float | None = None
    color_ebc: float | None = None
    extract_pct: float | None = None  # potential extract, fine grind
    diastatic_power_lintner: float | None = None
    descriptors: list[str] = Field(default_factory=list)


class Yeast(BaseModel):
    kind: IngredientKind = IngredientKind.YEAST
    canonical_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    lab: str | None = None  # White Labs, Wyeast, Lallemand...
    strain_code: str | None = None  # WLP001, US-05...
    species: str | None = None  # S. cerevisiae, S. pastorianus, Brett...
    attenuation_pct_min: float | None = None
    attenuation_pct_max: float | None = None
    temp_c_min: float | None = None
    temp_c_max: float | None = None
    flocculation: str | None = None  # low | medium | high
    phenolic_pof: bool | None = None  # POF+ => clove/pepper (hefeweizen, saison)
    alcohol_tolerance_pct: float | None = None
    ester_descriptors: list[str] = Field(default_factory=list)  # banana, pear, apple


class Barrel(BaseModel):
    """The core of spirits + barrel-aged beer provenance."""

    kind: IngredientKind = IngredientKind.BARREL
    canonical_id: str
    name: str
    wood: str | None = None  # American/French/Mizunara oak...
    prior_fill: str | None = None  # virgin | ex-bourbon | ex-sherry (oloroso/PX)...
    char_level: int | None = Field(None, ge=1, le=4)  # cooperage char #1-#4
    toast_level: str | None = None
    size_liters: float | None = None
    cooperage: str | None = None
    seasoning_months: int | None = None
    descriptors: list[str] = Field(default_factory=list)


class WaterProfile(BaseModel):
    kind: IngredientKind = IngredientKind.WATER_PROFILE
    canonical_id: str
    name: str  # "Burton-on-Trent", "Pilsen"
    calcium_ppm: float | None = None
    sulfate_ppm: float | None = None
    chloride_ppm: float | None = None
    sulfate_chloride_ratio: float | None = None  # >2 = hop-forward, <1 = malt-forward
    bicarbonate_ppm: float | None = None


class GenericIngredient(BaseModel):
    """Fruit, spice, botanical, adjunct, spirit grain — the long tail."""

    kind: IngredientKind
    canonical_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    descriptors: list[str] = Field(default_factory=list)
