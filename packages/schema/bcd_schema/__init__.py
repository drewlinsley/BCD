"""BCD canonical schema — the single source of truth for the data model.

Import from here; submodule layout may change but this surface is stable.
"""

from .api import (
    DetectedText,
    ProductSearchResponse,
    ScanResolveRequest,
    ScanResolveResponse,
    ScoredCandidate,
)
from .entities import (
    SKU,
    Brand,
    Category,
    ContainerType,
    MatchEvidence,
    Producer,
    Product,
    ProductSpec,
    Release,
    ResolvedProduct,
)
from .ingredients import (
    Barrel,
    GenericIngredient,
    Hop,
    HopOilFractions,
    IngredientKind,
    Malt,
    WaterProfile,
    Yeast,
)
from .profile import TasteProfile, WeeklyPrediction, WeeklyProfileDelta
from .provenance import (
    METHOD_CONFIDENCE_CEILING,
    ExtractionMethod,
    Provenance,
    Sourced,
)
from .recipe import (
    IngredientRole,
    ProcessStep,
    RecipeGraph,
    RecipeIngredient,
    Timing,
    TimingPhase,
)
from .sensory import SENSORY_AXES, SensorySource, SensoryVector
from .venue import Menu, MenuItem, Venue, VenueKind

__all__ = [
    "SENSORY_AXES",
    "METHOD_CONFIDENCE_CEILING",
    "Barrel",
    "Brand",
    "Category",
    "ContainerType",
    "DetectedText",
    "ExtractionMethod",
    "GenericIngredient",
    "Hop",
    "HopOilFractions",
    "IngredientKind",
    "IngredientRole",
    "Malt",
    "MatchEvidence",
    "Menu",
    "MenuItem",
    "ProcessStep",
    "Producer",
    "Product",
    "ProductSearchResponse",
    "ProductSpec",
    "Provenance",
    "RecipeGraph",
    "RecipeIngredient",
    "Release",
    "ResolvedProduct",
    "SKU",
    "ScanResolveRequest",
    "ScanResolveResponse",
    "ScoredCandidate",
    "Sourced",
    "SensorySource",
    "SensoryVector",
    "TasteProfile",
    "Timing",
    "TimingPhase",
    "Venue",
    "VenueKind",
    "WaterProfile",
    "WeeklyPrediction",
    "WeeklyProfileDelta",
    "Yeast",
]
