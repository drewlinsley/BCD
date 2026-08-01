"""Core catalog entities: Producer -> Brand -> Product -> Release -> SKU.

This is the resolved 'gold' shape. Every field that isn't a stable identifier is a
`Sourced[...]` so the app can show where each fact came from. Entity resolution merges
many silver-layer observations into one of these; `match_evidence` keeps merges auditable.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .provenance import Provenance, Sourced
from .recipe import RecipeGraph
from .sensory import SensoryVector


class Category(str, Enum):
    BEER = "beer"
    CIDER = "cider"
    WINE = "wine"
    SPIRIT = "spirit"
    RTD = "rtd"  # ready-to-drink / seltzer
    MEAD = "mead"
    SAKE = "sake"
    OTHER = "other"


class ContainerType(str, Enum):
    DRAFT = "draft"
    CAN = "can"
    BOTTLE = "bottle"
    CROWLER = "crowler"
    GROWLER = "growler"
    CASK = "cask"


class Producer(BaseModel):
    id: str
    name: str
    kind: str | None = None  # brewery | distillery | winery | cidery | contract
    country: str | None = None
    region: str | None = None
    lat: float | None = None
    lon: float | None = None
    parent_company: Sourced[str] | None = None  # who really owns it (ABI, Constellation)
    ttb_permit: str | None = None
    aliases: list[str] = Field(default_factory=list)
    website: str | None = None


class Brand(BaseModel):
    id: str
    producer_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)


class ProductSpec(BaseModel):
    """Observable facts. Everything sourced; everything nullable."""

    abv_pct: Sourced[float] | None = None
    ibu: Sourced[float] | None = None
    srm: Sourced[float] | None = None  # beer color
    og: Sourced[float] | None = None  # original gravity
    fg: Sourced[float] | None = None  # final gravity
    # spirit-specific
    proof: Sourced[float] | None = None
    age_statement_years: Sourced[float] | None = None
    entry_proof: Sourced[float] | None = None
    still_type: Sourced[str] | None = None  # pot | column | hybrid
    mash_bill_summary: Sourced[str] | None = None


class Product(BaseModel):
    """A named product line, e.g. 'Heady Topper'. Releases hang off it."""

    id: str
    brand_id: str
    producer_id: str
    category: Category
    name: str
    style: Sourced[str] | None = None  # 'New England IPA', 'Kentucky Straight Bourbon'
    style_bjcp_code: str | None = None
    spec: ProductSpec = Field(default_factory=ProductSpec)
    recipe: RecipeGraph = Field(default_factory=RecipeGraph)
    sensory: SensoryVector | None = None
    description: Sourced[str] | None = None
    aliases: list[str] = Field(default_factory=list)


class Release(BaseModel):
    """A specific batch/vintage/edition of a Product."""

    id: str
    product_id: str
    name: str | None = None  # 'Batch 3', '2026 Vintage', 'Barrel Pick — K&L'
    year: int | None = None
    limited: bool = False
    spec_overrides: ProductSpec = Field(default_factory=ProductSpec)


class SKU(BaseModel):
    """A purchasable unit. The thing a barcode or tap handle resolves to."""

    id: str
    product_id: str
    release_id: str | None = None
    container: ContainerType
    volume_ml: float | None = None
    pack_size: int | None = None
    upc: str | None = None
    gtin: str | None = None


class MatchEvidence(BaseModel):
    """Why two observations were merged into one entity. Makes ER reversible."""

    method: str  # 'exact_upc' | 'trigram+embedding' | 'llm_adjudication' | 'human'
    score: float
    left_ref: str
    right_ref: str
    note: str | None = None
    provenance: Provenance | None = None


class ResolvedProduct(BaseModel):
    """What the API returns from /v1/scan and /v1/product — a fully denormalized view."""

    product: Product
    producer: Producer
    brand: Brand
    releases: list[Release] = Field(default_factory=list)
    skus: list[SKU] = Field(default_factory=list)
    match_evidence: list[MatchEvidence] = Field(default_factory=list)
