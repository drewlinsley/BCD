"""Venue + menu — the bar/store side. A scan happens *at* a venue, and the venue's
known menu is the context that makes cold-path resolution fast and cheap.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .provenance import Provenance


class VenueKind(str, Enum):
    BAR = "bar"
    BREWERY_TAPROOM = "brewery_taproom"
    BOTTLE_SHOP = "bottle_shop"
    LIQUOR_STORE = "liquor_store"
    GROCERY = "grocery"
    RESTAURANT = "restaurant"


class Venue(BaseModel):
    id: str
    name: str
    kind: VenueKind
    lat: float | None = None
    lon: float | None = None
    address: str | None = None
    osm_id: str | None = None
    website: str | None = None


class MenuItem(BaseModel):
    product_ref: str | None = None  # resolved product id, or None if unresolved
    raw_name: str
    container: str | None = None
    price: float | None = None
    price_currency: str = "USD"
    size: str | None = None
    provenance: Provenance


class Menu(BaseModel):
    venue_id: str
    kind: str  # 'draft' | 'bottle_can' | 'full'
    items: list[MenuItem] = Field(default_factory=list)
    captured_at: str | None = None
