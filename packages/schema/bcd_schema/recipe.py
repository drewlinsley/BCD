"""RecipeGraph — the recoverable ingredient list + production process.

This is the structure the user described: 'a data structure that indexes each
individual part of the ingredient list and recipe and aging process.' Quantities are
usually null and that is fine — role, timing, and the canonical entity reference carry
most of the flavor signal, and every part is independently sourced.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .provenance import Provenance


class IngredientRole(str, Enum):
    BASE_MALT = "base_malt"
    SPECIALTY_MALT = "specialty_malt"
    BITTERING_HOP = "bittering_hop"
    FLAVOR_HOP = "flavor_hop"
    AROMA_HOP = "aroma_hop"
    DRY_HOP = "dry_hop"
    YEAST = "yeast"
    WATER_SALT = "water_salt"
    ADJUNCT = "adjunct"
    FRUIT = "fruit"
    SPICE = "spice"  # + botanicals
    BARREL = "barrel"
    MASH_GRAIN = "mash_grain"  # spirits mash bill component
    FINING = "fining"
    OTHER = "other"


class TimingPhase(str, Enum):
    MASH = "mash"
    FIRST_WORT = "first_wort"
    BOIL = "boil"
    WHIRLPOOL = "whirlpool"
    PRIMARY_FERMENT = "primary_ferment"
    DRY_HOP = "dry_hop"
    CONDITIONING = "conditioning"
    DISTILLATION = "distillation"
    AGING = "aging"
    PACKAGING = "packaging"


class Timing(BaseModel):
    phase: TimingPhase
    minutes_from_phase_start: float | None = None  # e.g. boil@60 -> 60
    day: int | None = None  # dry-hop day 4
    month: int | None = None  # aging month 18


class RecipeIngredient(BaseModel):
    """One indexed part of the ingredient list, pointing at a canonical entity."""

    role: IngredientRole
    entity_kind: str  # 'hop' | 'malt' | 'yeast' | ...
    entity_ref: str | None = None  # canonical_id once resolved; None while pending
    raw_name: str  # as written on the source ("Citra (US)")
    quantity: float | None = None
    unit: str | None = None  # g, kg, lb, %, oz, IBU-contribution
    percent_of_bill: float | None = None  # for grain/mash bills
    timing: Timing | None = None
    provenance: Provenance


class ProcessStep(BaseModel):
    """One step of the production process, ordered by `sequence`."""

    sequence: int
    phase: TimingPhase
    description: str
    temp_c: float | None = None
    duration_minutes: float | None = None
    # barrel/aging specifics (nullable — only present for the relevant phase)
    barrel_ref: str | None = None  # canonical barrel id
    warehouse_position: str | None = None  # rickhouse floor/row — affects the spirit
    provenance: Provenance


class RecipeGraph(BaseModel):
    """The full decomposed recipe for one product. Sparse by design."""

    ingredients: list[RecipeIngredient] = Field(default_factory=list)
    process_steps: list[ProcessStep] = Field(default_factory=list)

    @property
    def completeness(self) -> float:
        """Rough 0-1 signal of how much recipe we actually recovered.

        Used to prioritize enrichment and to temper recommendation confidence.
        """
        signals = 0
        if any(i.role.name.endswith("MALT") or i.role == IngredientRole.MASH_GRAIN
               for i in self.ingredients):
            signals += 1
        if any("HOP" in i.role.name for i in self.ingredients):
            signals += 1
        if any(i.role == IngredientRole.YEAST for i in self.ingredients):
            signals += 1
        if self.process_steps:
            signals += 1
        return signals / 4.0
