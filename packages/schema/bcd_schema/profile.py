"""TasteProfile — the personalization state that evolves weekly.

Updated by a per-user job: the week's interactions -> an LLM writes a natural-language
taste memo -> this structured object is updated -> a 'your week' card renders with 3
falsifiable predictions. Each confirmed/denied prediction is a training label.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .sensory import SensoryVector


class TasteProfile(BaseModel):
    user_id: str
    version: int = 0
    updated_at: str | None = None

    # affinities: canonical_id/style/axis -> weight in [-1, 1]
    ingredient_affinities: dict[str, float] = Field(default_factory=dict)
    style_affinities: dict[str, float] = Field(default_factory=dict)

    sensory_ideal: SensoryVector | None = None  # the taste centroid we recommend toward

    abv_band_min: float | None = None
    abv_band_max: float | None = None
    price_sensitivity: float | None = None  # 0 = insensitive, 1 = very
    novelty_appetite: float | None = None  # 0 = comfort picks, 1 = always new

    memo: str | None = None  # human-readable summary, shown on the weekly card


class WeeklyPrediction(BaseModel):
    """A falsifiable claim shown on the weekly card. Confirm/deny -> a training label."""

    text: str  # "You'll rate a hazy IPA over 4.0 this week"
    kind: str  # 'style' | 'ingredient' | 'venue' | 'abv'
    target_ref: str | None = None
    confidence: float = 0.5
    resolved: bool | None = None  # None=open, True=confirmed, False=denied


class WeeklyProfileDelta(BaseModel):
    user_id: str
    from_version: int
    to_version: int
    summary: str
    predictions: list[WeeklyPrediction] = Field(default_factory=list)
