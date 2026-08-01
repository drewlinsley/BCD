"""Provenance — the answer to 'data is not open or verifiable'.

Every recovered field in BCD carries one of these. It records *where* a fact came
from, *how* we extracted it, the exact supporting text, and how much we trust it.
The iOS app surfaces this as tappable chips on the product 'receipt'.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ExtractionMethod(str, Enum):
    """How a fact was obtained, ordered roughly from most to least authoritative."""

    STATED_BY_PRODUCER = "stated_by_producer"  # brewery/distillery's own words
    REGULATORY_FILING = "regulatory_filing"  # TTB COLA, EU e-label, etc.
    LABEL_OCR = "label_ocr"  # read off the physical label
    RETAILER_LISTING = "retailer_listing"  # shop/menu product page
    COMMUNITY_CLONE = "community_clone"  # homebrew clone recipe
    REVIEW_CONSENSUS = "review_consensus"  # aggregated from reviews
    LLM_INFERRED_FROM_STYLE_PRIOR = "llm_inferred_from_style_prior"  # weakest: a guess
    USER_CONTRIBUTED = "user_contributed"


# Default confidence floor per method. A concrete extraction can override, but this
# keeps inferred facts from ever masquerading as stated ones.
METHOD_CONFIDENCE_CEILING: dict[ExtractionMethod, float] = {
    ExtractionMethod.STATED_BY_PRODUCER: 1.0,
    ExtractionMethod.REGULATORY_FILING: 1.0,
    ExtractionMethod.LABEL_OCR: 0.9,
    ExtractionMethod.RETAILER_LISTING: 0.8,
    ExtractionMethod.COMMUNITY_CLONE: 0.6,
    ExtractionMethod.REVIEW_CONSENSUS: 0.6,
    ExtractionMethod.LLM_INFERRED_FROM_STYLE_PRIOR: 0.4,
    ExtractionMethod.USER_CONTRIBUTED: 0.7,
}


class Provenance(BaseModel):
    """Where a single fact came from. Attach one per recovered field."""

    source_id: str = Field(..., description="Registry id, e.g. 'ttb-cola-registry'.")
    url: str | None = Field(None, description="Exact page the fact was read from.")
    quote: str | None = Field(
        None, description="Verbatim supporting text — the receipt for the claim."
    )
    method: ExtractionMethod
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    extractor_version: str | None = Field(
        None, description="Model/pipeline version, so re-extractions are comparable."
    )

    @field_validator("confidence")
    @classmethod
    def _clamp_to_method_ceiling(cls, v: float, info) -> float:
        method = info.data.get("method")
        if method is not None:
            ceiling = METHOD_CONFIDENCE_CEILING[method]
            if v > ceiling:
                return ceiling
        return v


class Sourced[T](BaseModel):
    """A value plus its provenance. `Sourced[float]` is an ABV you can defend in court."""

    value: T
    provenance: Provenance
