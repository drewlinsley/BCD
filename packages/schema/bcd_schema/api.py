"""API contract models — the wire shapes shared conceptually with iOS BCDKit.

The Swift `Codable` structs in BCDKit mirror these field-for-field. Keep them in sync;
a future codegen step can emit the Swift from these, same as telemetry events.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .entities import ResolvedProduct
from .profile import TasteProfile


class DetectedText(BaseModel):
    """One OCR/barcode hit from the on-device scanner, with its frame bounds."""

    text: str
    kind: str = "text"  # 'text' | 'barcode'
    symbology: str | None = None  # for barcodes: 'ean13', 'upca', 'qr'
    # normalized bounding box 0-1 in image space, so the HUD can anchor the overlay
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    confidence: float | None = None


class ScanResolveRequest(BaseModel):
    detections: list[DetectedText]
    venue_id: str | None = None  # constrains matching to a known menu when present
    lat: float | None = None
    lon: float | None = None
    include_score: bool = True  # personalize with the caller's TasteProfile


class ScoredCandidate(BaseModel):
    """A resolved product for one detection, ranked, with a personal score + reason."""

    detection_index: int
    resolved: ResolvedProduct
    match_score: float  # how confident we are this is the right product
    personal_score: float | None = None  # 0-1 predicted enjoyment for this user
    reason: str | None = None  # one-line 'why' for the overlay
    cold_start: bool = False  # scored from chemistry alone (no reviews) — the moat


class ScanResolveResponse(BaseModel):
    candidates: list[ScoredCandidate] = Field(default_factory=list)
    unresolved_indices: list[int] = Field(default_factory=list)
    latency_ms: float | None = None
    # Whether more than one part of the frame agrees on some candidate — the label naming both
    # its maker and its drink, or naming one and printing a category that matches it. False
    # means we returned a guess off a single fragment, which reads the same as a confident
    # answer once it is an overlay. The client uses it to decide whether to ask the on-device
    # model, so a wrong-but-plausible row cannot quietly suppress the fallback built for exactly
    # that case: a Heady Topper can answered "Chemist" 11 frames running and never asked.
    corroborated: bool = False


class ProductSearchResponse(BaseModel):
    query: str
    results: list[ResolvedProduct] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """One taste verdict. Recorded as a `rating_submitted` telemetry event, then folded
    into the caller's TasteProfile — the thumbs the personalization loop learns from."""

    product_id: str
    rating: float = Field(..., ge=1.0, le=5.0)  # 1-5, neutral at 3
    aspects: dict[str, float] | None = None  # optional per-axis detail ("too sweet")


class FeedbackResponse(BaseModel):
    accepted: bool
    profile: TasteProfile  # echo the updated profile so the client can show the shift


class ScanVisionRequest(BaseModel):
    """One camera frame, for the labels OCR could not read.

    `detections` is what the on-device scanner *did* read of the same frame. It does not enter
    the matching — a clean reading needs no corroboration from a garbled one — but it is the
    only record of what the camera saw at the moment the picture was taken, and every
    diagnosis on this path so far has come from having exactly that.
    """

    image_b64: str
    media_type: str = "image/jpeg"
    detections: list[DetectedText] = Field(default_factory=list)
    venue_id: str | None = None
    lat: float | None = None
    lon: float | None = None


class ScanVisionResponse(ScanResolveResponse):
    """A resolve response, plus what the model claimed to see.

    `sightings` is the honest middle of the pipeline: a name here with no candidate beside
    it means the model read the can and the catalog does not have it — a different problem
    from the model reading nothing, and the two are indistinguishable from candidates alone.
    """

    sightings: list[str] = Field(default_factory=list)
    # The frame the server actually resolved, one entry per sighting and in the same
    # order, so `detection_index` addresses it. The client never built this frame — the
    # boxes are the model's — so it has to come back with the answers or the overlays
    # have nothing to anchor to.
    detections: list[DetectedText] = Field(default_factory=list)
    provider: str | None = None
    detail: str | None = None
