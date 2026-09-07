"""API contract models — the wire shapes shared conceptually with iOS BCDKit.

The Swift `Codable` structs in BCDKit mirror these field-for-field. Keep them in sync;
a future codegen step can emit the Swift from these, same as telemetry events.

Two granularities of scan input coexist:

  * `DetectedText` — one OCR line or barcode. The original per-line path; still accepted.
  * `DetectedObject` — one *physical thing* in frame (a can, a bottle, a tap handle) as the
    on-device coarse stage sees it: every OCR line that landed inside its box, its barcode
    if any, and the box itself. This is the unit the resolver should reason about — a can
    is one product, not five independent text fragments — and it is what lets the server
    say "not sure" instead of firing on the best-scoring fragment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .entities import ResolvedProduct


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


class DetectedObject(BaseModel):
    """One tracked object from the coarse stage: the union of evidence on a single can or
    bottle, accumulated over several frames so one-frame OCR jitter never reaches us."""

    id: str  # client-side track id; echoed back so the HUD can anchor the answer
    label: str | None = None  # coarse class from the segmenter: 'can' | 'bottle' | 'unknown'
    texts: list[str] = Field(default_factory=list)  # every OCR line seen on this object
    barcode: str | None = None
    symbology: str | None = None
    # normalized object box 0-1, origin top-left
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    frames_seen: int = 1  # temporal support behind this evidence
    confidence: float | None = None  # coarse detector confidence, if any


class ScanResolveRequest(BaseModel):
    detections: list[DetectedText] = Field(default_factory=list)  # per-line path
    objects: list[DetectedObject] = Field(default_factory=list)  # per-object path
    venue_id: str | None = None  # constrains matching to a known menu when present
    lat: float | None = None
    lon: float | None = None
    include_score: bool = True  # personalize with the caller's TasteProfile
    # Client override of the server's confidence floor for a 'resolved' verdict. Leave
    # unset for the server default; a HUD that would rather show nothing than something
    # wrong can raise it.
    min_match_score: float | None = None


class ScoredCandidate(BaseModel):
    """A resolved product for one detection or object, ranked, with a personal score."""

    detection_index: int | None = None  # set for per-line detections
    object_id: str | None = None  # set for per-object resolutions
    resolved: ResolvedProduct
    match_score: float  # how confident we are this is the right product (0-1)
    personal_score: float | None = None  # 0-1 predicted enjoyment for this user
    reason: str | None = None  # one-line 'why' for the overlay
    cold_start: bool = False  # scored from chemistry alone (no reviews) — the moat


ObjectStatus = Literal["resolved", "ambiguous", "unresolved"]


class ObjectResolution(BaseModel):
    """The server's verdict on one tracked object.

    * `resolved`   — `candidates[0]` clears the confidence floor AND the margin over the
                     runner-up; safe to overlay.
    * `ambiguous`  — plausible candidates exist but none is clearly right. The client's
                     fine stage (a closer OCR pass, then an on-device model constrained to
                     choose among *these* candidates) decides. Never overlay as-is.
    * `unresolved` — nothing plausible. Show nothing.
    """

    object_id: str
    status: ObjectStatus
    query: str  # the normalized text the server matched on — for debugging + telemetry
    candidates: list[ScoredCandidate] = Field(default_factory=list)  # best-first


class ScanResolveResponse(BaseModel):
    candidates: list[ScoredCandidate] = Field(default_factory=list)
    unresolved_indices: list[int] = Field(default_factory=list)
    objects: list[ObjectResolution] = Field(default_factory=list)
    latency_ms: float | None = None


class ProductSearchResponse(BaseModel):
    query: str
    results: list[ResolvedProduct] = Field(default_factory=list)


class LexiconResponse(BaseModel):
    """Vocabulary for the on-device OCR's custom-words hint: product, brand and producer
    names the recognizer should prefer over dictionary words. This is the single most
    effective fix for stylized label type — 'ALCHEMIST' stops becoming 'Chemist'."""

    words: list[str] = Field(default_factory=list)
