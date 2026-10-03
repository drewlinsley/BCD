"""API contract models — the wire shapes shared conceptually with iOS BCDKit.

The Swift `Codable` structs in BCDKit mirror these field-for-field. Keep them in sync;
a future codegen step can emit the Swift from these, same as telemetry events.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .entities import Category, ResolvedProduct
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


class DetectedObject(BaseModel):
    """One tracked object from the client's coarse stage: everything the camera read off a
    single can or bottle over several frames, plus its barcode and box.

    A can is one product, not five fragments. Sending the object rather than its lines
    lets the server judge all of its evidence together and hand back one verdict.
    """

    id: str  # client-side track id; echoed back so the HUD can anchor the answer
    label: str | None = None  # coarse class from the segmenter: 'can' | 'bottle' | 'object'
    texts: list[str] = Field(default_factory=list)  # every stable OCR line seen on it
    barcode: str | None = None
    symbology: str | None = None
    # normalized bounding box 0-1 in image space
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
    # Raise the floor a `resolved` object verdict must clear. None = the server default.
    # The client raises it when it has already pinned accurate OCR to the object and does
    # not want a weaker read to overwrite it.
    min_match_score: float | None = None


class ScoredCandidate(BaseModel):
    """A resolved product for one detection, ranked, with a personal score + reason."""

    # Which line of the request this answers, or -1 for an object-level answer (those
    # anchor to the object's own box, never to a line).
    detection_index: int = -1
    object_id: str | None = None  # set on the per-object path
    resolved: ResolvedProduct
    match_score: float  # how confident we are this is the right product
    personal_score: float | None = None  # 0-1 predicted enjoyment for this user
    reason: str | None = None  # one-line 'why' for the overlay
    cold_start: bool = False  # scored from chemistry alone (no reviews) — the moat


ObjectStatus = Literal["resolved", "ambiguous", "unresolved"]


class ObjectResolution(BaseModel):
    """The server's verdict on one tracked object.

    `resolved`   — safe to overlay; `candidates[0]` is the answer.
    `ambiguous`  — the evidence points at a shortlist but no single row accounts for it;
                   `candidates` carries the shortlist for the client's fine stage.
    `unresolved` — nothing the evidence supports; show nothing, keep reading.
    """

    object_id: str
    status: ObjectStatus
    query: str  # what the server actually matched, for diagnostics
    candidates: list[ScoredCandidate] = Field(default_factory=list)  # best-first


class ScanResolveResponse(BaseModel):
    candidates: list[ScoredCandidate] = Field(default_factory=list)
    unresolved_indices: list[int] = Field(default_factory=list)
    objects: list[ObjectResolution] = Field(default_factory=list)  # one per request object
    latency_ms: float | None = None
    # Whether more than one part of the frame agrees on some candidate — the label naming both
    # its maker and its drink, or naming one and printing a category that matches it. False
    # means we returned a guess off a single fragment, which reads the same as a confident
    # answer once it is an overlay. The client uses it to decide whether to ask the on-device
    # model, so a wrong-but-plausible row cannot quietly suppress the fallback built for exactly
    # that case: a Heady Topper can answered "Chemist" 11 frames running and never asked.
    corroborated: bool = False


class LexiconResponse(BaseModel):
    """Catalog vocabulary for the on-device recognizer's custom-words hint."""

    words: list[str] = Field(default_factory=list)


class ProductSearchResponse(BaseModel):
    query: str
    results: list[ResolvedProduct] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """One taste verdict. Recorded as a `rating_submitted` telemetry event, then folded
    into the caller's TasteProfile — the thumbs the personalization loop learns from."""

    product_id: str
    rating: float = Field(..., ge=1.0, le=5.0)  # 1-5, neutral at 3
    aspects: dict[str, float] | None = None  # optional per-axis detail ("too sweet")


class QuizAnswer(BaseModel):
    """One answer to the first-run quiz: a drink family, and how the drinker feels about it.

    `weight` is signed and bounded like a rating's — negative for "not for me", positive for
    "yes" — but the caller scales it below a real verdict's, because saying what you reach
    for is weaker evidence than saying what you thought of something you drank.
    """

    family: str = Field(..., min_length=1, max_length=32)
    weight: float = Field(..., ge=-1.0, le=1.0)


class QuizRequest(BaseModel):
    """A whole quiz in one call. Answering is a single act, not eight, and a half-sent quiz
    would leave a profile built from whichever questions happened to arrive."""

    answers: list[QuizAnswer] = Field(..., min_length=1, max_length=32)


class QuizDrink(BaseModel):
    """One question, as the server defines it — so the drinks can change without an app
    release, and so the app never has to know a flavour vector exists."""

    family: str
    prompt: str
    category: Category


class QuizResponse(BaseModel):
    drinks: list[QuizDrink] = Field(default_factory=list)


class WithdrawRequest(BaseModel):
    """Taking one verdict back. A POST with the id in the body rather than a DELETE with it
    in the path: product ids carry a colon (`bcd:the-alchemist-crusher`), and a body needs no
    escaping to survive the trip."""

    product_id: str = Field(..., min_length=1, max_length=128)


class FeedbackResponse(BaseModel):
    accepted: bool
    profile: TasteProfile  # echo the updated profile so the client can show the shift


class ContributionRequest(BaseModel):
    """A drink a drinker told us about, because the catalog could not place its label.

    This is the only route on which a user authors catalog data, so every field is bounded and
    nothing here is a `Product`: a contribution is a claim *towards* a row, not a row. It lands
    in bronze under its own source and is promoted by a curator, the same way a TTB filing is —
    a typed name must never become something the resolver can draw before someone has looked
    at it.

    `id` is the client's own id for the contribution, and it is what makes this route safe to
    retry: the bronze doc id is derived from it, so a phone that uploads, loses the response and
    uploads again leaves one document rather than two. It is therefore required — the server
    will not invent one, because an invented one would duplicate on every retry.

    `sightings` is what the camera read off the label at the moment the user tapped add. It is
    not shown back and not required; it is kept because it is the only record of what the phone
    actually saw on the one label the catalog missed, and a reviewer should be able to check a
    typed name against it.
    """

    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=120)
    category: Category
    maker: str | None = Field(default=None, max_length=120)
    abv_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    note: str | None = Field(default=None, max_length=500)
    sightings: list[str] = Field(default_factory=list, max_length=24)
    created_at: str | None = None  # when the phone captured it; the server records its own too


class ContributionResponse(BaseModel):
    """`accepted` means it is recorded and the phone may stop keeping it.

    `doc_id` is where it landed, so a submission can be found again without guessing, and
    `duplicate` says this id had already been recorded — an honest answer to a retry, and not
    an error, because the phone's job is done either way.
    """

    accepted: bool
    doc_id: str
    duplicate: bool = False


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
