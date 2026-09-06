"""Naming a label from the picture instead of from what OCR made of it.

Every failure this scan path has had comes down to one thing: the wordmark on a craft can
is a drawing, not type. VisionKit reads The Alchemist's Heady Topper as "FADY TOPPE",
"ROY TOPP", once "ПУТОРРЕ" — and no amount of trigram matching gets from those to a
catalog row, because the information is gone before the query starts. The on-device model
cannot help either: Apple's Foundation Models framework is text-only, so all it ever sees
is the same garbled fragments, which is why it spent 124 frames naming nothing.

A model that sees the image has the thing OCR threw away. So this module is the seam for
one: bytes in, product names out. What comes back is *only* a name — the facts still come
from the catalog, resolved by the same `Resolver` with the same corroboration guards, so a
vision sighting is a better query and never a source of truth.

Unconfigured is a first-class state. With no provider the endpoint reports that plainly
rather than failing, and the text path is untouched.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

# Roughly what a 1024px JPEG costs; a frame far over this is a client bug, not a photo, and
# uploading it would burn the latency budget before the model is even called.
MAX_IMAGE_BYTES = 5 * 1024 * 1024

_DEFAULT_MODEL = "claude-sonnet-5"
_ENDPOINT = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
# One camera frame, a handful of names. The cap is a cost ceiling, not a limit on the answer.
_MAX_TOKENS = 512
# Long enough for a shelf, short enough that the HUD is not waiting on a stall.
_TIMEOUT_S = 12.0

_PROMPT = """\
This is a photo of alcoholic drinks — bottles, cans, or a shelf of them.

Name every distinct product you can actually read on a label. For each one reply with an \
object: {"name": "<brand and product, as printed>", "box": [x, y, w, h]} where the box is \
the label's position in the image as fractions of width and height, 0 to 1.

Rules:
- Report only what is legibly printed in THIS image. Never infer a product from packaging \
style, colour, or what tends to sit beside it.
- If a label is present but you cannot read it, leave it out.
- One entry per distinct product, even when several cans of it are visible.
- Reply with a JSON array and nothing else. If you can read no label, reply exactly [].
"""


@dataclass(frozen=True)
class Sighting:
    """One product the model says it can read, and where it says it is."""

    name: str
    box: tuple[float, float, float, float] | None = None


class VisionProvider(Protocol):
    """The seam. Bytes in, names out — deliberately narrower than the model can do, so
    that swapping the model cannot widen what the scan path trusts it for."""

    @property
    def label(self) -> str: ...

    async def identify(self, image: bytes, media_type: str = "image/jpeg") -> list[Sighting]: ...


class AnthropicVision:
    """Claude over plain HTTP. `httpx` is already the repo's client, so this adds no
    dependency — and the Messages API surface used here is one POST with one image block."""

    def __init__(self, api_key: str, model: str | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        self._key = api_key
        self.model = model or os.environ.get("BCD_VISION_MODEL") or _DEFAULT_MODEL
        self._client = client

    @property
    def label(self) -> str:
        return f"anthropic:{self.model}"

    async def identify(self, image: bytes, media_type: str = "image/jpeg") -> list[Sighting]:
        body = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type,
                        "data": base64.b64encode(image).decode("ascii"),
                    }},
                    {"type": "text", "text": _PROMPT},
                ],
            }],
        }
        headers = {"x-api-key": self._key, "anthropic-version": _API_VERSION,
                   "content-type": "application/json"}
        if self._client is not None:
            resp = await self._client.post(_ENDPOINT, json=body, headers=headers,
                                           timeout=_TIMEOUT_S)
        else:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                resp = await client.post(_ENDPOINT, json=body, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
        text = "".join(part.get("text", "") for part in payload.get("content", [])
                       if part.get("type") == "text")
        return parse_sightings(text)


class StubVision:
    """Scripted provider for tests and for exercising the endpoint without a key."""

    def __init__(self, sightings: list[Sighting] | None = None,
                 label: str = "stub") -> None:
        self._sightings = sightings or []
        self._label = label
        self.calls: list[bytes] = []

    @property
    def label(self) -> str:
        return self._label

    async def identify(self, image: bytes, media_type: str = "image/jpeg") -> list[Sighting]:
        self.calls.append(image)
        return list(self._sightings)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_sightings(text: str) -> list[Sighting]:
    """Read the model's reply into sightings, tolerantly.

    Asking for "a JSON array and nothing else" gets one most of the time and a fenced block
    or a sentence of preamble the rest of it. The parser's job is to find the array; the
    validation below is what decides whether an entry is usable.
    """
    if not text:
        return []
    body = text.strip()
    fenced = _FENCE_RE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        rows = json.loads(body[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(rows, list):
        return []

    out: list[Sighting] = []
    seen: set[str] = set()
    for row in rows:
        name = row.get("name") if isinstance(row, dict) else row
        if not isinstance(name, str):
            continue
        name = " ".join(name.split()).strip(" \"'`*")
        # A name that is one or two characters is chrome the model volunteered, not a product.
        if len(name) < 3:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(Sighting(name=name, box=_box(row.get("box")) if isinstance(row, dict) else None))
    return out


def _box(raw: object) -> tuple[float, float, float, float] | None:
    """A box is advisory — the model estimates it, and a wrong one puts the overlay on the
    wrong can. Anything not four finite numbers describing a real rectangle inside the frame
    is dropped, and the caller falls back to laying the answers out itself."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    if not all(0.0 <= v <= 1.0 for v in (x, y)) or x + w > 1.001 or y + h > 1.001:
        return None
    return (x, y, min(w, 1.0), min(h, 1.0))


def provider_from_env() -> VisionProvider | None:
    """The configured provider, or None. Absent a key this returns None rather than raising:
    the vision path is an addition to the scan, and the scan has to keep working without it."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return None
    return AnthropicVision(key)
