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

# Opus by default. Sonnet is a third of the price and would be the obvious pick for a
# per-frame path, but which model reads a stylized wordmark well enough is the whole question
# this endpoint exists to answer, and picking the cheaper one before anybody has measured that
# is choosing the answer. `BCD_VISION_MODEL=claude-sonnet-5` is one line when the measurement
# says it holds.
_DEFAULT_MODEL = "claude-opus-5"
_ENDPOINT = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
# One camera frame, a handful of names. The cap is a cost ceiling, not a limit on the answer.
_MAX_TOKENS = 512
# Long enough for a shelf, short enough that the HUD is not waiting on a stall.
_TIMEOUT_S = 12.0

# --- local models ---------------------------------------------------------------------
_OLLAMA_URL = "http://localhost:11434/api/chat"
# 3B, and picked for reading text in pictures rather than for describing scenes — which is the
# whole job here. Overridable with BCD_VISION_MODEL like the hosted one.
_DEFAULT_LOCAL_MODEL = "qwen2.5vl:3b"
# Five minutes, and that is not a guess. Ollama has no Metal backend on an Intel Mac, so this
# runs on the CPU: measured on a 2018 i9, one 1024px frame costs ~111s cold — about 24s of it in
# the vision encoder (three 512-token batches) and the rest prefilling ~1050 image tokens through
# the model. A repeat of the same image lands in 25-45s off ollama's prompt cache, which is why a
# first measurement can look four times better than it is.
#
# 120s was the first guess here and it was wrong by a hair: every cold frame overran it, ollama
# aborted mid-generation, and the failure read as a timeout rather than as "too slow to ship".
# The timeout is sized to let a slow machine *answer*, because an answer at 111s is evidence and
# a timeout is not. It is emphatically not a claim that this path is fast enough to scan with.
_LOCAL_TIMEOUT_S = 300.0

# No boxes, and no negative instructions. A 3B model given the hosted model's prompt spends its
# output on coordinates it cannot estimate and rules it cannot follow; asked for a list of names
# it has a chance. The client already lays out answers that arrive without a box.
_LOCAL_PROMPT = """\
List every alcoholic drink label you can read in this photo.

For each one give the brand and product name exactly as printed, e.g. "Sierra Nevada Pale Ale".
Only list labels whose text you can actually read. If you cannot read any, return an empty list.
"""

# Ollama constrains generation to a schema when given one, which is what keeps a small model
# from narrating its way around the answer.
_LOCAL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
    },
    "required": ["labels"],
}

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


class OllamaVision:
    """A vision model running on this machine. No account, no key, no per-call cost.

    Same contract as the hosted provider — bytes in, names out — so everything downstream is
    unchanged: the names still go through `resolve_reading`, and a name the catalog cannot
    account for still draws nothing.
    """

    def __init__(self, model: str | None = None, url: str | None = None,
                 client: httpx.AsyncClient | None = None) -> None:
        self.model = model or os.environ.get("BCD_VISION_MODEL") or _DEFAULT_LOCAL_MODEL
        self.url = url or os.environ.get("BCD_OLLAMA_URL") or _OLLAMA_URL
        self._client = client

    @property
    def label(self) -> str:
        return f"ollama:{self.model}"

    async def identify(self, image: bytes, media_type: str = "image/jpeg") -> list[Sighting]:
        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": _LOCAL_PROMPT,
                "images": [base64.b64encode(image).decode("ascii")],
            }],
            "stream": False,
            "format": _LOCAL_SCHEMA,
            # Reading a label is not a creative task, and a small model wanders without this.
            "options": {"temperature": 0},
        }
        if self._client is not None:
            resp = await self._client.post(self.url, json=body, timeout=_LOCAL_TIMEOUT_S)
        else:
            async with httpx.AsyncClient(timeout=_LOCAL_TIMEOUT_S) as client:
                resp = await client.post(self.url, json=body)
        resp.raise_for_status()
        return parse_sightings(resp.json().get("message", {}).get("content", "") or "")


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
    """The configured provider, or None.

    Unconfigured returns None rather than raising: the vision path is an addition to the scan,
    and the scan has to keep working without it. `BCD_VISION_PROVIDER` chooses explicitly;
    absent that, a key selects the hosted model, because a key is only ever set on purpose.
    Nothing probes for a local server — a wrong guess here fails as silence, and the endpoint's
    `detail` saying "connection refused to localhost:11434" is worth more than a provider that
    quietly elected itself.
    """
    choice = (os.environ.get("BCD_VISION_PROVIDER") or "").strip().lower()
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if choice == "ollama":
        return OllamaVision()
    if choice == "anthropic":
        return AnthropicVision(key) if key else None
    return AnthropicVision(key) if key else None
