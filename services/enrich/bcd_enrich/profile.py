"""A per-product profile from a language model -- what THIS beer or spirit tastes like.

The style floor (`style_prior`) gives every product the vector of its style, which makes a Heady
Topper and a supermarket double IPA the same drink. For the products people actually scan that is
not good enough: a model that has read every brewery page, review and label knows Heady is soft,
fruity and Simcoe-forward and deceptively strong, and that Buffalo Trace is a caramel-and-vanilla
low-rye bourbon. This module asks it -- in a fixed rubric, into a fixed schema -- and writes what
comes back where the recommender and the detail screen read:

  * `product.sensory`            the 25 axes, source `llm_profile`, the model's own confidence;
  * `product.style`              when the row had none, or only a generic one ("Ale");
  * `product.description`        a one-sentence summary; the descriptors sit in its receipt;
  * `product.recipe.ingredients` the hops, grains, botanicals and barrels the maker has stated;
  * `product.spec.abv_pct`       when the row had none and the model knows the label's.

Everything it writes carries `ExtractionMethod.LLM_RECALLED`: recalled, not read, capped at 0.6
confidence -- one rank above a style guess, below anything with a URL. The model says whether it
actually knows the product or is only reading its name (`style_only`), and the apply step trusts a
style-only answer with less and never lets it state an ingredient or a strength. The raw answer is
kept in bronze (source `llm-profile`), so a change to these rules is a re-apply, not a re-purchase.

On the wire this is one POST per product to the Messages API with a forced tool call (the schema
is the tool's input), or one Message Batches submission for bulk at half the price. Plain `httpx`,
as `bcd_api.vision` does, so it adds no dependency. Run it with `python -m bcd_enrich.profile`.
Without an API account the same questions go out to a file (`--export`) and the answers -- written
by hand, or by a model in a chat session -- come back through one (`--answers`), validated and
applied exactly as an API answer would be.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import glob
import json
import os
import re
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
from bcd_ingest.store import BronzeDoc, doc_id, open_store
from bcd_schema import (
    SENSORY_AXES,
    ExtractionMethod,
    IngredientKind,
    IngredientRole,
    Product,
    Provenance,
    SensorySource,
    SensoryVector,
)
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ._env import load_dotenv
from .style_prior import _GENERIC_CLASS_STYLES, detect_style

_ENDPOINT = "https://api.anthropic.com/v1/messages"
_BATCH_ENDPOINT = "https://api.anthropic.com/v1/messages/batches"
_API_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-sonnet-5"
_MAX_TOKENS = 1024
_TIMEOUT_S = 60.0
#: Bumped whenever the rubric or the schema changes, so answers written under an older prompt can
#: be told apart (and re-asked) -- it rides along in bronze and in every provenance chip.
PROMPT_VERSION = "2026-09-17.1"
#: Where the raw answers live, in bronze.
SOURCE_ID = "llm-profile"
#: Where a submitted batch's custom-id -> product-id map waits for `--collect`.
BATCH_DIR = "data/profile_batches"

#: A provenance chip cannot claim more than this, whatever the model says of itself.
_CEILING = 0.6
#: A product the model only knows by its style is worth about what a named style prior is.
_STYLE_ONLY_MAX = 0.4
#: A strength is printed as a number; the model has to be this sure before one is written.
_ABV_MIN_CONFIDENCE = 0.7

# ---------------------------------------------------------------------------------------------
# the rubric
# ---------------------------------------------------------------------------------------------

_AXIS_NOTES: dict[str, str] = {
    "citrus": "lemon, grapefruit, orange peel",
    "tropical": "mango, pineapple, passion fruit, guava",
    "stone_fruit": "peach, apricot, plum",
    "berry": "raspberry, cherry, blackcurrant, red fruit",
    "floral": "rose, elderflower, orange blossom, hop florals",
    "herbal": "tea, sage, thyme, gentian, wormwood, juniper's green side",
    "piney_resinous": "pine, resin, dank",
    "grassy": "fresh-cut grass, hay, green, agave's vegetal side",
    "spicy_phenolic": "clove, pepper, cinnamon, rye spice, anise",
    "malty_bready": "bread, cracker, cereal, grain, biscuit",
    "caramel_toffee": "caramel, toffee, brown sugar, butterscotch",
    "roasted_coffee_choc": "coffee, dark chocolate, burnt, roast",
    "nutty": "almond, hazelnut, marzipan, walnut",
    "vanilla_oak": "vanilla, oak, coconut, cask character",
    "smoky_peat": "smoke, peat, ash, medicinal, mezcal smoke",
    "honey": "honey, mead-like sweetness",
    "banana_ester": "banana, bubblegum, pear esters",
    "funk_brett": "barnyard, leather, brett, wild",
    "sour_tart": "lactic or acetic acidity, tartness",
    "sweet": "perceived sweetness, residual sugar, liqueur sugar",
    "bitterness": "hop, botanical or amaro bitterness",
    "body_fullness": "thin and watery (0.1) to full and viscous (0.9)",
    "carbonation": "still (0.0) to lively (0.8); a spirit or liqueur is 0",
    "alcohol_warmth": "heat: ~0.15 at 4% abv, 0.3 at 6%, 0.5 at 9%, 0.8 at 40%, 0.95 at 55%+",
    "dryness_finish": "the finish: sweet and lingering (0.1) to bone dry (0.9)",
}
assert tuple(_AXIS_NOTES) == SENSORY_AXES

_EXAMPLE_KNOWN = {
    "recognition": "known", "confidence": 0.9, "style": "American pale ale", "abv_pct": 5.6,
    "summary": "Grapefruit and floral Cascade hops over a light caramel malt, moderately bitter "
               "with a clean, dry finish.",
    "descriptors": ["grapefruit", "floral", "pine", "caramel malt", "crisp"],
    "axes": {"citrus": 0.55, "floral": 0.4, "piney_resinous": 0.35, "grassy": 0.2,
             "malty_bready": 0.45, "caramel_toffee": 0.4, "bitterness": 0.55,
             "body_fullness": 0.4, "carbonation": 0.55, "alcohol_warmth": 0.25,
             "dryness_finish": 0.55},
    "ingredients": [
        {"name": "Cascade", "kind": "hop", "role": "aroma_hop"},
        {"name": "Magnum", "kind": "hop", "role": "bittering_hop"},
        {"name": "Perle", "kind": "hop", "role": "flavor_hop"},
        {"name": "Two-row pale malt", "kind": "malt", "role": "base_malt"},
        {"name": "Caramel malt", "kind": "malt", "role": "specialty_malt"},
    ],
    "basis": "the brewery lists its hops and malts; a benchmark of the style since 1980",
}
_EXAMPLE_STYLE_ONLY = {
    "recognition": "style_only", "confidence": 0.35, "style": "American amber ale",
    "abv_pct": None,
    "summary": "A typical amber ale: caramel and toasted malt with a modest hop bite and a "
               "medium body.",
    "descriptors": ["caramel", "toasted malt", "mild hop"],
    "axes": {"citrus": 0.2, "floral": 0.2, "malty_bready": 0.5, "caramel_toffee": 0.6,
             "nutty": 0.3, "bitterness": 0.4, "body_fullness": 0.45, "carbonation": 0.5,
             "alcohol_warmth": 0.25, "dryness_finish": 0.45},
    "ingredients": [],
    "basis": "style only; the name says amber and nothing else is on record",
}


def _full_axes(partial: dict[str, float]) -> dict[str, float]:
    return {a: partial.get(a, 0.0) for a in SENSORY_AXES}


def system_prompt() -> str:
    axes = "\n".join(f"- {a}: {note}" for a, note in _AXIS_NOTES.items())
    known = json.dumps({**_EXAMPLE_KNOWN, "axes": _full_axes(_EXAMPLE_KNOWN["axes"])})
    style_only = json.dumps({**_EXAMPLE_STYLE_ONLY,
                             "axes": _full_axes(_EXAMPLE_STYLE_ONLY["axes"])})
    return f"""\
You are a beverage sensory panelist with an encyclopedic knowledge of beers, spirits, ciders, \
sakes and wines: what breweries and distilleries publish about their products, how critics and \
drinkers describe them, and the chemistry behind each style.

You will be given one product from a catalog: its name, producer, category, the style or \
regulatory class it was filed under, and whatever else is on record. Record its tasting profile \
with the `record_profile` tool. Nothing else; no prose outside the tool call.

## Recognition -- say what you actually know
- "known": you recognise this specific product from this producer and can describe it as it \
is rather than as its style: big brands, cult beers, well-documented craft releases.
- "style_only": you do not know this particular product, but its name, producer and class tell \
you the style, and you are profiling a typical example of it, nudged only by what the name states \
outright (a "Double IPA" is stronger and more bitter; a "Coffee Stout" has coffee).
- "unknown": not even the style can be told. Rare: a filed class usually settles the style.
Never mistake a product for a similarly named one from another maker, and never upgrade \
"style_only" to "known" because the name sounds familiar.

## Confidence
0-1 for the profile as a whole. Known, famous and consistent: 0.8-0.9. Known but variable or \
hazily remembered (a barrel pick, a rotating series, a small brewery's one-off): 0.5-0.7. \
style_only: 0.3-0.45. unknown: below 0.2.

## Axes
Every axis 0-1: 0 absent, 0.2 a trace, 0.4 noticeable, 0.6 prominent, 0.8 and above defining. \
Aroma and flavour axes name only what a taster would report -- most drinks have three to six \
axes above 0.3, not fifteen. The five structure axes (bitterness, body_fullness, carbonation, \
alcohol_warmth, dryness_finish) always mean something; set all five, and let warmth follow the \
strength.
{axes}

## Ingredients
List only ingredients the producer has stated or that are widely and specifically documented for \
this product: named hop varieties, the grains of a mash bill, botanicals, barrel types with their \
prior fill, fruit, spices, a notable yeast. Never guess a hop bill from the style. An empty list \
is the right answer for most products. Put the source of the knowledge in `basis`, in a few \
words: "brewery lists Citra and Mosaic", "the distillery's low-rye mash bill", "style only".

## Style
The style as a drinker would name it: "New England IPA", "Kentucky straight bourbon", "London \
dry gin", "American lager", "amaro". Null only when unknown.

## ABV
Only the label's stated strength for this exact product, if you know it. Otherwise null -- never \
a typical-for-style number.

## Summary and descriptors
`summary` is one sentence about what it tastes like, without marketing. `descriptors` are three \
to six lower-case tasting words.

## Worked examples
Sierra Nevada Pale Ale (Sierra Nevada Brewing, Chico CA), filed as "Ale":
{known}

"Riverbend Amber" from Anytown Brewing Co., filed as "Ale", nothing else on record:
{style_only}
"""


_KINDS = [k.value for k in IngredientKind]
_ROLES = [r.value for r in IngredientRole]

PROFILE_TOOL: dict[str, Any] = {
    "name": "record_profile",
    "description": "Record the tasting profile of the product under consideration.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recognition": {"type": "string", "enum": ["known", "style_only", "unknown"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "style": {"type": ["string", "null"]},
            "abv_pct": {"type": ["number", "null"]},
            "summary": {"type": "string"},
            "descriptors": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            "axes": {
                "type": "object",
                "additionalProperties": False,
                "properties": {a: {"type": "number", "minimum": 0, "maximum": 1}
                               for a in SENSORY_AXES},
                "required": list(SENSORY_AXES),
            },
            "ingredients": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": _KINDS},
                        "role": {"type": "string", "enum": _ROLES},
                    },
                    "required": ["name", "kind", "role"],
                },
            },
            "basis": {"type": "string"},
        },
        "required": ["recognition", "confidence", "style", "abv_pct", "summary", "descriptors",
                     "axes", "ingredients", "basis"],
    },
}


# ---------------------------------------------------------------------------------------------
# the answer
# ---------------------------------------------------------------------------------------------

class ProfiledIngredient(BaseModel):
    name: str
    kind: IngredientKind = IngredientKind.OTHER
    role: IngredientRole = IngredientRole.OTHER

    @field_validator("name")
    @classmethod
    def _tidy(cls, v: str) -> str:
        return " ".join(v.split())


class ProductProfile(BaseModel):
    """What the model recorded, validated: clamped axes, lower-case descriptors, an ingredient
    list with no repeats. Unknown ingredient kinds and roles fall back to `other` rather than
    losing the row."""

    recognition: str = "unknown"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    style: str | None = None
    abv_pct: float | None = None
    summary: str = ""
    descriptors: list[str] = Field(default_factory=list)
    axes: dict[str, float] = Field(default_factory=dict)
    ingredients: list[ProfiledIngredient] = Field(default_factory=list)
    basis: str = ""

    @field_validator("recognition", mode="before")
    @classmethod
    def _known_values(cls, v: Any) -> str:
        v = str(v or "").strip().lower()
        return v if v in ("known", "style_only", "unknown") else "unknown"

    @field_validator("axes", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        for a in SENSORY_AXES:
            try:
                x = float((v or {}).get(a, 0.0))
            except (TypeError, ValueError):
                x = 0.0
            out[a] = round(min(1.0, max(0.0, x)), 3)
        return out

    @field_validator("descriptors", mode="before")
    @classmethod
    def _lower(cls, v: Any) -> list[str]:
        seen: list[str] = []
        for d in v or []:
            d = " ".join(str(d).split()).lower().strip(" .")
            if d and d not in seen:
                seen.append(d)
        return seen[:6]

    @field_validator("ingredients", mode="before")
    @classmethod
    def _dedupe(cls, v: Any) -> list[dict]:
        out: list[dict] = []
        names: set[str] = set()
        for ing in v or []:
            if not isinstance(ing, dict) or not str(ing.get("name", "")).strip():
                continue
            key = " ".join(str(ing["name"]).split()).lower()
            if key in names:
                continue
            names.add(key)
            row = dict(ing)
            if row.get("kind") not in _KINDS:
                row["kind"] = "other"
            if row.get("role") not in _ROLES:
                row["role"] = "other"
            out.append(row)
        return out

    @field_validator("abv_pct", mode="before")
    @classmethod
    def _plausible_abv(cls, v: Any) -> float | None:
        try:
            x = float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
        return x if x is not None and 0.0 < x < 96.0 else None

    @field_validator("style", mode="before")
    @classmethod
    def _tidy_style(cls, v: Any) -> str | None:
        s = " ".join(str(v or "").split()).strip()
        return s[:80] or None

    @property
    def known(self) -> bool:
        return self.recognition == "known"

    @property
    def usable(self) -> bool:
        """Whether there is a profile to write at all."""
        return self.recognition != "unknown" and any(self.axes.values())

    @property
    def sensory_confidence(self) -> float:
        """The vector's confidence: the model's own when it knows the product, else held to
        what a named style prior is worth -- and allowed below the detail screen's 0.30
        "tastes like" gate, so a guess about a bare "Ale" stays wallpaper."""
        if self.known:
            return round(max(0.45, min(0.9, self.confidence)), 2)
        return round(min(_STYLE_ONLY_MAX, max(0.2, self.confidence)), 2)

    def top_axes(self, n: int = 3) -> list[str]:
        ranked = sorted(self.axes.items(), key=lambda kv: (-kv[1], kv[0]))
        return [a for a, v in ranked[:n] if v > 0]


# ---------------------------------------------------------------------------------------------
# the question
# ---------------------------------------------------------------------------------------------

def describe(rec: dict, producer: dict | None, brand: dict | None) -> str:
    """The user turn: everything the catalog knows about the row, one fact per line."""
    lines = [f"Product: {rec.get('name', '')}"]
    if producer:
        where = ", ".join(x for x in (producer.get("city"), producer.get("region"),
                                       producer.get("country")) if x)
        kind = producer.get("kind")
        lines.append(f"Producer: {producer.get('name', '')}"
                     + (f" ({where})" if where else "") + (f" -- {kind}" if kind else ""))
    if brand and producer and brand.get("name") and \
            brand["name"].lower() != (producer.get("name") or "").lower() and \
            brand["name"].lower() not in (rec.get("name") or "").lower():
        lines.append(f"Brand: {brand['name']}")
    lines.append(f"Category: {rec.get('category', '')}")
    style = rec.get("style")
    if isinstance(style, dict) and style.get("value"):
        prov = style.get("provenance") or {}
        filed = prov.get("method") == ExtractionMethod.REGULATORY_FILING.value
        quote = (prov.get("quote") or "").strip()
        line = f"{'Filed style' if filed else 'Style on record'}: {style['value']}"
        if filed and quote and quote.lower() != str(style["value"]).lower():
            line += f" (registry class: {quote})"
        lines.append(line)
    spec = rec.get("spec") or {}
    abv = spec.get("abv_pct") if isinstance(spec, dict) else None
    if isinstance(abv, dict) and abv.get("value") is not None:
        lines.append(f"ABV on record: {abv['value']}%")
    age = spec.get("age_statement_years") if isinstance(spec, dict) else None
    if isinstance(age, dict) and age.get("value") is not None:
        lines.append(f"Age statement: {age['value']} years")
    recipe = rec.get("recipe") or {}
    known = [f"{i.get('raw_name')} ({i.get('entity_kind')})"
             for i in (recipe.get("ingredients") or []) if i.get("raw_name")]
    if known:
        lines.append("Ingredients on record: " + ", ".join(known[:12]))
    desc = rec.get("description")
    if isinstance(desc, dict) and desc.get("value"):
        lines.append(f"Description on record: {str(desc['value'])[:300]}")
    aliases = [a for a in rec.get("aliases") or [] if a and a != rec.get("name")]
    if aliases:
        lines.append("Also listed as: " + ", ".join(aliases[:4]))
    lines.append("")
    lines.append("Record this product's tasting profile.")
    return "\n".join(lines)


def request_params(user_text: str, model: str) -> dict[str, Any]:
    """The Messages API body -- also what one entry of a batch carries as `params`."""
    return {
        "model": model,
        "max_tokens": _MAX_TOKENS,
        "system": [{"type": "text", "text": system_prompt(),
                    "cache_control": {"type": "ephemeral"}}],
        "tools": [PROFILE_TOOL],
        "tool_choice": {"type": "tool", "name": PROFILE_TOOL["name"]},
        "messages": [{"role": "user", "content": user_text}],
    }


def tool_input(message: dict) -> dict:
    """The tool call's input out of a Messages API response."""
    for block in message.get("content") or []:
        if block.get("type") == "tool_use" and block.get("name") == PROFILE_TOOL["name"]:
            return dict(block.get("input") or {})
    raise ValueError(f"no {PROFILE_TOOL['name']} call in the answer "
                     f"(stop_reason={message.get('stop_reason')!r})")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 409, 429, 500, 502, 503, 529)
    return isinstance(exc, httpx.TransportError)


class Claude:
    """Claude over plain HTTP: one product per call, or a batch of them."""

    def __init__(self, api_key: str, model: str | None = None,
                 client: httpx.Client | None = None) -> None:
        self._key = api_key
        self.model = model or os.environ.get("BCD_ENRICH_MODEL") or _DEFAULT_MODEL
        self._client = client or httpx.Client(timeout=_TIMEOUT_S)

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._key, "anthropic-version": _API_VERSION,
                "content-type": "application/json"}

    @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def _post(self, url: str, body: dict) -> dict:
        resp = self._client.post(url, json=body, headers=self._headers)
        resp.raise_for_status()
        return resp.json()

    @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def _get(self, url: str) -> httpx.Response:
        resp = self._client.get(url, headers=self._headers)
        resp.raise_for_status()
        return resp

    def profile(self, user_text: str) -> tuple[dict, dict]:
        """One product, now: (the tool's input, the usage block)."""
        message = self._post(_ENDPOINT, request_params(user_text, self.model))
        return tool_input(message), dict(message.get("usage") or {})

    def submit_batch(self, items: Iterable[tuple[str, str]]) -> dict:
        """Many products, later: `items` are (custom_id, user_text). Returns the batch object;
        `id` is what `--collect` wants."""
        requests = [{"custom_id": cid, "params": request_params(text, self.model)}
                    for cid, text in items]
        return self._post(_BATCH_ENDPOINT, {"requests": requests})

    def batch(self, batch_id: str) -> dict:
        return self._get(f"{_BATCH_ENDPOINT}/{batch_id}").json()

    def batch_results(self, results_url: str) -> Iterator[tuple[str, dict | None, str | None]]:
        """(custom_id, tool input or None, error or None) per line of the results file."""
        for line in self._get(results_url).iter_lines():
            if not line.strip():
                continue
            row = json.loads(line)
            result = row.get("result") or {}
            if result.get("type") == "succeeded":
                try:
                    yield row["custom_id"], tool_input(result["message"]), None
                except ValueError as e:
                    yield row["custom_id"], None, str(e)
            else:
                err = result.get("error") or {}
                yield row["custom_id"], None, f"{result.get('type')}: {err.get('message', err)}"


# ---------------------------------------------------------------------------------------------
# the write
# ---------------------------------------------------------------------------------------------

def _provenance(profile: ProductProfile, model: str, quote: str) -> dict:
    return Provenance(
        source_id=SOURCE_ID,
        method=ExtractionMethod.LLM_RECALLED,
        confidence=min(_CEILING, profile.confidence),
        quote=quote[:500],
        extractor_version=f"{model}/{PROMPT_VERSION}",
    ).model_dump(mode="json")


def _generic_style(rec: dict) -> bool:
    """Whether the row's style says no more than its category does ("Ale", "Whiskey")."""
    style = rec.get("style")
    value = style.get("value") if isinstance(style, dict) else None
    if not value:
        return True
    detected = detect_style(rec.get("name") or "", rec.get("category"), class_type=value)
    return detected is None or detected in _GENERIC_CLASS_STYLES


def _replaceable_source(rec: dict, profile: ProductProfile) -> bool:
    """A model's profile replaces a prior, never a source that heard from drinkers."""
    current = rec.get("sensory")
    source = current.get("source") if isinstance(current, dict) else None
    if source in (None, SensorySource.STYLE_PRIOR.value, SensorySource.LLM_PROFILE.value):
        return True
    return source == SensorySource.CHEMISTRY_PRIOR.value and profile.known


def apply_profile(rec: dict, profile: ProductProfile, *, model: str) -> list[str]:
    """Write the profile into the gold record, in place. Returns the fields it changed."""
    changed: list[str] = []
    if not profile.usable:
        return changed

    if _replaceable_source(rec, profile):
        vec = SensoryVector(source=SensorySource.LLM_PROFILE,
                            confidence=profile.sensory_confidence,
                            axes={a: v for a, v in profile.axes.items() if v > 0})
        new = vec.model_dump(mode="json")
        if rec.get("sensory") != new:
            rec["sensory"] = new
            changed.append("sensory")

    if profile.style and _generic_style(rec) and \
            (profile.known or profile.confidence >= 0.5):
        current = rec.get("style") if isinstance(rec.get("style"), dict) else None
        if not current or current.get("value") != profile.style:
            quote = f"was: {current['value']}" if current and current.get("value") else \
                profile.basis
            rec["style"] = {"value": profile.style,
                            "provenance": _provenance(profile, model, quote)}
            changed.append("style")

    if profile.summary:
        current = rec.get("description") if isinstance(rec.get("description"), dict) else None
        method = ((current or {}).get("provenance") or {}).get("method")
        ours = method in (ExtractionMethod.LLM_RECALLED.value,
                          ExtractionMethod.LLM_INFERRED_FROM_STYLE_PRIOR.value)
        if (not current or ours) and (current or {}).get("value") != profile.summary:
            rec["description"] = {
                "value": profile.summary,
                "provenance": _provenance(profile, model,
                                          ", ".join(profile.descriptors) or profile.basis),
            }
            changed.append("description")

    if profile.known and profile.ingredients:
        recipe = rec.get("recipe")
        if not isinstance(recipe, dict):
            recipe = {"ingredients": [], "process_steps": []}
        rows = recipe.setdefault("ingredients", [])
        have = {" ".join(str(r.get("raw_name", "")).split()).lower() for r in rows}
        added = 0
        for ing in profile.ingredients:
            if ing.name.lower() in have:
                continue
            rows.append({
                "role": ing.role.value, "entity_kind": ing.kind.value, "entity_ref": None,
                "raw_name": ing.name, "quantity": None, "unit": None,
                "percent_of_bill": None, "timing": None,
                "provenance": _provenance(profile, model, profile.basis),
            })
            have.add(ing.name.lower())
            added += 1
        if added:
            rec["recipe"] = recipe
            changed.append(f"ingredients+{added}")

    if profile.known and profile.abv_pct is not None and \
            profile.confidence >= _ABV_MIN_CONFIDENCE:
        spec = rec.get("spec")
        if not isinstance(spec, dict):
            spec = {}
        if not (isinstance(spec.get("abv_pct"), dict) and
                spec["abv_pct"].get("value") is not None):
            spec["abv_pct"] = {"value": profile.abv_pct,
                               "provenance": _provenance(profile, model, profile.basis)}
            rec["spec"] = spec
            changed.append("abv")

    if changed:
        Product.model_validate(rec)  # never write a row the schema would refuse to read back
    return changed


def bronze_doc(product_id: str, user_text: str, answer: dict, *, model: str,
               usage: dict | None = None) -> BronzeDoc:
    return BronzeDoc(
        id=doc_id(SOURCE_ID, product_id), source_id=SOURCE_ID, natural_key=product_id,
        fetched_at=datetime.now(UTC).isoformat(), url=None,
        payload={"model": model, "prompt_version": PROMPT_VERSION, "question": user_text,
                 "answer": answer, "usage": usage or {}},
    )


# ---------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------

_CUSTOM_ID = re.compile(r"[^a-zA-Z0-9_-]")


def _scan_sightings(paths: Iterable[str]) -> collections.Counter[tuple[str, str]]:
    """(name, producer) of every candidate the HUD drew, by frames, out of the scan logs."""
    seen: collections.Counter[tuple[str, str]] = collections.Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for c in row.get("candidates") or []:
                    if c.get("name"):
                        seen[(c["name"], c.get("producer") or "")] += 1
    return seen


def _from_scans(store, paths: list[str]) -> list[dict]:
    """The catalog rows behind what the camera has drawn: an exact name match, and the
    producer's name when the log has one. Names the catalog cannot match are reported."""
    out: dict[str, dict] = {}
    for (name, producer), frames in _scan_sightings(paths).most_common():
        hits = store.products_named(name)
        if producer:
            by_maker = []
            for r in hits:
                maker = store.get_gold(r.get("producer_id") or "") or {}
                if (maker.get("name") or "").lower() == producer.lower():
                    by_maker.append(r)
            hits = by_maker or hits
        if not hits:
            print(f"  ? not in the catalog: {name!r} / {producer!r} ({frames} frames)")
        for r in hits:
            out.setdefault(r["id"], r)
    return list(out.values())


def select_products(store, *, ids: list[str], searches: list[str], from_scans: bool,
                    scan_glob: str, lineup: int, limit: int | None) -> list[dict]:
    rows: dict[str, dict] = {}
    for pid in ids:
        rec = store.get_gold(pid)
        if rec and rec.get("category"):
            rows[pid] = rec
        else:
            print(f"  ? no product {pid!r}")
    for q in searches:
        hits = store.search_gold_products(q, limit=1)
        if hits:
            rows[hits[0]["id"]] = hits[0]
        else:
            print(f"  ? nothing for {q!r}")
    if from_scans:
        for r in _from_scans(store, sorted(glob.glob(scan_glob))):
            rows[r["id"]] = r
    if lineup:
        for r in list(rows.values()):
            for sib in store.products_of(r.get("producer_id") or "", limit=lineup):
                rows.setdefault(sib["id"], sib)
    out = list(rows.values())
    return out[:limit] if limit is not None else out


def _answered(store) -> dict[str, BronzeDoc]:
    return {d.natural_key: d for d in store.iter_bronze(SOURCE_ID)}


def _report(rec: dict, profile: ProductProfile, changed: list[str]) -> str:
    axes = ", ".join(f"{a} {profile.axes[a]:.1f}" for a in profile.top_axes())
    flag = {"known": "K", "style_only": "s", "unknown": "?"}[profile.recognition]
    return (f"  [{flag} {profile.confidence:.2f}] {rec.get('name', '')!r}: "
            f"{profile.style or '-'} | {axes} | {len(profile.ingredients)} ingr"
            + (f" | wrote {', '.join(changed)}" if changed else " | nothing to write"))


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def run(args: argparse.Namespace) -> int:
    load_dotenv()
    store = open_store(root=args.root)
    model = args.model or os.environ.get("BCD_ENRICH_MODEL") or _DEFAULT_MODEL
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()

    if args.collect:
        return _collect(store, args, model, key)
    if args.answers:
        return _from_file(store, args, model)

    products = select_products(store, ids=args.ids, searches=args.search,
                               from_scans=args.from_scans, scan_glob=args.scans,
                               lineup=args.lineup, limit=args.limit)
    if not products:
        print("nothing selected: use --from-scans, --ids, or --search")
        return 1
    answered = _answered(store)

    if args.reapply:
        batch: list[tuple[str, str, dict]] = []
        for rec in products:
            doc = answered.get(rec["id"])
            if doc is None:
                continue
            profile = ProductProfile.model_validate(doc.payload.get("answer") or {})
            changed = apply_profile(rec, profile, model=doc.payload.get("model") or model)
            print(_report(rec, profile, changed))
            if changed and not args.dry_run:
                batch.append((rec["id"], "product", rec))
        if batch:
            store.put_gold_many(batch)
        print(f"re-applied {len(batch)} of {len(products)} from bronze")
        return 0

    todo = [r for r in products if args.force or r["id"] not in answered]
    print(f"→ {len(products)} products selected, {len(products) - len(todo)} already "
          f"answered, {len(todo)} to ask {model}")
    questions = []
    for rec in todo:
        producer = store.get_gold(rec.get("producer_id") or "")
        brand = store.get_gold(rec.get("brand_id") or "")
        questions.append((rec, describe(rec, producer, brand)))

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            for rec, q in questions:
                f.write(json.dumps({"id": rec["id"], "name": rec.get("name", ""),
                                    "question": q}) + "\n")
        print(f"wrote {len(questions)} questions to {args.export}")
        return 0

    if args.dry_run:
        sys_tokens = _estimate_tokens(system_prompt())
        per = sum(_estimate_tokens(q) for _, q in questions)
        print(f"system prompt ≈ {sys_tokens} tokens (cached after the first call); "
              f"questions ≈ {per} tokens; answers ≈ {350 * len(questions)} tokens")
        for _rec, q in questions[:args.show]:
            print("─" * 60)
            print(q)
        print("─" * 60)
        print(f"dry run: {len(questions)} questions, nothing asked, nothing written")
        return 0

    if not questions:
        print("nothing to ask")
        return 0
    if not key:
        print("ANTHROPIC_API_KEY is not set: put it in .env (never in a commit)")
        return 2
    claude = Claude(key, model)

    if args.batch:
        os.makedirs(BATCH_DIR, exist_ok=True)
        manifest: dict[str, str] = {}
        items = []
        for i, (rec, q) in enumerate(questions):
            cid = f"p{i:05d}-{_CUSTOM_ID.sub('_', rec['id'])}"[:64]
            manifest[cid] = rec["id"]
            items.append((cid, q))
        submitted = claude.submit_batch(items)
        path = os.path.join(BATCH_DIR, f"{submitted['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"model": model, "prompt_version": PROMPT_VERSION,
                       "submitted_at": datetime.now(UTC).isoformat(),
                       "custom_ids": manifest,
                       "questions": {cid: q for cid, q in items}}, f, indent=1)
        print(f"submitted batch {submitted['id']} ({len(items)} requests); "
              f"status {submitted.get('processing_status')}\n"
              f"collect with: python -m bcd_enrich.profile --collect {submitted['id']}")
        return 0

    usage: collections.Counter[str] = collections.Counter()
    writes: list[tuple[str, str, dict]] = []
    counts: collections.Counter[str] = collections.Counter()

    def ask(item: tuple[dict, str]) -> tuple[dict, str, dict | None, dict, str | None]:
        rec, q = item
        try:
            answer, used = claude.profile(q)
            return rec, q, answer, used, None
        except Exception as e:  # noqa: BLE001 - one bad row must not end the run
            return rec, q, None, {}, f"{type(e).__name__}: {e}"

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rec, q, answer, used, err in pool.map(ask, questions):
            if answer is None:
                print(f"  ! {rec.get('name', '')!r}: {err}")
                counts["failed"] += 1
                continue
            for k, v in used.items():
                if isinstance(v, int):
                    usage[k] += v
            store.put_bronze(bronze_doc(rec["id"], q, answer, model=model, usage=used))
            profile = ProductProfile.model_validate(answer)
            counts[profile.recognition] += 1
            changed = apply_profile(rec, profile, model=model)
            print(_report(rec, profile, changed), flush=True)
            if changed:
                writes.append((rec["id"], "product", rec))
    if writes:
        store.put_gold_many(writes)
    print("─" * 60)
    print(f"asked {len(questions)} in {time.time() - t0:.0f}s: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f"; wrote {len(writes)} products; tokens "
          + ", ".join(f"{k}={v}" for k, v in sorted(usage.items())))
    store.close()
    return 0


def _from_file(store, args: argparse.Namespace, model: str) -> int:
    """Answers written outside the API -- one JSON object per line, `{"id": ..., "answer":
    {...}}` -- go through the same validation and the same apply rules as an API answer, and
    into bronze the same way, with `via` saying where they came from."""
    questions = {}
    if args.export:
        with open(args.export, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    questions[row["id"]] = row.get("question", "")
    writes: list[tuple[str, str, dict]] = []
    tally: collections.Counter[str] = collections.Counter()
    with open(args.answers, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                pid, answer = row["id"], row["answer"]
            except (ValueError, KeyError, TypeError) as e:
                print(f"  ! line {n}: {type(e).__name__}: {e}")
                tally["failed"] += 1
                continue
            rec = store.get_gold(pid)
            if rec is None or not rec.get("category"):
                print(f"  ! line {n}: no product {pid!r}")
                tally["failed"] += 1
                continue
            profile = ProductProfile.model_validate(answer)
            tally[profile.recognition] += 1
            changed = apply_profile(rec, profile, model=model)
            if not args.quiet:
                print(_report(rec, profile, changed))
            if changed:
                writes.append((pid, "product", rec))
            if not args.dry_run:
                doc = bronze_doc(pid, questions.get(pid, ""), answer, model=model)
                doc.payload["via"] = args.via
                store.put_bronze(doc)
    if writes and not args.dry_run:
        store.put_gold_many(writes)
    print("─" * 60)
    print(f"read {sum(tally.values())} answers from {args.answers}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
          + f"; {'would write' if args.dry_run else 'wrote'} {len(writes)} products")
    store.close()
    return 0


def _collect(store, args: argparse.Namespace, model: str, key: str) -> int:
    path = os.path.join(BATCH_DIR, f"{args.collect}.json")
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except OSError:
        print(f"no manifest at {path}: was this batch submitted from this checkout?")
        return 1
    if not key:
        print("ANTHROPIC_API_KEY is not set: put it in .env (never in a commit)")
        return 2
    claude = Claude(key, manifest.get("model") or model)
    while True:
        batch = claude.batch(args.collect)
        status = batch.get("processing_status")
        counts = batch.get("request_counts") or {}
        print(f"batch {args.collect}: {status} "
              + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if status == "ended" or not args.wait:
            break
        time.sleep(args.wait)
    if status != "ended":
        return 3
    used_model = manifest.get("model") or model
    ids = manifest["custom_ids"]
    questions = manifest.get("questions") or {}
    writes: list[tuple[str, str, dict]] = []
    tally: collections.Counter[str] = collections.Counter()
    for cid, answer, err in claude.batch_results(batch["results_url"]):
        pid = ids.get(cid)
        rec = store.get_gold(pid) if pid else None
        if rec is None or answer is None:
            print(f"  ! {cid}: {err or 'product gone'}")
            tally["failed"] += 1
            continue
        store.put_bronze(bronze_doc(pid, questions.get(cid, ""), answer, model=used_model))
        profile = ProductProfile.model_validate(answer)
        tally[profile.recognition] += 1
        changed = apply_profile(rec, profile, model=used_model)
        if not args.quiet:
            print(_report(rec, profile, changed))
        if changed:
            writes.append((pid, "product", rec))
    if writes and not args.dry_run:
        store.put_gold_many(writes)
    print("─" * 60)
    print(f"collected {sum(tally.values())}: "
          + ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
          + f"; {'would write' if args.dry_run else 'wrote'} {len(writes)} products")
    store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bcd_enrich.profile",
        description="Ask a language model what each selected product tastes like, and write "
                    "the answer onto the catalog row.")
    ap.add_argument("--root", default="./data")
    sel = ap.add_argument_group("which products")
    sel.add_argument("--from-scans", action="store_true",
                     help="every product the camera has drawn, out of the scan logs")
    sel.add_argument("--scans", default="data/scans*.jsonl", help="the scan logs to read")
    sel.add_argument("--ids", nargs="*", default=[], help="product ids")
    sel.add_argument("--search", nargs="*", default=[], help="the top catalog hit for each")
    sel.add_argument("--lineup", type=int, default=0,
                     help="also each selected product's maker, up to N products")
    sel.add_argument("--limit", type=int, default=None)
    how = ap.add_argument_group("how")
    how.add_argument("--model", default=None,
                     help=f"default $BCD_ENRICH_MODEL or {_DEFAULT_MODEL}")
    how.add_argument("--workers", type=int, default=4, help="concurrent requests")
    how.add_argument("--dry-run", action="store_true",
                     help="print the questions and a token estimate; ask nothing, write nothing")
    how.add_argument("--show", type=int, default=3, help="questions to print in a dry run")
    how.add_argument("--force", action="store_true",
                     help="ask again about products that already have an answer in bronze")
    how.add_argument("--reapply", action="store_true",
                     help="re-run the apply rules over the answers already in bronze")
    how.add_argument("--batch", action="store_true",
                     help="submit through the Message Batches API instead of asking now")
    how.add_argument("--collect", metavar="BATCH_ID",
                     help="fetch a submitted batch's answers and apply them")
    how.add_argument("--export", metavar="FILE",
                     help="write the questions to FILE (JSONL) instead of asking; with "
                          "--answers, the questions to keep beside the answers in bronze")
    how.add_argument("--answers", metavar="FILE",
                     help="apply answers from FILE (JSONL of {id, answer}) instead of asking")
    how.add_argument("--via", default="session",
                     help="with --answers: where they came from, for the record")
    how.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                     help="with --collect: poll every N seconds until the batch has ended")
    how.add_argument("--quiet", action="store_true", help="with --collect: no per-row lines")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
