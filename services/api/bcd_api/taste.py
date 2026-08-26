"""Taste profile — turn behavior into a sensory centroid we can recommend toward.

The catalog half is done: every product carries a 25-axis SensoryVector. This is the
other half — given what a user rated, where do *they* sit in that same space? We build a
`sensory_ideal` by Rocchio relevance feedback (pull toward what they liked, push off what
they didn't), which keeps the profile in the product space so `nearest_by_sensory` can do
candidate generation directly against it.

Signals come from the telemetry event log, not a side table, so the profile a user gets is
identical whether their ratings arrived via the client's batch upload or the convenience
`/v1/feedback` endpoint. Two rules are load-bearing:

  * **Consent is enforced here, not assumed.** Only events carrying a personalization (or
    data-sharing) consent tier are allowed to shape a profile — an analytics-tier event is
    counted as if it never happened, per the tier contract in telemetry/events.yaml.
  * **Identity is the pseudonymous `install_id`**, never a durable account id.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from bcd_ingest.merge import get_product, resolve_id
from bcd_schema import (
    SENSORY_AXES,
    Product,
    SensorySource,
    SensoryVector,
    TasteProfile,
)

# Events that carry taste signal. A rating is the explicit ask; a list add is weaker and
# directional (saving something is interest, not a verdict). `scan_corrected_by_user` is
# deliberately absent: it labels *recognition*, not preference.
_RATING_EVENT = "rating_submitted"
_LIST_EVENT = "list_add"
_LIST_WEIGHTS = {"cellar": 0.6, "had_it": 0.5, "wishlist": 0.3, "want_to_try": 0.3}

#: The only events a profile is ever built from — used to filter the log on replay.
TASTE_EVENTS = frozenset({_RATING_EVENT, _LIST_EVENT})

# Consent tiers under which a profile may be built at all (see telemetry/events.yaml).
_PERSONALIZATION_CONSENT = frozenset({"personalization", "data_sharing"})

# Ratings are 1-5. 3 is "fine" — the pivot between a like and a dislike.
_RATING_NEUTRAL = 3.0
_RATING_SPAN = 2.0

# Rocchio: how hard dislikes push away. < 1 so dislikes inform the centroid without
# dominating it — one bad stout shouldn't erase everything roasty you enjoy.
_GAMMA = 0.4

# Confidence floor/step/ceiling for the derived ideal. It is still an inference, so it
# never claims more than a strong prior would.
_CONF_BASE = 0.3
_CONF_STEP = 0.08
_CONF_MAX = 0.9

_MIN_SIGNALS_FOR_BAND = 2


def signals_from_events(
    events: Iterable[dict[str, Any]], install_id: str
) -> dict[str, float]:
    """Collapse an event stream into one signed weight per product, in [-1, 1].

    Later events on the same product supersede earlier ones for ratings (a re-rate is a
    correction, not a second vote); list adds accumulate but stay capped.
    """
    ratings: dict[str, float] = {}
    lists: dict[str, float] = defaultdict(float)
    for ev in events:
        if ev.get("install_id") != install_id:
            continue
        if ev.get("consent_tier") not in _PERSONALIZATION_CONSENT:
            continue  # not consented for personalization — treat as if unrecorded
        pid = ev.get("product_id")
        if not pid:
            continue
        name = ev.get("name")
        if name == _RATING_EVENT:
            rating = ev.get("rating")
            if isinstance(rating, (int, float)):
                ratings[pid] = _rating_weight(float(rating))
        elif name == _LIST_EVENT:
            bump = _LIST_WEIGHTS.get(ev.get("list_kind") or "", 0.0)
            if bump:
                lists[pid] = min(1.0, lists[pid] + bump)

    signals = dict(lists)
    signals.update(ratings)  # an explicit rating always wins over an implicit list add
    return {pid: w for pid, w in signals.items() if w}


def _rating_weight(rating: float) -> float:
    """1-5 star rating -> signed weight in [-1, 1], neutral at 3."""
    return max(-1.0, min(1.0, (rating - _RATING_NEUTRAL) / _RATING_SPAN))


def build_profile(
    install_id: str,
    signals: dict[str, float],
    store: Any,
    previous: TasteProfile | None = None,
) -> TasteProfile:
    """Assemble a TasteProfile from signed per-product signals.

    Products the store doesn't know, or that carry no sensory vector, contribute nothing
    to the centroid but can still inform style affinity — a rating is never silently lost.
    """
    liked: list[tuple[float, list[float]]] = []
    disliked: list[tuple[float, list[float]]] = []
    style_weights: dict[str, list[float]] = defaultdict(list)
    liked_abvs: list[float] = []
    liked_styles: list[str] = []

    for pid, weight in signals.items():
        rec = get_product(store, pid)   # follows merge tombstones
        if not rec:
            continue
        try:
            product = Product.model_validate(rec)
        except Exception:
            continue
        if product.sensory is not None:
            bucket = liked if weight > 0 else disliked
            bucket.append((abs(weight), product.sensory.to_array()))
        style = product.style.value if product.style else None
        if style:
            style_weights[style].append(weight)
            if weight > 0:
                liked_styles.append(style)
        if weight > 0 and product.spec and product.spec.abv_pct:
            liked_abvs.append(float(product.spec.abv_pct.value))

    ideal = _centroid(liked, disliked)
    n = len(liked) + len(disliked)
    lo, hi = _abv_band(liked_abvs)

    return TasteProfile(
        user_id=install_id,
        version=(previous.version if previous else 0) + 1,
        updated_at=datetime.now(UTC).isoformat(),
        style_affinities={
            s: round(max(-1.0, min(1.0, sum(ws) / len(ws))), 3)
            for s, ws in style_weights.items()
        },
        sensory_ideal=(
            SensoryVector(
                source=SensorySource.RECONCILED,
                confidence=round(min(_CONF_MAX, _CONF_BASE + _CONF_STEP * n), 3),
                axes=ideal,
            )
            if ideal
            else None
        ),
        abv_band_min=lo,
        abv_band_max=hi,
        novelty_appetite=_novelty(liked_styles),
        memo=_memo(ideal, disliked),
    )


def _centroid(
    liked: list[tuple[float, list[float]]], disliked: list[tuple[float, list[float]]]
) -> dict[str, float]:
    """Rocchio relevance feedback, clipped back into the [0,1] product space.

    With no positive signal we return nothing rather than inventing a centroid from
    dislikes alone — "not that" doesn't locate a taste, and a bogus ideal would rank the
    whole catalog confidently wrong.
    """
    pos = _weighted_mean(liked)
    if pos is None:
        return {}
    neg = _weighted_mean(disliked)
    axes: dict[str, float] = {}
    for i, axis in enumerate(SENSORY_AXES):
        value = pos[i] - (_GAMMA * neg[i] if neg else 0.0)
        value = max(0.0, min(1.0, value))
        if value > 0:
            axes[axis] = round(value, 3)
    return axes


def _weighted_mean(items: list[tuple[float, list[float]]]) -> list[float] | None:
    total = sum(w for w, _ in items)
    if total <= 0:
        return None
    return [
        sum(w * vec[i] for w, vec in items) / total for i in range(len(SENSORY_AXES))
    ]


def _abv_band(abvs: list[float]) -> tuple[float | None, float | None]:
    """The ABV range they actually enjoy, padded. Needs a couple of points to mean
    anything — one 12% barleywine is not a band."""
    if len(abvs) < _MIN_SIGNALS_FOR_BAND:
        return (None, None)
    return (round(max(0.0, min(abvs) - 0.5), 1), round(max(abvs) + 0.5, 1))


def _novelty(liked_styles: list[str]) -> float | None:
    """Share of distinct styles among the things they liked: all-different reads as an
    explorer, all-the-same as a loyalist."""
    if len(liked_styles) < _MIN_SIGNALS_FOR_BAND:
        return None
    return round(len(set(liked_styles)) / len(liked_styles), 3)


def _memo(ideal: dict[str, float], disliked: list[tuple[float, list[float]]]) -> str | None:
    """One human-readable line for the weekly card. Names the axes actually driving the
    centroid, so the user can see (and argue with) what we think of them."""
    if not ideal:
        return None
    top = sorted(ideal.items(), key=lambda kv: kv[1], reverse=True)[:3]
    leans = ", ".join(_pretty(a) for a, _ in top)
    memo = f"You lean {leans}"
    neg = _weighted_mean(disliked)
    if neg:
        worst = max(range(len(SENSORY_AXES)), key=lambda i: neg[i])
        memo += f" — and away from {_pretty(SENSORY_AXES[worst])}"
    return memo + "."


def _pretty(axis: str) -> str:
    return axis.replace("_", " ")


# ---- persistence -------------------------------------------------------------------
# Profiles live in gold beside the catalog. `put_gold` only derives the pgvector column
# for entity_type='product', so a profile's ideal never leaks into product ANN results.

_PROFILE_ENTITY = "taste_profile"


def profile_key(install_id: str) -> str:
    return f"profile:{install_id}"


def load_profile(store: Any, install_id: str) -> TasteProfile | None:
    rec = store.get_gold(profile_key(install_id))
    if not rec:
        return None
    try:
        return TasteProfile.model_validate(rec)
    except Exception:
        return None


def save_profile(store: Any, profile: TasteProfile) -> None:
    store.put_gold(
        profile_key(profile.user_id), _PROFILE_ENTITY, profile.model_dump(mode="json")
    )


def _canonical(store: Any, events: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Rewrite each event's product_id to the id that still holds the product.

    Dedup merges rows away, and a rating pointing at a merged id would otherwise be dropped
    on rebuild — the user's signal silently disappearing because the catalog was tidied.
    Done BEFORE signal extraction so "a later rating supersedes an earlier one" keeps
    holding across a merge: rate row A, we merge A into B, rate B — that is one product with
    one verdict, not two. Events are copied, never mutated; the log stays what was recorded.
    """
    cache: dict[str, str] = {}
    for ev in events:
        pid = ev.get("product_id")
        if pid:
            if pid not in cache:
                cache[pid] = resolve_id(store, pid)
            if cache[pid] != pid:
                ev = {**ev, "product_id": cache[pid]}
        yield ev


def rebuild_profile(
    store: Any, events: Iterable[dict[str, Any]], install_id: str
) -> TasteProfile:
    """Recompute from the full event log and persist. Rebuilding wholesale (rather than
    nudging the stored vector) keeps the profile a pure function of consented events —
    so a withdrawn consent or a deleted event actually disappears from the result."""
    previous = load_profile(store, install_id)
    signals = signals_from_events(_canonical(store, events), install_id)
    profile = build_profile(install_id, signals, store, previous=previous)
    save_profile(store, profile)
    return profile
