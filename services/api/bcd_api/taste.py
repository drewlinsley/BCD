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
from bcd_schema.quiz import QUIZ_DRINKS

# Events that carry taste signal. A rating is the explicit ask; a list add is weaker and
# directional (saving something is interest, not a verdict). `scan_corrected_by_user` is
# deliberately absent: it labels *recognition*, not preference.
_RATING_EVENT = "rating_submitted"
#: Taking a verdict back. The log is append-only and the profile is a pure function of it,
#: so a withdrawal is a thing that HAPPENED rather than a thing erased: the rating stays in
#: the log and stops counting. That is what lets it travel the client's batch upload exactly
#: as a rating does, instead of needing a second, deleting path through the telemetry store.
_WITHDRAWN_EVENT = "rating_withdrawn"
#: The first-run quiz. Not a rating: nobody said they drank the thing, they said what they
#: reach for. It seeds the same centroid because a stated preference and a verdict are both
#: evidence about the same taste -- but it names a FAMILY, never a product, so it cannot put
#: a drink in `rated_products` and quietly bar it from the recommendations it exists to make.
_QUIZ_EVENT = "taste_quiz_answered"
_LIST_EVENT = "list_add"
_LIST_WEIGHTS = {"cellar": 0.6, "had_it": 0.5, "wishlist": 0.3, "want_to_try": 0.3}

#: A product the HUD drew. Weaker evidence than anything else here and the only kind that is
#: free: pointing a camera at a bottle is not a verdict on it, but it is not nothing either --
#: you are standing in front of it deciding. So it counts, positively, at a fraction of a
#: stated preference, and it is the signal that works for someone who skipped the quiz and has
#: rated nothing.
#:
#: There is no negative. You cannot read a dislike off a glance, and the shelf you walked past
#: sends no event at all.
_SCAN_EVENT = "scan_resolved"

#: What one scanned product is worth against a rating's 1.0 and a quiz answer's 0.5. Low on
#: purpose: scanning is how you ask what something IS, so plenty of scans are of drinks the
#: drinker will put straight back. Four of them are about one quiz answer.
_SCAN_WEIGHT = 0.15

#: The only events a profile is ever built from — used to filter the log on replay. The
#: withdrawal belongs here or replay would drop it and the rating it retracts would return.
TASTE_EVENTS = frozenset({_RATING_EVENT, _WITHDRAWN_EVENT, _QUIZ_EVENT, _LIST_EVENT,
                          _SCAN_EVENT})

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

# Before the memo will say a drinker avoids something, the drink they disliked has to be
# loud on that axis AND their own centroid has to be quiet on it. See `_aversion`.
_AVERSION_FLOOR = 0.5
_AVERSION_CEILING = 0.3


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
        elif name == _WITHDRAWN_EVENT:
            # Read in order with everything else, so withdrawing and then rating again
            # leaves the new verdict standing -- the same rule a re-rate already follows.
            # Only the rating goes: a list add is a separate signal and was not withdrawn.
            ratings.pop(pid, None)
        elif name == _LIST_EVENT:
            bump = _LIST_WEIGHTS.get(ev.get("list_kind") or "", 0.0)
            if bump:
                lists[pid] = min(1.0, lists[pid] + bump)

    signals = dict(lists)
    signals.update(ratings)  # an explicit rating always wins over an implicit list add
    return {pid: w for pid, w in signals.items() if w}


def quiz_from_events(
    events: Iterable[dict[str, Any]], install_id: str
) -> dict[str, float]:
    """The first-run quiz's answers, one signed weight per drink family.

    Read exactly as ratings are: consent-gated, this install only, and a later answer
    supersedes an earlier one, because retaking the quiz is a correction and not a second
    vote. An answer of 0 is "no opinion" and is dropped rather than stored as a neutral,
    so skipping a question says nothing instead of saying the middle.
    """
    out: dict[str, float] = {}
    for ev in events:
        if ev.get("name") != _QUIZ_EVENT or ev.get("install_id") != install_id:
            continue
        if ev.get("consent_tier") not in _PERSONALIZATION_CONSENT:
            continue
        family = ev.get("family")
        weight = ev.get("weight")
        if family in QUIZ_DRINKS and isinstance(weight, (int, float)):
            out[family] = max(-1.0, min(1.0, float(weight)))
    return {f: w for f, w in out.items() if w}


def scans_from_events(
    events: Iterable[dict[str, Any]], install_id: str
) -> dict[str, float]:
    """Products this person pointed a camera at, each at the same small positive weight.

    Read exactly as ratings and quiz answers are: consent-gated, this install only. Keyed by
    product so a product seen twenty times counts once — the HUD redraws every 350ms and the
    client already reports a product once per run, but the log spans months and a can looked
    at on two separate trips must not outvote a rating.

    Weight is flat rather than a function of how long it was on screen. Dwell is a measure of
    how hard the label was to read as much as of interest, and the honest unit here is "this
    was considered", which has no degrees.
    """
    out: dict[str, float] = {}
    for ev in events:
        if ev.get("name") != _SCAN_EVENT or ev.get("install_id") != install_id:
            continue
        if ev.get("consent_tier") not in _PERSONALIZATION_CONSENT:
            continue
        pid = ev.get("product_id")
        if pid:
            out[pid] = _SCAN_WEIGHT
    return out


def rated_products(
    store: Any, events: Iterable[dict[str, Any]], install_id: str
) -> set[str]:
    """Products this person has already passed a verdict on.

    A recommendation is something to drink next, and a drink you have rated is not one --
    "For you" led with a beer whose own row on the same screen showed the face for `spat it
    out` (2026-09-24). Ids are canonicalised the way `rebuild_profile` canonicalises them,
    so a verdict given before a merge still covers the row that survived it.

    Consent-gated exactly as the signals are: a rating that may not shape the profile may
    not quietly shape the list either. The client keeps its own copy of those and can hide
    them itself.
    """
    out: set[str] = set()
    for ev in _canonical(store, events):
        if ev.get("install_id") != install_id:
            continue
        if ev.get("consent_tier") not in _PERSONALIZATION_CONSENT:
            continue
        pid = ev.get("product_id")
        if not pid:
            continue
        name = ev.get("name")
        if name == _RATING_EVENT:
            out.add(pid)
        elif name == _WITHDRAWN_EVENT:
            # Withdrawn means never said, so it is something to drink next again. Without
            # this a mis-tap would quietly bar a drink from "For you" for good.
            out.discard(pid)
    return out


def _rating_weight(rating: float) -> float:
    """1-5 star rating -> signed weight in [-1, 1], neutral at 3."""
    return max(-1.0, min(1.0, (rating - _RATING_NEUTRAL) / _RATING_SPAN))


def build_profile(
    install_id: str,
    signals: dict[str, float],
    store: Any,
    previous: TasteProfile | None = None,
    quiz: dict[str, float] | None = None,
) -> TasteProfile:
    """Assemble a TasteProfile from signed per-product signals.

    Products the store doesn't know, or that carry no sensory vector, contribute nothing
    to the centroid but can still inform style affinity — a rating is never silently lost.

    `quiz` is the first-run answers, keyed by drink family. They enter the same buckets as
    a rating and nothing downstream knows the difference, which is the point: a drinker who
    has rated nothing has no centroid, and with no centroid `score` falls back to a style
    prior and `/v1/recommend` answers from a seed profile that is somebody else's taste.
    The quiz is what makes the first recommendation theirs.

    They are deliberately NOT products. A quiz answer carries the family's STYLE_PRIOR
    vector — the same vector the catalog's own style-filed rows carry — so it lands in the
    space the drinker's later ratings will, without claiming they drank anything.
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

    # Answers to the quiz, if it was taken. Weighted by the caller, which is where the rule
    # that a stated preference counts for less than a verdict lives.
    for family, weight in (quiz or {}).items():
        drink = QUIZ_DRINKS.get(family)
        if drink is None:
            continue
        prompt, _category, confidence, axes = drink
        vector = SensoryVector(source=SensorySource.STYLE_PRIOR,
                               confidence=confidence, axes=axes).to_array()
        (liked if weight > 0 else disliked).append((abs(weight), vector))
        style_weights[prompt].append(weight)
        if weight > 0:
            liked_styles.append(prompt)

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
    """One human-readable line for the taste card. Names the axes actually driving the
    centroid, so the user can see (and argue with) what we think of them."""
    if not ideal:
        return None
    top = sorted(ideal.items(), key=lambda kv: kv[1], reverse=True)[:3]
    leans = ", ".join(_pretty(a) for a, _ in top)
    memo = f"You lean {leans}"
    axis = _aversion(ideal, disliked)
    if axis:
        memo += f" — and away from {_pretty(axis)}"
    return memo + "."


def _aversion(
    ideal: dict[str, float], disliked: list[tuple[float, list[float]]]
) -> str | None:
    """The axis a dislike actually argues against — or None, when it argues against nothing.

    This used to be `max(neg)`: the loudest axis of whatever they disliked, named outright.
    That is only sound when someone's likes and dislikes look different. They usually don't.
    A drinker working through one style rates near-identical bottles, and the loudest axis of
    the bad one is then a note they love. Ours liked Heady Topper (tropical 0.70) and disliked
    a DDH IPA (tropical 0.75); the card told them they lean away from tropical while listing
    tropical among what they go for. Both halves came from the same two ratings.

    So an axis has to clear two bars to be named: the disliked drink is loud on it, and the
    drinker's own centroid is not. Two beers that differ only in degree clear neither, which
    is the correct outcome — a dislike that has not isolated anything gets no clause, rather
    than the best of a bad field.
    """
    neg = _weighted_mean(disliked)
    if not neg:
        return None
    best: str | None = None
    widest = 0.0
    for i, axis in enumerate(SENSORY_AXES):
        mine = ideal.get(axis, 0.0)
        if neg[i] < _AVERSION_FLOOR or mine >= _AVERSION_CEILING:
            continue
        if neg[i] - mine > widest:
            best, widest = axis, neg[i] - mine
    return best


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
    # One pass of the log, read twice: `_canonical` resolves merge tombstones and is not
    # free, and the quiz carries no product id to resolve.
    canonical = list(_canonical(store, events))
    # Scans first, so a real verdict on the same drink overwrites the glance at it. They go
    # in through `signals` rather than beside it because a scanned product IS a product and
    # carries its own vector: scan five IPAs and the centroid becomes IPA-shaped out of what
    # those five actually taste like, which is better evidence than the style's average and
    # works for all twenty-two shelves rather than the quiz's eight.
    signals = {**scans_from_events(canonical, install_id),
               **signals_from_events(canonical, install_id)}
    quiz = quiz_from_events(canonical, install_id)
    profile = build_profile(install_id, signals, store, previous=previous, quiz=quiz)
    save_profile(store, profile)
    return profile
