"""Physically merging duplicate product rows, and the redirects that outlive them.

Dedup deletes rows, but ids escape into places we do not control: telemetry events, a
user's list, a cached scan on a phone that has been offline for a week. A merge that simply
dropped the alias id would silently void all of them — a rating whose product no longer
exists just vanishes from the taste profile, and the user is never told why their
recommendations moved.

So a merge leaves a tombstone. The alias id keeps existing as a `product_redirect` row
pointing at the survivor, and `resolve_id()` follows it. Writing the tombstone *over* the
old row (same id, new entity_type) is what makes this atomic: the product disappears from
every `entity_type='product'` query in the same write that makes it redirectable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

REDIRECT_ENTITY = "product_redirect"

# A redirect chain is at most a couple of hops in practice (A merged into B, B later into
# C). The cap only exists so a cycle from a bad backfill cannot hang a request.
_MAX_HOPS = 8


def put_redirect(store: Any, old_id: str, new_id: str) -> None:
    """Leave `old_id` resolvable, pointing at `new_id`. Overwrites the row it replaces."""
    store.put_gold(old_id, REDIRECT_ENTITY, {"id": old_id, "redirects_to": new_id})


def _walk(store: Any, pid: str) -> tuple[str, dict | None]:
    """Follow tombstones from `pid`. Returns (id we ended on, its record or None).

    A live id costs exactly one lookup — the redirect check is a field on the record we
    already fetched, not a second query.
    """
    seen: set[str] = set()
    for _ in range(_MAX_HOPS):
        rec = store.get_gold(pid)
        if not isinstance(rec, dict):
            return pid, None                       # unknown id
        nxt = rec.get("redirects_to")
        if not nxt:
            return pid, rec                        # a live row
        if nxt == pid or nxt in seen:
            return pid, None                       # cyclic / self-referential tombstone
        seen.add(pid)
        pid = nxt
    return pid, None


def resolve_id(store: Any, pid: str) -> str:
    """The id that still holds this product. Unchanged when `pid` is live or unknown, so
    callers can treat the result as "the best id we have" and handle a miss as before."""
    return _walk(store, pid)[0]


def get_product(store: Any, pid: str) -> dict | None:
    """Fetch a product by an id that may have been merged away."""
    return _walk(store, pid)[1]


def collapse(pairs: dict[str, str]) -> dict[str, str]:
    """Point every alias at the FINAL survivor, so A->B->C becomes A->C and B->C."""
    def final(pid: str) -> str:
        seen: set[str] = set()
        while pid in pairs and pid not in seen:
            seen.add(pid)
            pid = pairs[pid]
        return pid
    return {a: final(c) for a, c in pairs.items() if final(c) != a}


# Fields the survivor inherits from the row it absorbs, when it has none of its own — so
# picking the "wrong" survivor cannot lose data.
_INHERIT = ("style", "spec", "recipe", "sensory", "description", "style_bjcp_code")


def _is_empty(v: Any) -> bool:
    """Whether a field carries no information. A default `ProductSpec()` serializes to a
    dict of Nones — truthy, but empty — so a plain falsy check would decide the survivor
    "already has a spec" and quietly drop the alias's ABV."""
    if v is None or v == "" or v == [] or v == {}:
        return True
    return isinstance(v, dict) and all(_is_empty(x) for x in v.values())


def _absorb(current: Any, incoming: Any) -> Any:
    """Keep what the survivor has, fill in what it lacks — per field, not per object."""
    if _is_empty(current):
        return incoming if not _is_empty(incoming) else current
    if isinstance(current, dict) and isinstance(incoming, dict):
        out = dict(current)
        for k, v in incoming.items():
            if _is_empty(out.get(k)) and not _is_empty(v):
                out[k] = v
        return out
    return current


def merge_products(store: Any, pairs: dict[str, str]) -> dict[str, int]:
    """Absorb each alias into its canonical: inherit missing fields, keep the alias name as
    an alias, repoint every SKU (so all barcodes still resolve), then tombstone the alias id.

    Idempotent: an alias that has already become a redirect is skipped, so this can be run
    repeatedly — which it must be, since each pass can expose new matches.
    """
    merges = collapse(pairs)
    stats = {"merged": 0, "skus_repointed": 0}
    if not merges:
        return stats

    skus = [s for s in store.iter_gold("sku") if s.get("product_id") in merges]
    for alias_id, canon_id in merges.items():
        alias = store.get_gold(alias_id)
        canon = store.get_gold(canon_id)
        if not alias or not canon or alias.get("redirects_to"):
            continue
        for key in _INHERIT:
            canon[key] = _absorb(canon.get(key), alias.get(key))
        names = set(canon.get("aliases") or []) | set(alias.get("aliases") or [])
        if alias.get("name") and alias["name"].casefold() != (canon.get("name") or "").casefold():
            names.add(alias["name"])
        canon["aliases"] = sorted(names)
        store.put_gold(canon_id, "product", canon)
        put_redirect(store, alias_id, canon_id)   # replaces the alias product row
        stats["merged"] += 1

    for sku in skus:
        sku["product_id"] = merges[sku["product_id"]]
        store.put_gold(sku["id"], "sku", sku)
        stats["skus_repointed"] += 1
    return stats


_PRODUCER_INHERIT = ("kind", "country", "region", "city", "lat", "lon",
                     "parent_company", "ttb_permit", "website")


def merge_producers(store: Any, pairs: dict[str, str]) -> dict[str, int]:
    """Fold each alias producer into its canonical, repointing everything that named it.

    Unlike a product merge there is no redirect row: nothing resolves a producer by id from
    outside, so the alias is simply removed once products and brands point elsewhere. The
    spellings are kept as aliases, since they are what a label actually says.

    A location is inherited rather than overwritten, and where a cluster disagrees the most
    common region wins — a company files from wherever it packages, and the address it uses
    most is the one worth showing.
    """
    merges = collapse(pairs)
    stats = {"merged": 0, "products_repointed": 0, "brands_repointed": 0}
    if not merges:
        return stats

    # Most common region per canonical, before anything is absorbed.
    regions: dict[str, Counter] = defaultdict(Counter)
    for alias_id, canon_id in merges.items():
        for pid in (alias_id, canon_id):
            rec = store.get_gold(pid)
            if rec and rec.get("region"):
                regions[canon_id][rec["region"]] += 1

    for alias_id, canon_id in merges.items():
        alias = store.get_gold(alias_id)
        canon = store.get_gold(canon_id)
        if not alias or not canon:
            continue
        for key in _PRODUCER_INHERIT:
            canon[key] = _absorb(canon.get(key), alias.get(key))
        names = set(canon.get("aliases") or []) | set(alias.get("aliases") or [])
        if alias.get("name") and alias["name"].casefold() != (canon.get("name") or "").casefold():
            names.add(alias["name"])
        canon["aliases"] = sorted(names)
        if regions[canon_id]:
            canon["region"] = regions[canon_id].most_common(1)[0][0]
        store.put_gold(canon_id, "producer", canon)
        store.delete_gold(alias_id)
        stats["merged"] += 1

    for kind, field, stat in (("product", "producer_id", "products_repointed"),
                              ("brand", "producer_id", "brands_repointed")):
        for rec in list(store.iter_gold(kind)):
            target = merges.get(rec.get(field))
            if target:
                rec[field] = target
                store.put_gold(rec["id"], kind, rec)
                stats[stat] += 1
    return stats
