"""Connector base — the contract every source implements.

Three stages map onto the medallion layers:
  fetch()      -> yields BronzeDoc (raw, immutable, with fetch metadata)
  normalize()  -> bronze -> silver records (per-source shape, still not canonical)
  promote()    -> silver -> gold canonical entities with Provenance attached

A connector declares its `source_id` (must match a registry yaml) and its `provides`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Any

from .store import BronzeDoc, Store


class Connector(ABC):
    source_id: str
    provides: tuple[str, ...] = ()

    def __init__(self, store: Store) -> None:
        self.store = store

    @abstractmethod
    async def fetch(self, limit: int | None = None) -> Iterable[BronzeDoc]:
        """Pull raw documents from the source. Must go through PoliteFetcher for
        Tier-D scrape sources; open APIs may use httpx directly."""

    @abstractmethod
    def normalize(self, doc: BronzeDoc) -> list[dict[str, Any]]:
        """Bronze -> silver: flatten/clean into per-source records tagged with
        entity_type. Return [] to skip a document."""

    @abstractmethod
    def promote(self) -> dict[str, int]:
        """Silver -> gold: build canonical entities with Provenance. Returns row counts
        by entity_type. Entity resolution (dedup/merge) hooks in here."""

    async def run(self, limit: int | None = None) -> dict[str, int]:
        """Full pipeline for this connector. Returns a summary of what landed."""
        n_bronze = 0
        async for doc in _stream(self.fetch(limit)):
            self.store.put_bronze(doc)
            n_bronze += 1
            for i, rec in enumerate(self.normalize(doc)):
                etype = rec.get("entity_type", "unknown")
                sid = f"{doc.id}:{etype}:{i}"
                self.store.put_silver(sid, self.source_id, etype, doc.id, rec)
        gold_counts = self.promote()
        # Brand-qualified names are derived from gold, so they go stale the moment a
        # promote lands new products or brands. Refreshing here means no caller has to
        # remember to.
        self.store.refresh_search_names()
        return {"bronze": n_bronze, **{f"gold_{k}": v for k, v in gold_counts.items()}}


async def _stream(maybe_iter: Any) -> AsyncIterator[BronzeDoc]:
    """Accept either a sync or an async iterable of BronzeDoc, and yield as they arrive.

    This used to build the whole list first. That is fine for a fixture and ruinous for a
    long crawl: a TTB walk over months of the registry would hold every document in memory
    and write nothing until the last one landed, so any failure — or an interrupt — threw
    away the entire run. Streaming means rows are durable as they are fetched, and a run
    that dies partway keeps what it already got.
    """
    if hasattr(maybe_iter, "__aiter__"):
        async for doc in maybe_iter:
            yield doc
    else:
        for doc in maybe_iter:
            yield doc
