"""Open Brewery DB connector — Tier A, open, no auth, no rate limit.

Provides the producer + venue spine on day one. Maps each brewery to a canonical
`Producer` gold entity with provenance. This is the connector the verification step runs
live: `make ingest SOURCE=openbrewerydb`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from bcd_schema import Producer

from ..base import Connector
from ..store import BronzeDoc, doc_id

API = "https://api.openbrewerydb.org/v1/breweries"


class OpenBreweryDBConnector(Connector):
    source_id = "openbrewerydb"
    provides = ("producer", "venue")

    async def fetch(self, limit: int | None = None) -> AsyncIterator[BronzeDoc]:
        per_page = 50
        fetched = 0
        page = 1
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                resp = await client.get(
                    API, params={"page": page, "per_page": per_page}
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                for row in batch:
                    yield BronzeDoc(
                        id=doc_id(self.source_id, row["id"]),
                        source_id=self.source_id,
                        natural_key=row["id"],
                        fetched_at="",  # store fills via put_bronze path; set below
                        url=f"{API}/{row['id']}",
                        payload=row,
                    )
                    fetched += 1
                    if limit is not None and fetched >= limit:
                        return
                page += 1

    def normalize(self, doc: BronzeDoc) -> list[dict[str, Any]]:
        r = doc.payload
        lat = _to_float(r.get("latitude"))
        lon = _to_float(r.get("longitude"))
        return [
            {
                "entity_type": "producer",
                "bronze_id": doc.id,
                "natural_key": r.get("id"),
                "name": r.get("name"),
                "kind": r.get("brewery_type"),
                "country": r.get("country"),
                "region": r.get("state_province"),
                "city": r.get("city"),
                "lat": lat,
                "lon": lon,
                "website": r.get("website_url"),
                "url": doc.url,
            }
        ]

    def promote(self) -> dict[str, int]:
        n = 0
        for rec in self.store.iter_silver("producer"):
            if not rec.get("name"):
                continue
            pid = f"prod:{self.source_id}:{rec['natural_key']}"
            producer = Producer(
                id=pid,
                name=rec["name"],
                kind=rec.get("kind"),
                country=rec.get("country"),
                region=rec.get("region"),
                city=rec.get("city"),
                lat=rec.get("lat"),
                lon=rec.get("lon"),
                website=rec.get("website"),
                # OBDB doesn't state ownership; parent_company is filled later from
                # Wikidata. Provenance for these fields is the source_id + bronze trace.
                parent_company=None,
            )
            self.store.put_gold(pid, "producer", producer.model_dump(mode="json"))
            n += 1
        return {"producer": n}


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
