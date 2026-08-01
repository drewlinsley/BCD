"""Thin async client for the Parallel.ai APIs BCD uses: FindAll, Monitor, Task, Search,
Extract. Only the surface we need. Reads PARALLEL_API_KEY from the environment.

Cost note baked in as a reminder: our own crawler does volume at ~$0.0001/page; Parallel
is for discovery, standing monitors, and hard extractions only.
"""

from __future__ import annotations

import os

import httpx

BASE = "https://api.parallel.ai"


class ParallelClient:
    def __init__(self, api_key: str | None = None, base: str = BASE) -> None:
        self.api_key = api_key or os.environ.get("PARALLEL_API_KEY", "")
        self.base = base

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "content-type": "application/json"}

    async def search(self, objective: str, queries: list[str],
                     processor: str = "base", max_results: int = 3) -> dict:
        """Search API — the cheapest confirmed-working call (~$0.001-0.005). Used by the
        dry-run as the end-to-end key check, and by discovery/enrichment for page-finding."""
        payload = {
            "objective": objective,
            "search_queries": queries,
            "processor": processor,
            "max_results": max_results,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base}/v1beta/search",
                                     headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def findall_preview(self, objective: str,
                              match_conditions: list[str] | None = None) -> dict:
        """FindAll preview ($0.10 fixed). NOTE: FindAll is a gated beta — this returns 401
        until the account is provisioned for it, even with a valid key (Search/Task work).
        Request FindAll access in the Parallel dashboard, then this lights up."""
        payload = {
            "objective": objective,
            "match_conditions": match_conditions or [],
            "generator": "preview",
            "match_limit": 5,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base}/v1beta/find_all",
                                     headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def create_monitor(self, objective: str, keywords: list[str],
                             frequency: str = "6h", processor: str = "base",
                             webhook_url: str | None = None) -> dict:
        payload = {
            "type": "event_stream",
            "objective": objective,
            "keywords": keywords,
            "frequency": frequency,
            "processor": processor,
        }
        if webhook_url:
            payload["webhook"] = {"url": webhook_url}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base}/v1beta/monitor",
                                     headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()
