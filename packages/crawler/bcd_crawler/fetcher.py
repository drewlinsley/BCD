"""PoliteFetcher — the only sanctioned way to pull a third-party page.

Wraps httpx with the CrawlPolicy and EvidenceLog so that robots, rate limits, identity,
and the audit trail are impossible to bypass by accident. Ingest connectors for Tier-D
sources fetch exclusively through this.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

import httpx

from .evidence import EvidenceLog, FetchEvent
from .policy import CrawlPolicy, Decision


class PoliteFetcher:
    def __init__(
        self,
        policy: CrawlPolicy,
        evidence: EvidenceLog,
        timeout: float = 20.0,
        max_defer_rounds: int = 5,
    ) -> None:
        self.policy = policy
        self.evidence = evidence
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": policy.user_agent},
        )
        self.max_defer_rounds = max_defer_rounds

    async def _ensure_robots(self, source_id: str, host: str, scheme: str) -> None:
        if not self.policy.robots_stale(host):
            return
        robots_url = f"{scheme}://{host}/robots.txt"
        try:
            resp = await self._client.get(robots_url)
            text = resp.text if resp.status_code == 200 else ""
        except httpx.HTTPError:
            text = ""  # unreachable robots => treat as empty (allow), but logged below
        self.policy.set_robots(host, text)

    async def get(self, source_id: str, url: str) -> httpx.Response | None:
        """Fetch `url` if policy allows. Returns the response, or None if denied.

        Every outcome is written to the evidence log.
        """
        parts = urlsplit(url)
        host = parts.netloc
        await self._ensure_robots(source_id, host, parts.scheme or "https")

        for _ in range(self.max_defer_rounds):
            result = self.policy.check(source_id, host, url)
            if result.decision == Decision.DEFER_RATE:
                await asyncio.sleep(result.wait_seconds)
                continue
            break

        if not result.allowed:
            self.evidence.write(
                FetchEvent(
                    source_id=source_id, url=url, host=host,
                    decision=result.decision.value, reason=result.reason,
                    user_agent=self.policy.user_agent,
                )
            )
            return None

        self.policy.record_request(host)
        try:
            resp = await self._client.get(url)
        except httpx.HTTPError as exc:
            self.evidence.write(
                FetchEvent(
                    source_id=source_id, url=url, host=host,
                    decision="error", reason=str(exc),
                    user_agent=self.policy.user_agent,
                )
            )
            return None

        self.evidence.write(
            FetchEvent(
                source_id=source_id, url=url, host=host,
                decision=Decision.ALLOW.value, reason=result.reason or "ok",
                status_code=resp.status_code, bytes_len=len(resp.content),
                user_agent=self.policy.user_agent,
            )
        )
        return resp

    async def aclose(self) -> None:
        await self._client.aclose()
