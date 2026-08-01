"""Crawl policy — the chosen posture ('everything public, robots.txt-gated') enforced
in code, not in a doc nobody reads.

Every fetch goes through `CrawlPolicy.check()`. It:
  - parses and honors robots.txt per host (via Protego),
  - enforces a per-host rate cap,
  - respects a per-source kill switch and a global crawl-cost guardrail,
  - and returns a decision that the fetcher logs to an append-only evidence trail.

Given the legal exposure of the posture, this module is the mitigation. Keep it strict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from protego import Protego


class Decision(str, Enum):
    ALLOW = "allow"
    DENY_ROBOTS = "deny_robots"
    DENY_KILLED = "deny_killed"  # source kill switch
    DENY_BUDGET = "deny_budget"  # global cost guardrail tripped
    DEFER_RATE = "defer_rate"  # allowed, but must wait


@dataclass
class CheckResult:
    decision: Decision
    wait_seconds: float = 0.0
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision in (Decision.ALLOW, Decision.DEFER_RATE)


@dataclass
class HostState:
    robots: Protego | None = None
    robots_fetched_at: float = 0.0
    last_request_at: float = 0.0
    request_count: int = 0


@dataclass
class CrawlPolicy:
    """Per-run policy object. One instance guards a whole crawl session."""

    user_agent: str
    contact_url: str
    default_rate_rps: float = 0.5  # 1 request / 2s per host unless overridden
    per_host_rate_rps: dict[str, float] = field(default_factory=dict)
    killed_sources: set[str] = field(default_factory=set)
    max_requests: int | None = None  # global guardrail; None = unlimited
    robots_ttl_seconds: float = 3600.0

    _hosts: dict[str, HostState] = field(default_factory=dict, init=False)
    _total_requests: int = field(default=0, init=False)

    # robots.txt is fetched by the caller (needs async I/O); we just cache the text.
    def set_robots(self, host: str, robots_txt: str) -> None:
        state = self._hosts.setdefault(host, HostState())
        state.robots = Protego.parse(robots_txt or "")
        state.robots_fetched_at = time.monotonic()

    def robots_stale(self, host: str) -> bool:
        state = self._hosts.get(host)
        if state is None or state.robots is None:
            return True
        return (time.monotonic() - state.robots_fetched_at) > self.robots_ttl_seconds

    def check(self, source_id: str, host: str, url: str) -> CheckResult:
        if source_id in self.killed_sources:
            return CheckResult(Decision.DENY_KILLED, reason=f"source '{source_id}' killed")

        if self.max_requests is not None and self._total_requests >= self.max_requests:
            return CheckResult(Decision.DENY_BUDGET, reason="global request budget exhausted")

        state = self._hosts.setdefault(host, HostState())

        if state.robots is not None and not state.robots.can_fetch(url, self.user_agent):
            return CheckResult(Decision.DENY_ROBOTS, reason="robots.txt disallow")

        # rate limiting — honor robots crawl-delay if it exceeds our cap
        rps = self.per_host_rate_rps.get(host, self.default_rate_rps)
        min_interval = 1.0 / rps if rps > 0 else 0.0
        if state.robots is not None:
            crawl_delay = state.robots.crawl_delay(self.user_agent)
            if crawl_delay:
                min_interval = max(min_interval, float(crawl_delay))

        now = time.monotonic()
        elapsed = now - state.last_request_at if state.last_request_at else min_interval
        if elapsed < min_interval:
            return CheckResult(
                Decision.DEFER_RATE,
                wait_seconds=min_interval - elapsed,
                reason=f"rate cap {rps} rps",
            )
        return CheckResult(Decision.ALLOW)

    def record_request(self, host: str) -> None:
        """Call after a fetch actually goes out, to advance rate + budget counters."""
        state = self._hosts.setdefault(host, HostState())
        state.last_request_at = time.monotonic()
        state.request_count += 1
        self._total_requests += 1
