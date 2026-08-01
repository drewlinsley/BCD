"""Crawl policy — the posture-enforcement invariants. If these fail, we crawl illegally."""

from __future__ import annotations

from bcd_crawler import CrawlPolicy, Decision

UA = "BCDBot/0.1 (+https://github.com/drewlinsley/BCD)"


def _policy(**kw) -> CrawlPolicy:
    return CrawlPolicy(user_agent=UA, contact_url="mailto:x@example.com", **kw)


def test_robots_disallow_is_denied():
    pol = _policy()
    pol.set_robots("example.com", "User-agent: *\nDisallow: /private")
    r = pol.check("src", "example.com", "https://example.com/private/page")
    assert r.decision == Decision.DENY_ROBOTS
    assert not r.allowed


def test_robots_allow_passes():
    pol = _policy()
    pol.set_robots("example.com", "User-agent: *\nDisallow: /private")
    r = pol.check("src", "example.com", "https://example.com/public/page")
    assert r.allowed


def test_kill_switch_blocks_source():
    pol = _policy(killed_sources={"banned"})
    pol.set_robots("example.com", "")
    r = pol.check("banned", "example.com", "https://example.com/x")
    assert r.decision == Decision.DENY_KILLED


def test_global_budget_guardrail():
    pol = _policy(max_requests=1)
    pol.set_robots("example.com", "")
    assert pol.check("s", "example.com", "https://example.com/a").allowed
    pol.record_request("example.com")
    r = pol.check("s", "example.com", "https://example.com/b")
    assert r.decision == Decision.DENY_BUDGET


def test_rate_limit_defers_then_allows():
    pol = _policy(default_rate_rps=1000.0)  # tiny interval so the test is fast
    pol.set_robots("example.com", "")
    assert pol.check("s", "example.com", "https://example.com/a").allowed
    pol.record_request("example.com")
    r = pol.check("s", "example.com", "https://example.com/a")
    # immediately after a request, must defer briefly
    assert r.decision == Decision.DEFER_RATE
    assert r.wait_seconds > 0


def test_robots_staleness_ttl():
    pol = _policy(robots_ttl_seconds=0.0)
    pol.set_robots("example.com", "")
    assert pol.robots_stale("example.com")  # ttl 0 => always stale => refetch
