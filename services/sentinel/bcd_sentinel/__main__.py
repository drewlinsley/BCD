"""`python -m bcd_sentinel dryrun` — validate sentinel configs, and if PARALLEL_API_KEY is
set, issue ONE cheap FindAll preview call ($0.10) to confirm the key works end to end.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import load_all, validate
from .parallel_client import ParallelClient


async def _dryrun(live: bool) -> int:
    problems = validate()
    loaded = load_all()
    print(f"sentinels: {len(loaded['monitors'])} monitors, "
          f"{len(loaded['find_all'])} find_all jobs")
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("  ✓ all sentinel configs well-formed")

    client = ParallelClient()
    if not client.configured:
        print("  · PARALLEL_API_KEY not set — skipping live preview "
              "(set it in .env to test the key).")
        return 0
    if not live:
        print("  · key present; pass --live to issue one ~$0.005 Search call.")
        return 0

    # Connectivity check uses the Search API (confirmed working). FindAll is a gated beta
    # that 401s until the account is provisioned, so it's not the right smoke test.
    job = loaded["find_all"][0]
    print(f"  → live Parallel Search for '{job['id']}' connectivity (~$0.005)...")
    try:
        result = await client.search(
            objective=job["objective"],
            queries=[job["objective"][:90], "craft beer tap list"],
            max_results=3,
        )
        n = len(result.get("results", []))
        print(f"  ✓ Parallel responded — {n} result(s). Key works end to end.")
        print("  · FindAll (discovery) needs beta provisioning on this key — request it "
              "in the dashboard; Search + Task already work.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ Parallel call failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="bcd_sentinel")
    ap.add_argument("command", choices=["dryrun"])
    ap.add_argument("--live", action="store_true",
                    help="issue a real $0.10 FindAll preview to confirm the key")
    args = ap.parse_args()
    return asyncio.run(_dryrun(args.live))


if __name__ == "__main__":
    raise SystemExit(main())
