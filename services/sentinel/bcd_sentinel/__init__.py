"""Sentinel orchestration — loads sentinels/*.yaml and drives Parallel Monitor + FindAll.

Thin, dependency-light loader/validator here; the network calls to Parallel go through
`ParallelClient`. A dry-run validates configs and issues a single cheap FindAll `preview`
call to confirm the API key works end to end (`make sentinel-dryrun`).
"""

from __future__ import annotations

import glob
import os

import yaml

SENTINEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                             "sentinels"))


def load_all() -> dict[str, list[dict]]:
    """Return {'monitors': [...], 'find_all': [...]} across every sentinels/*.yaml."""
    out: dict[str, list[dict]] = {"monitors": [], "find_all": []}
    for path in sorted(glob.glob(os.path.join(SENTINEL_DIR, "*.yaml"))):
        with open(path, encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}
        defaults = spec.get("defaults", {})
        for m in spec.get("monitors", []):
            out["monitors"].append({**defaults, **m, "_file": os.path.basename(path)})
        for fa in spec.get("find_all", []):
            out["find_all"].append({**defaults, **fa, "_file": os.path.basename(path)})
    return out


def validate() -> list[str]:
    """Return a list of problems; empty means all sentinel configs are well-formed."""
    problems: list[str] = []
    loaded = load_all()
    seen: set[str] = set()
    for kind in ("monitors", "find_all"):
        for job in loaded[kind]:
            jid = job.get("id")
            if not jid:
                problems.append(f"{job.get('_file')}: {kind} entry missing id")
                continue
            if jid in seen:
                problems.append(f"duplicate sentinel id '{jid}'")
            seen.add(jid)
            if not job.get("objective"):
                problems.append(f"{jid}: missing objective")
    return problems


__all__ = ["load_all", "validate", "SENTINEL_DIR"]
