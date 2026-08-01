#!/usr/bin/env python3
"""Validate every data/registry/sources/*.yaml against data/registry/schema.json.

`make validate-registry`. Exits non-zero on the first invalid file. A missing license or
legal_basis is a hard failure — that is the guardrail that keeps the crawl posture honest.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import yaml
from jsonschema import Draft7Validator

HERE = os.path.dirname(__file__)
SCHEMA = os.path.normpath(os.path.join(HERE, "..", "data", "registry", "schema.json"))
SOURCES = os.path.normpath(os.path.join(HERE, "..", "data", "registry", "sources"))


def main() -> int:
    with open(SCHEMA, encoding="utf-8") as f:
        validator = Draft7Validator(json.load(f))

    files = sorted(glob.glob(os.path.join(SOURCES, "*.yaml")))
    if not files:
        print(f"no source files in {SOURCES}", file=sys.stderr)
        return 1

    errors = 0
    ids: set[str] = set()
    for path in files:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        name = os.path.basename(path)
        if doc.get("id") in ids:
            print(f"✗ {name}: duplicate id '{doc.get('id')}'")
            errors += 1
        ids.add(doc.get("id"))
        if doc.get("id") != name[:-5]:
            print(f"✗ {name}: id '{doc.get('id')}' != filename")
            errors += 1
        for err in validator.iter_errors(doc):
            loc = "/".join(str(p) for p in err.path) or "(root)"
            print(f"✗ {name}: {loc}: {err.message}")
            errors += 1

    if errors:
        print(f"\n{errors} error(s) across {len(files)} files")
        return 1
    print(f"✓ {len(files)} source files valid")
    _summary(files)
    return 0


def _summary(files: list[str]) -> None:
    from collections import Counter
    tiers: Counter = Counter()
    provides: Counter = Counter()
    blocked = sentinels = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        tiers[doc["tier"]] += 1
        for p in doc["provides"]:
            provides[p] += 1
        if doc["status"] == "blocked":
            blocked += 1
        if doc.get("sentinel"):
            sentinels += 1
    print(f"  tiers: {dict(sorted(tiers.items()))}")
    print(f"  sentinels: {sentinels} · blocked (ToS): {blocked}")
    print(f"  top provides: {dict(provides.most_common(6))}")


if __name__ == "__main__":
    raise SystemExit(main())
