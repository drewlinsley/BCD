"""Append-only fetch evidence log.

Every fetch decision — allowed or denied — is written here as one JSON line. This is
the audit trail that lets us prove, per URL, what robots.txt said and when we fetched.
Given the crawl posture, this is not optional bookkeeping; it's the paper trail.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


@dataclass
class FetchEvent:
    source_id: str
    url: str
    host: str
    decision: str
    reason: str
    status_code: int | None = None
    bytes_len: int | None = None
    user_agent: str = ""
    ts: str = ""


class EvidenceLog:
    """Thread-safe JSONL appender."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: FetchEvent) -> None:
        if not event.ts:
            event.ts = datetime.now(UTC).isoformat()
        line = json.dumps(asdict(event), separators=(",", ":"))
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
