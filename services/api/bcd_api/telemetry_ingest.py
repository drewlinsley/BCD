"""Telemetry collector — our own, not a vendor SDK.

Accepts event batches from the client's durable queue and appends them to a local JSONL
sink (dev) / partitioned warehouse (prod). Validates each event against the codegen'd
event names so client and warehouse cannot drift. Runs the flywheel: scan corrections
land here as self-labeled training data.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime

# Loaded lazily from telemetry/events.yaml via the codegen'd allowlist. Kept minimal
# here; the authoritative set is telemetry/events.yaml.
_FALLBACK_EVENTS = {
    "session_start", "session_end", "scan_session_start", "scan_frame_batch",
    "scan_candidate_shown", "scan_resolved", "scan_corrected_by_user",
    "overlay_impression", "overlay_tap", "product_view", "provenance_expanded",
    "rating_submitted", "list_add", "alert_fired", "alert_converted",
    "agent_task_created", "agent_task_completed", "purchase_intent",
    "paywall_shown", "paywall_converted", "weekly_profile_delta_shown",
}


class TelemetryCollector:
    def __init__(self, root: str = "./data", allowed: set[str] | None = None) -> None:
        self.sink = os.path.join(root, "telemetry_events.jsonl")
        os.makedirs(root, exist_ok=True)
        self.allowed = allowed or _load_allowed() or _FALLBACK_EVENTS

    def ingest(self, batch: dict | bytes) -> list[dict]:
        """Append the batch's valid events and return the accepted ones, so a caller can
        react to what actually landed (e.g. refresh a taste profile) without re-decoding."""
        events = _decode(batch)
        received_at = datetime.now(UTC).isoformat()
        accepted: list[dict] = []
        with open(self.sink, "a", encoding="utf-8") as f:
            for ev in events:
                name = ev.get("name")
                if name not in self.allowed:
                    # Reject unknown events loudly rather than silently storing drift.
                    continue
                ev["_received_at"] = received_at
                f.write(json.dumps(ev, separators=(",", ":")) + "\n")
                accepted.append(ev)
        return accepted

    def iter_events(self, names: set[str] | None = None) -> Iterator[dict]:
        """Replay the sink, oldest first. The taste profile is rebuilt from this, so the
        log is the source of truth for personalization — not a side table that can drift."""
        if not os.path.exists(self.sink):
            return
        with open(self.sink, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn trailing write shouldn't poison the replay
                if names is None or ev.get("name") in names:
                    yield ev


def _decode(batch: dict | bytes) -> list[dict]:
    if isinstance(batch, (bytes, bytearray)):
        try:
            batch = json.loads(gzip.decompress(batch))
        except OSError:
            batch = json.loads(batch)
    if isinstance(batch, dict):
        return batch.get("events", [])
    if isinstance(batch, list):
        return batch
    return []


def _load_allowed() -> set[str] | None:
    """Read event names from telemetry/events.yaml if present."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "telemetry",
                        "events.yaml")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return None
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        return {e["name"] for e in spec.get("events", [])}
    except Exception:
        return None
