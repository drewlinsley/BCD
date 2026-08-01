# 05 — Telemetry

"Massive and total" usage telemetry — with the one engineering move that keeps it honest: **[telemetry/events.yaml](../telemetry/events.yaml) is the single source of truth**, and `make codegen` emits both the Swift enum ([TelemetryEvents.swift](../ios/BCDKit/Sources/BCDKit/Generated/TelemetryEvents.swift)) and the Python warehouse allowlist ([generated_events.py](../services/telemetry/bcd_telemetry/generated_events.py)). Client and warehouse **cannot drift**; `make codegen && git diff --exit-code` proves it in CI.

## The flywheel

`scan_frame_batch` (derived OCR strings + latency + resolution outcome) plus `scan_corrected_by_user` (the user overrode our resolution) is a **self-labeling training set** for the resolver. Every correction improves the model that made the mistake. Instrument this before anything else.

## Event taxonomy (21 events)

| Group | Events |
|---|---|
| session | `session_start`, `session_end` |
| **scan funnel** | `scan_session_start`, `scan_frame_batch`, `scan_candidate_shown`, `scan_resolved`, `scan_corrected_by_user` |
| overlays / detail | `overlay_impression`, `overlay_tap`, `product_view`, `provenance_expanded` |
| ratings / lists | `rating_submitted`, `list_add` |
| alerts / agents | `alert_fired`, `alert_converted`, `agent_task_created`, `agent_task_completed` |
| commerce | `purchase_intent`, `paywall_shown`, `paywall_converted` |
| weekly evolution | `weekly_profile_delta_shown` |

`provenance_expanded` is worth calling out — it measures whether verifiability actually matters to users (do they tap the chips?), which validates the entire data thesis.

## Client (durable, offline-safe)

[TelemetryQueue](../ios/BCDKit/Sources/BCDKit/Telemetry.swift) is a disk-backed, batched actor: events persist across launches until the collector acknowledges them, so **events are never dropped**. Batches are gzipped and retried. The queue only clears *after* the sink accepts — verified by a test.

## Own collector, not a vendor SDK

The behavioral data **is** the monetizable asset; shipping it to a third party gives it away. [TelemetryCollector](../services/api/bcd_api/telemetry_ingest.py) validates every event against the codegen'd allowlist and rejects unknown names (no silent schema drift), then appends to JSONL (dev) / a partitioned warehouse (prod).

## Privacy — designed in from commit one

Alcohol consumption is health-adjacent and counts as sensitive under CPRA/GDPR. This is not retrofittable, so it's here from the start:

- **Three separate consent tiers** — `analytics`, `personalization`, `data_sharing` — each an independent opt-in. An event tagged `personalization` is **silently dropped** unless that tier is granted (enforced in `TelemetryQueue`, verified by a test).
- **Pseudonymous, rotatable on-device install id.** No account required to use the app.
- **Raw camera frames are never uploaded** — by construction, they are not an event field. Only derived OCR strings, under the personalization tier.
- **ATT prompt**, a documented retention schedule, and working **export / delete** endpoints (surfaced in the You tab).

The data-licensing and ad-targeting business models are only legal if this architecture exists first — see [06-legal.md](06-legal.md).
