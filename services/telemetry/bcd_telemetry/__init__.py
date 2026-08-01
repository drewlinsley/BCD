"""Telemetry warehouse-side package. Event names come from generated_events.py, which is
codegen'd from telemetry/events.yaml — never hand-edit either the Swift or Python binding.
"""

try:
    from .generated_events import EVENT_NAMES, EVENT_TIERS
except ImportError:  # not yet generated
    EVENT_NAMES = frozenset()
    EVENT_TIERS = {}

__all__ = ["EVENT_NAMES", "EVENT_TIERS"]
