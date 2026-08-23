"""Connector naming — the part that decides what an overlay actually says."""

from __future__ import annotations

from bcd_ingest.connectors.ttb_cola import _display_name


def test_ttb_name_leads_with_brand_when_fanciful_is_generic():
    # TTB's fanciful_name is often just the class/type ("Pale Ale"); the overlay must carry the
    # brand, or it reads as a generic descriptor and trigram-matches any stray OCR of that style.
    assert _display_name("Sierra Nevada", "Pale Ale") == "Sierra Nevada Pale Ale"
    assert (_display_name("Buffalo Trace", "Kentucky Straight Bourbon Whiskey")
            == "Buffalo Trace Kentucky Straight Bourbon Whiskey")


def test_ttb_name_skips_prepend_when_brand_already_present():
    assert _display_name("Sierra Nevada", "Sierra Nevada Torpedo") == "Sierra Nevada Torpedo"


def test_ttb_name_brand_alone_when_no_fanciful():
    assert _display_name("Lagavulin", None) == "Lagavulin"
    assert _display_name("Lagavulin", "") == "Lagavulin"
