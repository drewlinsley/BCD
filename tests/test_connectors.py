"""Connector naming + config — small units that shape what lands and what an overlay says."""

from __future__ import annotations

from bcd_ingest.connectors.openfoodfacts import (
    OpenFoodFactsConnector,
    _is_nonbeverage,
    _is_weak_name,
)
from bcd_ingest.connectors.openfoodfacts import _display_name as _off_name
from bcd_ingest.connectors.ttb_cola import _display_name


def test_off_country_filter_from_env(monkeypatch):
    # BCD_OFF_COUNTRY targets a market (e.g. US) instead of OFF's Euro-first ordering, without
    # threading a flag through the generic registry/CLI.
    monkeypatch.setenv("BCD_OFF_COUNTRY", "united-states")
    assert OpenFoodFactsConnector(store=None).country == "united-states"
    # An explicit argument still wins over the env var.
    assert OpenFoodFactsConnector(store=None, country="france").country == "france"


def test_off_country_none_by_default(monkeypatch):
    monkeypatch.delenv("BCD_OFF_COUNTRY", raising=False)
    assert OpenFoodFactsConnector(store=None).country is None


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


# ---- OFF weak-name gate: bare classes + non-Latin foreign-market rows ----

def test_off_bare_spirit_classes_are_weak():
    # "Tequila"/"Mezcal"/"Bourbon" alone name no product any more than "Gin" does; they slipped
    # the gate before and landed as bare, false-positive-prone catalog rows.
    for w in ("Tequila", "tequila", "Mezcal", "Bourbon", "Scotch", "Brandy"):
        assert _is_weak_name(w), w


def test_off_non_latin_names_are_weak():
    # Cyrillic / Hebrew / Arabic product_names can't render in the HUD or match Latin OCR.
    for w in ("Куантро Ликер крепкий", "הוגרדן פחית", "كوكا ميني", "Бира"):
        assert _is_weak_name(w), w


def test_off_real_short_names_stay_strong():
    for ok in ("1664", "J&B", "Bud", "OB"):
        assert not _is_weak_name(ok), ok


def test_off_recovers_non_latin_name_to_latin_brand():
    # A non-Latin name with a Latin brand collapses to the brand alone — no script-mixing.
    assert _off_name("Виски ирландский купажированный Джемесон", "Jameson") == "Jameson"
    assert _off_name("הוגרדן פחית", "Hoegaarden") == "Hoegaarden"
    assert _off_name("Куантро Ликер крепкий", "Cointreau") == "Cointreau"


def test_off_drops_unsalvageable_rows():
    # Non-Latin name + non-Latin/empty brand, or a bare class with no brand -> nothing usable.
    assert _off_name("Ариана светло", "Ариана") is None
    assert _off_name("Светла бира", "") is None
    assert _off_name("Tequila", "") is None


def test_off_keeps_ascii_qualifier_but_drops_non_latin_one():
    # A pure-ASCII age/style qualifier is kept ("Glenfiddich 15"); a non-Latin one is dropped so
    # the label never mixes scripts ("Stella Artois", not "Stella Artois Бира 5%").
    assert _off_name("15", "Glenfiddich") == "Glenfiddich 15"
    assert _off_name("Gin", "Hendrick's") == "Hendrick's Gin"
    assert _off_name("Бира 5%", "Stella Artois") == "Stella Artois"


def test_off_soda_double_tagged_as_beer_is_nonbeverage():
    # OFF double-tags a soft drink with a beer category; the gate drops it, real styles survive.
    assert _is_nonbeverage("كوكا ميني", "Beers, Sodas")
    assert _is_nonbeverage("Fizzy Orange", "Soft drink")
    assert not _is_nonbeverage("Chocolate Stout", "Beers, Stouts")
