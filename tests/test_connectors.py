"""Connector naming + config — small units that shape what lands and what an overlay says."""

from __future__ import annotations

from datetime import date

import pytest
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


# ---- generic names get anchored on their brand ---------------------------------------

@pytest.mark.parametrize("pn, brand, expected", [
    # multi-word category names: many producers sell each, so they must be told apart
    ("Blended Canadian Whiskey", "Crown Royal", "Crown Royal Blended Canadian Whiskey"),
    ("Rhum blanc agricole", "HSE", "HSE Rhum blanc agricole"),
    ("London Dry Gin", "Beefeater", "Beefeater London Dry Gin"),
    ("Bière Blonde", "Lidl", "Lidl Bière Blonde"),
    # a real product name is left alone, even when it contains category words
    ("Punk IPA", "BrewDog", "Punk IPA"),
    ("London Pride", "Fuller's", "London Pride"),
    ("Lagunitas IPA", "Lagunitas", "Lagunitas IPA"),
    # brand already stated in the name -> no stutter
    ("Cerveza Heineken", "Heineken", "Cerveza Heineken"),
])
def test_generic_names_are_brand_qualified(pn, brand, expected):
    assert _off_name(pn, brand) == expected


def test_generic_name_without_a_brand_is_left_as_is():
    # Nothing to anchor on; the row still exists, it just stays generic.
    assert _off_name("Blended Scotch Whisky", "") == "Blended Scotch Whisky"


def _ttb_page(first: int, last: int, total: int) -> str:
    """A result page shaped like the registry's: a range header and `last - first + 1` rows."""
    rows = "".join(
        "<tr>"
        f'<td><a href="viewColaDetails.do?ttbid={26000000000000 + n}">{n}</a></td>'
        "<td>DSP-X-1</td><td>1</td><td>08/24/2026</td><td></td>"
        f"<td>BRAND {n}</td><td>06</td><td>MICHIGAN</td><td>301</td><td>VODKA</td>"
        "</tr>"
        for n in range(first, last + 1)
    )
    return f"<html><body>{first} to {last} of {total}<table>{rows}</table></body></html>"


def test_ttb_pagination_stops_on_an_exact_multiple_of_the_page_size():
    # The registry's paginator clamps: ask past the end and it re-serves the last page
    # forever. With 140 records and 20 to a page the final page is full, so "a short page
    # is the last page" never fires and the walk spins, re-fetching rows it already has.
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    pages = [_ttb_page(1 + i * 20, 20 + i * 20, 140) for i in range(7)]
    served: list[str] = []

    class _Client:
        def request(self, method, url, data=None):
            # Past the end, hand back the last page again — exactly what TTB does.
            page = pages[len(served)] if len(served) < len(pages) else pages[-1]
            served.append(page)

            class _R:
                text = page

                @staticmethod
                def raise_for_status():
                    return None

            return _R()

    conn = TTBColaConnector(store=None, use_fixture=False)
    rows = list(conn._walk_day(_Client(), date(2026, 8, 24), "100", "699"))

    assert len(rows) == 140, "every record once"
    assert len({r["ttb_id"] for r in rows}) == 140, "and none of them twice"
    assert len(served) == 7, "no request past the last page"


def test_ttb_pagination_stops_when_the_header_is_missing():
    # No range header (a layout change, an error page): fall back to the short-page rule
    # rather than paging forever.
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    page = _ttb_page(1, 5, 5).replace("1 to 5 of 5", "")

    class _Client:
        def request(self, method, url, data=None):
            class _R:
                text = page

                @staticmethod
                def raise_for_status():
                    return None

            return _R()

    conn = TTBColaConnector(store=None, use_fixture=False)
    assert len(list(conn._walk_day(_Client(), date(2026, 8, 24), "100", "699"))) == 5


def test_ttb_parse_range_reads_the_result_header():
    from bcd_ingest.connectors.ttb_cola import parse_range

    assert parse_range("<b>121 to 140 of 140</b>") == (121, 140, 140)
    assert parse_range("no results") is None


def test_ttb_category_covers_the_registry_vocabulary():
    # TTB's class/type descriptions are its own vocabulary, not ours. A thin map filed
    # 4,019 products under `other` — including every stout, porter, brandy, liqueur and
    # agave spirit — where they can be neither styled nor recommended.
    from bcd_ingest.connectors.ttb_cola import _category_of

    for text, want in [
        ("Stout", "beer"), ("Porter", "beer"), ("Ale", "beer"),
        ("Malt Beverages Specialities - Flavored", "beer"),
        ("Agave Spirits", "spirit"), ("Mezcal Fb", "spirit"),
        ("Cognac (brandy) Fb", "spirit"), ("Apple Brandy (calvados)", "spirit"),
        ("Triple Sec", "spirit"), ("Peppermint Schnapps", "spirit"),
        ("Dairy Cream Liqueur/cordial", "spirit"), ("Neutral Spirits - Grain", "spirit"),
        ("Anisette, Ouzo, Ojen", "spirit"), ("Bitters - Beverage", "spirit"),
        ("Sake - Imported", "sake"),
    ]:
        assert _category_of(text).value == want, text


def test_ttb_single_malt_is_a_whisky_not_a_malt_beverage():
    # The map is matched as substrings, first hit wins, so order is load-bearing:
    # "Straight American Single Malt" carries no whisky word and must not be reached by
    # the beer entry that "malt" would otherwise suggest.
    from bcd_ingest.connectors.ttb_cola import _category_of

    assert _category_of("Straight American Single Malt").value == "spirit"
    assert _category_of("Single Malt Scotch Whisky").value == "spirit"
    assert _category_of("Malt Beverages Specialities - Flavored").value == "beer"
