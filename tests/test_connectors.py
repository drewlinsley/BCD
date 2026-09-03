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
    return (f"<html><body>{first} to {last} of {total} "
            f"(Total Matching Records: {total})<table>{rows}</table></body></html>")


def test_ttb_pagination_stops_on_an_exact_multiple_of_the_page_size():
    # The registry's paginator clamps: ask past the end and it re-serves the last page
    # forever. With 140 records and 20 to a page the final page is full, so "a short page
    # is the last page" never fires and the walk spins, re-fetching rows it already has.
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    pages = [_ttb_page(1 + i * 20, 20 + i * 20, 140) for i in range(7)]
    fetched: list[str] = []

    class _Client:
        def request(self, method, url, data=None):
            # Page 1 is already in the caller's hand, so the Nth request serves page N+1.
            # Past the end, hand back the last page again — exactly what TTB does.
            i = len(fetched) + 1
            page = pages[i] if i < len(pages) else pages[-1]
            fetched.append(page)

            class _R:
                text = page

                @staticmethod
                def raise_for_status():
                    return None

            return _R()

    conn = TTBColaConnector(store=None, use_fixture=False)
    rows = list(conn._paginate(_Client(), pages[0], "08/24/2026", "100", "699"))

    assert len(rows) == 140, "every record once"
    assert len({r["ttb_id"] for r in rows}) == 140, "and none of them twice"
    assert len(fetched) == 6, "six more requests after the page already in hand"


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
    assert len(list(conn._paginate(_Client(), page, "08/24/2026", "100", "699"))) == 5


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


def _ttb_csv(n: int, start: int = 1) -> str:
    head = ("TTB ID,Permit No.,Serial Number,Completed Date,Fanciful Name,"
            "Brand Name,Origin,Origin Desc,Class/Type,Class/Type Desc")
    rows = "\n".join(
        f"'{26000000000000 + i}',DSP-X-1,{i},08/24/2026,,BRAND {i},06,MICHIGAN,301,VODKA"
        for i in range(start, start + n)
    )
    return f"{head}\n{rows}\n"


def _search_page(total: int) -> str:
    return f"<html><body>Total Matching Records: {total}<table></table></body></html>"


class _Recorder:
    """Serves a scripted reply per request and records what was asked for."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def request(self, method, url, data=None):
        self.calls.append((method, url, dict(data or {})))
        text = self.replies.pop(0)

        class _R:
            @staticmethod
            def raise_for_status():
                return None

        _R.text = text
        return _R()


def test_ttb_uses_the_csv_export_when_the_window_fits():
    # One request for the whole window instead of one per twenty rows.
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    client = _Recorder([_search_page(140), _ttb_csv(140)])
    conn = TTBColaConnector(store=None, use_fixture=False)
    rows = list(conn._walk_range(client, date(2026, 8, 20), date(2026, 8, 24), "100", "699"))

    assert len(rows) == 140
    assert rows[0]["ttb_id"] == "26000000000001", "quotes stripped from the id"
    assert rows[0]["brand_name"] == "BRAND 1"
    assert len(client.calls) == 2, "one search, one export — no paging"


def test_ttb_bisects_a_window_the_export_would_truncate():
    # The export silently returns ~1,000 rows for a search matching more, so an oversized
    # window must be halved on the search's own stated total, never exported and trusted.
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    client = _Recorder([
        _search_page(2400),                    # 5-day window: too big
        _search_page(900), _ttb_csv(900, 1),   # first half fits
        _search_page(800), _ttb_csv(800, 901),  # second half fits
    ])
    conn = TTBColaConnector(store=None, use_fixture=False)
    rows = list(conn._walk_range(client, date(2026, 8, 20), date(2026, 8, 24), "100", "699"))

    assert len(rows) == 1700
    assert len({r["ttb_id"] for r in rows}) == 1700, "halves must not overlap"
    spans = [(c[2].get("searchCriteria.dateCompletedFrom"),
              c[2].get("searchCriteria.dateCompletedTo"))
             for c in client.calls if c[0] == "POST"]
    assert spans == [("08/20/2026", "08/24/2026"),
                     ("08/20/2026", "08/22/2026"),
                     ("08/23/2026", "08/24/2026")], "halves must tile the window exactly"


def test_ttb_falls_back_to_paging_when_the_export_comes_back_short():
    # Belt and braces: if the export truncates anyway, the row count disagrees with the
    # total the search reported, and we page the table rather than accept silent loss.
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    client = _Recorder([
        _ttb_page(1, 20, 40),    # search says 40, and carries rows 1-20 as TTB's does
        _ttb_csv(25),            # export gives 25 — short
        _ttb_page(21, 40, 40),   # ...so page the table instead
    ])
    conn = TTBColaConnector(store=None, use_fixture=False)
    rows = list(conn._walk_range(client, date(2026, 8, 24), date(2026, 8, 24), "100", "699"))

    assert len(rows) == 40, "all rows come from the table, not the short export"
    assert len({r["ttb_id"] for r in rows}) == 40


def test_ttb_empty_window_costs_one_request():
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    client = _Recorder([_search_page(0)])
    conn = TTBColaConnector(store=None, use_fixture=False)
    assert list(conn._walk_range(client, date(2026, 8, 20), date(2026, 8, 24), "1", "2")) == []
    assert len(client.calls) == 1, "no export for an empty result set"


def test_ttb_parse_total_reads_the_stated_match_count():
    from bcd_ingest.connectors.ttb_cola import parse_total

    assert parse_total("<b>Total Matching Records: 29811</b>") == 29811
    assert parse_total("no records were found") == 0


_CSV_HEAD = ("TTB ID,Permit No.,Serial Number,Completed Date,Fanciful Name,"
             "Brand Name,Origin,Origin Desc,Class/Type,Class/Type Desc")


def test_ttb_export_repairs_a_comma_split_row():
    # TTB does not quote a field containing a comma, so a fanciful name splits in two and
    # shifts every column after it — putting a state in the date column and a class code
    # in the id. Real row from the registry.
    from bcd_ingest.connectors.ttb_cola import parse_export

    csv_text = (f"{_CSV_HEAD}\n"
                "'25255001000115',TX-I-21177,25RMCL,09/12/2025,COYOTÁ CAPÓN,"
                " TOBALÁ CAPÓN,REAL MINERO,81,MEXICO,983,AGAVE SPIRITS\n")
    (row,) = parse_export(csv_text)
    assert row["ttb_id"] == "25255001000115"
    assert row["completed_date"] == "09/12/2025"
    assert row["brand_name"] == "REAL MINERO"
    assert row["fanciful_name"] == "COYOTÁ CAPÓN, TOBALÁ CAPÓN"
    assert row["origin_desc"] == "MEXICO"
    assert row["class_type"] == "AGAVE SPIRITS"


def test_ttb_export_keeps_commas_in_the_class_description():
    # The tail is found from the RIGHTMOST numeric pair, because the class description
    # carries commas of its own and must not be mistaken for the shifted middle.
    from bcd_ingest.connectors.ttb_cola import parse_export

    csv_text = (f"{_CSV_HEAD}\n"
                "'25255001000999',DSP-CA-1,25X,03/04/2025,SOME, FANCY, NAME,"
                "BRANDCO,05,CALIFORNIA,255,ANISETTE, OUZO, OJEN\n")
    (row,) = parse_export(csv_text)
    assert row["brand_name"] == "BRANDCO"
    assert row["fanciful_name"] == "SOME, FANCY, NAME"
    assert row["origin_desc"] == "CALIFORNIA"
    assert row["class_type"] == "ANISETTE, OUZO, OJEN"


def test_ttb_export_rejoins_a_record_split_by_a_newline():
    # A newline inside an unquoted field tears one record across two physical lines, so
    # records are segmented on the id column, not on line breaks.
    from bcd_ingest.connectors.ttb_cola import parse_export

    csv_text = (f"{_CSV_HEAD}\n"
                "'25269001000226',BR-TX-20212,25WASC,09/29/2025,,THE WANDERING SCAPEGOAT\n"
                "BARREL AGED SOUR ALE,44,TEXAS,902,ALE\n"
                "'25269001000340',BR-OH-20080,25FBQU,11/13/2025,FIG BELGIAN QUAD,"
                "URBAN ARTIFACT,35,OHIO,902,ALE\n")
    rows = parse_export(csv_text)
    assert len(rows) == 2, "a torn record is one record, and does not eat the next one"
    assert rows[0]["brand_name"] == "THE WANDERING SCAPEGOAT BARREL AGED SOUR ALE"
    assert rows[0]["origin_desc"] == "TEXAS"
    assert rows[1]["ttb_id"] == "25269001000340"
    assert rows[1]["brand_name"] == "URBAN ARTIFACT"


def test_ttb_export_drops_a_row_it_cannot_trust():
    # Better to lose one row than to store a shifted one: a shifted row is not visibly
    # wrong downstream, it is just a product with a state where its date should be.
    from bcd_ingest.connectors.ttb_cola import parse_export

    csv_text = f"{_CSV_HEAD}\n'25255001000115',TX-I-1,25X,NOT-A-DATE,,BRAND,81,MEXICO,983,X\n"
    assert parse_export(csv_text) == []


def test_ttb_export_repairs_an_import_whose_origin_code_is_not_numeric():
    # Origin codes are not all numeric — Moldova files under "6J" — so the repair anchors
    # on the three-digit class code alone. Real row; requiring a numeric origin code lost
    # every import that also carried a comma in its class description.
    from bcd_ingest.connectors.ttb_cola import parse_export

    csv_text = (f"{_CSV_HEAD}\n"
                "'26232001000301',IL-I-21021,260040,08/26/2026,SURPRISE 10 YEARS AGED,"
                "KVINT,6J,MOLDOVA,588,OTHER GRAPE BRANDY (PISCO, GRAPPA) FB\n")
    (row,) = parse_export(csv_text)
    assert row["ttb_id"] == "26232001000301"
    assert row["fanciful_name"] == "SURPRISE 10 YEARS AGED"
    assert row["brand_name"] == "KVINT"
    assert row["origin_code"] == "6J"
    assert row["origin_desc"] == "MOLDOVA"
    assert row["class_type"] == "OTHER GRAPE BRANDY (PISCO, GRAPPA) FB"


def _error_page() -> str:
    """The registry's error page: HTTP 200, no stated total, so it reads as zero rows."""
    return ("<html><head><title>TTB Online - COLAs Online - Error Message</title></head>"
            "<body><h1>Error</h1><p>An error has occurred in the system.</p>"
            "<div class='errorbox'>Unable to process request.</div></body></html>")


def test_ttb_error_page_is_not_an_empty_window():
    from bcd_ingest.connectors.ttb_cola import is_error_page, parse_total

    page = _error_page()
    assert parse_total(page) == 0, "no total on the page, so the count alone cannot tell"
    assert is_error_page(page), "...which is exactly why it needs its own test"
    assert not is_error_page(_search_page(0)), "a real empty result set is not an error"


def test_ttb_bisects_an_error_page_rather_than_losing_the_window():
    # The registry errors on some ranges and returns HTTP 200 with no total, which read as
    # an empty window and skipped it in silence: 1992 answered 0 for the year while its
    # quarters held thousands. The halves that do not contain the offending row still answer.
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    client = _Recorder([
        _error_page(),                        # 5-day window: errors
        _search_page(3), _ttb_csv(3, 1),      # first half answers
        _search_page(2), _ttb_csv(2, 4),      # second half answers
    ])
    conn = TTBColaConnector(store=None, use_fixture=False)
    rows = list(conn._walk_range(client, date(2026, 8, 20), date(2026, 8, 24), "100", "699"))

    assert len(rows) == 5, "the window is recovered, not skipped"
    spans = [(c[2].get("searchCriteria.dateCompletedFrom"),
              c[2].get("searchCriteria.dateCompletedTo"))
             for c in client.calls if c[0] == "POST"]
    assert spans == [("08/20/2026", "08/24/2026"),
                     ("08/20/2026", "08/22/2026"),
                     ("08/23/2026", "08/24/2026")], "halves tile the window exactly"


def test_ttb_error_page_on_a_single_day_is_reported_not_swallowed(capsys):
    # A day cannot be halved further. It is a genuine hole, and the run has to say so —
    # the whole point of the change is that this stops looking like an empty day.
    from bcd_ingest.connectors.ttb_cola import TTBColaConnector

    client = _Recorder([_error_page()])
    conn = TTBColaConnector(store=None, use_fixture=False)
    rows = list(conn._walk_range(client, date(1992, 9, 29), date(1992, 9, 29), "100", "699"))

    assert rows == []
    assert len(client.calls) == 1, "nothing to export, and no half to try"
    out = capsys.readouterr().out
    assert "09/29/1992" in out and "error page" in out


def test_ttb_export_reads_the_eight_digit_ids_of_the_early_registry():
    # TTB widened the id from eight digits to fourteen. A floor of ten matched no record
    # boundary in a pre-2000 export, so it parsed as empty and the walk paged the HTML
    # twenty rows at a time instead — correct, but fifty times the requests.
    from bcd_ingest.connectors.ttb_cola import parse_export

    csv_text = (f"{_CSV_HEAD}\n"
                "'92093118',MI-I-532,92-153,09/28/1992,,LAPHROAIG,5K,SCOTLAND,"
                "153,SINGLE MALT SCOTCH WHISKY\n")
    (row,) = parse_export(csv_text)
    assert row["ttb_id"] == "92093118"
    assert row["brand_name"] == "LAPHROAIG"
    assert row["completed_date"] == "09/28/1992"


def test_ttb_export_repairs_a_serial_number_written_with_a_thousands_separator():
    # The repair used to read the completed date at a fixed index 3, on the stated
    # assumption that the first four columns never carry a comma. An early serial is
    # written "21,142", which splits and pushes the date to index 4 -- so index 3 held
    # "142", the date check failed, and the row was dropped. Exactly one row per window
    # of the 1980s, every time, which the count mismatch then paid for by paging.
    from bcd_ingest.connectors.ttb_cola import parse_export

    csv_text = (f"{_CSV_HEAD}\n"
                "'88033310',PHI-RB-523,21,142,03/31/1988,,SCHMIDTS,"
                "21,PENNSYLVANIA,142,MALT BEVERAGE\n")
    (row,) = parse_export(csv_text)
    assert row["ttb_id"] == "88033310"
    assert row["serial"] == "21,142", "the separator is put back, not dropped"
    assert row["completed_date"] == "03/31/1988"
    assert row["brand_name"] == "SCHMIDTS"
    assert row["class_type_code"] == "142"


def test_ttb_export_still_repairs_a_comma_in_the_name_when_the_serial_is_clean():
    # The common case must be unchanged: date at index 3, comma in the fanciful name.
    from bcd_ingest.connectors.ttb_cola import parse_export

    csv_text = (f"{_CSV_HEAD}\n"
                "'26051001000586',DSP-CA-1,26-1,08/24/2026,COYOTA CAPON, TOBALA CAPON,"
                "BRAND X,06,CALIFORNIA,301,VODKA\n")
    (row,) = parse_export(csv_text)
    assert row["fanciful_name"] == "COYOTA CAPON, TOBALA CAPON"
    assert row["completed_date"] == "08/24/2026"
    assert row["brand_name"] == "BRAND X"
