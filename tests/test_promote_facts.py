"""The gates on a strength a drinker typed about a drink the catalog already has.

Each gate is a rule about when somebody's word may be written onto a row half a million people
read. They are the same gates a contributed DRINK clears, plus the two the open-data harvest
earned from rows that were actually wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest", "services/enrich", "scripts"):
    sys.path.insert(0, str(ROOT / p))

promote_facts = pytest.importorskip("promote_facts")
Claim = promote_facts.Claim


class FakeStore:
    """Enough store for the plan: rows by id, and bronze it never reads here."""

    def __init__(self, rows: dict[str, dict]):
        self.rows = rows
        self.written: dict[str, dict] = {}

    def get_gold(self, gid):
        return self.rows.get(gid)

    def put_gold(self, gid, entity_type, record):
        self.written[gid] = record


def _row(pid: str, name: str, category: str = "beer", abv: dict | None = None) -> dict:
    spec: dict = {}
    if abv is not None:
        spec["abv_pct"] = abv
    return {"id": pid, "brand_id": "b", "producer_id": "p", "category": category,
            "name": name, "spec": spec, "recipe": {"ingredients": []}, "aliases": []}


def _sourced(value: float, method: str) -> dict:
    return {"value": value,
            "provenance": {"source_id": "s", "url": None, "quote": None, "method": method,
                           "confidence": 0.9, "extracted_at": "2026-01-01T00:00:00Z",
                           "extractor_version": None}}


def _plan(rows, groups):
    return promote_facts.plan(FakeStore(rows), groups)


# ---- how many people said it -------------------------------------------------------------

def test_one_account_is_a_claim_not_a_fact():
    rows = {"p:1": _row("p:1", "Nameless Pale")}
    promote, held = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 5.2, "t1")]})
    assert promote == []
    assert any("fewer than 2 accounts" in k for k in held)


def test_two_accounts_agreeing_earns_it():
    rows = {"p:1": _row("p:1", "Nameless Pale")}
    promote, _ = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 5.2, "t1"),
                                                   Claim("acct:b", 5.2, "t2")]})
    assert [(p.pid, p.value, p.accounts) for p in promote] == [("p:1", 5.2, 2)]
    assert promote[0].was == "empty"


def test_a_rounding_difference_still_agrees():
    # Both are reading the same printed number; 4.5 against 4.6 is a rounding difference.
    rows = {"p:1": _row("p:1", "Nameless Pale")}
    promote, _ = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 4.5, "t1"),
                                                   Claim("acct:b", 4.6, "t2")]})
    assert promote and promote[0].value == pytest.approx(4.6, abs=0.1)


def test_a_real_disagreement_is_held_not_averaged():
    # Averaging would write a number neither of them said.
    rows = {"p:1": _row("p:1", "Nameless Pale")}
    promote, held = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 4.5, "t1"),
                                                      Claim("acct:b", 8.0, "t2")]})
    assert promote == []
    assert "the accounts disagree" in held


# ---- what is possible ----------------------------------------------------------------------

def test_a_strength_impossible_for_the_category_is_held():
    rows = {"p:1": _row("p:1", "Nameless Pale", category="beer")}
    promote, held = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 45.0, "t1"),
                                                      Claim("acct:b", 45.0, "t2")]})
    assert promote == []
    assert "impossible for the category" in held


def test_the_same_number_is_fine_on_a_spirit():
    rows = {"p:1": _row("p:1", "Bottled In Bond", category="spirit")}
    promote, _ = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 50.0, "t1"),
                                                   Claim("acct:b", 50.0, "t2")]})
    assert promote and promote[0].value == 50.0


def test_a_row_whose_name_says_otherwise_is_held():
    # The name got there off a filing, which outranks anybody's typing.
    rows = {"p:1": _row("p:1", "Carlsberg Unfiltered 5.0 Degre Alcool")}
    promote, held = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 4.2, "t1"),
                                                      Claim("acct:b", 4.2, "t2")]})
    assert promote == []
    assert "the row's own name says otherwise" in held


def test_a_row_whose_name_agrees_is_promoted():
    rows = {"p:1": _row("p:1", "India Pale Ale 6.6%")}
    promote, _ = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 6.6, "t1"),
                                                   Claim("acct:b", 6.6, "t2")]})
    assert promote and promote[0].value == 6.6


# ---- what may be written over ----------------------------------------------------------------

def test_a_style_guess_gives_way_to_someone_who_read_the_tin():
    rows = {"p:1": _row("p:1", "Nameless Pale",
                        abv=_sourced(5.0, "llm_inferred_from_style_prior"))}
    promote, _ = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 5.4, "t1"),
                                                   Claim("acct:b", 5.4, "t2")]})
    assert promote and promote[0].was == "llm_inferred_from_style_prior"


@pytest.mark.parametrize("method", ["regulatory_filing", "stated_by_producer", "label_ocr",
                                    "open_database", "retailer_listing"])
def test_anything_that_outranks_a_drinker_is_left_alone(method):
    rows = {"p:1": _row("p:1", "Nameless Pale", abv=_sourced(5.0, method))}
    promote, held = _plan(rows, {("p:1", "abv_pct"): [Claim("acct:a", 7.7, "t1"),
                                                      Claim("acct:b", 7.7, "t2")]})
    assert promote == []
    assert "already holds something that outranks a drinker" in held


def test_the_replaceable_set_comes_from_the_ladder():
    # Derived, not listed, so it cannot drift when a method is added to the schema.
    from bcd_schema.provenance import METHOD_CONFIDENCE_CEILING, ExtractionMethod
    ceiling = METHOD_CONFIDENCE_CEILING[ExtractionMethod.USER_CONTRIBUTED]
    for method in promote_facts._REPLACEABLE:
        assert METHOD_CONFIDENCE_CEILING[ExtractionMethod(method)] < ceiling
    assert ExtractionMethod.OPEN_DATABASE.value not in promote_facts._REPLACEABLE


# ---- what gets written -------------------------------------------------------------------------

def test_what_is_written_says_who_said_so():
    rows = {"p:1": _row("p:1", "Nameless Pale")}
    store = FakeStore(rows)
    promote, _ = promote_facts.plan(store, {("p:1", "abv_pct"): [Claim("acct:a", 5.2, "t1"),
                                                                 Claim("acct:b", 5.2, "t2")]})
    assert promote_facts.write(store, promote) == 1

    abv = store.written["p:1"]["spec"]["abv_pct"]
    assert abv["value"] == 5.2
    prov = abv["provenance"]
    assert prov["method"] == "user_contributed"
    assert prov["confidence"] == 0.7, "the ladder's ceiling, not a number invented here"
    assert "2 accounts" in prov["quote"]
    assert prov["extractor_version"] == promote_facts.VERSION


def test_a_row_that_vanished_between_plan_and_write_is_skipped():
    store = FakeStore({})
    written = promote_facts.write(store, [promote_facts.Promotion(
        pid="p:gone", name="Gone", category="beer", value=5.0, accounts=2, was="empty")])
    assert written == 0
