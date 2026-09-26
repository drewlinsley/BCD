"""Importers are not producers.

A COLA filed on an import permit says who brought a bottle in, not who made it, and the permit
pass above these names every permit after its best-filed brand. On `CT-I-15074` that put
"Johnnie Walker" on a bottle of Lagavulin. These pin the rule that fixes it and the two places
it must refuse to fire: a real distillery's notice, and a brand that names nobody.

Postgres-only, because the pass is SQL. Skips cleanly with no server, like `test_pg_store`.
"""

from __future__ import annotations

import os

import pytest

psycopg = pytest.importorskip("psycopg")

from bcd_ingest.permit_producers import (  # noqa: E402
    rekey_importers_to_brands,
    rekey_to_permits,
)
from bcd_ingest.pg_store import PostgresStore, _normalize_dsn  # noqa: E402
from bcd_ingest.store import BronzeDoc  # noqa: E402

_URL = os.environ.get("BCD_DATABASE_URL", "postgresql://localhost:5432/bcd")
_SCHEMA = "bcd_test_permits"


@pytest.fixture()
def pg():
    try:
        admin = psycopg.connect(_normalize_dsn(_URL), autocommit=True, connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 — any connect failure -> skip, not fail
        pytest.skip(f"Postgres not reachable at {_URL}: {exc}")
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
    admin.close()
    store = PostgresStore(_URL, search_path=f"{_SCHEMA},public")
    try:
        yield store
    finally:
        with store._conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
        store.close()


def _cola(store: PostgresStore, ttb_id: str, permit: str, brand: str, *,
          fanciful: str = "", origin: str = "SCOTLAND") -> None:
    """One filing, in both the layer that owns the permit number and the layer the app reads."""
    store.put_bronze(BronzeDoc(
        id=f"ttb:{ttb_id}", source_id="ttb-cola-registry", natural_key=ttb_id,
        fetched_at="2026-09-25T00:00:00Z", url=None,
        payload={"ttb_id": ttb_id, "permit_no": permit, "brand_name": brand,
                 "fanciful_name": fanciful, "origin_desc": origin}))
    store.put_gold(f"ttb:{ttb_id}", "product", {
        "id": f"ttb:{ttb_id}", "name": (fanciful or brand).title(), "category": "spirit",
        "brand_id": f"brand:ttb-cola-registry:{brand.lower().replace(' ', '-')}",
        "producer_id": None})


def _rekey(store: PostgresStore) -> dict[str, int]:
    rekey_to_permits(_URL, search_path=f"{_SCHEMA},public")
    return rekey_importers_to_brands(_URL, search_path=f"{_SCHEMA},public")


def _producer_of(store: PostgresStore, ttb_id: str) -> tuple[str, str | None]:
    """(producer name, region) as the app would read it off the product."""
    rec = store.get_gold(f"ttb:{ttb_id}")
    pr = store.get_gold(rec["producer_id"])
    return pr["name"], pr.get("region")


# ---- what may name a producer -------------------------------------------------------------

def test_a_brand_field_holding_a_product_name_names_nobody(pg):
    """The gate that stops this fix recreating what the permit rekey cured. Where a filer leaves
    `fanciful_name` empty the whole product name goes in `brand_name`, so "LAGAVULIN 12 YR" is
    filed as a brand -- and keyed on that, 29,893 of 55,154 new producers would have held one
    product and shared its name. A brand earns the field by appearing somewhere with its product
    named separately from it."""
    _cola(pg, "1", "CT-I-15074", "LAGAVULIN", fanciful="16 YEAR OLD")
    _cola(pg, "2", "CT-I-15074", "LAGAVULIN 12 YR")
    _rekey(pg)
    assert _producer_of(pg, "1")[0] == "Lagavulin"
    assert pg.get_gold("ttb:2")["producer_id"] == ""


def test_a_brand_earns_the_field_anywhere_in_the_registry(pg):
    """Whether a string is a brand is a fact about the string, not about the permit it was filed
    on -- so the evidence is read across every filing, including the distillery's own."""
    _cola(pg, "1", "DSP-KY-113", "WELLER", fanciful="ANTIQUE 107", origin="KENTUCKY")
    _cola(pg, "2", "NY-I-500", "WELLER", origin="KENTUCKY")
    _rekey(pg)
    assert _producer_of(pg, "2")[0] == "Weller"


# ---- the bug ------------------------------------------------------------------------------

def test_an_import_permits_other_brands_do_not_name_the_bottle(pg):
    """CT-I-15074 really does hold both, and Johnnie Walker really did win the vote."""
    _cola(pg, "1", "CT-I-15074", "JOHNNIE WALKER", fanciful="BLUE LABEL")
    _cola(pg, "2", "CT-I-15074", "JOHNNIE WALKER", fanciful="BLACK LABEL")
    _cola(pg, "3", "CT-I-15074", "JOHNNIE WALKER", fanciful="RED LABEL")
    _cola(pg, "4", "CT-I-15074", "LAGAVULIN", fanciful="16 YEAR OLD")
    _rekey(pg)
    assert _producer_of(pg, "4")[0] == "Lagavulin"
    assert _producer_of(pg, "1")[0] == "Johnnie Walker"


def test_a_brand_filed_by_several_importers_is_one_producer(pg):
    """The point of keying on the brand: Lagavulin is filed by seven importers, and splitting
    its range between them would be a different wrong answer from naming it after one of them."""
    _cola(pg, "1", "CT-I-15074", "LAGAVULIN", fanciful="16 YEAR OLD")
    _cola(pg, "2", "CA-I-23238", "LAGAVULIN", fanciful="DISTILLERS EDITION")
    _cola(pg, "3", "NY-I-21645", "LAGAVULIN", fanciful="25 YEAR OLD")
    _rekey(pg)
    assert len({pg.get_gold(f"ttb:{i}")["producer_id"] for i in "123"}) == 1


def test_a_distillery_keeps_its_permit(pg):
    """`DSP-KY-113` is Buffalo Trace's plant. It made the bottle; the rule must not fire."""
    _cola(pg, "1", "DSP-KY-113", "BUFFALO TRACE", fanciful="KOSHER RYE", origin="KENTUCKY")
    _cola(pg, "2", "DSP-KY-113", "EAGLE RARE", fanciful="10 YEAR OLD", origin="KENTUCKY")
    _rekey(pg)
    for i in "12":
        assert pg.get_gold(f"ttb:{i}")["producer_id"].startswith(
            "prod:ttb-cola-registry:permit:dsp-ky-113")


def test_the_class_letter_is_read_from_the_second_position(pg):
    """A permit's first token is a state, or a city ("PHI-I-4477"), or missing ("I 1214").
    Reading the letter from the front would call `I`-prefixed permits importers and miss the
    rest -- and would catch `IL-CI-104`, an Illinois plant, by its leading I."""
    _cola(pg, "1", "PHI-I-4477", "FERNET BRANCA", fanciful="AMARO", origin="ITALY")
    _cola(pg, "2", "I 1214", "HENNESSY", fanciful="VS", origin="FRANCE")
    _cola(pg, "3", "IL-CI-104", "KOVAL", fanciful="RYE", origin="ILLINOIS")
    _rekey(pg)
    assert pg.get_gold("ttb:1")["producer_id"] == "prod:ttb-cola-registry:brand:fernet-branca"
    assert pg.get_gold("ttb:2")["producer_id"] == "prod:ttb-cola-registry:brand:hennessy"
    assert pg.get_gold("ttb:3")["producer_id"].startswith(
        "prod:ttb-cola-registry:permit:il-ci-104")


# ---- what it refuses to do ----------------------------------------------------------------

def test_a_brand_that_names_nobody_is_not_promoted(pg):
    """Filers really do enter "LLC" as a brand, and an incorporation word is not a maker. Nor is
    the import permit it was filed on, so the row is left with no producer rather than either."""
    _cola(pg, "1", "NY-I-500", "LLC", fanciful="AMARO")
    _cola(pg, "2", "NY-I-500", "AMARO MONTENEGRO", fanciful="LIQUEUR", origin="ITALY")
    _rekey(pg)
    assert pg.get_gold("ttb:1")["producer_id"] == ""
    assert _producer_of(pg, "2")[0] == "Amaro Montenegro"


def test_a_country_needs_four_fifths_of_the_brand_to_agree(pg):
    """Taking the commonest origin across an import PERMIT is what put Scotland on Crown Royal --
    a Diageo permit mostly carries Islay. Per BRAND the mode is trustworthy, so the rule is a
    supermajority rather than the mode.

    Not unanimity either: one filer typing "Virginia" on a Lagavulin is enough to refuse a brand
    the registry says Scotland about 84 times, and that cost 1,194 brands their country."""
    _cola(pg, "1", "CT-I-325", "TALISKER", fanciful="10 YEAR OLD", origin="SCOTLAND")
    _cola(pg, "2", "CT-I-325", "TALISKER", fanciful="SKYE", origin="SCOTLAND")
    _cola(pg, "3", "CT-I-325", "TALISKER", fanciful="STORM", origin="SCOTLAND")
    _cola(pg, "4", "CT-I-325", "TALISKER", fanciful="DARK STORM", origin="SCOTLAND")
    _cola(pg, "5", "CT-I-325", "TALISKER", fanciful="57 NORTH", origin="VIRGINIA")
    # a name two unrelated drinks share: half and half, and no country is the honest answer
    _cola(pg, "6", "CT-I-325", "LEGACY", fanciful="BLENDED", origin="SCOTLAND")
    _cola(pg, "7", "CT-I-325", "LEGACY", fanciful="REPOSADO", origin="MEXICO")
    _rekey(pg)
    assert _producer_of(pg, "1") == ("Talisker", "Scotland")   # 4 of 5
    assert _producer_of(pg, "6")[1] is None                    # 1 of 2


def test_the_pass_is_idempotent(pg):
    """Re-run after any backfill. The second run must find nothing left to move."""
    _cola(pg, "1", "CT-I-15074", "LAGAVULIN", fanciful="16 YEAR OLD")
    _cola(pg, "2", "DSP-KY-113", "BUFFALO TRACE", origin="KENTUCKY")
    _rekey(pg)
    before = {i: pg.get_gold(f"ttb:{i}")["producer_id"] for i in "12"}
    assert rekey_importers_to_brands(
        _URL, search_path=f"{_SCHEMA},public")["products_relinked"] == 0
    assert {i: pg.get_gold(f"ttb:{i}")["producer_id"] for i in "12"} == before


def test_an_emptied_import_permit_stops_being_a_producer(pg):
    """Its rows all moved to their brands, so it names nothing and must not survive in the
    producer table -- the resolver matches label text against every name in it."""
    _cola(pg, "1", "CT-I-15074", "LAGAVULIN", fanciful="16 YEAR OLD")
    _rekey(pg)
    assert pg.get_gold("prod:ttb-cola-registry:permit:ct-i-15074") is None


def test_an_emptied_row_is_still_a_product_the_app_can_resolve(pg):
    """The regression this nearly shipped. `Product.producer_id` is a required `str`, so writing
    a JSON null (or dropping the key) makes `_hydrate` fail validation and the row disappears
    from search and from the scan entirely -- 31,602 of them. The empty string validates, and
    every reader already treats it as no producer: "Unknown" here, an omitted line in Discover,
    and -1 in the label index."""
    from bcd_api.resolver import Resolver
    from bcd_schema import Product

    _cola(pg, "1", "CT-I-15074", "LAGAVULIN", fanciful="16 YEAR OLD")
    _cola(pg, "2", "CT-I-15074", "LAGAVULIN 12 YR")
    _rekey(pg)

    rec = pg.get_gold("ttb:2")
    assert rec["producer_id"] == ""
    Product.model_validate(rec)                       # would raise on null or a missing key
    resolved = Resolver(pg)._hydrate(rec)
    assert resolved is not None and resolved.producer.name == "Unknown"
    assert resolved.product.name == "Lagavulin 12 Yr"


def test_a_producer_given_by_hand_is_not_overruled(pg):
    """This pass takes rows away from importers. It is not entitled to overrule anything else,
    and a filing on an import permit is no reason to.

    One row in the catalog was in this state -- `Ramazzotti Aperitivo Rosato`, hand-linked to the
    curated house `prod:bcd:ramazzotti` by an earlier merge. Pulling it onto a brand producer
    emptied the house, and the house is what the resolver's producer path corroborates a
    Ramazzotti label against: 22 frames of the replay log went dark for that one row.
    """
    _cola(pg, "1", "CT-I-265", "RAMAZZOTTI", fanciful="APERITIVO ROSATO", origin="ITALY")
    pg.put_gold("prod:bcd:ramazzotti", "producer",
                {"id": "prod:bcd:ramazzotti", "name": "Fratelli Ramazzotti"})
    rec = pg.get_gold("ttb:1")
    rec["producer_id"] = "prod:bcd:ramazzotti"
    pg.put_gold("ttb:1", "product", rec)

    rekey_importers_to_brands(_URL, search_path=f"{_SCHEMA},public")
    assert pg.get_gold("ttb:1")["producer_id"] == "prod:bcd:ramazzotti"
