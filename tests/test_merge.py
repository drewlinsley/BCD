"""Merge redirects — ids that escaped into telemetry keep working after dedup."""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.taste import rebuild_profile
from bcd_ingest.merge import (
    REDIRECT_ENTITY,
    collapse,
    get_product,
    merge_products,
    put_redirect,
    resolve_id,
)
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    SKU,
    Category,
    ExtractionMethod,
    Product,
    ProductSpec,
    Provenance,
    SensorySource,
    SensoryVector,
    Sourced,
)

PROV = Provenance(source_id="t", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)


def _product(pid, name, axes=None, abv=None):
    return Product(
        id=pid, brand_id="brand:x", producer_id="prod:x", category=Category.BEER, name=name,
        spec=ProductSpec(abv_pct=Sourced[float](value=abv, provenance=PROV) if abv else None),
        sensory=(SensoryVector(source=SensorySource.STYLE_PRIOR, confidence=0.25, axes=axes)
                 if axes else None),
    )


@pytest.fixture()
def store():
    s = MedallionStore(root=tempfile.mkdtemp())
    s.put_gold("p:keep", "product",
               _product("p:keep", "Cerveza Heineken").model_dump(mode="json"))
    s.put_gold("p:gone", "product",
               _product("p:gone", "Heineken Cerveza", {"malty_bready": 0.6}, 5.0)
               .model_dump(mode="json"))
    s.put_gold("sku:1", "sku",
               SKU(id="sku:1", product_id="p:gone", container="can", upc="123")
               .model_dump(mode="json"))
    yield s
    s.close()


# ---- resolution ----------------------------------------------------------------------

def test_live_id_resolves_to_itself(store):
    assert resolve_id(store, "p:keep") == "p:keep"


def test_unknown_id_is_returned_unchanged(store):
    assert resolve_id(store, "p:never-existed") == "p:never-existed"


def test_merged_id_resolves_to_the_survivor(store):
    put_redirect(store, "p:gone", "p:keep")
    assert resolve_id(store, "p:gone") == "p:keep"
    assert get_product(store, "p:gone")["name"] == "Cerveza Heineken"


def test_chained_merges_resolve_to_the_final_survivor(store):
    put_redirect(store, "p:a", "p:gone")
    put_redirect(store, "p:gone", "p:keep")
    assert resolve_id(store, "p:a") == "p:keep"


def test_a_cycle_terminates_instead_of_hanging(store):
    put_redirect(store, "p:x", "p:y")
    put_redirect(store, "p:y", "p:x")
    assert resolve_id(store, "p:x") in {"p:x", "p:y"}
    assert get_product(store, "p:x") is None


def test_collapse_points_every_alias_at_the_final_survivor():
    assert collapse({"a": "b", "b": "c"}) == {"a": "c", "b": "c"}


# ---- merging -------------------------------------------------------------------------

def test_merge_absorbs_repoints_and_tombstones(store):
    stats = merge_products(store, {"p:gone": "p:keep"})
    assert stats == {"merged": 1, "skus_repointed": 1}

    assert [p["id"] for p in store.iter_gold("product")] == ["p:keep"]
    keep = store.get_gold("p:keep")
    assert keep["sensory"]["axes"] == {"malty_bready": 0.6}     # inherited, not lost
    assert keep["spec"]["abv_pct"]["value"] == 5.0
    assert "Heineken Cerveza" in keep["aliases"]                # still findable by old name
    assert store.get_gold("sku:1")["product_id"] == "p:keep"    # barcode follows
    assert store.get_gold("p:gone")["redirects_to"] == "p:keep"


def test_tombstone_is_not_a_product(store):
    merge_products(store, {"p:gone": "p:keep"})
    assert store.get_gold("p:gone")["redirects_to"]
    assert "p:gone" not in {p["id"] for p in store.iter_gold("product")}
    assert [r["id"] for r in store.iter_gold(REDIRECT_ENTITY)] == ["p:gone"]


def test_merging_twice_is_a_no_op(store):
    merge_products(store, {"p:gone": "p:keep"})
    assert merge_products(store, {"p:gone": "p:keep"})["merged"] == 0


# ---- the regression this exists to prevent -------------------------------------------

def _rating(pid, rating, install="demo"):
    return {"name": "rating_submitted", "install_id": install,
            "consent_tier": "personalization", "product_id": pid, "rating": rating}


def test_a_rating_survives_its_product_being_merged_away(store):
    """Without redirects the rated row is simply gone and the signal vanishes silently."""
    events = [_rating("p:gone", 5.0)]
    before = rebuild_profile(store, events, "demo")
    assert before.sensory_ideal is not None

    merge_products(store, {"p:gone": "p:keep"})
    after = rebuild_profile(store, events, "demo")
    assert after.sensory_ideal is not None, "rating was dropped when its product merged"
    assert after.sensory_ideal.axes == before.sensory_ideal.axes


def test_ratings_on_both_sides_of_a_merge_are_one_verdict(store):
    """Rate a row, merge it away, rate the survivor: that is one product with one opinion,
    and the later rating must win rather than averaging against its own earlier self."""
    merge_products(store, {"p:gone": "p:keep"})
    profile = rebuild_profile(store, [_rating("p:gone", 5.0), _rating("p:keep", 1.0)], "demo")
    assert profile.sensory_ideal is None      # net verdict is a dislike -> no centroid
    assert profile.style_affinities == {}
