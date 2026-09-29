"""Discover, one shelf at a time.

Gin, bourbon and vodka are three questions. `rank_catalog` can only answer whichever one the
drinker's centroid sits nearest -- so these test the thing that makes the other twenty-one
reachable, and the line between a shelf ranked FOR someone and a shelf merely shown to them.
"""

from __future__ import annotations

import tempfile

import pytest
from bcd_api.recommend import rank_family, shelf_vector
from bcd_api.resolver import Resolver
from bcd_ingest.store import MedallionStore
from bcd_schema import (
    Category,
    ExtractionMethod,
    Product,
    Provenance,
    SensorySource,
    SensoryVector,
    Sourced,
    TasteProfile,
)
from bcd_schema.family import Family, styles_in

PROV = Provenance(source_id="test", method=ExtractionMethod.REGULATORY_FILING, confidence=1.0)
#: An IPA drinker: citrus and pine, no juniper anywhere near it.
IPA_TASTE = {"citrus": 0.8, "tropical": 0.7, "piney_resinous": 0.6, "bitterness": 0.6}


def _sv(source, confidence, **axes):
    return SensoryVector(source=source, confidence=confidence, axes=axes)


def _product(pid, name, style, sensory, category=Category.SPIRIT, producer="house"):
    return Product(id=pid, brand_id=f"brand:{pid}", producer_id=f"prod:{producer}",
                   category=category, name=name,
                   style=Sourced[str](value=style, provenance=PROV),
                   sensory=sensory).model_dump(mode="json")


@pytest.fixture()
def store():
    s = MedallionStore(root=tempfile.mkdtemp())
    s.put_gold("prod:house", "producer", {"id": "prod:house", "name": "A House"})
    # Two gins with profiles of their own. The citrus one is nearer an IPA taste; the juniper
    # one is the better-evidenced bottle.
    s.put_gold("gin:citrus", "product", _product(
        "gin:citrus", "Citrus Gin", "London Dry Gin",
        _sv(SensorySource.LLM_PROFILE, 0.6, citrus=0.8, piney_resinous=0.3)))
    s.put_gold("gin:juniper", "product", _product(
        "gin:juniper", "Juniper Gin", "Gin",
        _sv(SensorySource.REVIEW_CONSENSUS, 0.9, piney_resinous=0.9, herbal=0.5)))
    # 3 anonymous gins carrying the style's centroid -- the shelf's ties.
    for i in range(3):
        s.put_gold(f"gin:floor{i}", "product", _product(
            f"gin:floor{i}", f"Registry Gin {i}", "Gin",
            _sv(SensorySource.STYLE_PRIOR, 0.3, piney_resinous=0.5, herbal=0.4)))
    # A bourbon, so the gin queries have something to wrongly return.
    s.put_gold("bourbon:one", "product", _product(
        "bourbon:one", "A Bourbon", "Straight Bourbon",
        _sv(SensorySource.LLM_PROFILE, 0.7, vanilla_oak=0.8, caramel_toffee=0.6)))
    yield s
    s.close()


@pytest.fixture()
def resolver(store):
    return Resolver(store)


@pytest.fixture()
def ipa_drinker():
    return TasteProfile(user_id="u", version=1,
                        sensory_ideal=_sv(SensorySource.RECONCILED, 0.5, **IPA_TASTE))


GIN = styles_in(Family.GIN)


# ---- which shelves are about the drinker ---------------------------------------------------

def test_a_shelf_they_have_never_rated_on_is_not_ranked_for_them(ipa_drinker):
    assert shelf_vector(ipa_drinker, rated_in=False) is None


def test_a_shelf_they_have_rated_on_is(ipa_drinker):
    assert shelf_vector(ipa_drinker, rated_in=True) is not None


def test_asking_to_guess_across_styles_ranks_the_rest_too(ipa_drinker):
    assert shelf_vector(ipa_drinker, rated_in=False, cross_style=True) is not None


def test_with_no_profile_nothing_is_ranked_for_anyone():
    assert shelf_vector(None, rated_in=True, cross_style=True) is None


# ---- what an unranked shelf may say ---------------------------------------------------------

def test_an_unrated_shelf_carries_no_score(store, resolver, ipa_drinker):
    """The number is read as "how much you'd like it", and the one thing known for certain is
    that nobody has worked that out -- they have never rated anything on this shelf."""
    shelf = rank_family(store, resolver, ipa_drinker, store.best_known_in_family(GIN, 10),
                        personal=False, rated_in=False)
    assert shelf["basis"] == "unrated"
    assert shelf["results"]
    assert all(r["score"] is None and r["reason"] is None for r in shelf["results"])


def test_an_unrated_shelf_leads_with_what_the_catalog_knows_best(store, resolver, ipa_drinker):
    """Drinkers' consensus outranks a guess. Juniper Gin is the better-evidenced bottle and the
    further one from an IPA taste, so this also proves the taste is not quietly consulted."""
    shelf = rank_family(store, resolver, ipa_drinker, store.best_known_in_family(GIN, 10),
                        personal=False, rated_in=False)
    assert shelf["results"][0]["name"] == "Juniper Gin"


def test_a_rated_shelf_says_it_is_theirs_and_scores(store, resolver, ipa_drinker):
    rows = store.nearest_in_family(ipa_drinker.sensory_ideal.to_array(), GIN, 10)
    shelf = rank_family(store, resolver, ipa_drinker, rows, personal=True, rated_in=True)
    assert shelf["basis"] == "yours"
    assert all(r["score"] is not None for r in shelf["results"])


def test_guessing_across_styles_is_named_as_a_guess(store, resolver, ipa_drinker):
    """`cross` and `yours` must not be one word. One was earned on this shelf and the other was
    borrowed from another, and the screen says so differently."""
    rows = store.nearest_in_family(ipa_drinker.sensory_ideal.to_array(), GIN, 10)
    shelf = rank_family(store, resolver, ipa_drinker, rows, personal=True, rated_in=False)
    assert shelf["basis"] == "cross"


# ---- the shelf queries themselves -----------------------------------------------------------

def test_a_shelf_holds_only_its_own_styles(store, ipa_drinker):
    """The bourbon is nothing like an IPA, but neither is a gin -- so a filter that leaked would
    not be caught by looking at the order."""
    rows = store.nearest_in_family(ipa_drinker.sensory_ideal.to_array(), GIN, 10)
    assert rows and all("Gin" in r["name"] for r in rows)


def test_a_shelf_ignores_the_rows_carrying_its_styles_average(store, ipa_drinker):
    """Gin is 9,659 rows in the live catalog of which 178 have a profile; the rest hold the
    style's centroid and sit at one distance. Ranking those ranks nine thousand ties and returns
    an arbitrary six."""
    rows = store.nearest_in_family(ipa_drinker.sensory_ideal.to_array(), GIN, 10)
    assert not any("Registry" in r["name"] for r in rows)
    assert not any("Registry" in r["name"] for r in store.best_known_in_family(GIN, 10))


def test_a_shelf_nobody_can_name_is_empty_not_everything(store, ipa_drinker):
    """An empty style list is a family the caller could not name, which is not every family."""
    assert store.nearest_in_family(ipa_drinker.sensory_ideal.to_array(), [], 10) == []
    assert store.best_known_in_family([], 10) == []


def test_a_ranked_shelf_puts_the_nearest_first(store, ipa_drinker):
    """Citrus Gin over Juniper Gin for an IPA drinker -- the reverse of the unranked order, so
    the two orders are provably different things."""
    rows = store.nearest_in_family(ipa_drinker.sensory_ideal.to_array(), GIN, 10)
    assert rows[0]["name"] == "Citrus Gin"


def test_a_drink_already_judged_is_not_suggested_again(store, resolver, ipa_drinker):
    rows = store.nearest_in_family(ipa_drinker.sensory_ideal.to_array(), GIN, 10)
    shelf = rank_family(store, resolver, ipa_drinker, rows, personal=True, rated_in=True,
                        exclude={"gin:citrus"})
    assert "gin:citrus" not in {r["product_id"] for r in shelf["results"]}


def test_shelves_fetched_together_stay_with_their_shelf(store, ipa_drinker):
    """`shelves_many` answers positionally, and a screen that mixed them would put bourbon under
    Gin. Asked here with one ranked shelf and one not, which is what Discover really sends."""
    ideal = ipa_drinker.sensory_ideal.to_array()
    gin, bourbon = store.shelves_many(
        [(GIN, ideal), (styles_in(Family.BOURBON), None)], limit=10)
    assert all("Gin" in r["name"] for r in gin)
    assert [r["name"] for r in bourbon] == ["A Bourbon"]
