"""Producer location linking — precision guards, earned from real false positives."""

from __future__ import annotations

import tempfile

import pytest
from bcd_ingest.producer_link import link, name_key, plan
from bcd_ingest.store import MedallionStore


@pytest.fixture()
def store():
    s = MedallionStore(root=tempfile.mkdtemp())
    yield s
    s.close()


def _obdb(pid, name, city, region, country):
    return (f"prod:openbrewerydb:{pid}",
            {"id": f"prod:openbrewerydb:{pid}", "name": name, "city": city,
             "region": region, "country": country})


def _local(pid, name):
    return (f"prod:openfoodfacts:{pid}",
            {"id": f"prod:openfoodfacts:{pid}", "name": name})


def _seed(s, rows):
    for gid, rec in rows:
        s.put_gold(gid, "producer", rec)


# ---- name keys ----------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("The Alchemist LLC", "The Alchemist Brewery"),   # trade suffixes are not identity
    ("Athletic Brewing Co", "Athletic Brewing Company"),
    ("Dark Horse Brewing Co.", "Dark Horse Brewery"),
])
def test_trade_suffixes_do_not_separate_a_business(a, b):
    assert name_key(a) == name_key(b)


@pytest.mark.parametrize("name", ["Brewing Company", "The Brewery", "LLC", "Unknown"])
def test_names_with_nothing_distinctive_refuse_to_key(name):
    # "Unknown" is Open Food Facts' placeholder for a missing brand. If it keyed, every
    # unattributed drink in the catalog would inherit one brewery's address.
    assert name_key(name) is None


# ---- matching -----------------------------------------------------------------------

def test_links_location_onto_a_producer_that_has_none(store):
    _seed(store, [_obdb("a", "Dogfish Head Craft Brewery", "Milton", "Delaware",
                        "United States"),
                  _local("dfh", "Dogfish Head")])
    assert link(store, apply=True) == {"matched": 1, "written": 1}
    rec = store.get_gold("prod:openfoodfacts:dfh")
    assert (rec["city"], rec["region"]) == ("Milton", "Delaware")


def test_a_name_two_breweries_share_is_left_alone(store):
    # "Broken Spoke" exists in more than one state; guessing between them attaches a
    # plausible-looking wrong city, which is worse than showing none.
    _seed(store, [_obdb("a", "Broken Spoke Brewing", "Austin", "Texas", "United States"),
                  _obdb("b", "Broken Spoke Brewery", "Denver", "Colorado", "United States"),
                  _local("bs", "Broken Spoke")])
    assert plan(store) == []


def test_containment_matches_only_when_the_extra_words_name_a_site(store):
    # Real: the brewery behind Heady Topper is listed as "Alchemist Cannery", so exact key
    # equality misses it — but "Groggs Pinnacle Brewing" must not claim the vodka "Pinnacle",
    # because there the extra token *is* the identity.
    _seed(store, [_obdb("a", "Alchemist Cannery", "Stowe", "Vermont", "United States"),
                  _obdb("b", "Groggs Pinnacle Brewing Co", "Helper", "Utah", "United States"),
                  _local("alc", "The Alchemist LLC"),
                  _local("pin", "Pinnacle")])
    linked = {rec["id"] for rec, _ in plan(store)}
    assert linked == {"prod:openfoodfacts:alc"}


def test_link_is_idempotent(store):
    _seed(store, [_obdb("a", "Bell's Brewery, Inc", "Galesburg", "Michigan", "United States"),
                  _local("bel", "Bell's")])
    assert link(store, apply=True)["written"] == 1
    assert link(store, apply=True) == {"matched": 0, "written": 0}
