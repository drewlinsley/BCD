"""Substring / cross-source entity resolution — high precision, no false merges.

Cases are the real catalog pairs the audit surfaced: two true cross-source dupes among twelve
containments, the rest coincidental style-word overlaps that must NOT merge.
"""
from __future__ import annotations

from bcd_ingest.dedup import find_duplicate_merges, find_substring_merges

# The real cross-source neighbourhood: two true dupes + the Lagavulin/Buffalo-Trace style-word
# fan-in + the demo-IPA false hits.
CATALOG = [
    {"id": "bcd-demo:heady", "name": "Heady Topper", "brand": "Heady Topper"},
    {"id": "ttb:heady", "name": "The Alchemist Heady Topper", "brand": "The Alchemist"},
    {"id": "off:sn-pale", "name": "Pale Ale", "brand": "Sierra Nevada"},
    {"id": "ttb:sn-pale", "name": "Sierra Nevada Pale Ale", "brand": "Sierra Nevada"},
    {"id": "ttb:lagavulin", "name": "Lagavulin 16 Year Old Single Malt Scotch Whisky",
     "brand": "Lagavulin"},
    {"id": "off:ardbeg", "name": "10 y. o. whisky 46%", "brand": "ARDBEG"},
    {"id": "off:cardhu", "name": "Scotch Whisky 1824", "brand": "Cardhu"},
    {"id": "off:bowmore", "name": "Single Malt Scotch Whisky", "brand": "Bowmore"},
    {"id": "ttb:buffalo", "name": "Buffalo Trace Kentucky Straight Bourbon Whiskey",
     "brand": "Buffalo Trace"},
    {"id": "off:fourroses", "name": "Whiskey 40%", "brand": "FOUR ROSES"},
    {"id": "bcd-demo:wcr", "name": "West Coast Reference IPA", "brand": "West Coast Reference IPA"},
    {"id": "off:420", "name": "420 IPA", "brand": "Sweetwater Brewing Company"},
    {"id": "off:hefe", "name": "Hefeweizen", "brand": "Unknown"},
    {"id": "bcd-demo:bavhefe", "name": "Bavarian Hefeweizen", "brand": "Bavarian Hefeweizen"},
]


def test_finds_exactly_the_two_true_cross_source_dupes():
    merges = set(find_substring_merges(CATALOG))
    assert merges == {
        ("bcd-demo:heady", "ttb:heady"),   # brand echoes name -> compatible
        ("off:sn-pale", "ttb:sn-pale"),    # bare "Pale Ale" rescued by matching brand
    }


def test_rejects_style_word_fan_in():
    # Ardbeg / Cardhu / Bowmore / Four Roses share only "scotch/whisky/bourbon" with the TTB
    # bottles — a different distillery is a different product.
    merges = find_substring_merges(CATALOG)
    aliased = {a for a, _ in merges}
    for wrong in ("off:ardbeg", "off:cardhu", "off:bowmore", "off:fourroses"):
        assert wrong not in aliased


def test_rejects_brand_mismatch_and_bare_style():
    merges = find_substring_merges(CATALOG)
    aliased = {a for a, _ in merges}
    assert "off:420" not in aliased      # Sweetwater != West Coast Reference IPA
    assert "off:hefe" not in aliased     # bare "Hefeweizen", no brand -> nothing to anchor


def test_ambiguous_multiple_supersets_are_left_alone():
    # A bare brand covered by several distinct products has no unique canonical -> skip, never
    # collapse "Absolut" into whichever flavour sorts first.
    cat = [
        {"id": "off:absolut", "name": "Absolut", "brand": "Absolut"},
        {"id": "off:absolut-vodka", "name": "Absolut Vodka", "brand": "Absolut"},
        {"id": "off:absolut-razz", "name": "Absolut Raspberri", "brand": "Absolut"},
    ]
    assert find_substring_merges(cat, cross_source_only=False) == []


def test_cross_source_only_flag():
    # Same-source containment is out of scope by default; opening the flag still requires the guards.
    cat = [
        {"id": "off:a", "name": "Guinness", "brand": "Guinness"},
        {"id": "off:b", "name": "Guinness Draught Stout", "brand": "Guinness"},
    ]
    assert find_substring_merges(cat, cross_source_only=True) == []
    assert find_substring_merges(cat, cross_source_only=False) == [("off:a", "off:b")]


# ---- equivalence pass: same product, written differently ------------------------------
# Real rows from the OFF catalog. The traps are deliberate: identical names that are NOT the
# same beer, and generic names shared by rival producers.
EQUIV = [
    {"id": "off:hein-a", "name": "Heineken Cerveza", "brand": "Heineken"},
    {"id": "off:hein-b", "name": "Cerveza Heineken", "brand": "Heineken"},
    # same name, genuinely different breweries -> must stay apart
    {"id": "off:radler-a", "name": "Natur Radler", "brand": "Hoepfner"},
    {"id": "off:radler-b", "name": "Natur Radler", "brand": "Gösser"},
    # bare category + a brand-less row: adopting it would be a guess
    {"id": "off:irish-a", "name": "Irish whiskey", "brand": "Unknown"},
    {"id": "off:irish-b", "name": "Irish Whiskey", "brand": "Bushmills"},
    # bare category, but both carry the same brand punctuated differently -> a real dupe
    {"id": "off:lawson-a", "name": "Blended Scotch Whisky", "brand": "William Lawson's"},
    {"id": "off:lawson-b", "name": "blended scotch whisky", "brand": "william lawsons"},
    # rival producers of the same rhum category
    {"id": "off:rhum-a", "name": "Rhum blanc agricole", "brand": "Rhum J.M"},
    {"id": "off:rhum-b", "name": "Rhum blanc agricole", "brand": "HSE"},
    # a mis-scraped brand must not win the survivor slot
    {"id": "off:coors-a", "name": "Coors Light", "brand": "Coors"},
    {"id": "off:coors-b", "name": "Coors Light", "brand": "OLD TRAPPER"},
]


def _pairs(catalog=EQUIV):
    return set(find_duplicate_merges(catalog))


def test_merges_word_order_variants():
    assert ("off:hein-a", "off:hein-b") in _pairs() or ("off:hein-b", "off:hein-a") in _pairs()


def test_same_name_different_brewery_is_not_a_duplicate():
    """Hoepfner and Gösser both sell a "Natur Radler" and they are different beers."""
    merged = {a for a, _ in _pairs()} | {c for _, c in _pairs()}
    assert "off:radler-a" not in merged
    assert "off:radler-b" not in merged


def test_bare_category_name_does_not_adopt_a_brandless_row():
    """"Irish Whiskey" names a class; the unbranded row could be anyone's."""
    assert not {p for p in _pairs() if "irish" in p[0]}


def test_bare_category_merges_when_both_brands_are_the_same():
    assert ("off:lawson-a", "off:lawson-b") in _pairs() or \
           ("off:lawson-b", "off:lawson-a") in _pairs()


def test_rival_producers_of_a_category_stay_apart():
    assert not {p for p in _pairs() if "rhum" in p[0]}


def test_survivor_is_the_row_whose_brand_matches_its_name():
    """Preferring merely *having* a brand files Coors Light under a jerky company."""
    coors = [p for p in _pairs() if "coors" in p[0]]
    assert coors == [("off:coors-b", "off:coors-a")]


# ---- containment: the relaxation must not let a product collapse to its brand ---------

def test_age_statement_in_another_language_still_merges():
    cat = [
        {"id": "off:lag", "name": "Lagavulin 16 ans", "brand": "Isle of Islay"},
        {"id": "ttb:lag", "name": "Lagavulin 16 Year Old Single Malt Scotch Whisky",
         "brand": "Lagavulin"},
    ]
    assert set(find_substring_merges(cat)) == {("off:lag", "ttb:lag")}


def test_different_age_statements_are_different_bottles():
    cat = [
        {"id": "off:lag8", "name": "Lagavulin 8", "brand": "Lagavulin"},
        {"id": "ttb:lag16", "name": "Lagavulin 16 Year Old Single Malt Scotch", "brand": "Lagavulin"},
    ]
    assert find_substring_merges(cat) == []


def test_style_words_are_identity_not_noise():
    """Both traps the relaxation originally fell into: a product must not reduce to the
    brand/region word it shares with an unrelated sibling."""
    cat = [
        {"id": "ttb:snpale", "name": "Sierra Nevada Pale Ale", "brand": "Sierra Nevada"},
        {"id": "off:juicy", "name": "Juicy Little Thing Hazy IPA", "brand": "Sierra Nevada"},
        {"id": "bcd-demo:bavhefe", "name": "Bavarian Hefeweizen", "brand": "Bavarian Hefeweizen"},
        {"id": "off:bavamber", "name": "Bavarian Amber Lager", "brand": "Red Oak"},
    ]
    assert find_substring_merges(cat) == []


# --- producers ---------------------------------------------------------------------------

def _prod(pid, name, region=None):
    return {"id": pid, "name": name, "region": region}


def test_producer_spellings_of_one_company_collapse():
    # TTB names a producer from the brand on each filing, so one brewery arrives once per
    # spelling it has ever used. A differing location does not make it two companies —
    # Lagunitas cans in more than one state.
    from bcd_ingest.dedup import find_producer_merges

    rows = [
        _prod("p:lagunitas", "Lagunitas", "California"),
        _prod("p:lag-brew-co", "Lagunitas Brewing Co", "California"),
        _prod("p:the-lag-brew-co", "The Lagunitas Brewing Co.", "Minnesota"),
        _prod("p:lag-brew-company", "Lagunitas Brewing Company", "California"),
    ]
    merges = dict(find_producer_merges(rows))
    assert set(merges) == {"p:lag-brew-co", "p:the-lag-brew-co", "p:lag-brew-company"}
    assert set(merges.values()) == {"p:lagunitas"}, "the plainest spelling survives"


def test_a_brewery_and_a_distillery_are_not_one_company():
    # They may share a name and be unrelated: Mammoth Brewing is in California, Mammoth
    # Distilling in Michigan. A row that says neither cannot be assigned to either.
    from bcd_ingest.dedup import find_producer_merges

    rows = [
        _prod("p:mam-brew", "Mammoth Brewing", "California"),
        _prod("p:mam-brew-co", "Mammoth Brewing Co.", "California"),
        _prod("p:mam-dist", "Mammoth Distilling", "Michigan"),
        _prod("p:mam-dist-co", "Mammoth Distilling Co", "Michigan"),
        _prod("p:mam", "Mammoth", None),
    ]
    merges = dict(find_producer_merges(rows))
    assert merges == {"p:mam-brew-co": "p:mam-brew", "p:mam-dist-co": "p:mam-dist"}
    assert "p:mam" not in merges, "an unqualified row is left alone, not guessed at"


def test_a_trade_word_inside_a_name_is_identity_not_noise():
    # Stripping trade words anywhere turns "Ale House Brewing Co" into "house", which then
    # collects unrelated companies in seven states. Only a TRAILING one is noise.
    from bcd_ingest.dedup import find_producer_merges, producer_core

    assert producer_core("Ale House Brewing Co") == "ale house"
    assert producer_core("Amber Ale") == "amber ale"
    rows = [
        _prod("p:ale-house", "Ale House"),
        _prod("p:ale-house-brew", "Ale House Brewing Co"),
        _prod("p:house", "House"),
        _prod("p:amber-ale", "Amber Ale"),
        _prod("p:amber-lager", "Amber Lager"),
    ]
    merges = dict(find_producer_merges(rows))
    assert merges == {"p:ale-house-brew": "p:ale-house"}


def test_a_name_the_ascii_fold_erases_is_never_grouped():
    # Ten producers write their names in Greek or Korean. Folded to ASCII they all become
    # the empty string, which would merge them into one company.
    from bcd_ingest.dedup import find_producer_merges, producer_core

    assert producer_core("Άλφα") == ""
    rows = [_prod("p:alfa", "Άλφα"), _prod("p:eza", "Εζα"), _prod("p:semi", ";")]
    assert find_producer_merges(rows) == []


def test_producer_merge_repoints_products_and_brands_and_keeps_a_region():
    import tempfile

    from bcd_ingest.dedup import find_producer_merges
    from bcd_ingest.merge import merge_producers
    from bcd_ingest.store import MedallionStore

    store = MedallionStore(root=tempfile.mkdtemp())
    rows = [
        _prod("p:lag", "Lagunitas", None),
        _prod("p:lag-brew", "Lagunitas Brewing Co", "California"),
        _prod("p:lag-brew-2", "The Lagunitas Brewing Company", "California"),
        _prod("p:lag-mn", "Lagunitas Brewing Co.", "Minnesota"),
    ]
    for r in rows:
        store.put_gold(r["id"], "producer", r)
    store.put_gold("prod:1", "product", {"id": "prod:1", "name": "IPA", "producer_id": "p:lag-mn"})
    store.put_gold("b:1", "brand", {"id": "b:1", "name": "Lagunitas", "producer_id": "p:lag-brew"})

    stats = merge_producers(store, dict(find_producer_merges(rows)))
    assert stats["merged"] == 3
    assert stats["products_repointed"] == 1
    assert stats["brands_repointed"] == 1

    assert store.get_gold("p:lag-mn") is None, "absorbed rows are removed"
    survivor = store.get_gold("p:lag")
    assert survivor["region"] == "California", "the region it files from most often"
    assert "Lagunitas Brewing Co" in survivor["aliases"]
    assert store.get_gold("prod:1")["producer_id"] == "p:lag"
    assert store.get_gold("b:1")["producer_id"] == "p:lag"
    store.close()


def test_a_misspelled_corporate_suffix_still_folds():
    # Filers misspell their own suffix, and each misspelling forks a company into its own
    # producer. All real, from the registry.
    from bcd_ingest.dedup import producer_core

    for name in ["The Lagunitas Brewing Comany", "Lagunitas Brewing Compnay"]:
        assert producer_core(name) == "lagunitas", name
    assert producer_core("Siluria Brewing Companuy, LLC") == "siluria"
    assert producer_core("The Illusionist Destillery") == "illusionist"
    assert producer_core("Tenth Ward Distiilling") == "tenth ward"


def test_fuzzy_suffix_matching_does_not_eat_short_identity_words():
    # Only long trailing tokens are fuzzy-matched, so a short real word is never mistaken
    # for a typo of a suffix.
    from bcd_ingest.dedup import producer_core

    assert producer_core("Anchor Steam") == "anchor steam"
    assert producer_core("Amber Ale") == "amber ale"
    assert producer_core("Ale House Brewing Co") == "ale house"
    assert producer_core("Sierra Nevada Brewing Company") == "sierra nevada"
