"""The two gates on promoting a drinker's typed drink into the catalog.

Set by the user 2026-10-12: it must not already be in the catalog, misspellings included, and
two DIFFERENT accounts must have told us about it.

No store here. `promote` is a pure function of the contributions and the shortlist of catalog
rows handed to it, which is the point -- the policy is testable without 534,104 rows behind it.
"""

from __future__ import annotations

from bcd_ingest.dedup import match_typed_name, same_word
from bcd_ingest.promote import (
    IN_CATALOG,
    NEEDS_A_LOOK,
    NEW,
    Contribution,
    group_contributions,
    judge,
    plan,
)


def _c(name, account, *, maker=None, category="beer", abv=None, at="2026-10-01T00:00:00Z",
       doc=None):
    return Contribution(doc_id=doc or f"doc:{account}:{name}", account_id=account, name=name,
                        category=category, maker=maker, abv_pct=abv, received_at=at)


def _row(pid, name, brand=""):
    return {"id": pid, "name": name, "brand": brand}


# ---- is it already in the catalog ---------------------------------------------------------


def test_a_typo_still_finds_the_row():
    """The gate the user asked for first. Somebody typing a label into a phone drops letters."""
    assert match_typed_name("Heady Toper", None,
                            [_row("ttb:1", "The Alchemist Heady Topper")]) == ["ttb:1"]


def test_a_less_specific_name_finds_the_row():
    """Nobody types the brewery when the can says the beer."""
    assert match_typed_name("Hazy Little Thing", None,
                            [_row("ttb:1", "Sierra Nevada Hazy Little Thing")]) == ["ttb:1"]


def test_one_match_means_it_is_already_there_and_is_not_promoted():
    g = group_contributions([_c("Hazy Little Thing", "acct:a"),
                             _c("Hazy Little Thing", "acct:b")])[0]
    judged = judge(g, [_row("ttb:1", "Sierra Nevada Hazy Little Thing")])
    assert judged.verdict == IN_CATALOG
    # Two accounts agreed and it STILL does not promote, because it exists.
    assert len(judged.accounts) == 2
    assert not judged.promotable
    assert "already in the catalog" in judged.held_because


def test_several_matches_are_held_for_a_look_not_guessed():
    """The real case. A bare "Crusher" in this catalog covers The Alchemist's, another
    brewery's own bare "Crusher", and a dozen Bone Crushers; choosing one would hang the wrong
    producer and the wrong flavour profile off the can in somebody's hand."""
    g = group_contributions([_c("Crusher", "acct:a"), _c("Crusher", "acct:b")])[0]
    judged = judge(g, [_row("bcd:alch", "The Alchemist Crusher", "the alchemist"),
                       _row("ttb:9", "Crusher", "crusher"),
                       _row("ttb:7", "Bone Crusher Stout", "bone crusher")])
    assert judged.verdict == NEEDS_A_LOOK
    assert not judged.promotable
    assert len(judged.catalog) > 1
    assert "does not say which" in judged.reason


def test_nothing_matching_is_new():
    g = group_contributions([_c("Petit Mutant", "acct:a")])[0]
    assert judge(g, [_row("ttb:1", "Sierra Nevada Pale Ale")]).verdict == NEW


def test_a_name_that_identifies_a_class_is_held():
    """"Blended Scotch Whisky" is sold by forty companies. Promoting it makes a row the
    resolver would draw for any of them."""
    g = group_contributions([_c("Blended Scotch Whisky", "acct:a", category="spirits"),
                             _c("Blended Scotch Whisky", "acct:b", category="spirits")])[0]
    judged = judge(g, [])
    assert judged.verdict == NEEDS_A_LOOK
    assert "class of drink" in judged.reason
    assert not judged.promotable


def test_a_name_with_no_identifying_word_is_held():
    g = group_contributions([_c("J&B", "acct:a", category="spirits"),
                             _c("J&B", "acct:b", category="spirits")])[0]
    assert judge(g, []).verdict == NEEDS_A_LOOK


def test_a_typed_age_is_not_the_row_without_one():
    """Ages are identity. The 16-year-old is not the bottle called plain Lagavulin."""
    assert match_typed_name("Lagavulin 16", None, [_row("ttb:1", "Lagavulin")]) == []
    # The other way round is fine: a typed bare name can be the aged row...
    assert match_typed_name("Lagavulin", None, [_row("ttb:1", "Lagavulin 16")]) == ["ttb:1"]
    # ...and when it reaches three of them, that is three, which the caller holds.
    many = [_row("ttb:1", "Lagavulin 8"), _row("ttb:2", "Lagavulin 16"),
            _row("ttb:3", "Lagavulin 25")]
    assert len(match_typed_name("Lagavulin", None, many)) == 3


def test_a_shared_category_word_is_not_a_match():
    assert match_typed_name("Double IPA", None, [_row("ttb:1", "Odell Double IPA")]) == []


def test_a_different_brewery_with_the_same_beer_name_is_not_a_match():
    """Hoepfner and Gösser both sell a Natur Radler and they are different beers."""
    assert match_typed_name("Natur Radler", "Gosser",
                            [_row("ttb:1", "Natur Radler", "Hoepfner")]) == []


def test_a_name_must_read_as_a_phrase_not_just_share_words():
    """The looseness this had. "Hazy Little Thing" and "Juicy Little Thing Hazy IPA" share
    every word of the former, so set containment called them one drink -- but the words are
    neither together nor in that order, and in the second one "Hazy" is a description while
    "Juicy" is the name. On the live catalog this took a typed "Hazy Little Thing" from 5
    candidate rows to 3, dropping the two that were not that beer."""
    assert match_typed_name("Hazy Little Thing", None,
                            [_row("off:1", "Juicy Little Thing Hazy IPA", "sierra nevada")]) == []
    assert match_typed_name(
        "Hazy Little Thing", None,
        [_row("ttb:1", "My Favorite Thing Little Big Star Hazy Double India Pale Ale")]) == []


def test_extra_words_at_either_end_are_still_fine():
    """Which is the whole point of containment: a brand in front and a style behind are what a
    less specific name is missing."""
    assert match_typed_name("Hazy Little Thing", None,
                            [_row("ttb:1", "Sierra Nevada Hazy Little Thing")]) == ["ttb:1"]
    assert match_typed_name("Hazy Little Thing", None,
                            [_row("off:1", "Hazy Little Thing IPA")]) == ["off:1"]
    assert match_typed_name("Hazy Little Thing", None,
                            [_row("ttb:2", "Sierra Nevada Nitro Hazy Little Thing")]) == ["ttb:2"]


def test_the_brand_may_be_the_words_in_front():
    """A label reads "TITO'S HANDMADE VODKA" and the catalog splits it into a brand row and a
    product named "Handmade Vodka", so the phrase has to be built the way `search_name` builds
    it before the words can be looked for in order."""
    assert match_typed_name("Titos Handmade Vodka", None,
                            [_row("ttb:1", "Handmade Vodka", "Titos")]) == ["ttb:1"]


def test_reordering_the_words_is_not_a_match_and_that_is_the_trade():
    """"Branca Fernet" will not reach "Fernet Branca", and a window that allowed any order
    would not fix the bug above: in "Juicy Little Thing Hazy IPA" the words of "Hazy Little
    Thing" DO occupy three consecutive positions. Order is what separates them. A reordered
    typing reads as new, which the two-account gate then holds for a person to look at."""
    assert match_typed_name("Branca Fernet", None, [_row("ttb:1", "Fernet Branca")]) == []


def test_a_compound_word_is_not_a_typo():
    """`difflib` rewards one word being contained in another, so "crusher" and "funcrusher"
    score 0.824 and read as the same word mistyped. They are two words."""
    assert not same_word("crusher", "funcrusher")
    assert match_typed_name("Crusher", None, [_row("ttb:1", "Funcrusher")]) == []
    # A plural still is a typo's distance away, and is the same drink.
    assert same_word("banger", "bangers")


def test_short_words_must_match_exactly():
    """Measured: at 0.80 fuzzy matching catches half the typos in a four-letter word at three
    to five times the false-pair rate, so under five letters a word has to be right."""
    assert same_word("topper", "toper")
    assert not same_word("bock", "back")
    assert not same_word("gose", "rose")
    assert same_word("lager", "lagre")


# ---- two different accounts ---------------------------------------------------------------


def test_two_accounts_promote_it():
    g = plan([_c("Petit Mutant", "acct:a"), _c("Petit Mutant", "acct:b")], {})[0]
    assert g.verdict == NEW
    assert len(g.accounts) == 2
    assert g.promotable


def test_one_account_twice_is_still_one_account():
    """The route is idempotent per account and client id, so one phone can file the same drink
    twice in a minute. Two entries from one person is not two people agreeing."""
    g = plan([_c("Petit Mutant", "acct:a", doc="doc:1"),
              _c("Petit Mutant", "acct:a", doc="doc:2")], {})[0]
    assert len(g.members) == 2
    assert len(g.accounts) == 1
    assert not g.promotable
    assert "1 account" in g.held_because


def test_one_account_alone_is_held_and_says_what_is_missing():
    g = plan([_c("Petit Mutant", "acct:a")], {})[0]
    assert not g.promotable
    assert "1 more needed" in g.held_because


def test_a_typo_from_a_second_person_is_the_same_drink():
    """Otherwise the gate never opens: two people typing a label will not type it identically,
    and counting spellings instead of drinks would hold a corroborated drink forever."""
    groups = plan([_c("Petit Mutant", "acct:a"), _c("Petit Mutnat", "acct:b")], {})
    assert len(groups) == 1
    assert groups[0].promotable
    assert len(groups[0].spellings) == 2


def test_the_fuller_spelling_becomes_the_name():
    """The one the resolver can place. A bare beer name reaches far more rows than a name
    carrying its brewery."""
    groups = plan([_c("Heady Topper", "acct:a"),
                   _c("The Alchemist Heady Topper", "acct:b")], {})
    assert groups[0].name == "The Alchemist Heady Topper"


def test_the_most_agreed_spelling_wins_over_the_longer_one():
    groups = plan([_c("Petit Mutant", "acct:a"), _c("Petit Mutant", "acct:b"),
                   _c("Petit Mutant Sour", "acct:c")], {})
    assert groups[0].name == "Petit Mutant"


def test_two_different_drinks_do_not_group():
    groups = plan([_c("Petit Mutant", "acct:a"), _c("Focal Banger", "acct:b")], {})
    assert len(groups) == 2
    assert not any(g.promotable for g in groups)


def test_a_permissive_name_does_not_chain_two_drinks_together():
    """"Crusher" covers both, but Bone Crusher Stout and Hop Crusher are not one drink, and a
    cluster grown through the loose member would promote a drink nobody named."""
    groups = group_contributions([_c("Bone Crusher Stout", "acct:a"),
                                  _c("Crusher", "acct:b"),
                                  _c("Hop Crusher", "acct:c")])
    assert len(groups) >= 2
    assert all(len(g.accounts) < 3 for g in groups)


def test_drinks_in_different_categories_never_group():
    groups = plan([_c("Mutant", "acct:a", category="beer"),
                   _c("Mutant", "acct:b", category="spirits")], {})
    assert len(groups) == 2


# ---- what the promoted row would carry ----------------------------------------------------


def test_an_abv_is_kept_only_when_they_agree():
    agree = plan([_c("Petit Mutant", "acct:a", abv=8.0),
                  _c("Petit Mutant", "acct:b", abv=8.0)], {})[0]
    assert agree.abv_pct == 8.0
    # Two different numbers for one drink is not a fact about it, and the mean is a third
    # figure nobody typed.
    clash = plan([_c("Petit Mutant", "acct:a", abv=8.0),
                  _c("Petit Mutant", "acct:b", abv=6.5)], {})[0]
    assert clash.abv_pct is None


def test_a_shorter_real_name_becomes_an_alias():
    """What the can says. A scan reading "Heady Topper" should land on the row."""
    g = plan([_c("The Alchemist Heady Topper", "acct:a"),
              _c("Heady Topper", "acct:b")], {})[0]
    assert g.name == "The Alchemist Heady Topper"
    assert g.aliases == ("Heady Topper",)


def test_a_typo_never_becomes_an_alias():
    """It is not written on anything. Indexing it would put a string no label carries into the
    thing the recognizer matches against, which makes the index worse in exchange for one
    person's thumb."""
    g = plan([_c("Petit Mutant", "acct:a"), _c("Petit Mutant", "acct:b"),
              _c("Petit Mutnat", "acct:c")], {})[0]
    assert g.name == "Petit Mutant"
    assert "Petit Mutnat" in g.spellings       # still recorded as evidence...
    assert g.aliases == ()                     # ...but never as a name


def test_a_maker_nobody_erases():
    g = plan([_c("Petit Mutant", "acct:a", maker="The Alchemist", at="2026-10-01T00:00:00Z"),
              _c("Petit Mutant", "acct:b", at="2026-10-02T00:00:00Z")], {})[0]
    assert g.maker == "The Alchemist"


def test_promotable_groups_come_first():
    groups = plan([_c("Alone", "acct:a"),
                   _c("Petit Mutant", "acct:b"), _c("Petit Mutant", "acct:c")], {})
    assert groups[0].name == "Petit Mutant"
    assert groups[0].promotable
    assert not groups[1].promotable
