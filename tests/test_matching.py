"""Label-text matching — the decision layer that keeps OCR garbage off the HUD."""

from __future__ import annotations

from bcd_api.matching import (
    AMBIGUOUS_MIN_SCORE,
    RESOLVE_MIN_SCORE,
    Identity,
    coverage,
    object_query,
    score_identity,
    token_similarity,
    tokenize,
    verdict,
)

HEADY = Identity(product_name="Heady Topper", brand_name="Heady",
                 producer_name="The Alchemist")
FOCAL = Identity(product_name="Focal Banger", brand_name="Focal Banger",
                 producer_name="The Alchemist")
SIXTY = Identity(product_name="60 Minute IPA", brand_name="60 Minute",
                 producer_name="Dogfish Head Craft Brewery")
PLAIN_IPA = Identity(product_name="IPA", brand_name="IPA", producer_name="Some Brewing Co")


def q(*fragments: str) -> list[str]:
    return tokenize(object_query(list(fragments)))


def test_tokenize_and_object_query_merge_fragments():
    assert tokenize("THE ALCHEMIST — Heady Topper, 16 FL OZ") == \
        ["the", "alchemist", "heady", "topper", "16", "fl", "oz"]
    assert object_query(["HEADY", "TOPPER", "heady topper"]) == "heady topper"


def test_token_similarity_tolerates_ocr_typos_but_not_fragments():
    assert token_similarity("topper", "topper") == 1.0
    assert token_similarity("toppfr", "topper") >= 0.8  # one substituted glyph
    assert token_similarity("chemist", "alchemist") >= 0.8  # dropped leading letters
    assert token_similarity("mist", "alchemist") == 0.0  # not the same word
    assert token_similarity("ipa", "ipl") == 0.0  # short tokens must be exact


def test_generic_tokens_barely_count_toward_coverage():
    # "IPA" is on half the cans in the fridge: it must not identify "60 Minute IPA".
    assert coverage(["ipa"], tokenize("60 Minute IPA")) < 0.2
    assert coverage(["60", "minute", "ipa"], tokenize("60 Minute IPA")) == 1.0


def test_full_label_read_resolves():
    s = score_identity(q("THE ALCHEMIST", "HEADY", "TOPPER", "16 FL OZ"), HEADY)
    assert s >= RESOLVE_MIN_SCORE


def test_the_partners_failure_case_does_not_resolve():
    # What the HUD actually saw on a Heady Topper can. Producer evidence only -> at most
    # a shortlist for the fine stage, never a fired overlay.
    frags = q("Chemist", "hop chemist", "Mist")
    heady, focal = score_identity(frags, HEADY), score_identity(frags, FOCAL)
    assert heady < RESOLVE_MIN_SCORE and focal < RESOLVE_MIN_SCORE
    assert verdict(sorted([heady, focal], reverse=True)) in ("ambiguous", "unresolved")
    assert verdict(sorted([heady, focal], reverse=True)) != "resolved"


def test_producer_only_evidence_is_ambiguous_between_siblings():
    frags = q("THE ALCHEMIST", "WATERBURY VT")
    scores = sorted([score_identity(frags, HEADY), score_identity(frags, FOCAL)], reverse=True)
    assert scores[0] >= AMBIGUOUS_MIN_SCORE
    assert verdict(scores) == "ambiguous"  # two Alchemist beers, no name evidence


def test_ocr_typo_in_name_still_resolves():
    s = score_identity(q("HEADY", "TOPPFR"), HEADY)
    assert s >= RESOLVE_MIN_SCORE


def test_generic_only_name_needs_producer():
    assert score_identity(q("IPA"), PLAIN_IPA) < RESOLVE_MIN_SCORE
    assert score_identity(q("IPA", "Some Brewing Co"), PLAIN_IPA) >= RESOLVE_MIN_SCORE


def test_verdict_requires_margin():
    assert verdict([0.9, 0.2]) == "resolved"
    assert verdict([0.9, 0.85]) == "ambiguous"  # two near-identical candidates
    assert verdict([0.1]) == "unresolved"
    assert verdict([]) == "unresolved"
    assert verdict([0.7], min_score=0.8) == "ambiguous"  # client raised the floor
