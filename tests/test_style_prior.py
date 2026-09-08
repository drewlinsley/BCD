"""Style prior — every product gets a plausible, correctly-shaped sensory vector + ABV."""
from __future__ import annotations

import pytest
from bcd_enrich.style_prior import abv_from_style, detect_style, sensory_from_style
from bcd_schema import SENSORY_AXES, Category, SensorySource


def _top_axis(sv):
    return max(sv.axes.items(), key=lambda kv: kv[1])[0]


@pytest.mark.parametrize("name, cat, style", [
    ("Lagunitas IPA", Category.BEER, "ipa"),
    ("Hazy Little Thing IPA", Category.BEER, "neipa"),
    ("Founders Imperial Stout", Category.BEER, "imperial_stout"),
    ("Pilsner Urquell", Category.BEER, "pilsner"),
    ("Hoegaarden Witbier", Category.BEER, "wheat"),
    ("Grey Goose Vodka", Category.SPIRIT, "vodka"),
    ("Hendrick's Gin", Category.SPIRIT, "gin"),
    ("Lagavulin 16 Single Malt Scotch", Category.SPIRIT, "peated_scotch"),
    ("Buffalo Trace Kentucky Bourbon", Category.SPIRIT, "bourbon"),
    ("Cointreau", Category.SPIRIT, "triple_sec"),
    ("Rhum Blanc Agricole", Category.SPIRIT, "white_rum"),
])
def test_detect_style(name, cat, style):
    assert detect_style(name, cat) == style


def test_category_gates_ambiguous_words():
    # "Blanc/Blanche" is a witbier in beer, a white rum in spirits.
    assert detect_style("Blanche de Bruxelles", Category.BEER) == "wheat"
    assert detect_style("Bacardi Carta Blanca", Category.SPIRIT) == "white_rum"


def test_unknown_style_falls_back_to_category_centroid():
    assert detect_style("Some Mystery Brew", Category.BEER) == "beer"
    assert detect_style("Unnamed Spirit", Category.SPIRIT) == "spirit"
    assert detect_style("whatever", None) is None


def test_vector_shape_and_source():
    sv = sensory_from_style("West Coast IPA", Category.BEER)
    assert sv is not None
    assert sv.source == SensorySource.STYLE_PRIOR
    assert 0.0 < sv.confidence <= 0.4
    assert len(sv.to_array()) == len(SENSORY_AXES) == 25
    assert all(0.0 <= v <= 1.0 for v in sv.to_array())


def test_signature_axes_are_dominant():
    assert _top_axis(sensory_from_style("Guinness Draught Stout", Category.BEER)) \
        == "roasted_coffee_choc"
    assert _top_axis(sensory_from_style("Laphroaig 10", Category.SPIRIT)) == "smoky_peat"
    # Vodka is near-neutral: alcohol warmth outweighs any flavor axis.
    assert _top_axis(sensory_from_style("Absolut Vodka", Category.SPIRIT)) == "alcohol_warmth"


def test_non_alcoholic_zeroes_warmth():
    reg = sensory_from_style("Jupiler", Category.BEER)
    na = sensory_from_style("Jupiler 0,0%", Category.BEER)
    assert reg.axes.get("alcohol_warmth", 0) >= 0.0
    assert na.axes["alcohol_warmth"] == 0.0
    assert abv_from_style("Jupiler 0,0%", Category.BEER) == 0.4


def test_unknown_returns_none():
    assert sensory_from_style("mystery", None) is None
    assert abv_from_style("mystery", None) is None


def test_abv_prior_reasonable():
    assert abv_from_style("Lagunitas IPA", Category.BEER) == pytest.approx(6.5)
    assert abv_from_style("Tito's Vodka", Category.SPIRIT) == 40.0
    assert 4.0 <= abv_from_style("Bud Light Lager", Category.BEER) <= 5.5
