"""Style prior — every product gets a plausible, correctly-shaped sensory vector + ABV."""
from __future__ import annotations

import pytest
from bcd_enrich.style_prior import (
    _CENTROIDS,
    abv_from_style,
    adjuncts,
    agave_marks,
    detect_style,
    sensory_from_style,
)
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


# --- what the name says is IN the beer, not just what kind it is (2026-09-22) ---

@pytest.mark.parametrize("name, style", [
    ("Firestone Walker Wee Heavy", "scottish_ale"),
    ("Schlafly Kölsch", "kolsch"),
    ("Ballast Point West Coast IPA", "west_coast_ipa"),
    ("Founders Dirty Bastard Scotch Ale", "scottish_ale"),
    ("Sierra Nevada Bigfoot Barleywine", "barleywine"),
    ("21st Amendment Back in Black IPA", "black_ipa"),
    ("Founders All Day Session IPA", "session_ipa"),
    ("Left Hand Milk Stout", "sweet_stout"),
    ("Samuel Smith Oatmeal Stout", "oatmeal_stout"),
    ("Uerige Altbier", "altbier"),
    ("Köstritzer Schwarzbier", "schwarzbier"),
    ("Fuller's ESB", "esb"),
    ("Weihenstephaner Dunkelweizen", "dunkelweizen"),
    ("Paulaner Festbier", "festbier"),
    ("Genesee Cream Ale", "cream_ale"),
    ("Terrapin Rye Pale Ale", "rye_beer"),
])
def test_the_name_states_a_style_the_rules_had_no_centroid_for(name, style):
    assert detect_style(name, Category.BEER) == style


def test_a_short_style_word_is_read_whole():
    """The spirits patterns taught this the hard way: a bare substring hides inside a longer
    word. "esb" is in "Desby", "alt" in "Walter" -- so these rules are anchored."""
    assert detect_style("Desby Brewing Pale Ale", Category.BEER) == "pale_ale"
    assert detect_style("Walter's Amber", Category.BEER) == "amber"


def test_the_filed_class_wins_unless_the_name_says_a_kind_of_it():
    """A TTB class is a real filing, so it outranks a guess off the name -- but a "Milk Stout"
    filed as "Stout" is still a stout, and reading the lactose only sharpens it."""
    assert detect_style("Milk Stout", Category.BEER, class_type="Stout") == "sweet_stout"
    # A barleywine filed as a stout is a conflict, not a refinement: trust the filing.
    assert detect_style("Barleywine", Category.BEER, class_type="Stout") == "stout"


def test_nitro_is_how_a_beer_is_poured_not_what_is_in_it():
    """Guinness is a dry stout on nitrogen, not a sweet one -- so nitro softens the bubbles
    and fills the body without adding the lactose sweetness of a milk stout."""
    assert detect_style("Guinness Draught Nitro Stout", Category.BEER) == "stout"
    flat = sensory_from_style("Draught Stout", Category.BEER)
    nitro = sensory_from_style("Nitro Draught Stout", Category.BEER)
    assert nitro.axes["carbonation"] < flat.axes["carbonation"]
    assert nitro.axes["body_fullness"] > flat.axes["body_fullness"]
    assert nitro.axes.get("sweet", 0) == flat.axes.get("sweet", 0)


def test_an_adjunct_moves_the_styles_axes():
    plain = sensory_from_style("Porter", Category.BEER)
    loaded = sensory_from_style("Barrel-Aged Coconut Coffee Porter", Category.BEER)
    assert adjuncts("Barrel-Aged Coconut Coffee Porter") == ["coffee", "coconut", "barrel"]
    assert loaded.axes["roasted_coffee_choc"] > plain.axes["roasted_coffee_choc"]
    assert loaded.axes["nutty"] > plain.axes["nutty"]
    assert loaded.axes["vanilla_oak"] > plain.axes.get("vanilla_oak", 0)
    # Still a guess about the product, however much the name says, and the recommender reads
    # anything at or above 0.45 as knowing it.
    assert plain.confidence < loaded.confidence < 0.45
    assert all(0.0 <= v <= 1.0 for v in loaded.to_array())


def test_the_prior_is_absolute_so_running_it_again_changes_nothing():
    """The agave character pass added a delta to whatever the row already carried, so a second
    run doubled it. This reads only the name and the style's own centroid: re-deriving it is a
    no-op, and a row can be re-profiled without knowing whether it was profiled before."""
    name = "Barrel-Aged Coconut Coffee Porter"
    once = sensory_from_style(name, Category.BEER)
    twice = sensory_from_style(name, Category.BEER)
    assert once.axes == twice.axes and once.confidence == twice.confidence


def test_a_fruit_in_somebody_s_name_is_not_a_fruit_in_the_beer():
    """Blackberry Farm is a brewery in Tennessee; Pecan Street is in Austin. The word only
    counts as an ingredient if some mention of it is not an address."""
    assert adjuncts("Blackberry Farm Brewery Ampersand Saison") == []
    assert adjuncts("Pecan Street Brewing Ride On Helles Lager") == []
    assert adjuncts("Cocoa Beach Oktoberfest") == []
    assert adjuncts("Blackberry Farm Strawberry Buckwheat") == ["berry_fruit"]
    # Said twice, meant the second time.
    assert "coconut" in adjuncts("Coconut Tree Barrel Aged Imperial Coconut Stout")


def test_adjuncts_are_read_for_beer_only():
    """A flavored spirit already has a centroid of its own -- `flavored_whiskey` is where the
    honey went -- so reading the name again would count it twice."""
    spirit = "Wild Turkey American Honey"
    assert sensory_from_style(spirit, Category.SPIRIT).axes == \
        _CENTROIDS[detect_style(spirit, Category.SPIRIT)]
    assert sensory_from_style("Honey Wheat Ale", Category.BEER).axes["honey"] > \
        sensory_from_style("Wheat Ale", Category.BEER).axes.get("honey", 0)


def test_an_agave_label_states_its_maturation_and_the_vector_follows():
    """61% of the 17,286 agave rows say how long they rested, which agave, or how they were
    made -- and none of it reached the vector: 11,889 tequila rows shared 7 vectors, 3,912
    mezcal rows shared 3, and the recommender shows one entry per vector (2026-09-22). Oak is
    the big one; a blanco and an extra añejo are the same distillate and not the same drink."""
    def oak(n):
        return sensory_from_style(n, Category.SPIRIT, style_hint="Tequila Fb").axes

    # a blanco carries no oak axis at all -- nothing in it ever touched wood
    ladder = [oak(f"Casa Probe {e}").get("vanilla_oak", 0.0)
              for e in ("Blanco", "Gold", "Reposado", "Anejo", "Extra Anejo")]
    assert ladder == sorted(ladder), ladder
    assert ladder[0] == 0.0 and ladder[-1] > 0.5
    # and the raw agave goes the other way
    assert oak("Casa Probe Blanco")["grassy"] > oak("Casa Probe Extra Anejo")["grassy"]


def test_a_bottle_rested_once_so_only_one_maturation_is_read():
    """Ordering inside the `age` family is what keeps a brand word from being read as an
    expression -- the shape that once read the PLATA in "Rio de Plata Tequila Añejo"."""
    age = {"blanco", "reposado", "anejo", "extra_anejo", "cristalino", "gold"}
    for name, want in [("Cinco Blancos Anejo", "anejo"),
                       ("Cinco Blancos Reposado", "reposado"),
                       ("Don Ramon Plata Cristalino Platinum", "cristalino"),
                       ("Corrido Cristalino Blanco", "cristalino"),
                       ("Herrncia De Plata Anejo", "anejo"),
                       ("Degollado Silver 100% Agave", "blanco")]:
        assert [m for m in agave_marks(name) if m in age] == [want], name


def test_oro_is_somebody_s_name_before_it_is_a_colour():
    """`oro` unanchored hides inside "adoro" (200 extra rows); anchored it is still a brand
    word far more often than an expression, so gold yields to any stated maturation and to
    "de oro" outright."""
    assert agave_marks("Carreta De Oro Añejo") == ["anejo"]
    assert agave_marks("Gran Cava De Oro Blanco") == ["blanco"]
    assert agave_marks("Cava De Oro 25th Anniversary Limited Edition") == []
    assert agave_marks("Adoro Silver") == ["blanco"]
    assert agave_marks("Sombrero Gold") == ["gold"]


def test_varietals_and_processes_stack_but_a_whiskey_is_not_read_this_way():
    """An ensamble really is several agaves, and a pechuga really is also a capón. But
    "reposado" on a whiskey is somebody else's word, so the marks are gated on the STYLE."""
    marks = agave_marks("5 Sentidos Espadin y Tobala Pechuga")
    assert "espadin" in marks and "tobala" in marks and "pechuga" in marks
    whiskey = "Old Probe Reposado Barrel Finished"
    assert detect_style(whiskey, Category.SPIRIT, class_type="Straight Bourbon Whisky") == "bourbon"
    assert sensory_from_style(whiskey, Category.SPIRIT, style_hint="Straight Bourbon Whisky") \
        .axes == _CENTROIDS["bourbon"]


def test_reading_a_label_twice_lands_in_the_same_place():
    """Each mark moves the STYLE'S OWN centroid, so the answer is a function of (style, name)
    and re-running the enrichment recomputes it instead of adding a second helping."""
    n, hint = "Real Minero Arroqueño Pechuga Anejo", "Agave Spirits"
    first = sensory_from_style(n, Category.SPIRIT, style_hint=hint).axes
    assert sensory_from_style(n, Category.SPIRIT, style_hint=hint).axes == first
    assert all(0.0 <= v <= 1.0 for v in first.values())
