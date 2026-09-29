"""Style prior — every product gets a plausible, correctly-shaped sensory vector + ABV."""
from __future__ import annotations

import pytest
from bcd_enrich.style_prior import (
    _CENTROIDS,
    abv_from_style,
    adjuncts,
    agave_marks,
    detect_style,
    readable_style,
    sensory_from_style,
    whisky_marks,
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
    """The beer table is beer's OWN vocabulary. A spirit's name is read by the spirit tables
    instead, so no word is ever counted twice from two tables. "Pumpkin" is a beer adjunct and
    no whisky's mark, so on a bourbon it must move nothing at all."""
    spirit = "Old Probe Pumpkin Bourbon"
    assert adjuncts(spirit) == ["pumpkin"] and whisky_marks(spirit) == []
    assert sensory_from_style(spirit, Category.SPIRIT).axes == _CENTROIDS["bourbon"]
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
    whiskey = "Old Probe Reposado Bourbon"
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


def test_a_whisky_label_states_its_age_and_its_cask():
    """The catalog's worst collapse: 42,147 style-prior whisky rows held 12 vectors between
    seven styles, and all 4,527 single malt Scotches shared one (2026-09-24). Only 21% of these
    names state anything -- but the ones that do are the bottles people ask for."""
    def ax(n):
        return sensory_from_style(n, Category.SPIRIT, style_hint="Single Malt Scotch Whisky").axes

    ladder = [ax(f"Probe {a}").get("vanilla_oak", 0.0)
              for a in ("5 Year", "12 Year", "18 Year", "30 Year")]
    assert ladder == sorted(ladder), ladder
    # a sherry cask puts fruit in it that no bare centroid has
    assert ax("Probe 12 Year Oloroso Sherry Cask")["stone_fruit"] > ax("Probe 12 Year")\
        .get("stone_fruit", 0.0)
    # an unpeated Scotch carries no smoke axis at all
    assert ax("Probe Heavily Peated")["smoky_peat"] > ax("Probe").get("smoky_peat", 0.0)


def test_an_age_statement_is_a_number_and_the_largest_one_is_the_claim():
    """"Batch No. 2 21 Yr" is a 21-year whisky. The catalog states ages up to 49."""
    assert whisky_marks("Ad Rattray Batch No. 2 21 Yr") == ["age_vintage"]
    assert whisky_marks("Alexander Murray Highland 49 Yo") == ["age_vintage"]
    assert whisky_marks("Probe 12 Year Old") == ["age_mid"]
    assert whisky_marks("Probe 6 Yr") == ["age_young"]
    # a bare number is not an age
    assert whisky_marks("Old Charter 8") == []


def test_the_whisky_guards_are_the_ones_the_catalog_earned():
    """Each of these is a real row that a looser pattern claimed. `char\\w*` reaches "Charles"
    and "Charlottesville"; `peat\\w*` reaches "Peatside"; and "Port Dundas" is a distillery in
    Glasgow, so port has to sit next to a cask word to count as one."""
    assert whisky_marks("Cape Charles Distillery") == []
    assert whisky_marks("Bowman Brothers Charlottesville") == []
    assert whisky_marks("Ad Rattray Peatside 6 Yr") == ["age_young"]
    assert whisky_marks("Barrel To Bottle Port Dundas") == []
    assert whisky_marks("1876 Port Barrel Finish") == ["port"]
    assert whisky_marks("Buzzard's Roost Char #1") == ["charred_oak"]


def test_a_whisky_rested_in_one_cask_at_one_strength():
    """`cask`, `strength`, `peat` and `grain` are exclusive families: table order decides, so a
    named cask beats the generic "finished in" and Islay beats a bare "peated"."""
    m = whisky_marks("15 Stars Sherry Cask Finished")
    assert "sherry" in m and "cask_finish" not in m
    assert "heavily_peated" in whisky_marks("12 Yr Old 100% Islay Single Cask")
    assert "peated" not in whisky_marks("Abomination Heavily Peated Malt")
    # but a flavoured bottling may state several flavours at once
    assert set(whisky_marks("Austin Peppered Maple Blood Orange Bourbon")) >= \
        {"maple", "pepper_flavor", "citrus_flavor"}


def test_smoke_is_a_mark_a_label_states_not_a_style():
    """"smoky"/"smoke" were keywords for `peated_scotch`, so 520 rows reached an Islay centroid
    at smoky_peat 0.9 with no peat on the label: "1930 Smoked Rum", "Archrival Smoked Gin",
    "Alma Loca Original Smoked Margarita", the whole Ole Smoky moonshine line (2026-09-24).
    Smoke now reaches the vector as a whisky MARK, on top of whatever the row really is."""
    for name, hint, want in [("1930 Smoked Rum", "Rum Specialties", "rum"),
                             ("Archrival Smoked Gin", "Distilled Gin", "gin"),
                             ("Ole Smoky Dill Pickle Moonshine",
                              "Other Specialties & Proprietaries", "herbal_liqueur"),
                             ("Arby's Smoked Bourbon", "Bourbon Whisky", "bourbon")]:
        assert detect_style(f"{hint} {name}", Category.SPIRIT, class_type=hint) == want, name
    # peat itself, and the distilleries famous for it, still name a peated Scotch
    for name in ("Whiskey Del Bac Ode To Islay", "Nikka Yoichi Peaty & Salty",
                 "Laphroaig 10", "Westland Peat Week"):
        assert detect_style(name, Category.SPIRIT) == "peated_scotch", name
    # and a smoked bourbon is a bourbon WITH smoke in it, not a Scotch
    smoked = sensory_from_style("Arby's Smoked Bourbon", Category.SPIRIT,
                                style_hint="Bourbon Whisky").axes
    assert "smoked" in whisky_marks("Arby's Smoked Bourbon")
    assert smoked["smoky_peat"] == 0.25
    assert smoked["vanilla_oak"] == _CENTROIDS["bourbon"]["vanilla_oak"]


def test_a_single_malt_is_not_a_scotch_unless_the_label_says_so():
    """"single malt" was a keyword for `scotch`, so 566 rows printed "Scotch Whisky" on those
    two words alone -- 259 saying American, Oregon or Texas on the label and 271 saying nothing
    about origin at all (2026-09-25). Single malt is how a whisky is made, one distillery and
    all malted barley. Scotch is a place."""
    def style(name, code):
        return detect_style(f"{code} {name}", Category.SPIRIT, class_type=code)

    # made that way, somewhere else
    assert style("Bull Run Distilling Co Oregon Single Malt", "Whisky") == "single_malt"
    assert style("Westland American Single Malt", "Whisky") == "single_malt"
    assert readable_style("Whisky", "single_malt") == "Single Malt Whisky"
    # the registry files American single malts as their own thing; print what it filed
    for code in ("American Single Malt Whiskey", "Straight American Single Malt"):
        assert style("Probe Single Malt", code) == "single_malt"
        assert readable_style(code, "single_malt") == "American Single Malt"
    # a Scotch filing, a Scottish region, or the word itself still names a Scotch
    assert style("Kilkerran Single Malt", "Scotch Whisky Fb") == "scotch"
    assert style("Old Pulteney Highland Single Malt", "Whisky") == "scotch"
    assert style("Loch Haim Single Malt Scotch Whisky", "Whisky") == "scotch"
    assert readable_style("Single Malt Scotch Whisky", "scotch") == "Single Malt Scotch"
    # and an origin the registry states outright wins over the method
    assert style("Tyrconnel Single Malt Irish Whiskey", "Irish Whisky Fb") == "irish_whiskey"


def test_a_pure_malt_is_a_blend_and_not_a_single_malt():
    """A looser "malt whisky" keyword claims 155 names, 87 of which say PURE, BLENDED or VATTED
    malt -- malts from several distilleries, the opposite of a single malt. Only the phrase
    itself counts; the 81 rows filed as class "Malt Whisky" are read from the filing."""
    assert detect_style("Japanese blended whisky Suntory Pure Malt Whiskey", Category.SPIRIT,
                        class_type="Japanese blended whisky") == "whiskey"
    assert detect_style("Whisky Koshiji Pure Malt Whisky 10 Years", Category.SPIRIT,
                        class_type="Whisky") == "whiskey"
    assert detect_style("Malt Whisky Foundry Single Malt", Category.SPIRIT,
                        class_type="Malt Whisky") == "single_malt"
    assert readable_style("Malt Whisky", "single_malt") == "Malt Whisky"


def test_a_single_malt_carries_its_own_centroid_and_reads_its_own_label():
    """It is not a Scotch's vector: more malt and more caramel, because what files here is
    usually matured in new charred oak rather than a refill cask, and none of the honeyed
    softness age in a used butt gives. And it is a whisky, so the whisky marks apply."""
    assert _CENTROIDS["single_malt"] != _CENTROIDS["scotch"]
    assert _CENTROIDS["single_malt"]["malty_bready"] > _CENTROIDS["scotch"]["malty_bready"]
    assert _CENTROIDS["single_malt"]["honey"] < _CENTROIDS["scotch"]["honey"]
    aged = sensory_from_style("Probe 18 Year Single Malt Sherry Cask", Category.SPIRIT,
                              style_hint="Whisky").axes
    young = sensory_from_style("Probe Single Malt", Category.SPIRIT, style_hint="Whisky").axes
    assert aged["vanilla_oak"] > young["vanilla_oak"]
    assert aged["stone_fruit"] > young["stone_fruit"]
