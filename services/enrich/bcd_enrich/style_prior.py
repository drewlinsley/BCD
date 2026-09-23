"""Style prior — a defensible SensoryVector (and ABV) for a product from its style alone.

The chemistry prior (`sensory_from_recipe`) needs a hop/malt bill, which most OFF rows don't
carry — so it yields nothing for the bulk of the catalog and the recommendation index stays dark.
This fills the gap with the weakest-but-universal signal the schema already names: a BJCP/Meilgaard
STYLE centroid. Every "IPA" gets a hoppy/citrus/bitter baseline, every stout a roasty/coffee one,
every vodka a near-neutral warmth. It is explicitly the lowest-confidence source
(`SensorySource.STYLE_PRIOR`, `ExtractionMethod.LLM_INFERRED_FROM_STYLE_PRIOR`) so the reconciler
and the UI can always tell a guess from a stated fact.

Detection is keyword-over-name, category-gated (a beer's "blanc" is a witbier; a spirit's is a
white rum), most-specific first. Unknown styles fall back to a broad beer/spirit centroid so every
product becomes scoreable rather than invisible.
"""
from __future__ import annotations

import re
import unicodedata

from bcd_schema import Category, SensorySource, SensoryVector

# style key -> sensory axes (0-1). Only non-zero axes listed; the rest default to 0.
_CENTROIDS: dict[str, dict[str, float]] = {
    # --- beer ---
    "neipa": {"tropical": 0.85, "citrus": 0.7, "stone_fruit": 0.5, "bitterness": 0.4,
              "body_fullness": 0.65, "malty_bready": 0.35, "dryness_finish": 0.2, "sweet": 0.3,
                  "carbonation": 0.45},
    "dipa": {"citrus": 0.7, "tropical": 0.65, "piney_resinous": 0.6, "bitterness": 0.85,
             "malty_bready": 0.4, "caramel_toffee": 0.3, "alcohol_warmth": 0.5,
             "dryness_finish": 0.6, "body_fullness": 0.55, "carbonation": 0.45},
    "ipa": {"citrus": 0.65, "tropical": 0.5, "piney_resinous": 0.55, "bitterness": 0.75,
            "malty_bready": 0.35, "grassy": 0.3, "dryness_finish": 0.55, "body_fullness": 0.4,
                "carbonation": 0.5},
    "pale_ale": {"citrus": 0.5, "piney_resinous": 0.4, "bitterness": 0.55, "malty_bready": 0.45,
                 "caramel_toffee": 0.3, "floral": 0.3, "body_fullness": 0.4, "carbonation": 0.5},
    "imperial_stout": {"roasted_coffee_choc": 0.9, "caramel_toffee": 0.55, "malty_bready": 0.5,
                       "bitterness": 0.55, "body_fullness": 0.85, "alcohol_warmth": 0.6,
                       "sweet": 0.45, "vanilla_oak": 0.35, "nutty": 0.35, "carbonation": 0.3},
    "stout": {"roasted_coffee_choc": 0.85, "caramel_toffee": 0.45, "malty_bready": 0.5,
              "bitterness": 0.5, "body_fullness": 0.7, "nutty": 0.35, "sweet": 0.35,
                  "carbonation": 0.4},
    "porter": {"roasted_coffee_choc": 0.65, "caramel_toffee": 0.55, "malty_bready": 0.55,
               "nutty": 0.4, "bitterness": 0.4, "body_fullness": 0.6, "sweet": 0.35,
                   "carbonation": 0.45},
    "wheat": {"banana_ester": 0.7, "spicy_phenolic": 0.55, "citrus": 0.4, "malty_bready": 0.4,
              "carbonation": 0.6, "body_fullness": 0.45, "sweet": 0.3, "floral": 0.3},
    "tripel": {"spicy_phenolic": 0.55, "banana_ester": 0.5, "honey": 0.4, "caramel_toffee": 0.35,
               "alcohol_warmth": 0.45, "sweet": 0.4, "body_fullness": 0.5, "dryness_finish": 0.4,
                   "carbonation": 0.75},
    "belgian_dark": {"caramel_toffee": 0.55, "stone_fruit": 0.45, "spicy_phenolic": 0.45,
                     "banana_ester": 0.4, "malty_bready": 0.5, "alcohol_warmth": 0.5,
                     "sweet": 0.45, "body_fullness": 0.6, "carbonation": 0.65},
    "saison": {"spicy_phenolic": 0.6, "herbal": 0.45, "grassy": 0.4, "citrus": 0.35,
               "funk_brett": 0.3, "carbonation": 0.65, "dryness_finish": 0.65,
               "body_fullness": 0.35},
    "sour": {"sour_tart": 0.85, "funk_brett": 0.4, "citrus": 0.45, "berry": 0.45,
             "carbonation": 0.6, "dryness_finish": 0.55, "sweet": 0.25, "body_fullness": 0.3},
    "amber": {"caramel_toffee": 0.6, "malty_bready": 0.55, "nutty": 0.35, "bitterness": 0.4,
              "body_fullness": 0.45, "floral": 0.25, "carbonation": 0.45},
    "brown": {"nutty": 0.6, "caramel_toffee": 0.55, "malty_bready": 0.55,
              "roasted_coffee_choc": 0.3, "body_fullness": 0.45, "sweet": 0.3, "carbonation": 0.45},
    "bock": {"caramel_toffee": 0.6, "malty_bready": 0.6, "roasted_coffee_choc": 0.35, "nutty": 0.35,
             "body_fullness": 0.6, "alcohol_warmth": 0.35, "sweet": 0.4, "carbonation": 0.45},
    "pilsner": {"malty_bready": 0.5, "grassy": 0.4, "herbal": 0.35, "floral": 0.3,
                "bitterness": 0.45, "carbonation": 0.65, "dryness_finish": 0.55,
                "body_fullness": 0.3},
    "helles": {"malty_bready": 0.55, "honey": 0.3, "grassy": 0.3, "bitterness": 0.3,
               "carbonation": 0.6, "body_fullness": 0.35, "dryness_finish": 0.4},
    "radler": {"citrus": 0.65, "sweet": 0.5, "sour_tart": 0.4, "carbonation": 0.65,
               "bitterness": 0.15, "body_fullness": 0.25},
    "lager": {"malty_bready": 0.4, "grassy": 0.3, "bitterness": 0.3, "carbonation": 0.6,
              "dryness_finish": 0.5, "body_fullness": 0.3},
    "beer": {"malty_bready": 0.45, "bitterness": 0.35, "carbonation": 0.55, "body_fullness": 0.4,
             "dryness_finish": 0.4, "grassy": 0.25},
    # --- beer styles the name states and the rules had no centroid for (2026-09-22). Every
    # one of these was sitting on its parent's centroid or on the broad "beer" fallback. ---
    "west_coast_ipa": {"piney_resinous": 0.75, "citrus": 0.65, "bitterness": 0.8,
                       "tropical": 0.35, "grassy": 0.35, "malty_bready": 0.3,
                       "caramel_toffee": 0.2, "dryness_finish": 0.7, "body_fullness": 0.35,
                           "carbonation": 0.55},
    "black_ipa": {"citrus": 0.55, "piney_resinous": 0.55, "roasted_coffee_choc": 0.5,
                  "bitterness": 0.7, "malty_bready": 0.4, "caramel_toffee": 0.3,
                  "body_fullness": 0.5, "dryness_finish": 0.55, "carbonation": 0.5},
    "session_ipa": {"citrus": 0.6, "tropical": 0.45, "piney_resinous": 0.45, "bitterness": 0.6,
                    "grassy": 0.3, "malty_bready": 0.25, "body_fullness": 0.25,
                    "dryness_finish": 0.6, "carbonation": 0.55},
    "brut_ipa": {"citrus": 0.55, "tropical": 0.5, "floral": 0.4, "bitterness": 0.4,
                 "dryness_finish": 0.9, "carbonation": 0.75, "body_fullness": 0.2, "sweet": 0.05},
    "cold_ipa": {"citrus": 0.6, "piney_resinous": 0.55, "tropical": 0.45, "bitterness": 0.65,
                 "malty_bready": 0.25, "dryness_finish": 0.7, "carbonation": 0.6,
                 "body_fullness": 0.3},
    "sweet_stout": {"roasted_coffee_choc": 0.75, "sweet": 0.6, "caramel_toffee": 0.5,
                   "body_fullness": 0.8, "malty_bready": 0.45, "nutty": 0.35, "bitterness": 0.35,
                   "carbonation": 0.3, "vanilla_oak": 0.25},
    "oatmeal_stout": {"roasted_coffee_choc": 0.75, "caramel_toffee": 0.45, "malty_bready": 0.5,
                      "body_fullness": 0.8, "nutty": 0.4, "bitterness": 0.45, "sweet": 0.4,
                      "carbonation": 0.35},
    "barleywine": {"caramel_toffee": 0.75, "malty_bready": 0.65, "stone_fruit": 0.5,
                   "alcohol_warmth": 0.8, "sweet": 0.55, "body_fullness": 0.8,
                   "vanilla_oak": 0.35, "bitterness": 0.45, "nutty": 0.35, "carbonation": 0.3},
    "scottish_ale": {"caramel_toffee": 0.75, "malty_bready": 0.65, "sweet": 0.5, "nutty": 0.35,
                     "body_fullness": 0.7, "alcohol_warmth": 0.5, "smoky_peat": 0.15,
                     "bitterness": 0.25, "carbonation": 0.35},
    "winter_warmer": {"spicy_phenolic": 0.6, "caramel_toffee": 0.6, "malty_bready": 0.55,
                      "stone_fruit": 0.35, "sweet": 0.45, "alcohol_warmth": 0.5,
                      "body_fullness": 0.6, "nutty": 0.3, "carbonation": 0.45},
    "esb": {"caramel_toffee": 0.5, "malty_bready": 0.55, "bitterness": 0.5, "herbal": 0.4,
            "nutty": 0.3, "floral": 0.3, "body_fullness": 0.45, "dryness_finish": 0.5,
            "carbonation": 0.4},
    "rye_beer": {"spicy_phenolic": 0.55, "malty_bready": 0.5, "bitterness": 0.5, "citrus": 0.35,
                 "caramel_toffee": 0.3, "dryness_finish": 0.55, "body_fullness": 0.45,
                     "carbonation": 0.5},
    "dunkelweizen": {"banana_ester": 0.7, "spicy_phenolic": 0.5, "caramel_toffee": 0.45,
                     "malty_bready": 0.5, "roasted_coffee_choc": 0.25, "carbonation": 0.6,
                     "body_fullness": 0.5, "sweet": 0.35},
    "schwarzbier": {"roasted_coffee_choc": 0.5, "malty_bready": 0.5, "caramel_toffee": 0.3,
                    "bitterness": 0.35, "carbonation": 0.55, "body_fullness": 0.4,
                    "dryness_finish": 0.5},
    "altbier": {"malty_bready": 0.55, "caramel_toffee": 0.45, "nutty": 0.4, "bitterness": 0.5,
                "body_fullness": 0.45, "carbonation": 0.5, "dryness_finish": 0.5},
    "kolsch": {"malty_bready": 0.4, "floral": 0.3, "grassy": 0.3, "bitterness": 0.3,
               "carbonation": 0.6, "dryness_finish": 0.55, "body_fullness": 0.3,
               "stone_fruit": 0.2},
    "festbier": {"malty_bready": 0.6, "caramel_toffee": 0.35, "honey": 0.3, "bitterness": 0.3,
                 "carbonation": 0.55, "body_fullness": 0.45, "dryness_finish": 0.45},
    "cream_ale": {"malty_bready": 0.45, "sweet": 0.35, "body_fullness": 0.35, "carbonation": 0.55,
                  "bitterness": 0.25, "dryness_finish": 0.4, "grassy": 0.2},
    # --- spirits ---
    "peated_scotch": {"smoky_peat": 0.9, "vanilla_oak": 0.5, "malty_bready": 0.35, "honey": 0.3,
                      "alcohol_warmth": 0.7, "dryness_finish": 0.55, "body_fullness": 0.5},
    "scotch": {"vanilla_oak": 0.6, "malty_bready": 0.4, "honey": 0.45, "nutty": 0.35,
               "caramel_toffee": 0.4, "alcohol_warmth": 0.7, "dryness_finish": 0.5, "sweet": 0.3},
    "irish_whiskey": {"vanilla_oak": 0.5, "honey": 0.45, "malty_bready": 0.35,
                      "caramel_toffee": 0.4, "alcohol_warmth": 0.6, "sweet": 0.35,
                      "dryness_finish": 0.4, "nutty": 0.3},
    "bourbon": {"vanilla_oak": 0.75, "caramel_toffee": 0.6, "honey": 0.4,
                "roasted_coffee_choc": 0.3, "nutty": 0.35, "alcohol_warmth": 0.75,
                "sweet": 0.45, "spicy_phenolic": 0.3},
    "rye": {"spicy_phenolic": 0.6, "vanilla_oak": 0.55, "caramel_toffee": 0.4, "herbal": 0.35,
            "alcohol_warmth": 0.7, "dryness_finish": 0.5, "nutty": 0.3},
    "whiskey": {"vanilla_oak": 0.6, "caramel_toffee": 0.5, "honey": 0.4, "alcohol_warmth": 0.7,
                "nutty": 0.3, "sweet": 0.35, "dryness_finish": 0.45},
    "spiced_rum": {"spicy_phenolic": 0.55, "vanilla_oak": 0.5, "caramel_toffee": 0.5, "honey": 0.35,
                   "sweet": 0.6, "alcohol_warmth": 0.6, "banana_ester": 0.3},
    "aged_rum": {"caramel_toffee": 0.6, "vanilla_oak": 0.5, "honey": 0.4, "banana_ester": 0.35,
                 "sweet": 0.55, "alcohol_warmth": 0.65, "nutty": 0.3, "body_fullness": 0.45},
    "white_rum": {"sweet": 0.45, "alcohol_warmth": 0.6, "dryness_finish": 0.5, "floral": 0.3,
                  "caramel_toffee": 0.2, "banana_ester": 0.3},
    "rum": {"caramel_toffee": 0.45, "vanilla_oak": 0.35, "sweet": 0.55, "honey": 0.35,
            "alcohol_warmth": 0.6, "banana_ester": 0.3},
    "gin": {"herbal": 0.7, "floral": 0.5, "piney_resinous": 0.45, "citrus": 0.45,
            "spicy_phenolic": 0.35, "alcohol_warmth": 0.6, "dryness_finish": 0.6, "grassy": 0.3},
    "vodka": {"alcohol_warmth": 0.7, "dryness_finish": 0.6, "sweet": 0.1},
    "mezcal": {"smoky_peat": 0.65, "herbal": 0.5, "spicy_phenolic": 0.4, "citrus": 0.35,
               "alcohol_warmth": 0.7, "dryness_finish": 0.5, "grassy": 0.35},
    "tequila": {"herbal": 0.55, "spicy_phenolic": 0.4, "citrus": 0.4, "grassy": 0.35,
                "alcohol_warmth": 0.7, "dryness_finish": 0.55, "floral": 0.3},
    # The registry's "Agave Spirits" is a catch-all, not a style: mezcal, raicilla, sotol,
    # bacanora and every wild-agave distillate that may not legally be called tequila file
    # under it, and all 2,381 of them were being handed tequila's centroid (2026-09-22).
    # This sits where the bucket's middle actually is -- nearer mezcal than tequila, because
    # most of what files here is pit-roasted rather than autoclaved: some smoke, more raw
    # vegetal agave, and not the floral roundness a column-distilled tequila gets. A name
    # that says mezcal or tequila outright is refined off this by `_AGAVE_SPECIFIC`; what
    # stays is what the label would not narrow.
    "agave_spirit": {"herbal": 0.6, "grassy": 0.45, "smoky_peat": 0.35, "spicy_phenolic": 0.4,
                     "citrus": 0.35, "alcohol_warmth": 0.72, "dryness_finish": 0.55,
                     "floral": 0.2},
    "brandy": {"vanilla_oak": 0.6, "stone_fruit": 0.55, "caramel_toffee": 0.5, "honey": 0.4,
               "sweet": 0.45, "alcohol_warmth": 0.7, "body_fullness": 0.45},
    "triple_sec": {"citrus": 0.85, "sweet": 0.7, "floral": 0.35, "alcohol_warmth": 0.4},
    "anise": {"herbal": 0.75, "spicy_phenolic": 0.6, "sweet": 0.4, "alcohol_warmth": 0.65,
              "dryness_finish": 0.4},
    "cream_liqueur": {"sweet": 0.8, "roasted_coffee_choc": 0.45, "vanilla_oak": 0.4, "nutty": 0.35,
                      "alcohol_warmth": 0.3, "body_fullness": 0.6},
    "amaro": {"herbal": 0.7, "sour_tart": 0.4, "citrus": 0.4, "sweet": 0.45, "spicy_phenolic": 0.4,
              "alcohol_warmth": 0.5, "dryness_finish": 0.5},
    "liqueur": {"sweet": 0.8, "honey": 0.4, "vanilla_oak": 0.3, "caramel_toffee": 0.35,
                "alcohol_warmth": 0.4},
    "spirit": {"alcohol_warmth": 0.65, "dryness_finish": 0.5, "vanilla_oak": 0.25, "sweet": 0.2},
    # --- what the TTB registry files by the tens of thousands, and the keyword rules had no
    # centroid for: half a million rows arrived with a class code and nothing here to map it
    # to (2026-09-17). ---
    "ale": {"malty_bready": 0.5, "caramel_toffee": 0.3, "bitterness": 0.45, "citrus": 0.3,
            "floral": 0.25, "body_fullness": 0.45, "carbonation": 0.5, "dryness_finish": 0.45},
    "flavored_malt": {"sweet": 0.6, "citrus": 0.45, "berry": 0.4, "tropical": 0.3,
                      "carbonation": 0.65, "bitterness": 0.1, "malty_bready": 0.15,
                      "body_fullness": 0.3, "dryness_finish": 0.3, "sour_tart": 0.25},
    "malt_liquor": {"malty_bready": 0.45, "sweet": 0.5, "alcohol_warmth": 0.5, "carbonation": 0.5,
                    "bitterness": 0.2, "body_fullness": 0.4, "dryness_finish": 0.3},
    "na_beer": {"malty_bready": 0.45, "sweet": 0.35, "grassy": 0.25, "bitterness": 0.25,
                "carbonation": 0.55, "body_fullness": 0.25, "dryness_finish": 0.35},
    "sake": {"malty_bready": 0.4, "sweet": 0.35, "floral": 0.35, "stone_fruit": 0.3,
             "banana_ester": 0.25, "alcohol_warmth": 0.35, "dryness_finish": 0.45,
             "body_fullness": 0.35, "carbonation": 0.1},
    "cider": {"stone_fruit": 0.6, "sour_tart": 0.45, "sweet": 0.4, "floral": 0.3,
              "carbonation": 0.6, "dryness_finish": 0.5, "body_fullness": 0.3,
              "alcohol_warmth": 0.25},
    "flavored_vodka": {"citrus": 0.45, "berry": 0.4, "sweet": 0.5, "floral": 0.25,
                       "alcohol_warmth": 0.55, "dryness_finish": 0.4},
    "flavored_gin": {"herbal": 0.5, "berry": 0.45, "citrus": 0.45, "sweet": 0.5, "floral": 0.4,
                     "alcohol_warmth": 0.5, "dryness_finish": 0.4},
    "flavored_rum": {"sweet": 0.65, "tropical": 0.5, "citrus": 0.35, "vanilla_oak": 0.35,
                     "caramel_toffee": 0.3, "alcohol_warmth": 0.5, "banana_ester": 0.3},
    "flavored_whiskey": {"sweet": 0.6, "vanilla_oak": 0.5, "honey": 0.45, "caramel_toffee": 0.45,
                         "spicy_phenolic": 0.35, "alcohol_warmth": 0.55, "stone_fruit": 0.3},
    "fruit_liqueur": {"sweet": 0.75, "stone_fruit": 0.5, "berry": 0.5, "citrus": 0.4,
                      "floral": 0.3, "alcohol_warmth": 0.35, "body_fullness": 0.45},
    "coffee_liqueur": {"roasted_coffee_choc": 0.85, "sweet": 0.75, "vanilla_oak": 0.4,
                       "caramel_toffee": 0.4, "alcohol_warmth": 0.35, "body_fullness": 0.55},
    "nut_liqueur": {"nutty": 0.8, "sweet": 0.7, "vanilla_oak": 0.35, "caramel_toffee": 0.4,
                    "stone_fruit": 0.3, "alcohol_warmth": 0.35, "body_fullness": 0.5},
    "herbal_liqueur": {"herbal": 0.75, "spicy_phenolic": 0.5, "sweet": 0.55, "honey": 0.35,
                       "citrus": 0.3, "alcohol_warmth": 0.5, "dryness_finish": 0.4},
    "mint_liqueur": {"herbal": 0.8, "sweet": 0.7, "spicy_phenolic": 0.3, "alcohol_warmth": 0.35,
                     "dryness_finish": 0.35},
    "chocolate_liqueur": {"roasted_coffee_choc": 0.8, "sweet": 0.75, "vanilla_oak": 0.35,
                          "nutty": 0.3, "alcohol_warmth": 0.35, "body_fullness": 0.55},
    "bitters": {"herbal": 0.85, "spicy_phenolic": 0.6, "citrus": 0.4, "sour_tart": 0.3,
                "sweet": 0.3, "alcohol_warmth": 0.6, "dryness_finish": 0.6},
    "neutral_spirit": {"alcohol_warmth": 0.8, "dryness_finish": 0.6},
    # --- wine / other ---
    "wine": {"berry": 0.5, "stone_fruit": 0.4, "sour_tart": 0.4, "vanilla_oak": 0.35, "sweet": 0.35,
             "alcohol_warmth": 0.45, "dryness_finish": 0.55, "floral": 0.35},
}

# Typical ABV per style — a weak prior, only ever used to fill a missing value.
_ABV: dict[str, float] = {
    "neipa": 6.5, "dipa": 8.2, "ipa": 6.5, "pale_ale": 5.2, "imperial_stout": 9.0, "stout": 5.5,
    "porter": 5.5, "wheat": 5.0, "tripel": 8.5, "belgian_dark": 7.5, "saison": 6.0, "sour": 4.5,
    "amber": 5.3, "brown": 5.2, "bock": 6.8, "pilsner": 4.8, "helles": 4.9, "radler": 2.5,
    "lager": 4.8, "beer": 5.0, "peated_scotch": 43.0, "scotch": 43.0, "irish_whiskey": 40.0,
    "bourbon": 45.0, "rye": 45.0, "whiskey": 43.0, "spiced_rum": 37.5, "aged_rum": 40.0,
    "white_rum": 40.0, "rum": 40.0, "gin": 42.0, "vodka": 40.0, "mezcal": 45.0, "tequila": 40.0,
    "agave_spirit": 45.0,
    "brandy": 40.0, "triple_sec": 40.0, "anise": 40.0, "cream_liqueur": 17.0, "amaro": 25.0,
    "liqueur": 25.0, "spirit": 40.0, "wine": 12.5,
    "west_coast_ipa": 7.0, "black_ipa": 6.5, "session_ipa": 4.5, "brut_ipa": 6.5,
    "cold_ipa": 6.5, "sweet_stout": 5.5, "oatmeal_stout": 5.4, "barleywine": 10.5,
    "scottish_ale": 7.5,
    "winter_warmer": 7.0, "esb": 5.2, "rye_beer": 6.0, "dunkelweizen": 5.3,
    "schwarzbier": 4.9, "altbier": 4.8, "kolsch": 4.8, "festbier": 5.8,
    "cream_ale": 4.8,
    "ale": 5.5, "flavored_malt": 5.0, "malt_liquor": 7.5, "na_beer": 0.4, "sake": 15.0,
    "cider": 5.5, "flavored_vodka": 35.0, "flavored_gin": 30.0, "flavored_rum": 35.0,
    "flavored_whiskey": 35.0,
    "fruit_liqueur": 20.0, "coffee_liqueur": 20.0, "nut_liqueur": 24.0, "herbal_liqueur": 30.0,
    "mint_liqueur": 25.0, "chocolate_liqueur": 20.0, "bitters": 40.0, "neutral_spirit": 40.0,
}

# (style, keyword-substrings). Order = priority; most specific first, within each category.
# Styles the name states outright, read as whole words and before the substring rules below.
# The old rules match on bare substrings, which is right for a loose word like "stout" and
# wrong for a short one -- "esb" inside "Desby", "alt" inside "Walter". These are anchored, and
# each may name what disqualifies it: an imperial milk stout is an imperial stout first.
_BEER_SPECIFIC: list[tuple[str, str, str | None]] = [
    ("barleywine", r"\bbarley ?wines?\b", None),
    ("black_ipa", r"\bblack (ipa|i\.p\.a)\b|\bcascadian dark\b", None),
    ("brut_ipa", r"\bbrut (ipa|i\.p\.a)\b", None),
    ("cold_ipa", r"\bcold (ipa|i\.p\.a)\b", None),
    ("session_ipa", r"\bsession (ipa|i\.p\.a|india pale)\b", None),
    ("west_coast_ipa",
     r"\bwest coast\b(?=.*\b(ipa|i\.p\.a|india pale)\b)"
     r"|\b(ipa|i\.p\.a|india pale)\b(?=.*\bwest coast\b)", None),
    # Lactose, oats or plain sweetness -- but "nitro" is how a beer is poured, not what is in
    # it, and Guinness is a dry stout on nitrogen.
    ("oatmeal_stout", r"\boat(meal)? stouts?\b", r"\b(imperial|russian)\b"),
    ("sweet_stout", r"\b(milk|sweet|cream|lactose) stouts?\b", r"\b(imperial|russian)\b"),
    ("scottish_ale", r"\bwee heavy\b|\bscotch ale\b|\bscottish (ale|export)\b", None),
    ("winter_warmer", r"\bwinter warmer\b|\b(christmas|holiday|yule) ale\b", None),
    ("esb", r"\besb\b|\bextra special bitter\b|\bbest bitter\b|\bspecial bitter\b", None),
    ("dunkelweizen", r"\bdunkel ?wei(s|z|ss)en\b|\bdunkel ?weisse\b", None),
    ("schwarzbier", r"\bschwarz ?bier\b|\bblack lager\b", None),
    ("altbier", r"\balt ?bier\b|\bsticke\b", None),
    ("kolsch", r"\bk(o|\u00f6|oe)lsch\b", None),
    ("festbier", r"\bfest ?bier\b", None),
    ("cream_ale", r"\bcream ale\b", None),
    ("rye_beer", r"\brye (ale|beer|lager|pale ale)\b|\broggen\w*\b", None),
]
_BEER_SPECIFIC_RE = [(s, re.compile(p), re.compile(u) if u else None)
                     for s, p, u in _BEER_SPECIFIC]

# What each anchored style is a kind of. A specific TTB class is a real filing and outranks the
# name, except where the name says a KIND of what was filed -- a "Milk Stout" filed as "Stout"
# is still a stout, and reading the lactose off the name only sharpens it.
_BEER_PARENT: dict[str, str] = {
    "west_coast_ipa": "ipa", "black_ipa": "ipa", "session_ipa": "ipa", "brut_ipa": "ipa",
    "cold_ipa": "ipa", "sweet_stout": "stout", "oatmeal_stout": "stout",
    "schwarzbier": "lager", "festbier": "lager",
    "kolsch": "ale", "altbier": "ale", "cream_ale": "ale", "esb": "ale", "scottish_ale": "ale",
    "winter_warmer": "ale", "barleywine": "ale", "rye_beer": "ale", "dunkelweizen": "wheat",
}


def _specific_beer_style(n: str, cls: str | None) -> str | None:
    """The anchored style the name states, or None. A specific class filed with the label wins
    unless the name names a kind of it."""
    for style, pat, unless in _BEER_SPECIFIC_RE:
        if not pat.search(n) or (unless is not None and unless.search(n)):
            continue
        if cls is None or cls in _GENERIC_CLASS_STYLES or _BEER_PARENT.get(style) == cls:
            return style
        return None
    return None


_BEER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("neipa", ("hazy", "juicy", "neipa", "new england", "haze")),
    ("dipa", ("double ipa", "dipa", "imperial ipa", "triple ipa", "double i.p.a",
              "double india pale", "imperial india pale", "triple india pale")),
    ("ipa", ("ipa", "india pale", "i.p.a", " apa")),
    ("pale_ale", ("pale ale", "american pale", "apa ")),
    ("imperial_stout", ("imperial stout", "russian imperial", "impy")),
    ("stout", ("stout",)),
    ("porter", ("porter",)),
    ("wheat", ("hefe", "weiss", "weizen", "witbier", "wit ", "white ale", "belgian white",
               "blanche", "wheat", "blanc", "weisse", "hoegaarden")),
    ("tripel", ("tripel", "triple")),
    ("belgian_dark", ("dubbel", "quadrupel", "quad", "abbey", "abbaye", "trappist", "grimbergen",
                      "leffe", "belgian strong")),
    ("saison", ("saison", "farmhouse")),
    ("sour", ("sour", "gose", "lambic", "berliner", "kriek", "gueuze", "wild ale")),
    ("amber", ("amber", "red ale", "irish red", " rouge")),
    ("brown", ("brown ale", "nut brown", " brown")),
    ("bock", ("doppelbock", "bock", "dunkel", "schwarz", "dark lager")),
    ("pilsner", ("pilsner", "pilsener", "pils", "urquell")),
    ("helles", ("helles", "kellerbier", "keller", "märzen", "marzen", "oktoberfest")),
    ("radler", ("radler", "shandy")),
    ("lager", ("lager", "light", "lite", "premium", "especial", "cerveza", "pale lager",
               "blonde", "blond", "pils", "birra", "bier", "biere")),
]
# The two agave spirits the catch-all class does NOT get to swallow, read as whole words.
# Anchoring is the whole point here: the bare substring "tequila" is inside `tequilana`, the
# agave SPECIES, and all 15 rows in the catch-all whose name contains "tequila" are that
# species, not the category -- a mezcal from agave tequilana is still a mezcal. `\btequilas?\b`
# matches none of them, which is the only honest way to ask the question.
#
# Mezcal is claimed by the word itself or by an agave varietal only mezcal is made from.
# Raicilla, sotol and bacanora are NOT mezcal -- different plants, different places, their own
# denominations -- so they are named as what disqualifies a varietal reading and keep the
# catch-all centroid. That guard is earned: "Oroza Raicilla Ensamble" is a raicilla.
_AGAVE_SPECIFIC: list[tuple[str, str, str | None]] = [
    ("mezcal", r"\b(mez|mes)cal\w*\b", None),
    ("mezcal",
     r"\b(espadin\w*|espadilla|tobala|toba(x|s|z)iche|cui(sh|x)e|(madre|bi)cui(sh|x)e"
     r"|tepe(z|x)tate|arroqueno|papalome\w*|papalote|maguey\w*|cupreata|salmiana|karwinskii"
     r"|potatorum|jabali|sierra negra|coyota|mexicano|ensamble|barril)\b",
     r"\b(raicilla|sotol\w*|bacanora)\b"),
    ("tequila", r"\btequilas?\b", None),
]
_AGAVE_SPECIFIC_RE = [(s, re.compile(p), re.compile(u) if u else None)
                      for s, p, u in _AGAVE_SPECIFIC]

#: What each agave style is a kind of -- the same parent rule `_BEER_PARENT` states for beer.
_AGAVE_PARENT: dict[str, str] = {"mezcal": "agave_spirit", "tequila": "agave_spirit"}

#: Beer and agave together: the one map `detect_style` and `readable_style` both read, so a
#: sub-style the name states can never print one thing and vector another.
_PARENT_STYLE: dict[str, str] = {**_BEER_PARENT, **_AGAVE_PARENT}


def _specific_agave_style(n: str, cls: str | None) -> str | None:
    """The agave spirit the name states, or None. A real Tequila or Mezcal filing outranks the
    name; the catch-all class does not, which is the whole point. The parent check is what says
    so, and is kept even though `detect_style` only calls this for the catch-all today."""
    for style, pat, unless in _AGAVE_SPECIFIC_RE:
        if not pat.search(n) or (unless is not None and unless.search(n)):
            continue
        if cls is None or cls in _GENERIC_CLASS_STYLES or _AGAVE_PARENT.get(style) == cls:
            return style
        return None
    return None


_SPIRIT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("peated_scotch", ("islay", "peated", "peat", "laphroaig", "lagavulin", "ardbeg", "smoky",
                       "smoke")),
    ("scotch", ("single malt", "scotch", "speyside", "highland", "glen", "macallan", "ecosse",
                "chivas", "ballantine", "grant", "famous grouse", "johnnie walker")),
    ("irish_whiskey", ("irish", "jameson", "tullamore", "bushmills", "irlandais")),
    ("bourbon", ("bourbon", "tennessee", "kentucky", "jack daniel", "buffalo trace", "four roses",
                 "maker", "wild turkey", "jim beam")),
    ("rye", ("rye whiskey", "rye whisky", " rye")),
    ("whiskey", ("whiskey", "whisky", "whisk")),
    ("spiced_rum", ("spiced", "kraken", "captain morgan")),
    ("aged_rum", ("añejo", "anejo", "dark rum", "gold rum", "aged rum", "negrita", "reserva",
                  "carta oro", "oro")),
    ("white_rum", ("white rum", "silver rum", "carta blanca", "superior rum", "light rum",
                   "platinum", "blanco rum", "rhum blanc", "rhum agricole blanc")),
    ("rum", ("rum", "rhum", " ron ", "ron ", "bacardi")),
    ("gin", ("gin", "london dry", "hendrick", "bombay", "tanqueray", "beefeater")),
    ("vodka", ("vodka", "smirnoff", "absolut", "svedka", "ketel", "grey goose", "stoli", "vodca")),
    ("mezcal", ("mezcal", "mescal")),
    ("tequila", ("tequila", "reposado", "patron", "jose cuervo", "don julio", "espolon", "1800",
                 "blanco tequila", "silver tequila")),
    ("brandy", ("brandy", "cognac", "armagnac", "calvados", "hennessy", "remy martin")),
    ("triple_sec", ("cointreau", "triple sec", "grand marnier", "curacao", "curaçao")),
    ("anise", ("absinthe", "ouzo", "pastis", "sambuca", "anis", "raki", "arak")),
    ("cream_liqueur", ("baileys", "irish cream", "cream liqueur")),
    ("amaro", ("amaro", "aperol", "campari", "vermouth", "fernet", "cynar", "martini",
               "aperitivo", "aperitif", "bitter")),
    ("liqueur", ("liqueur", "likör", "likor", "licor", "schnapps", "kahlua", "coffee liqueur")),
]

# TTB class/type text -> style key. The registry's own vocabulary, matched as substrings of
# the description with its bottling suffix removed ("Scotch Whisky Fb" is foreign-bottled
# Scotch, "Tequila Usb" is US-bottled tequila). Order is load-bearing: specific before the
# general word it contains. Two kinds of class: a SPECIFIC one names the style outright and
# outranks anything the product's name says ("Absolut Citron" under `Vodka - Other Flavored`
# is a flavored vodka, whatever the brand keyword makes of it); a GENERIC one ("Ale", "Beer",
# "Whisky", "Malt Beverages Specialities - Flavored") is a bucket, and the name decides
# first -- a fruit IPA filed as a flavored malt beverage is still an IPA.
_CLASS_RULES: list[tuple[str, tuple[str, ...]]] = [
    # beer
    ("na_beer", ("near beer", "non alcoholic", "non-alcoholic", "cereal beverage")),
    ("malt_liquor", ("malt liquor",)),
    ("stout", ("stout",)),
    ("porter", ("porter",)),
    ("flavored_malt", ("malt beverages specialities - flavored",
                       "malt beverage specialties - flavored",
                       "malt beverages specialties - flavored", "flavored malt")),
    ("ale", ("ale",)),
    ("lager", ("lager",)),
    ("beer", ("beer", "malt beverage")),
    # whisky
    ("peated_scotch", ("islay",)),
    ("scotch", ("scotch", "single malt scotch")),
    ("irish_whiskey", ("irish whisk",)),
    ("bourbon", ("bourbon", "corn whisk")),
    ("rye", ("rye whisk",)),
    ("flavored_whiskey", ("whisky (flavored)", "whiskey (flavored)", "flavored whisk",
                          "liqueurs (whisky)", "liqueurs (whiskey)")),
    ("whiskey", ("whisk", "canadian", "single malt")),
    # agave. "Agave Spirits" covered every agave distillate that is NOT tequila and was
    # mapped to `tequila` by the bare "agave" keyword. It is a bucket -- but a bucket of one
    # family, so it keeps a centroid of its own rather than joining `_GENERIC_CLASS_STYLES`:
    # turning the loose spirit keywords loose on these names read "Teodoro" as gold rum,
    # "Derrumbes" as rum, "Glenns Creek" as Scotch and "The Original" as gin. The name is
    # read instead by the anchored `_AGAVE_SPECIFIC` rules. Tequila and mezcal are real
    # filings and stay specific; "agave" is matched last so they win the ones they name.
    ("mezcal", ("mezcal",)),
    ("tequila", ("tequila",)),
    ("agave_spirit", ("agave",)),
    # rum
    ("flavored_rum", ("rum other flavored", "flavored rum", "liqueurs (rum)", "spiced rum")),
    ("white_rum", ("rum (white)", "rum white", "white rum")),
    ("aged_rum", ("rum (gold)", "rum gold", "gold rum", "rum (dark)", "aged rum", "rum dark")),
    ("rum", ("rum",)),
    # gin, vodka, neutral
    ("flavored_gin", ("gin - flavored", "flavored gin", "gin flavored", "sloe gin")),
    ("gin", ("gin",)),
    ("flavored_vodka", ("vodka - ", "vodka flavored", "flavored vodka", "liqueurs (vodka)")),
    ("vodka", ("vodka",)),
    ("neutral_spirit", ("neutral spirit", "grain spirit")),
    # brandy
    ("brandy", ("brandy", "cognac", "armagnac", "calvados", "pisco", "grappa", "slivovitz",
                "eau de vie", "eau-de-vie")),
    # liqueurs and the rest
    ("cream_liqueur", ("cream liqueur", "creme or creams", "cremes or creams", "dairy cream")),
    ("coffee_liqueur", ("coffee", "cafe")),
    ("chocolate_liqueur", ("creme de cacao", "chocolate")),
    ("mint_liqueur", ("creme de menthe", "peppermint", "mint")),
    ("nut_liqueur", ("amaretto", "nut liqueur", "noyaux", "almond", "hazelnut")),
    ("anise", ("anisette", "ouzo", "ojen", "absinthe", "sambuca", "anis", "arack", "arak", "raki")),
    ("triple_sec", ("triple sec", "curacao", "orange liqueur")),
    ("bitters", ("bitters",)),
    ("herbal_liqueur", ("herb", "seeds", "specialties & proprietaries",
                        "specialities & proprietaries", "cordial", "kummel")),
    ("fruit_liqueur", ("fruit", "peels", "schnapps")),
    ("amaro", ("amaro", "vermouth", "aperitif", "aperitivo")),
    ("liqueur", ("liqueur",)),
    ("sake", ("sake",)),
    ("cider", ("cider", "perry")),
    ("wine", ("wine", "champagne", "sparkling", "mead")),
    ("spirit", ("other spirits", "spirits")),
]

#: Classes that are buckets rather than styles: the product's name decides first.
_GENERIC_CLASS_STYLES = frozenset({"ale", "beer", "flavored_malt", "whiskey", "rum",
                                   "gin", "vodka", "brandy", "liqueur", "herbal_liqueur",
                                   "fruit_liqueur", "spirit"})

_BOTTLING_SUFFIX = re.compile(r"\s+(fb|usb|bib)\b\*?$")


def normalize_class(class_type: str | None) -> str:
    """A TTB class/type description as the rules read it: casefolded, the bottling suffix
    (FB foreign-bottled, USB US-bottled, BIB bottled-in-bond) and stray asterisks gone."""
    c = _norm(class_type or "").strip().rstrip("*").strip()
    return _BOTTLING_SUFFIX.sub("", c).strip()


def class_style(class_type: str | None) -> str | None:
    """The style key a TTB class/type description names, or None for no description."""
    c = normalize_class(class_type)
    if not c:
        return None
    for style, kws in _CLASS_RULES:
        if any(k in c for k in kws):
            return style
    return None


# What a person would call the style, from what the registry calls it. The detail screen
# printed the class code as filed -- "Other Rum Gold Usb" on a bottle of Gosling's, reported
# from the camera as "should be ID'd as rum" (2026-09-16). The bottling suffix is dropped and
# the registry's bucketing words ("Other", "Specialties") go; a style the NAME detects more
# precisely than the class (an IPA filed as "Ale") is printed as that style.
_STYLE_NAMES: dict[str, str] = {
    "neipa": "New England IPA", "dipa": "Double IPA", "ipa": "IPA", "pale_ale": "Pale Ale",
    "imperial_stout": "Imperial Stout", "stout": "Stout", "porter": "Porter", "wheat": "Wheat Beer",
    "tripel": "Tripel", "belgian_dark": "Belgian Dark Ale", "saison": "Saison", "sour": "Sour Ale",
    "amber": "Amber Ale", "brown": "Brown Ale", "bock": "Bock", "pilsner": "Pilsner",
    "helles": "Helles", "radler": "Radler", "lager": "Lager", "beer": "Beer", "ale": "Ale",
    "west_coast_ipa": "West Coast IPA", "black_ipa": "Black IPA",
    "session_ipa": "Session IPA", "brut_ipa": "Brut IPA", "cold_ipa": "Cold IPA",
    "sweet_stout": "Sweet Stout", "oatmeal_stout": "Oatmeal Stout",
    "barleywine": "Barleywine",
    "scottish_ale": "Scottish Ale", "winter_warmer": "Winter Warmer", "esb": "ESB",
    "rye_beer": "Rye Beer", "dunkelweizen": "Dunkelweizen",
    "schwarzbier": "Schwarzbier", "altbier": "Altbier", "kolsch": "Kölsch",
    "festbier": "Festbier", "cream_ale": "Cream Ale",
    "flavored_malt": "Flavored Malt Beverage", "malt_liquor": "Malt Liquor",
    "na_beer": "Non-Alcoholic Beer", "peated_scotch": "Peated Scotch", "scotch": "Scotch Whisky",
    "irish_whiskey": "Irish Whiskey", "bourbon": "Bourbon", "rye": "Rye Whiskey",
    "whiskey": "Whiskey", "flavored_whiskey": "Flavored Whiskey", "spiced_rum": "Spiced Rum",
    "aged_rum": "Gold Rum", "white_rum": "White Rum", "flavored_rum": "Flavored Rum", "rum": "Rum",
    "gin": "Gin", "flavored_gin": "Flavored Gin", "vodka": "Vodka",
    "flavored_vodka": "Flavored Vodka",
    "neutral_spirit": "Neutral Spirit", "mezcal": "Mezcal", "tequila": "Tequila",
    "agave_spirit": "Agave Spirit",
    "brandy": "Brandy", "triple_sec": "Orange Liqueur", "anise": "Anise Spirit",
    "cream_liqueur": "Cream Liqueur", "coffee_liqueur": "Coffee Liqueur",
    "chocolate_liqueur": "Chocolate Liqueur", "mint_liqueur": "Mint Liqueur",
    "nut_liqueur": "Nut Liqueur", "herbal_liqueur": "Herbal Liqueur",
    "fruit_liqueur": "Fruit Liqueur", "bitters": "Bitters", "amaro": "Amaro", "liqueur": "Liqueur",
    "spirit": "Spirit", "sake": "Sake", "cider": "Cider", "wine": "Wine",
}

# Class descriptions whose readable name is better said outright than derived.
_CLASS_NAMES: dict[str, str] = {
    "single malt scotch whisky": "Single Malt Scotch",
    "straight bourbon whisky": "Straight Bourbon",
    "straight bourbon whisky blends": "Straight Bourbon",
    "bourbon whisky": "Bourbon",
    "blended bourbon whisky": "Blended Bourbon",
    "straight rye whisky": "Straight Rye",
    "straight rye whisky blends": "Straight Rye",
    "rye whisky": "Rye Whiskey",
    "blended whisky": "Blended Whiskey",
    "canadian whisky": "Canadian Whisky",
    "irish whisky": "Irish Whiskey",
    "corn whisky": "Corn Whiskey",
    "american single malt whiskey": "American Single Malt",
    "london dry distilled gin": "London Dry Gin",
    "london dry gin": "London Dry Gin",
    "cognac (brandy)": "Cognac",
    "armagnac (brandy)": "Armagnac",
    "apple brandy (calvados)": "Calvados",
    "apple brandy": "Apple Brandy",
    "plum brandy (slivovitz)": "Slivovitz",
    "other grape brandy (pisco, grappa)": "Grape Brandy",
    "agave spirits": "Agave Spirit",
    "tequila": "Tequila",
    "mezcal": "Mezcal",
    "specialties & proprietaries": "Specialty",
    "other specialties & proprietaries": "Specialty",
    "specialities & proprietaries": "Specialty",
    "sake - imported": "Sake",
    "sake - imported flavored": "Flavored Sake",
    "sake - domestic flavored": "Flavored Sake",
    "other (herbs & seeds)": "Herbal Liqueur",
    "other herb & seed cordials/liqueurs": "Herbal Liqueur",
    "herbs and seeds schnapps liqueur": "Herbal Schnapps",
    "herbs & seeds schnapps liqueur": "Herbal Schnapps",
    "fruits & peels schnapps liqueur": "Fruit Schnapps",
    "peppermint schnapps": "Peppermint Schnapps",
    "coffee (cafe) liqueur": "Coffee Liqueur",
    "other liqueur (creme or creams)": "Cream Liqueur",
    "other liqueur (cremes or creams)": "Cream Liqueur",
    "dairy cream liqueur/cordial": "Cream Liqueur",
    "creme de cacao brown": "Crème de Cacao",
    "creme de cacao white": "Crème de Cacao",
    "creme de menthe green": "Crème de Menthe",
    "creme de menthe white": "Crème de Menthe",
    "anisette, ouzo, ojen": "Anisette",
    "bitters - beverage": "Bitters",
    "triple sec": "Triple Sec",
    "curacao": "Curaçao",
    "amaretto": "Amaretto",
    "malt beverages specialities - flavored": "Flavored Malt Beverage",
    "malt beverages specialities": "Malt Beverage Specialty",
    "malt beverages": "Malt Beverage",
    "cereal beverages - near beer (non alcoholic)": "Non-Alcoholic Beer",
    "malt liquor": "Malt Liquor",
    "neutral spirits - grain": "Grain Neutral Spirit",
    "vodka 80-89 proof": "Vodka",
    "vodka 100 proof up": "Vodka",
    "diluted vodka": "Vodka",
    "fruit flavored liqueurs": "Fruit Liqueur",
    "other fruits & peels liqueurs": "Fruit Liqueur",
    "other fruit & peels liqueurs": "Fruit Liqueur",
    "whisky specialties": "Whisky",
    "rum specialties": "Rum",
    "vodka specialties": "Vodka",
    "gin specialties": "Gin",
    "other spirits": "Spirit",
}


#: The registry's bucketing words, dropped from a generic class before it is printed.
_CLASS_FILLER = frozenset({"other", "specialties", "specialities", "specialty", "domestic",
                           "foreign", "foriegn", "imported", "u.s.", "ur.s.", "us", "-"})
_CLASS_PAREN = re.compile(r"\((\w+)\)")


def readable_style(class_type: str | None, detected: str | None = None) -> str | None:
    """What to print for a product's style: a named style the product's name detects when the
    class is only a bucket, else the class in plain words, else the registry's text with
    its suffix dropped."""
    c = normalize_class(class_type)
    cls = class_style(class_type)
    # The same rule `detect_style` uses, so the words on the screen and the vector behind them
    # cannot disagree: the name's style prints when the class is only a bucket, or when the name
    # says a KIND of what was filed -- a "Milk Stout" filed as "Stout" prints as a milk stout.
    if detected and detected in _STYLE_NAMES and detected not in _GENERIC_CLASS_STYLES and (
            cls is None or cls in _GENERIC_CLASS_STYLES or _PARENT_STYLE.get(detected) == cls):
        return _STYLE_NAMES[detected]
    if c in _CLASS_NAMES:
        return _CLASS_NAMES[c]
    if cls is not None and cls not in _GENERIC_CLASS_STYLES:
        return _STYLE_NAMES.get(cls)
    if cls is not None:
        # A generic class: the registry's words, tidied. "Other Rum (white)" is a white rum,
        # "Whisky Specialties" is whisky.
        paren = _CLASS_PAREN.search(c)
        base = _CLASS_PAREN.sub("", c)
        words = ([paren.group(1)] if paren else []) + [w for w in base.split()
                                                       if w not in _CLASS_FILLER]
        tidy = " ".join(w.title() for w in words)
        return tidy or _STYLE_NAMES.get(cls)
    return (class_type or "").strip() or None


_NA_MARKERS = ("0,0", "0.0", "alcohol free", "alcohol-free", "alkoholfrei", "sans alcool",
               "non alcoholic", "non-alcoholic", "sin alcohol", "analcolico", "0 %", "0%",
               "alkoholfri", "cero", "tostada 0")


def _norm(s: str) -> str:
    d = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in d if not unicodedata.combining(c)).casefold()


def is_non_alcoholic(name: str) -> bool:
    n = (name or "").casefold()
    if any(m in n for m in _NA_MARKERS):
        return True
    return bool(re.search(r"\b0[.,]0\b", n))


def detect_style(name: str, category: Category | str | None,
                 class_type: str | None = None) -> str | None:
    """Best style key for a product, or None if even the category is unknown.

    A specific TTB class decides outright (see `_CLASS_RULES`); a generic one leaves it to
    the name, and answers only when the name says nothing. Category gates the name keywords
    so a beer's "blanc" reads as a witbier and a spirit's as a white rum; the registry's
    "other" (aperitivos, amari, the specialties) reads the spirit rules."""
    n = _norm(name)
    cat = category.value if isinstance(category, Category) else (category or "")
    cat = str(cat).lower()
    cls = class_style(class_type)
    if cat == "beer":
        specific = _specific_beer_style(n, cls)
        if specific is not None:
            return specific
    # Only the agave catch-all asks the name. Replaying the whole catalog showed why the
    # wider gate is wrong: a name may MENTION an agave spirit without being one, and the
    # spirit rules already read those right -- "Chattanooga Whiskey Tequila Barrel Finished"
    # is a whiskey, "Voyager Series Mezcal Barrel Rested" a gin, "Luma Vodka Mexicano" a
    # vodka. Every one of them kept its class and lost it to the anchored rules (2026-09-22).
    if cls == "agave_spirit":
        specific = _specific_agave_style(n, cls)
        if specific is not None:
            return specific
    if cls is not None and cls not in _GENERIC_CLASS_STYLES:
        return cls
    rules = (_BEER_RULES if cat == "beer" else
             _SPIRIT_RULES if cat in ("spirit", "other") else [])
    for style, kws in rules:
        if any(k in n for k in kws):
            return style
    if cls is not None:
        return cls
    # Nothing matched: fall back to the broad category centroid so the row is still scoreable.
    return {"beer": "beer", "spirit": "spirit", "other": "spirit", "wine": "wine",
            "sake": "sake", "cider": "cider", "mead": "wine", "rtd": "flavored_malt"}.get(cat)


# What a beer's name says was PUT IN IT, over and above its style. A coconut coffee stout and a
# plain stout are not the same drink, and until now they shared a centroid -- one vector stood
# for 100,953 beer rows. Each adjunct nudges the style's own axes, so the answer is a function
# of (style, name) alone: re-running the prior recomputes it from the centroid and lands in the
# same place, rather than adding a second helping to whatever the row already carried.
#
# An entry may name what disqualifies it, because a fruit word is usually a place: Orange
# County brews no orange beer.
_ADJUNCTS: list[tuple[str, str, str | None]] = [
    ("coffee", r"\b(coffee|espresso|cold brew|mocha|latte|cappuccino|affogato)\b", None),
    ("chocolate", r"\b(chocolate|cocoa|cacao|fudge|brownie)\b", None),
    ("vanilla", r"\bvanilla\b", None),
    ("coconut", r"\bcoconut\b", None),
    ("nut", r"\b(peanut|pecan|hazelnut|almond|walnut|pistachio|nut brown|hazel ?nut)\b", None),
    ("maple", r"\bmaple\b", r"\bmaple (street|ave|avenue|road|lane|leaf|city|grove|valley)\b"),
    ("honey", r"\bhoney\b", r"\bhoney (badger|bee|moon|hole|do)\b"),
    ("pumpkin", r"\bpumpkins?\b", None),
    ("spice", r"\b(cinnamon|nutmeg|clove|chai|gingerbread|allspice|cardamom|jalapeno|habanero"
              r"|chipotle|chili|chile|ancho|ginger)\b", r"\bginger ?(bread )?man\b"),
    ("citrus_fruit", r"\b(lemon|lime|orange|grapefruit|tangerine|yuzu|citrus|clementine"
                     r"|mandarin|blood orange)\b",
     r"\b(orange|lemon|lime) (county|street|st|ave|avenue|road|blossom|glen|hill|grove|crush)\b"),
    ("tropical_fruit", r"\b(mango|guava|passion ?fruit|pineapple|papaya|lychee|dragon ?fruit"
                       r"|tropical)\b", None),
    ("berry_fruit", r"\b(raspberry|strawberry|blueberry|blackberry|cranberry|boysenberry"
                    r"|currant|berries|berry|acai|elderberry)\b",
     r"\b(berry|blueberry|strawberry) (hill|farm|street|lane|road)\b"),
    ("stone_fruit", r"\b(peach|apricot|cherry|plum|nectarine|cherries)\b",
     r"\b(cherry|peach) (hill|street|st|creek|tree|blossom|grove|lane|point|wood)\b"),
    ("banana", r"\bbananas?\b", None),
    ("barrel", r"\bbarrel[- ]?aged\b|\b(bourbon|whiskey|whisky|rum|wine|tequila|brandy) barrel"
               r"\b|\bbarrel[- ]?ag(e|ing)\b", None),
    ("smoked", r"\bsmoked\b|\brauch\w*\b", None),
    ("pastry", r"\b(pastry|smoothie|slushy|slushie|milkshake|dessert|cheesecake|cobbler|sundae"
               r"|donut|doughnut|pancake|tiramisu|s'?mores)\b", None),
    ("nitro", r"\bnitro\b", None),
    ("dry_hopped", r"\bddh\b|\bdouble dry[- ]?hopped\b|\bdry[- ]?hopped\b", None),
]
_ADJUNCTS_RE = [(k, re.compile(p), re.compile(u) if u else None) for k, p, u in _ADJUNCTS]

# axis -> how far the adjunct moves it, from the style's centroid. Clamped into 0..1.
_ADJUNCT_DELTAS: dict[str, dict[str, float]] = {
    "coffee": {"roasted_coffee_choc": +0.25, "bitterness": +0.10, "dryness_finish": +0.05},
    "chocolate": {"roasted_coffee_choc": +0.22, "sweet": +0.12, "body_fullness": +0.08},
    "vanilla": {"vanilla_oak": +0.30, "sweet": +0.12, "body_fullness": +0.05},
    "coconut": {"nutty": +0.25, "sweet": +0.12, "body_fullness": +0.08, "vanilla_oak": +0.08},
    "nut": {"nutty": +0.30, "sweet": +0.08},
    "maple": {"caramel_toffee": +0.20, "sweet": +0.18},
    "honey": {"honey": +0.30, "sweet": +0.12},
    "pumpkin": {"spicy_phenolic": +0.25, "malty_bready": +0.10, "sweet": +0.10},
    "spice": {"spicy_phenolic": +0.25},
    "citrus_fruit": {"citrus": +0.30, "sour_tart": +0.10, "dryness_finish": +0.05},
    "tropical_fruit": {"tropical": +0.30, "sweet": +0.10},
    "berry_fruit": {"berry": +0.32, "sour_tart": +0.10},
    "stone_fruit": {"stone_fruit": +0.30, "sweet": +0.08},
    "banana": {"banana_ester": +0.25, "sweet": +0.08},
    "barrel": {"vanilla_oak": +0.30, "caramel_toffee": +0.15, "alcohol_warmth": +0.15,
               "body_fullness": +0.10},
    "smoked": {"smoky_peat": +0.40},
    "pastry": {"sweet": +0.25, "body_fullness": +0.15, "vanilla_oak": +0.10,
               "bitterness": -0.10, "dryness_finish": -0.15},
    "nitro": {"body_fullness": +0.15, "carbonation": -0.30},
    "dry_hopped": {"citrus": +0.12, "tropical": +0.12, "grassy": +0.08, "piney_resinous": +0.08},
}


# A fruit word followed by one of these is a place, and the place is somebody's name: Blackberry
# Farm brews no blackberry beer, Pecan Street Brewing no pecan one, and "Nutmeg State" is
# Connecticut. Checked on EVERY occurrence, not the first -- Highland Park's "Coconut Tree Barrel
# Aged Imperial Coconut Stout" says coconut twice and means it the second time.
_PLACE_AFTER = re.compile(
    r"^[ \-',]*(farms?|hills?|street|st|ave|avenue|roads?|rd|lane|county|valley|creeks?|groves?"
    r"|ridge|points?|springs?|city|mountains?|islands?|bays?|parks?|works|trails?|cellars?"
    r"|tavern|pub|district|station|junction|landing|crossing|corners?|square|trees?|woods?"
    r"|blossom|gardens?|alley|rivers?|lakes?|beach|state|cove|glen|heights|manor|meadows?"
    r"|orchard|brewing|brewery|brewhouse|brewers)\b")


def adjuncts(name: str) -> list[str]:
    """What the name says went into the beer, in table order -- `[]` when it says nothing."""
    n = _norm(name)
    out = []
    for k, pat, unless in _ADJUNCTS_RE:
        spans = [m for m in pat.finditer(n)]
        if not spans or (unless is not None and unless.search(n)):
            continue
        # every mention is somebody's address, so nobody put it in the beer
        if all(_PLACE_AFTER.match(n[m.end():]) for m in spans):
            continue
        out.append(k)
    return out


def sensory_from_style(name: str, category: Category | str | None,
                       style_hint: str | None = None) -> SensoryVector | None:
    """A STYLE_PRIOR SensoryVector for the product, or None when no style can be inferred. A
    non-alcoholic marker zeroes alcohol_warmth so a 0.0% reads distinct from its full sibling.
    `style_hint` is what the catalog says the style is -- a TTB class/type description, or a
    style word off an Open Food Facts row -- and is read as a class first, then as words."""
    style = detect_style(f"{style_hint or ''} {name}", category, class_type=style_hint)
    if style is None:
        return None
    axes = dict(_CENTROIDS[style])
    # What the name says is in it, on top of the style. Read off the name alone -- the TTB
    # class says "Flavored Malt Beverage", never which fruit -- and only for beer, where the
    # adjunct is the difference between two rows that would otherwise share a vector.
    cat = category.value if isinstance(category, Category) else (category or "")
    added = adjuncts(name) if str(cat).lower() == "beer" else []
    for key in added:
        for axis, delta in _ADJUNCT_DELTAS[key].items():
            axes[axis] = round(min(1.0, max(0.0, axes.get(axis, 0.0) + delta)), 3)
    if is_non_alcoholic(name) or style == "na_beer":
        axes["alcohol_warmth"] = 0.0
        axes["body_fullness"] = round(axes.get("body_fullness", 0.3) * 0.7, 3)
    # Broad category fallbacks are even weaker than a named style; a name that states what is
    # in the beer is a better-determined guess than one that states only a style, and ranks
    # ahead of it -- but it is still a guess, and stays under the 0.45 the recommender reads
    # as knowing the product.
    conf = 0.25 if style in ("beer", "spirit", "wine") else 0.35
    if added:
        conf = min(0.44, conf + 0.05)
    return SensoryVector(source=SensorySource.STYLE_PRIOR, confidence=conf, axes=axes)


def abv_from_style(name: str, category: Category | str | None,
                   style_hint: str | None = None) -> float | None:
    """Typical ABV for the inferred style, or None. 0.4 for a detected non-alcoholic product."""
    if is_non_alcoholic(name):
        return 0.4
    style = detect_style(f"{style_hint or ''} {name}", category, class_type=style_hint)
    return _ABV.get(style) if style else None
