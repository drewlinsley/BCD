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
              "body_fullness": 0.65, "malty_bready": 0.35, "dryness_finish": 0.2, "sweet": 0.3},
    "dipa": {"citrus": 0.7, "tropical": 0.65, "piney_resinous": 0.6, "bitterness": 0.85,
             "malty_bready": 0.4, "caramel_toffee": 0.3, "alcohol_warmth": 0.5,
             "dryness_finish": 0.6, "body_fullness": 0.55},
    "ipa": {"citrus": 0.65, "tropical": 0.5, "piney_resinous": 0.55, "bitterness": 0.75,
            "malty_bready": 0.35, "grassy": 0.3, "dryness_finish": 0.55, "body_fullness": 0.4},
    "pale_ale": {"citrus": 0.5, "piney_resinous": 0.4, "bitterness": 0.55, "malty_bready": 0.45,
                 "caramel_toffee": 0.3, "floral": 0.3, "body_fullness": 0.4},
    "imperial_stout": {"roasted_coffee_choc": 0.9, "caramel_toffee": 0.55, "malty_bready": 0.5,
                       "bitterness": 0.55, "body_fullness": 0.85, "alcohol_warmth": 0.6,
                       "sweet": 0.45, "vanilla_oak": 0.35, "nutty": 0.35},
    "stout": {"roasted_coffee_choc": 0.85, "caramel_toffee": 0.45, "malty_bready": 0.5,
              "bitterness": 0.5, "body_fullness": 0.7, "nutty": 0.35, "sweet": 0.35},
    "porter": {"roasted_coffee_choc": 0.65, "caramel_toffee": 0.55, "malty_bready": 0.55,
               "nutty": 0.4, "bitterness": 0.4, "body_fullness": 0.6, "sweet": 0.35},
    "wheat": {"banana_ester": 0.7, "spicy_phenolic": 0.55, "citrus": 0.4, "malty_bready": 0.4,
              "carbonation": 0.6, "body_fullness": 0.45, "sweet": 0.3, "floral": 0.3},
    "tripel": {"spicy_phenolic": 0.55, "banana_ester": 0.5, "honey": 0.4, "caramel_toffee": 0.35,
               "alcohol_warmth": 0.45, "sweet": 0.4, "body_fullness": 0.5, "dryness_finish": 0.4},
    "belgian_dark": {"caramel_toffee": 0.55, "stone_fruit": 0.45, "spicy_phenolic": 0.45,
                     "banana_ester": 0.4, "malty_bready": 0.5, "alcohol_warmth": 0.5,
                     "sweet": 0.45, "body_fullness": 0.6},
    "saison": {"spicy_phenolic": 0.6, "herbal": 0.45, "grassy": 0.4, "citrus": 0.35,
               "funk_brett": 0.3, "carbonation": 0.65, "dryness_finish": 0.65,
               "body_fullness": 0.35},
    "sour": {"sour_tart": 0.85, "funk_brett": 0.4, "citrus": 0.45, "berry": 0.45,
             "carbonation": 0.6, "dryness_finish": 0.55, "sweet": 0.25, "body_fullness": 0.3},
    "amber": {"caramel_toffee": 0.6, "malty_bready": 0.55, "nutty": 0.35, "bitterness": 0.4,
              "body_fullness": 0.45, "floral": 0.25},
    "brown": {"nutty": 0.6, "caramel_toffee": 0.55, "malty_bready": 0.55,
              "roasted_coffee_choc": 0.3, "body_fullness": 0.45, "sweet": 0.3},
    "bock": {"caramel_toffee": 0.6, "malty_bready": 0.6, "roasted_coffee_choc": 0.35, "nutty": 0.35,
             "body_fullness": 0.6, "alcohol_warmth": 0.35, "sweet": 0.4},
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
    "brandy": 40.0, "triple_sec": 40.0, "anise": 40.0, "cream_liqueur": 17.0, "amaro": 25.0,
    "liqueur": 25.0, "spirit": 40.0, "wine": 12.5,
}

# (style, keyword-substrings). Order = priority; most specific first, within each category.
_BEER_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("neipa", ("hazy", "juicy", "neipa", "new england", "haze")),
    ("dipa", ("double ipa", "dipa", "imperial ipa", "triple ipa", "double i.p.a")),
    ("ipa", ("ipa", "india pale", "i.p.a", " apa")),
    ("pale_ale", ("pale ale", "american pale", "apa ")),
    ("imperial_stout", ("imperial stout", "russian imperial", "impy")),
    ("stout", ("stout",)),
    ("porter", ("porter",)),
    ("wheat", ("hefe", "weiss", "weizen", "witbier", "wit ", "white ale", "blanche", "wheat",
               "blanc", "weisse", "hoegaarden")),
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
    ("amaro", ("amaro", "aperol", "campari", "vermouth", "fernet", "cynar", "martini")),
    ("liqueur", ("liqueur", "likör", "likor", "licor", "schnapps", "kahlua", "coffee liqueur")),
]

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


def detect_style(name: str, category: Category | str | None) -> str | None:
    """Best style key for a product, or None if even the category is unknown. Category gates the
    keyword set so a beer's "blanc" reads as a witbier and a spirit's as a white rum."""
    n = _norm(name)
    cat = category.value if isinstance(category, Category) else (category or "")
    cat = str(cat).lower()
    rules = _BEER_RULES if cat == "beer" else _SPIRIT_RULES if cat == "spirit" else []
    for style, kws in rules:
        if any(k in n for k in kws):
            return style
    # Nothing matched: fall back to the broad category centroid so the row is still scoreable.
    if cat == "beer":
        return "beer"
    if cat == "spirit":
        return "spirit"
    if cat == "wine":
        return "wine"
    return None


def sensory_from_style(name: str, category: Category | str | None,
                       style_hint: str | None = None) -> SensoryVector | None:
    """A STYLE_PRIOR SensoryVector for the product, or None when no style can be inferred. A
    non-alcoholic marker zeroes alcohol_warmth so a 0.0% reads distinct from its full sibling."""
    style = detect_style(f"{style_hint or ''} {name}", category)
    if style is None:
        return None
    axes = dict(_CENTROIDS[style])
    if is_non_alcoholic(name):
        axes["alcohol_warmth"] = 0.0
        axes["body_fullness"] = round(axes.get("body_fullness", 0.3) * 0.7, 3)
    # Broad category fallbacks are even weaker than a named style.
    conf = 0.25 if style in ("beer", "spirit", "wine") else 0.35
    return SensoryVector(source=SensorySource.STYLE_PRIOR, confidence=conf, axes=axes)


def abv_from_style(name: str, category: Category | str | None) -> float | None:
    """Typical ABV for the inferred style, or None. 0.4 for a detected non-alcoholic product."""
    if is_non_alcoholic(name):
        return 0.4
    style = detect_style(name, category)
    return _ABV.get(style) if style else None
