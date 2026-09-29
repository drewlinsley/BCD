"""Drink families — the level a person actually browses at.

`Category` is too coarse (`spirit` holds gin, bourbon and vodka, which are three different
recommendations) and `style` is too fine: the catalog carries 622 of them, and `Bourbon`,
`Straight Bourbon` and `Kentucky straight bourbon` are one shelf. A family sits between.

Built from measured counts, not from a spirits education. The catalog is concentrated enough
for a hand table to be honest about its coverage: the top 20 styles are 73% of products with a
style, the top 50 are 91%, the top 100 are 98%. Everything below that falls through to `None`,
which means "no family", not "some other family" — see `family_of`.

The table is keyed on the style string alone and ignores `category`, deliberately. 620 Tequila
rows, 363 Bourbon rows and 512 Amaro rows are filed under category `other`, and a drinker
looking for tequila should find those too. The style is the more reliable of the two fields.
"""

from __future__ import annotations

from enum import Enum


class Family(str, Enum):
    """What shelf a drink is on. The `label` is what a person is shown."""

    # spirits
    VODKA = "vodka"
    GIN = "gin"
    BOURBON = "bourbon"
    RYE = "rye"
    SCOTCH = "scotch"
    IRISH = "irish"
    WHISKEY = "whiskey"  # everything whisky that is not one of the four above
    AGAVE = "agave"
    RUM = "rum"
    BRANDY = "brandy"
    LIQUEUR = "liqueur"
    AMARO = "amaro"
    # beer
    IPA = "ipa"
    PALE_ALE = "pale_ale"
    STOUT = "stout"
    PORTER = "porter"
    LAGER = "lager"
    WHEAT = "wheat"
    SOUR = "sour"
    BELGIAN = "belgian"
    # other
    CIDER = "cider"
    SAKE = "sake"

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS: dict[Family, str] = {
    Family.VODKA: "Vodka",
    Family.GIN: "Gin",
    Family.BOURBON: "Bourbon",
    Family.RYE: "Rye",
    Family.SCOTCH: "Scotch",
    Family.IRISH: "Irish whiskey",
    Family.WHISKEY: "Whiskey",
    Family.AGAVE: "Tequila & mezcal",
    Family.RUM: "Rum",
    Family.BRANDY: "Brandy & cognac",
    Family.LIQUEUR: "Liqueurs",
    Family.AMARO: "Amaro & bitters",
    Family.IPA: "IPA",
    Family.PALE_ALE: "Pale & amber ale",
    Family.STOUT: "Stout",
    Family.PORTER: "Porter",
    Family.LAGER: "Lager & pilsner",
    Family.WHEAT: "Wheat beer",
    Family.SOUR: "Sour",
    Family.BELGIAN: "Belgian ale",
    Family.CIDER: "Cider",
    Family.SAKE: "Sake",
}

# Styles that name a family, lowercased. A style may only appear once.
#
# What is deliberately absent is as load-bearing as what is here. `Ale` (96,751 rows), `Beer`
# (29,052), `Flavored Malt Beverage` (60,765), `Specialty` (23,036) and `Malt Beverage` are TTB
# class names, not styles: they say a filing was for a malt beverage and nothing about what is
# in the can. Mapping `Ale` to a family would put 96,751 unknown beers on somebody's shelf under
# a name none of them earned, which is the same mistake as naming an importer as a producer.
# They get no family and do not appear in a family's list.
_STYLE_FAMILY: dict[str, Family] = {}


def _claim(family: Family, *styles: str) -> None:
    for style in styles:
        key = style.lower()
        if key in _STYLE_FAMILY:
            raise ValueError(f"{style!r} already claimed by {_STYLE_FAMILY[key]}")
        _STYLE_FAMILY[key] = family


_claim(Family.VODKA, "vodka", "flavored vodka", "grain neutral spirit", "neutral spirit",
       "potato vodka")
_claim(Family.GIN, "gin", "london dry gin", "distilled gin", "flavored gin", "genever",
       "old tom gin", "sloe gin")
_claim(Family.BOURBON, "bourbon", "straight bourbon", "kentucky straight bourbon",
       "corn whiskey", "straight corn whiskey", "wheated bourbon", "bottled in bond bourbon")
_claim(Family.RYE, "rye whiskey", "straight rye", "straight rye whiskey", "rye whisky",
       "malt rye whiskey")
_claim(Family.SCOTCH, "single malt scotch", "scotch whisky", "blended scotch",
       "blended scotch whisky", "blended malt scotch", "single grain scotch", "islay single malt")
_claim(Family.IRISH, "irish whiskey", "single pot still", "irish single malt")
_claim(Family.WHISKEY, "whisky", "whiskey", "blended whiskey", "flavored whiskey",
       "canadian whisky", "straight whisky", "single malt whisky", "american single malt",
       "blended whisky", "japanese whisky", "japanese single malt", "straight malt whiskey",
       "malt whiskey", "light whiskey", "spirit whiskey", "straight wheat whiskey",
       "wheat whiskey")
_claim(Family.AGAVE, "tequila", "mezcal", "agave spirit", "blanco tequila", "reposado tequila",
       "anejo tequila", "sotol", "raicilla")
_claim(Family.RUM, "gold rum", "white rum", "rum", "spiced rum", "flavored rum", "dark rum",
       "aged rum", "rhum agricole", "cachaca", "cachaça", "overproof rum")
_claim(Family.BRANDY, "cognac", "armagnac", "grape brandy", "french brandy", "brandy",
       "calvados", "california brandy", "fruit brandy", "apple brandy", "spanish grape brandy",
       "italian grape brandy", "portuguese grape brandy", "slivovitz", "pisco", "grappa",
       "eau de vie", "kirschwasser", "pear brandy", "cherry brandy", "peach brandy")
_claim(Family.LIQUEUR, "fruit liqueur", "cream liqueur", "herbal liqueur", "coffee liqueur",
       "triple sec", "amaretto", "crème de menthe", "creme de menthe", "crème de cacao",
       "creme de cacao", "anisette", "anise spirit", "fruit schnapps", "peppermint schnapps",
       "herbal schnapps", "absinthe", "sambuca", "ouzo", "limoncello", "nut liqueur",
       "chocolate liqueur", "orange liqueur", "whisky liqueur", "whiskey liqueur")
_claim(Family.AMARO, "amaro", "bitters", "aperitif", "vermouth", "quinquina", "fernet")

_claim(Family.IPA, "ipa", "new england ipa", "double ipa", "west coast ipa", "session ipa",
       "black ipa", "imperial ipa", "triple ipa", "hazy double ipa", "wheat ipa",
       "american ipa", "brut ipa", "milkshake ipa", "cold ipa", "rye ipa", "belgian ipa")
_claim(Family.PALE_ALE, "pale ale", "amber ale", "esb", "brown ale", "american pale ale",
       "single-hop pale ale", "blonde ale", "cream ale", "red ale", "irish red ale",
       "scottish ale", "barleywine", "winter warmer", "old ale", "strong ale", "altbier",
       "kölsch", "kolsch", "extra special bitter", "bitter", "english pale ale")
_claim(Family.STOUT, "stout", "sweet stout", "imperial stout", "oatmeal stout", "dry stout",
       "milk stout", "pastry stout", "russian imperial stout", "irish stout")
_claim(Family.PORTER, "porter", "baltic porter", "robust porter")
_claim(Family.LAGER, "lager", "pilsner", "helles", "bock", "schwarzbier", "festbier",
       "doppelbock", "märzen", "marzen", "oktoberfest", "vienna lager", "dunkel",
       "american lager", "light lager", "maibock", "rauchbier", "czech pilsner", "dortmunder",
       "radler", "shandy")
_claim(Family.WHEAT, "wheat beer", "hefeweizen", "witbier", "american wheat", "dunkelweizen",
       "weizenbock", "berliner weisse", "gose", "rye beer")
_claim(Family.SOUR, "sour ale", "sour", "lambic", "gueuze", "flanders red", "wild ale",
       "fruited sour", "kettle sour")
_claim(Family.BELGIAN, "saison", "tripel", "belgian dark ale", "dubbel", "quadrupel",
       "belgian strong ale", "belgian pale ale", "farmhouse ale", "abbey ale")

# Cider is here for completeness and will usually show nothing: all 333 rows the catalog files
# under category `cider` carry a NULL style, and only 11 rows anywhere say "Rosé cider". A family
# earns its place in an answer by having scorable rows, not by being in this table.
_claim(Family.CIDER, "cider", "hard cider", "perry", "fruit cider", "rosé cider", "rose cider")
_claim(Family.SAKE, "sake", "flavored sake", "junmai", "ginjo", "daiginjo", "nigori")


def family_of(style: str | None) -> Family | None:
    """The family a style belongs to, or None when the catalog does not say.

    None is a real answer and the common one: a row whose style is `Ale` or `Flavored Malt
    Beverage` has been classified by a regulator, not described. Such a row can still be
    scanned, resolved, scored and recommended in the flat list — it simply has no shelf to be
    browsed from, and putting it on the wrong one would be worse than leaving it off.
    """
    if not style:
        return None
    return _STYLE_FAMILY.get(style.strip().lower())


def styles_in(family: Family) -> list[str]:
    """Every style that names this family, for the query that fetches its rows."""
    return sorted(s for s, f in _STYLE_FAMILY.items() if f is family)


#: Every family, in the order a list of shelves should be read when nothing is known about the
#: drinker. Beer first because the catalog is 337k beer against 158k spirits.
FAMILIES: tuple[Family, ...] = (
    Family.IPA, Family.PALE_ALE, Family.LAGER, Family.STOUT, Family.PORTER, Family.WHEAT,
    Family.SOUR, Family.BELGIAN,
    Family.BOURBON, Family.RYE, Family.SCOTCH, Family.IRISH, Family.WHISKEY,
    Family.GIN, Family.VODKA, Family.AGAVE, Family.RUM, Family.BRANDY,
    Family.LIQUEUR, Family.AMARO,
    Family.CIDER, Family.SAKE,
)
