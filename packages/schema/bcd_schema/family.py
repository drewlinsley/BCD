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

Below the table sits one rule, not a second table. `Peated Scotch`, `Blended Bourbon` and
`Hazy IPA` are styles the table already knows wearing a qualifier it does not, and there were
5,712 rows of them on no shelf at all (measured 2026-09-30). `_qualified` reads the last style
word; `shelf_styles` is how a shelf gets filled with the spellings the catalog really holds,
because knowing `Peated Scotch` is a scotch and never showing one is its own kind of broken.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import Enum
from functools import lru_cache


class Group(str, Enum):
    """The aisle a shelf is in. Two levels rather than one because twenty-two shelves in a flat
    list is a scroll, and because "beer" and "spirits" is the division a drinker already has:
    someone who wants a gin is not half-deciding between gin and stout on the way there."""

    BEER = "beer"
    SPIRITS = "spirits"
    CIDER = "cider"
    SAKE = "sake"

    @property
    def label(self) -> str:
        return {Group.BEER: "Beer", Group.SPIRITS: "Spirits",
                Group.CIDER: "Cider", Group.SAKE: "Sake"}[self]


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

    @property
    def group(self) -> Group:
        return _GROUPS[self]


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

#: Which aisle each shelf is in. Cider and sake are each their own aisle rather than sharing an
#: "Other": neither is a beer and neither is a spirit, and a drinker looking for sake is looking
#: for sake, not for the drawer of things that fitted nowhere. An aisle holding one shelf is
#: shown as one row, not as a shelf inside an aisle of the same name.
_GROUPS: dict[Family, Group] = {
    **{f: Group.BEER for f in (Family.IPA, Family.PALE_ALE, Family.STOUT, Family.PORTER,
                               Family.LAGER, Family.WHEAT, Family.SOUR, Family.BELGIAN)},
    **{f: Group.SPIRITS for f in (Family.VODKA, Family.GIN, Family.BOURBON, Family.RYE,
                                  Family.SCOTCH, Family.IRISH, Family.WHISKEY, Family.AGAVE,
                                  Family.RUM, Family.BRANDY, Family.LIQUEUR, Family.AMARO)},
    Family.CIDER: Group.CIDER,
    Family.SAKE: Group.SAKE,
}

#: Aisles in the order they are read when nothing is known about the drinker. The catalog is
#: 337k beer against 158k spirits, and cider and sake are small but are not leftovers.
GROUPS: tuple[Group, ...] = (Group.BEER, Group.SPIRITS, Group.CIDER, Group.SAKE)


def families_in(group: Group) -> tuple[Family, ...]:
    """The shelves in one aisle, in `FAMILIES` order."""
    return tuple(f for f in FAMILIES if f.group is group)


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

# Spellings the last-word rule below gets wrong or cannot see, measured on the live catalog
# 2026-09-30. `Curaçao` (175 rows) names no family word at all; `Liqueur & Brandy` (64) and
# `Rock & Rye, Rum & Brandy (Etc.)` (55) are TTB cordial classes whose last word is `brandy`,
# which is a listed ingredient and not the shelf; `Belgian-style strong ale` (12) ends in a
# style that belongs to a different shelf than the beer does. Each is a real spelling with rows
# behind it, not a style anybody imagined.
_claim(Family.LIQUEUR, "curaçao", "curacao", "liqueur & brandy",
       "rock & rye, rum & brandy (etc.)")
_claim(Family.WHISKEY, "moonshine", "flavored moonshine", "japanese blended malt")
_claim(Family.BELGIAN, "belgian-style strong ale", "belgian strong golden ale")
_claim(Family.SOUR, "spontaneously fermented ale", "flanders-style sour brown ale")


#: Head nouns that name a family only when something else qualifies them. Kept out of
#: `_STYLE_FAMILY` on purpose, because the bare word and the qualified one go to different
#: places: `Ale` alone is 98,440 rows of TTB class name and must stay shelfless, while
#: `Scotch ale` and `Pumpkin ale` are beers; `Scotch` alone is not a style anybody files, while
#: `Peated Scotch` is a whisky. A marker speaks only when the style says more than the marker.
_MARKERS: dict[str, Family] = {
    "ale": Family.PALE_ALE,
    "wheat ale": Family.WHEAT,
    "scotch": Family.SCOTCH,
    "liqueur": Family.LIQUEUR,
    "cordial": Family.LIQUEUR,
    "schnapps": Family.LIQUEUR,
}

_PHRASES: dict[str, Family] = {**_STYLE_FAMILY, **_MARKERS}
if len(_PHRASES) != len(_STYLE_FAMILY) + len(_MARKERS):
    raise ValueError("a marker is also a claimed style; it belongs in one table, not both")


def _word(phrase: str) -> re.Pattern[str]:
    """`phrase`, as whole words, allowing a plural.

    Anchored at BOTH ends. A style word that may run on matches inside another word -- `gin`
    lives in `ginger` and in `original`, `scotch` in `butterscotch` -- which the cask-finish
    pass learned the expensive way, and a fallback that guesses is exactly where that mistake
    would be free to happen again.
    """
    body = re.escape(phrase)
    # brandy -> brandies, liqueur -> liqueurs. The catalog pluralises a style word often
    # enough to matter: without this, `Gin Liqueurs` reads as a gin.
    plural = re.escape(phrase[:-1]) + "ies" if phrase.endswith("y") else body + "(?:e?s)?"
    return re.compile(rf"(?<![a-z0-9])(?:{body}|{plural})(?![a-z0-9])")


_PATTERNS: tuple[tuple[re.Pattern[str], str, Family], ...] = tuple(
    (_word(phrase), phrase, family) for phrase, family in _PHRASES.items())

#: The aisle a row's own `category` puts it in. Only ever used to REJECT a guess.
_CATEGORY_GROUP: dict[str, Group] = {"beer": Group.BEER, "spirit": Group.SPIRITS,
                                     "cider": Group.CIDER, "sake": Group.SAKE}


@lru_cache(maxsize=8192)
def _qualified(style: str, group: Group | None) -> Family | None:
    """The family named by the last style word in a style nobody claimed outright.

    `Peated Scotch`, `Blended Bourbon`, `Hazy IPA`, `Barrel-aged imperial stout`, `Cognac VS`
    and the whole flavored-brandy run are each a style the table already knows, wearing a
    qualifier it does not. Measured on the live catalog 2026-09-30: 5,712 rows across 436 such
    styles, every one of them on no shelf at all.

    Read the RIGHTMOST one, because English puts the head noun last -- which is what makes
    `Apricot Brandy` a brandy and `Cinnamon whisky liqueur` a liqueur -- and among phrases
    ending there prefer the longest, so `straight bourbon` beats `bourbon` and `sour ale` beats
    `ale`.

    A phrase spanning the WHOLE string is ignored: that is the exact lookup's job, and refusing
    it here is what keeps `Ale` shelfless while `Scotch ale` reaches the ale shelf.

    `group` comes from the row's own category and only ever rejects a candidate. Exact claims
    stay category-blind on purpose -- 620 Tequila rows are filed under category `other` -- but a
    guess does not get that benefit: without the guard a beer filed `Scotch ale` would be read
    as a whisky.
    """
    best: tuple[int, int] | None = None
    found: Family | None = None
    for pattern, phrase, family in _PATTERNS:
        if group is not None and family.group is not group:
            continue
        for m in pattern.finditer(style):
            if m.start() == 0 and m.end() == len(style):
                continue
            rank = (m.end(), len(phrase))
            if best is None or rank > best:
                best, found = rank, family
    return found


def family_of(style: str | None, category: str | None = None) -> Family | None:
    """The family a style belongs to, or None when the catalog does not say.

    Answered from the table first, and only then by reading the style's last style word
    (`_qualified`). `category` is optional and only ever narrows that second answer; the table
    itself never consults it.

    None is a real answer and the common one: a row whose style is `Ale` or `Flavored Malt
    Beverage` has been classified by a regulator, not described. Such a row can still be
    scanned, resolved, scored and recommended in the flat list — it simply has no shelf to be
    browsed from, and putting it on the wrong one would be worse than leaving it off.
    """
    if not style:
        return None
    key = style.strip().lower()
    exact = _STYLE_FAMILY.get(key)
    if exact is not None:
        return exact
    return _qualified(key, _CATEGORY_GROUP.get((category or "").strip().lower()))


def styles_in(family: Family) -> list[str]:
    """Every style the TABLE gives this family, lowercased."""
    return sorted(s for s, f in _STYLE_FAMILY.items() if f is family)


def shelf_styles(rows: Iterable[tuple[str | None, str | None]]) -> dict[Family, list[str]]:
    """Bucket a catalog's own `(category, style)` pairs onto shelves, lowercased.

    `styles_in` answers from the table alone, which is right for a test and wrong for a query.
    The shelf a drinker opens is filled by `lower(style) = ANY(...)`, so a style the table
    reaches only through `_qualified` is on nobody's shelf until its own spelling is in that
    list: the Scotch shelf would call itself yours the moment you rated a `Peated Scotch` and
    then not contain one. Pass the spellings the catalog actually holds.

    A style whose answer depends on which category it is filed under is dropped rather than
    guessed at. The query filters on the style alone, so it cannot put the `Scotch ale` beers on
    one shelf and a `Scotch ale`-filed whisky on another, and a shelf with the wrong drink on it
    is worse than a shelf without that spelling.
    """
    answers: dict[str, set[Family | None]] = {}
    for category, style in rows:
        if not style or not style.strip():
            continue
        answers.setdefault(style.strip().lower(), set()).add(family_of(style, category))
    out: dict[Family, list[str]] = {}
    for style, families in answers.items():
        if len(families) != 1:
            continue
        family = next(iter(families))
        if family is not None:
            out.setdefault(family, []).append(style)
    return {family: sorted(styles) for family, styles in out.items()}


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
