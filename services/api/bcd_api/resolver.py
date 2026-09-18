"""Scan resolution + personal scoring — the server half of the latency-critical path.

Kept deliberately simple and dependency-light here (LIKE match + a transparent
chemistry-based cold-start scorer) so it runs on the laptop store. In production the
match step is Postgres trigram + pgvector ANN, and scoring blends the learned
ingredient->sensory model with the user's TasteProfile. The *shape* is what the iOS
client codes against and what we optimize behind.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from bcd_ingest.dedup import _PRODUCER_SUFFIX, is_generic_token, search_name
from bcd_ingest.store import Store, _cosine
from bcd_schema import (
    SENSORY_AXES,
    Brand,
    DetectedObject,
    DetectedText,
    ObjectResolution,
    Producer,
    Product,
    ResolvedProduct,
    ScanResolveRequest,
    ScanResolveResponse,
    ScoredCandidate,
    SensoryVector,
    TasteProfile,
)

# A text detection resolves to a product only if its name-match clears this floor. Tuned
# against the real catalog: real beers (Heineken 1.0, Krombacher 0.69, a "GUINNESS DRAUGHT
# 440ML" line 0.56) clear it; OCR chrome ("12 FL OZ" 0.38, "BREWED AND BOTTLED BY" 0.33) does
# not — so noise resolves to nothing instead of a confident wrong beer.
_MIN_MATCH = 0.5
# A very short catalog name ("J&B", "1664") is low-information and trigram-matches garbled OCR
# far too easily, so it must clear a near-exact bar instead of the normal floor. Observed live: a
# mangled Heady-Topper-can frame matched the scotch "J&B" at exactly 0.5.
_SHORT_MIN_MATCH = 0.8
_SHORT_NAME_LEN = 5
# Cap overlays per frame so a busy shelf can't bury the HUD (the client caps + anchors too).
_MAX_CANDIDATES = 8
# Rows the frame's lines name between them (`LabelIndex.match_frame`), on top of each line's
# own matches.
_FRAME_CANDIDATES = 8

# Score bands for the overlay's one-line 'why'. Above _STRONG_MATCH we claim a match;
# below _MILD_MATCH we say so plainly rather than dressing up a miss.
_STRONG_MATCH = 0.8
_MILD_MATCH = 0.6

# A detection is a *product identity* only if it carries a real word — a run of >=3 letters (a
# brand or name token). A bare number is label chrome, not a name: "15" off a 15th-anniversary
# can, "40" for proof, "500" for mL, "5" for %ABV. Trigram-matching those resolves to whatever
# junk shares the digits (a can's "15" once matched a spirit literally named "15"), so they must
# resolve to nothing. The one numeric exception is a 4+-digit run that IS the identity ("1664").
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_LONGNUM_RE = re.compile(r"^[0-9]{4,}$")


def _is_identity_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _WORD_RE.search(t):
        return True
    return bool(_LONGNUM_RE.match(t.replace(" ", "")))


# Even a line that clears the score floor can be a *coincidental* trigram window rather than a
# real name hit. OCR mangles the ubiquitous Surgeon-General warning ("...impairs your ability to
# drive A CAR OR operate machinery") into a fragment like "BACAR OR", which word-similarity-matches
# "Bacardi" at 0.625 — HIGHER than a legitimately-embedded "GUINNESS DRAUGHT 440ML" line scores
# (0.56). Measured against the real catalog, no similarity threshold separates the two. What does
# separate them is token agreement: a genuine match carries an OCR token that *is* a name word
# ("GUINNESS" == "Guinness"), while the garble only has a truncation ("BACAR" vs "Bacardi" ~ 0.56,
# "OR" vs "Bacardi" = 0). So a text match must also carry a real name token (>=4 letters) that
# closely matches some OCR token — otherwise the "brand" is only a coincidental sub-window.
_TOKEN_SUPPORT_MIN = 0.85
_MIN_NAME_TOKEN_LEN = 4
_TOKEN_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)  # runs of >=2 unicode letters


def _norm_token(s: str) -> str:
    """Casefold and strip diacritics so 'Bière' and 'BIERE' compare equal."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


# A possessive is one word. The catalog says "Tito's", the label prints TITO'S, and the
# recognizer reads it TITOS as often as TITO'S -- and split at the apostrophe, "tito's" was
# the token "tito", which TITOS is not a read of (a letter gained). The three OFF rows for
# the bottle, one of them spelt "Titos", had covered for it; merged into one row spelt with
# the apostrophe, the bottle stopped proving itself (2026-09-17). Gosling's, Lawson's and
# every other possessive brand read the same way. Stripped before tokenizing, on both sides.
_APOSTROPHES = str.maketrans("", "", "'\u2019\u2018`")


def _unapostrophed(s: str) -> str:
    return (s or "").translate(_APOSTROPHES)


def _tokens(s: str) -> list[str]:
    return [_norm_token(t) for t in _TOKEN_RE.findall(_unapostrophed(s))]


# The recognizer picks a script per line by what the letterforms most resemble, and a stylized
# Latin wordmark resembles other alphabets: on one can of Heady Topper, in one session, HEADY
# TOPPER arrived as "ЯДУ ТОРР", "АДУ ТОРО" and "ГАДУ ТОРРА". Asking the scanner for en-US does
# not stop it -- VisionKit's language list is a preference, not a pin, and the non-Latin share
# of frames was 11.5% with it and 9-16% without. The shapes it read are right; only the
# alphabet is wrong, and every Cyrillic letter has the Latin letter it is drawn like, which is
# why the recognizer chose it. Mapped back, "ЯДУ ТОРР" is "RDY TOPP": a read of HEADY TOPPER as
# good as any the Latin model gives, and one the maker pick can use. Hangul and CJK reads have
# no such map and stay what they are, which is nothing.
#
# Applied to the camera's reading, not to the catalog, because the catalog is Latin: the
# eighteen Cyrillic-named rows in it are OFF imports of Bulgarian and Russian products a US
# shelf does not hold, and a Latin read of their labels is the trade for a Latin read of ours.
_CONFUSABLE = str.maketrans({
    "А": "A", "Б": "B", "В": "B", "Г": "T", "Д": "D", "Е": "E", "Ё": "E", "Ж": "X", "З": "E",
    "И": "N", "Й": "N", "К": "K", "Л": "A", "М": "M", "Н": "H", "О": "O", "П": "N", "Р": "P",
    "С": "C", "Т": "T", "У": "Y", "Ф": "O", "Х": "X", "Ц": "U", "Ч": "Y", "Ш": "W", "Щ": "W",
    "Ъ": "B", "Ы": "BI", "Ь": "B", "Э": "E", "Ю": "IO", "Я": "R",
    "Ѕ": "S", "І": "I", "Ї": "I", "Ј": "J", "Є": "E", "Ґ": "T", "Ў": "Y", "Ђ": "D", "Ћ": "H",
    "Љ": "A", "Њ": "H", "Џ": "U",
    "а": "a", "б": "b", "в": "b", "г": "t", "д": "d", "е": "e", "ё": "e", "ж": "x", "з": "e",
    "и": "n", "й": "n", "к": "k", "л": "a", "м": "m", "н": "h", "о": "o", "п": "n", "р": "p",
    "с": "c", "т": "t", "у": "y", "ф": "o", "х": "x", "ц": "u", "ч": "y", "ш": "w", "щ": "w",
    "ъ": "b", "ы": "bi", "ь": "b", "э": "e", "ю": "io", "я": "r",
    "ѕ": "s", "і": "i", "ї": "i", "ј": "j", "є": "e", "ґ": "t", "ў": "y", "ђ": "d", "ћ": "h",
    "љ": "a", "њ": "h", "џ": "u",
})


def _latin(text: str) -> str:
    """The camera's reading, in the alphabet the label is printed in."""
    return (text or "").translate(_CONFUSABLE)


@lru_cache(maxsize=200_000)
def _trigrams(token: str) -> frozenset[str]:
    # pg_trgm-style padding: two leading spaces + one trailing, then 3-grams. Mirrors the
    # store's similarity() closely enough to calibrate one threshold across both. Cached:
    # a frame of twenty lines compares a few hundred distinct tokens a few hundred thousand
    # times, and building the set was a third of a slow frame (2026-09-16).
    padded = f"  {token} "
    return frozenset(padded[i:i + 3] for i in range(len(padded) - 2))


def _trigram_sim(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


#: How much of the line a candidate must account for before a frame with nothing to
#: corroborate against is allowed to certify itself. Measured on real frames: the fragments
#: top out at 0.667 ("Chemist" against "CHEMIST VER") and the whole-label reads start at
#: 0.862, so 0.7 sits in the gap rather than on either population.
_ACCOUNTS_FOR_LINE = 0.7


def _flatten(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _unapostrophed(s).lower()).strip()


#: Words and numbers of a name, for asking whether a line prints the name whole. Not `_tokens`:
#: that drops digits, and "Stella Artois 0.0%" is not Stella Artois, "Sierra Nevada 6 & Out"
#: is not the pale ale. Three letters, not four: "Ice" is what tells `Miller High Life Ice`
#: from the beer, "New Day" what tells `Little Willow New Day` from the brewery's line.
_NAME_WORD_RE = re.compile(r"[^\W_]{3,}|\d+", re.UNICODE)


def _name_words(name: str) -> list[str]:
    """Every word of the name but a company suffix -- category words included. The first
    cut dropped those as it drops them everywhere else, and what was left of `West Coast
    Wheat` was "west coast", which a double IPA's can printed as WEST COAST STYLE; what was
    left of `Toppling Goliath Brewing Co. Zz Hop` was the brewery, which the brewery's own
    line printed whole (2026-09-15). The words a name shares with its category are not what
    picks it off a shelf, but they are part of the name, and a line that prints the name
    prints them."""
    out = []
    for w in _NAME_WORD_RE.findall(_unapostrophed(name).lower()):
        w = _norm_token(w)
        if w not in _PRODUCER_SUFFIX and w not in out:
            out.append(w)
    return out


#: A "line" this long is a paragraph -- the back-label essay, the Surgeon General's warning
#: (about forty words) -- and nothing on a label is named inside a paragraph. The largest
#: block a label prints as one line, Miller High Life's ("BREWING COMPANY MILWAUKEE PREMIUM
#: Miller BREWED HIGH LIFE EST 1903 The Champagne of Beers 12 FLUID OUNCES 1.355 LITERS"), is
#: about twenty.
_PARAGRAPH_WORDS = 30


def _reads_the_name(name: str, line: str, *, loose: bool = False) -> bool:
    """Whether a line prints a multi-word name whole, in order and together, whatever else is
    on it.

    Every word and number of the name, not only the identifying ones: the first cut of this
    took `_identifying_tokens` and drew `Sierra Nevada 6 & Out` off SIERRA NEVADA PALE ALE
    thirteen times, the 6 and the OUT being too short to ask for. In order and within one
    stray word of each other: "Miller BREWED HIGH LIFE" is the name with a flourish between;
    "we have worked so / you MUST pour" on the back of a Heady Topper can is not `Must Have`,
    and a paragraph is where two common words will always end up near each other.

    `loose` reads each word the way `_read_as` does -- letters lost at one end allowed --
    for a phrase the recognizer garbles word by word, like a maker's line in small type."""
    words = _name_words(name)
    if len(words) < _MIN_SELF_PROOF_TOKENS or sum(len(w) for w in words) < _MIN_SELF_PROOF_CHARS:
        return False
    toks = [_norm_token(w) for w in _NAME_WORD_RE.findall(_unapostrophed(line).lower())]
    if len(toks) > _PARAGRAPH_WORDS:
        return False

    def read(w: str, tok: str) -> bool:
        if loose:
            return _read_as(w, {tok})
        return _trigram_sim(w, tok) >= _TOKEN_SUPPORT_MIN

    span = len(words) + 1
    for start in range(max(1, len(toks) - span + 1)):
        window = toks[start:start + span]
        at = 0
        for w in words:                      # in the name's order, each after the last
            while at < len(window) and not read(w, window[at]):
                at += 1
            if at >= len(window):
                break
            at += 1
        else:
            return True
    return False


def _accounts_for_the_line(name: str, line: str, *, threshold: float | None = None) -> bool:
    """Whether the candidate is the whole of what was read, or only a piece of it.

    Containment cannot tell the difference: `word_similarity` is 1.0 for ANY name wholly
    inside the line, so a catalog row named "Mist" scores 1.0 against "ACHE MIST-VERM" --
    which is THE ALCHEMIST VERMONT with the wordmark split mid-word by the recognizer. The
    can produced "ALCHE MIST VERM" and "ACHE MISTVERN" too, and every one of those pieces is
    a real product name someone has registered.

    Plain similarity is the measure that penalises what the name leaves out, which is exactly
    the question here. It also, correctly, denies the exemption to "Draught Stout" read off
    "GUINNESS DRAUGHT STOUT": that row is not the whole label either, and the frame should
    say so rather than certify itself.
    """
    return _trigram_sim(_flatten(name), _flatten(line)) >= (threshold or _ACCOUNTS_FOR_LINE)



# Proving a product off one line alone is a stronger claim than "this row accounts for what
# the line says", so it answers to a higher bar. At 0.7 a rim fragment reading "CHEMIST-VE"
# (0.73) certified a distillery named `Chemist` on every other tick. The closest real label has
# to come is "HEADY TOPPER THE ALCHEMIST" against its catalog name, at 0.86.
_SELF_PROOF_SIM = 0.8

# ...and a name with too little identifying substance cannot name a product on its own however
# exactly it matches, because a short garble matches something exactly: "BALE" off a Focal
# Banger can is a perfect 1.00 against a catalog row called `Bale`. Five characters keeps
# "STONE IPA" -- the least substantial label the recogniser is meant to know -- and drops the
# four-letter coincidences the log is full of: `Bale`, `Vern`, `Mist`, `Topo`, `Ver`.
_MIN_SELF_PROOF_CHARS = 5
# ...and it has to be a phrase, not a word -- see `_is_whole_label` for the measurement.
_MIN_SELF_PROOF_TOKENS = 2


def _identifying_tokens(name: str) -> list[str]:
    """Name tokens that could actually pick this product off a shelf: long enough to be a real
    word, and not a category or packaging word every other label carries too.

    Nor a word every producer's name carries. Nearly every can prints BREWING COMPANY, and a
    producer named "Pariah Brewing Company" was agreed with by a Miller High Life can on the
    word COMPANY -- which, with COLORS off the fine print "NO COLORS OR FLAVORS FROM ARTIFICIAL
    SOURCES", made two lines naming a beer called `Colors` (2026-09-14)."""
    return [t for t in _tokens(name)
            if len(t) >= _MIN_NAME_TOKEN_LEN and not is_generic_token(t)
            and t not in _PRODUCER_SUFFIX]


def _token_supported(query: str, name: str) -> bool:
    """True if an *identifying* name token closely matches some OCR token — evidence the brand
    word is actually present in the line, not a coincidental trigram window.

    Agreement on a category word is not evidence. A Heady Topper can carries "AMERICAN DOUBLE
    IPA" and "DRINK FROM THE CAN"; matching on "double" and "drink" pulled in an unrelated hazy
    IPA and a product named "Life drink", and both outranked the real beer because all three sat
    within 0.012 of each other just above the floor. A name with nothing identifying in it
    ("J&B", "1664", "Hazy Double IPA") has nothing to anchor on and defers to the raised floor
    the caller applies instead.

    Closely: near-exact, or the word with its first or last letter lost -- the letter a
    stylized face loses most. CAMPARI arrived as CAMPAR on thirty frames of thirty, a 0.67
    against its own name, and the row was never so much as a candidate (2026-09-15). One
    letter, not the five-letter affix `_read_as` allows: BACAR, off "...drive A CAR OR..."
    in the warning, starts the way BACARDI starts, and this gate exists to refuse it. This
    gate decides what may be evidence; what it proves is decided downstream, and nothing
    there loosens with it."""
    name_tokens = [t for t in _tokens(name) if len(t) >= _MIN_NAME_TOKEN_LEN]
    if not name_tokens:
        return True
    identifying = _identifying_tokens(name)
    if not identifying:
        # Real words, but every one of them is a category word: "Ipa Ipa", "Irish Whiskey".
        # Such a name can only be what the label names if the label is equally generic. When
        # the line does carry something specific the row cannot account for, the agreement is
        # a coincidence — "DOGFISH HEAD 60 MINUTE IPA" resolved to a row literally named "Ipa
        # Ipa", because containment scores it 1.0 and so the raised floor this used to defer
        # to never bit.
        return not _identifying_tokens(query)
    q_tokens = set(_tokens(query))
    return any(_trigram_sim(nt, qt) >= _TOKEN_SUPPORT_MIN
               or (len(qt) >= _MAKER_HYPOTHESIS_WORD and qt in (nt[:-1], nt[1:]))
               for nt in identifying for qt in q_tokens)


# ---- frame-level corroboration ----
# A real label corroborates itself. A can carries its brewery *and* its beer, so the product
# it names is named by more than one line: "THE ALCHEMIST" and "HEADY TOPPER" agree. Label
# chrome does not corroborate — "PINT" names exactly one catalog row ("Pint Cake") and nothing
# else on the can agrees with it.
#
# Scoring each line in isolation cannot see that difference. Word-similarity asks how well a
# line matches *part of* a name, not how much of the name it accounts for, so "PINT" is a
# flawless hit inside "Pint Cake" and scores 1.00 — identical to the real beer. At 4.7k
# products that was harmless because no row was named "Pint Cake"; at 363k every common word
# stamped on a can (PINT, CAN, DRINK, DOUBLE) is a perfect word-match for *something*, four
# candidates tie at 1.00, and the right one ranks by luck. Counting how many distinct lines
# name a candidate is what separates the beer from the chrome.
_UNCORROBORATED = 0.75
# Below this many identity-bearing lines there is no corroborating evidence to be had, so a
# lone hit is not evidence of weakness — a barcode or a single clean brand line must stay
# confident. The penalty applies only where other lines *could* have agreed and none did.
_MIN_FRAME_FOR_PENALTY = 2


# Packaging and measure words. The shared dedup vocabulary already knows the *style* words a
# label carries ("american", "double", "ipa", "stout", "drink"); it does not know the words that
# describe the container, because a container word is a perfectly good part of a catalog *name*
# and dedup must not fold "Proper Pint" into "Proper". Here the question is different — whether
# an OCR line is worth matching at all — so the resolver keeps its own list rather than widening
# dedup's and changing how the catalog merges.
#
# This is where "PINT" was getting in. It is not a beer, it is the size of the can, and matching
# it cost 1.5s to return "Pint Cake".
_PACKAGING = {
    "pint", "pints", "can", "cans", "canned", "bottle", "bottled", "bottles",
    "draft", "draught", "keg", "growler", "crowler", "ounce", "ounces",
    "milliliter", "milliliters", "litre", "litres", "liter", "liters",
    "pack", "sixpack", "contents", "volume", "net", "vol", "alc",
}


def _worth_matching(text: str) -> bool:
    """Whether an OCR line could name a product at all — asked *before* the trigram query.

    A line built only from container words names a size, not a drink, so matching it can only
    produce a coincidence — "PINT" cost 1.5s to return "Pint Cake". Filtering before the query
    is what makes it cheap: a short common word is also the most expensive thing to match,
    because it matches tens of thousands of rows.

    Deliberately narrower than "has nothing identifying in it". A line of pure *category* words
    must still be matched, because a catalog name can be pure category too: "FML Hazy Double
    IPA" is a real product and a clean read of it has to resolve. Those lines are already
    handled — `_token_supported` lets a generic line match a generic name and nothing else — so
    widening this to category words costs recall and buys nothing the frame does not already
    fix by ranking."""
    meaningful = [t for t in _tokens(text) if len(t) >= _MIN_NAME_TOKEN_LEN]
    if not meaningful:
        return True                      # nothing to judge; leave it to the existing guards
    return any(t not in _PACKAGING for t in meaningful)


# ---- the producer path ----
# When the camera cannot read the product name, it can often still read the maker. A Heady
# Topper can's wordmark is a wavy psychedelic script: across 38 live frames Vision returned the
# beer's own name 3 times out of 100 lines — once as Cyrillic, "АДУ ТОРИ" — while the small rim
# print, "ALCHEMIST-VER…", came through 23 times. The brewery is the readable half of that
# label, and a brewery with two products is a far narrower answer than 363k rows.
_PRODUCER_MATCH_MIN = 0.6
# Past this many products in the hinted category, the label has identified a *maker* and not a
# drink. Offering a guess then would be inventing one.
_PRODUCER_MAX_PRODUCTS = 4
# A producer hit is indirect evidence, so it must never outrank a product the label actually
# names — only the nothing it is competing against.
_PRODUCER_EVIDENCE = 0.6

# Telling a maker's beers apart by the *shape* of an unreadable wordmark.
#
# A stylized can OCRs its maker in plain type and its own name as garble: THE ALCHEMIST comes
# through, HEADY TOPPER arrives as "FADY TOPPE", "ADY TOPP", "ROY TOP". Against the whole
# catalog that garble is noise -- it is how `Roy!` and `Deadeye` got drawn. Against one
# brewery's twenty beers it is not: measured over 245 logged frames with the maker read,
# "heady topper" scores 0.22-0.47 against its garble and ~0.06 against "focal banger", and a
# Focal Banger can reverses that at 0.30-0.64. The floor and margin below picked 100 of those
# frames, 82 Heady and 18 Focal, and got none wrong; the other 145 had no wordmark to score.
#
# Three exclusions are what make the numbers mean anything, each earned from a false pick:
# category and packaging chrome ("ALE / ALC. 8% BY VOL" scored 0.43 against a beer called
# `Alena`), the maker's own tokens (a garbled ALCHEMIST resembles `Alena` too), and two-letter
# fragments ("AL" off "ALC." scored 0.29). And a one-word name needs a near read where a
# two-word name needs a resemblance, because a lone garbled word resembles many things and two
# words agreeing on two words is the corroboration a label gives -- the same rule as
# `_MIN_SELF_PROOF_TOKENS`, applied to the other side of the match.
_MAKER_PICK_MIN = 0.20          # a two-word name, read as garble
_MAKER_PICK_MIN_LONE = 0.40     # a one-word name has to be nearly read
_MAKER_PICK_MARGIN = 0.12       # over the runner-up among the maker's beers
_MAKER_PICK_MAX_PRODUCTS = 60   # a brewery's catalog; past this it is a distributor
_MAKER_PICK_WINDOW = 3          # a name is one to three consecutive tokens
_MAKER_PICK_MIN_TOKEN = 3       # "AL" and "DY" carry nothing on their own
# A word in a window that shares nothing with any word of the name -- nor with the name
# written as one word -- is a different word, not a garble of one, and the window is not a
# reading of the name however the rest of it scores. "casa pombata" -- CASA FONDATA off a
# Ramazzotti label, the second word misread -- is a 0.24 against `casa comerci` on the
# strength of CASA alone, over the two-word floor, and named a Sardinian beer under a
# one-word maker the same label had spelt by accident (2026-09-16). The bar is a trigram
# or two, not resemblance: "NYTOPPANDY" is HEADY TOPPER stacked and read as one word, and
# shares only TOP and OPP with it. Held to words of five letters or more: "ROY", "FADY" and
# "DY" are what a stylized HEADY looks like to the recognizer.
_MAKER_WINDOW_WORD_MIN = 0.1
_MAKER_WINDOW_WORD_LEN = 5
_MAKER_TOKEN_SIM = 0.5          # what counts as a (garbled) read of the maker's own name
# The maker line is garbled too. The can prints THE ALCHEMIST and the scanner reads
# "CHEMIST-VER" sixty times for every four "THE ALCHEMIST", and the producer guards --
# built so that "BACAR" cannot reach Bacardi -- rightly refuse that as a read of the maker.
# So the pick starts from a looser *hypothesis*: any maker the line resembles at all, tried
# and discarded unless the wordmark then names one of its beers by a margin. Two weak reads
# that agree are one strong read; a weak read that agrees with nothing stays nothing.
#
# How much resemblance: "CHEMIST-VER", the read the can gives most, is a 0.38 against `The
# Alchemist`, and "HEMIST-VER" -- one more letter gone -- a 0.33. Measured over the 985
# frames in the scan log, with the wordmark contest as the only thing standing between a
# hypothesis and the screen: a floor of 0.40 drew Heady Topper on 16 frames, 0.35 on 66,
# 0.30 on 71, and none of the three drew anything wrong. The floor is what the maker line
# reads at, not what a clean read would score.
_MAKER_HYPOTHESIS_MIN = 0.30
_MAKER_HYPOTHESES = 6           # makers a line, or a word of it, may be tried against
# A maker's line is a few words: THE ALCHEMIST, DAVIDE CAMPARI MILANO, a brewery and its
# town. A line longer than this is a paragraph -- an appliance sticker, the back label --
# and matching a whole paragraph against the producer table costs a window scan per row
# per word of it; a fridge of stickers put a frame at 2.4 s (2026-09-16). Its words are
# still tried one at a time, which is how the maker is found in a grouped line anyway.
_MAKER_LINE_MAX_TOKENS = 12
# The maker line carries more than the maker: "CHEMIST-VERMONT" is the name with the town
# after it, and matched as a whole it resembles `Vermont Ice`, `Vermont Distillers` and five
# more Vermont producers better than it resembles `The Alchemist` (0.38) -- which then never
# made the six. So each word of the line long enough to be a name nominates makers on its
# own as well: "chemist" reaches `The Alchemist` and "vermont" the Vermont producers, and
# the wordmark contest sorts them out. Four letters is "mist", which reaches nothing worth
# trying; five is where a fragment starts to be a word.
_MAKER_HYPOTHESIS_WORD = 5


def _affix_read(line: str, maker_name: str) -> bool:
    """Whether some word of the line is a word of the maker's name with letters lost at one
    end -- the recognizer's failure on stylized type, and the only resemblance a hypothesis
    may rest on.

    "CHEMIST", "HEMIST", "ACHEMIST" and "FICHEMIST" all end the way ALCHEMIST ends; "CAMPAR"
    starts the way CAMPARI starts. "LOURE" shares three letters with `Money Lure` and is a
    garble of FLAVORS, and the resemblance the matcher scored it at (0.38) nominated that
    brewery off a can's fine print, whose next line then named `Colorado Fisherman` by shape
    (2026-09-14). A trigram score cannot tell those apart; the shape of the loss can."""
    # ...and on a word of the maker's that is the maker's: "Aperi" starts the way APERITIVO
    # starts, and a producer registered as `Terrativo Aperitivo` was hypothesised off a
    # bottle of Campari on the word for what is in it (2026-09-15).
    makers = _identifying_tokens(maker_name)
    for w in _tokens(line):
        for m in makers:
            if w == m:
                return True
            if len(w) >= _MAKER_HYPOTHESIS_WORD and len(m) >= _MAKER_HYPOTHESIS_WORD:
                k = _MAKER_HYPOTHESIS_WORD
                if w[:k] == m[:k] or w[-k:] == m[-k:]:
                    return True
    return False
# A candidate whose category *contradicts* the label's own fine print. Not merely unsupported —
# the frame says one thing and the row says another, which is evidence against, not absence of
# evidence. "A CHEMIST VER" off this can matched a distillery's `Chemist` at 1.00 while the same
# frame read "ALE"; 17 of that maker's 19 products are spirits. 'other' is never a contradiction,
# because it means the catalog does not know, not that it disagrees.
_CATEGORY_CONTRADICTS = 0.5

# The category words a label prints in its fine print. This is the one part of a stylized can
# the OCR reads reliably — "ALE / ALC. 8% BY VOL / 1 PINT" came through on 25 of those 100
# lines, unfailingly, while the brand did not. It cannot name a product, but it names a
# category, and that is exactly what separates the right maker from the wrong one here:
# Alchemist makes 1 beer, while Chemist makes 17 spirits and Cocktail Chemist 7.
_CATEGORY_WORDS = {
    "beer": {"ale", "lager", "stout", "porter", "pilsner", "pilsener", "ipa", "beer",
             "bock", "saison", "gose", "witbier", "weisse", "hefeweizen", "kolsch",
             "brew", "brewed", "malt", "pale", "amber", "dunkel", "tripel", "dubbel"},
    "spirit": {"vodka", "gin", "whiskey", "whisky", "rum", "tequila", "bourbon", "brandy",
               "cognac", "mezcal", "liqueur", "scotch", "rye", "absinthe", "schnapps",
               "distilled", "proof"},
    "wine": {"wine", "vino", "chardonnay", "merlot", "cabernet", "riesling", "rose",
             "prosecco", "champagne", "sauvignon", "pinot", "syrah", "zinfandel"},
}


# Finer than the category: what kind of spirit. A label that says RUM has said which of the
# catalog's spirits it can be, and a row whose registered class is a gin is not one of
# them -- however whole the line prints that row's name. A 1984 filing for a London dry gin
# called `Black Seal` was proven by "BLACK SEAL / 80 PROOF / BERMUDA BLACK RUM" on every
# frame of a bottle of Gosling's, and once the rum's own rows were merged into one it was
# the only name the object had (2026-09-15). Synonyms are grouped so RHUM and RON are rum.
#
# Within the row's own category only. Across categories the label's word is as likely a
# slogan or a garble as a fact -- "The Champagne of Beers" names a wine on every Miller
# can, and BEERS arrives as BETT -- and that judgement stays the category rule's, which
# marks the score down rather than closing the door. A label that says RUM has read the
# word that matters cleanly, in the fine print, where the recognizer is at its best.
_KIND_FAMILIES = (
    ("spirit", {"rum", "rhum", "ron"}), ("spirit", {"gin"}), ("spirit", {"vodka"}),
    ("spirit", {"whiskey", "whisky", "bourbon", "scotch", "rye"}),
    ("spirit", {"tequila", "mezcal"}),
    ("spirit", {"brandy", "cognac", "armagnac", "calvados", "grappa"}),
    ("spirit", {"liqueur", "liquore", "schnapps", "amaro", "aperitivo", "aperitif", "vermouth",
                "vermut", "sambuca", "limoncello"}),
    ("spirit", {"absinthe"}),
)
_KIND_OF = {w: i for i, (_, fam) in enumerate(_KIND_FAMILIES) for w in fam}


def _kinds(text: str) -> set[int]:
    """The kinds of drink a piece of text names."""
    return {_KIND_OF[t] for t in _tokens(text) if t in _KIND_OF}


def _kind_contradicts(resolved: ResolvedProduct, frame_kinds: set[int]) -> bool:
    """Whether the label named what is in the bottle, in the row's own category, and the
    row is something else. Silence on either side is not a contradiction: a filing with no
    class, or a can that prints no kind word, is unknown, not wrong."""
    p = resolved.product
    cat = p.category.value if p.category else None
    said = {k for k in frame_kinds if _KIND_FAMILIES[k][0] == cat}
    if not said:
        return False
    own = _kinds(p.name or "") | _kinds(p.style.value if p.style else "")
    return bool(own) and not (own & said)


def _category_hint(detections) -> str | None:
    """The category the label's own fine print names, or None if it says nothing or disagrees.

    Counted across the whole frame rather than taken from the first match, because a single
    word is easy to misread and a can carries several ("ALE", "ALC", "1 PINT"). A tie means the
    label is ambiguous and the hint is withheld — a wrong category filter is worse than none.
    """
    votes: dict[str, int] = {}
    for det in detections:
        for tok in _tokens(det.text):
            for cat, words in _CATEGORY_WORDS.items():
                if tok in words:
                    votes[cat] = votes.get(cat, 0) + 1
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _candidate_vocabulary(resolved: ResolvedProduct) -> list[str]:
    """The identifying words that would name this product on a label: its own name, its brand
    and its producer. The producer is what carries the signal — "THE ALCHEMIST" is the line
    that tells the real Heady Topper apart from a one-word coincidence."""
    seen: dict[str, None] = {}
    # Aliases are the other names the label prints. A merge keeps every absorbed row's name
    # here, and that is what put "Bombay Sapphire Vapour Infused London Dry Gin" -- the words
    # actually on the bottle -- beside a row named without them. Without it, a frame that read
    # VAPOUR INFUSED gave that line's support to `East Vapour Infused London Dry Gin`, a
    # different gin by the same house whose name happens to print those words, and East won
    # three of the thirty-nine Bombay frames in the log while EAST appeared in none of them.
    for part in (resolved.product.name, resolved.brand.name, resolved.producer.name,
                 *(resolved.product.aliases or [])):
        for t in _identifying_tokens(part or ""):
            seen[t] = None
    return list(seen)


def _own_vocabulary(resolved: ResolvedProduct) -> tuple[list[str], bool]:
    """The identifying words of a product's name that are not its maker's -- what tells it
    from its siblings -- and whether it has any.

    A name that is nothing but the maker's (`Miller High Life`, by Miller High Life) is its
    own: the flagship is named for the house, and its words are the words that name it. The
    catalog also holds 164,000 rows whose brand is their whole label -- a filing with no
    fanciful name -- and for those the same fallback says every word is the beer's own, when
    most are the brewery's. Callers get the flag so they can hold that case to the whole
    name (`_reads_every_word`) rather than to any word of it."""
    makers = set()
    for part in _maker_names(resolved):
        makers.update(_identifying_tokens(part))
    named = _identifying_tokens(resolved.product.name or "")
    own = [t for t in named if t not in makers]
    flagship = not own
    if flagship:
        own = list(named)
    # The other names a merge left on the row are the label's words too ("Bombay Sapphire
    # Vapour Infused London Dry Gin" beside a row named without them) -- more of the
    # bottle's own vocabulary, never a substitute for it.
    for alias in resolved.product.aliases or []:
        own += [t for t in _identifying_tokens(alias) if t not in makers and t not in own]
    return own, flagship


def _read_as(word: str, read_toks: set[str]) -> bool:
    """Whether the frame read this word: exactly or nearly, or -- for a word long enough to
    survive it -- with letters lost at one end, the recognizer's failure on stylized type
    that `_affix_read` allows a maker's name. TOPPLING arrives as PLING and PPLING off the
    brewery line of a Dino Break can, and a name held to every word must not lose the beer
    to the way the can's typeface loses its first letters.

    Lost, never gained, and what is left has to be the word: the read is no longer than
    the word, and matches its start or its end letter for letter, one substitution allowed
    (RAMAZZOTTI arrived as RAMAZZON and RAMAZZOI: the type thins at the end and the last
    letter it does read is a guess). ALCHEMIST ends the way `Chemist` ends, and the rim of
    a Heady Topper can read whole would otherwise have read a distillery's one-word name;
    DRINKEY -- DRINK with the next word's first letters run on, off a can that prints DRINK
    FROM THE CAN -- would have read `Drinky`, and FARMSTOCK, sharing its last five letters
    with `Campstock`, would have read a rye that was not on the shelf (2026-09-16). The
    first cut anchored five letters at one end and let the rest be anything."""
    k = _MAKER_HYPOTHESIS_WORD
    for r in read_toks:
        if _trigram_sim(word, r) >= _TOKEN_SUPPORT_MIN:
            return True
        if len(word) >= k and k <= len(r) <= len(word):
            n = len(r)
            if _substitutions(word[:n], r) <= 1 or _substitutions(word[-n:], r) <= 1:
                return True
    return False


def _substitutions(a: str, b: str) -> int:
    """Letters that differ between two strings of one length."""
    return sum(x != y for x, y in zip(a, b, strict=True))


def _unread(words: list[str], read_toks: set[str]) -> list[str]:
    """The words the frame did not read."""
    return [w for w in words if not _read_as(w, read_toks)]


def _reads_every_word(name: str, read_toks: set[str]) -> bool:
    """Whether every identifying word of the name was read somewhere in the frame."""
    return not _unread(_identifying_tokens(name), read_toks)


def _agreeing_lines(vocab: list[str], line_tokens: list[list[str]]) -> frozenset[int]:
    """Which lines of the frame carry a word of this vocabulary -- the evidence a candidate
    rests on, by line rather than by count."""
    return frozenset(i for i, toks in enumerate(line_tokens)
                     if any(_trigram_sim(v, t) >= _TOKEN_SUPPORT_MIN for v in vocab for t in toks))


def _is_business_name(resolved: ResolvedProduct) -> bool:
    """Whether the row is named for its maker, suffix and all, and for nothing else: a
    permit filed under the brewery's name with no beer on it. The catalog holds one such row
    for most of the breweries in it -- 164,000 rows are named exactly as their brand -- and
    the ones that carry a trade suffix are companies, not drinks: `Toppling Goliath Brewing
    Co.` was drawn beside `Toppling Goliath Brewing Co. Dino Break` off the brewery line of a
    Dino Break can, and alone when only that line was in view (2026-09-15). A maker's name
    without a suffix (`Sierra Nevada`, `Campari`) may be a flagship's, and stays."""
    name = _tokens(resolved.product.name)
    if not any(t in _PRODUCER_SUFFIX for t in name):
        return False
    core = {t for t in name if t not in _PRODUCER_SUFFIX}
    if not core:
        return False
    return any(core == {t for t in _tokens(m) if t not in _PRODUCER_SUFFIX}
               for m in _maker_names(resolved))


def _maker_names(resolved: ResolvedProduct) -> list[str]:
    """The brand and producer names the catalog actually holds. `_hydrate` stands in a brand
    named for the product when the row has none, and that placeholder would make every row
    its own maker."""
    return [m.name for m in (resolved.brand, resolved.producer)
            if m is not None and m.id != "unknown" and m.name]


def _maker_ids(resolved: ResolvedProduct) -> set[str]:
    return {m.id for m in (resolved.brand, resolved.producer)
            if m is not None and m.id != "unknown"}


def _house_names_the_label(resolved: ResolvedProduct, detections: list[DetectedText],
                           read_toks: set[str]) -> bool:
    """Whether a one-word label is proven by its house's line.

    A single word is not a label (`_MIN_SELF_PROOF_TOKENS`), and a word read twice is one
    piece of evidence (`_frame_support`) -- and some labels are one word. CAMPARI is the
    whole of the name on the bottle, and under it, in small type, the house: DAVIDE CAMPARI
    MILANO. The camera read the wordmark as CAMPAR on every one of thirty frames and the
    house as "Davide Campan MILAN", "Davide Campani MILANO", and the bottle drew nothing
    (2026-09-15). The phrase rule is met the way the label meets it: a row whose name is
    its house's word, read, beside its house's full name -- two or more words of it, read
    as a phrase on one line. The maker's phrase is the second word the name does not have.

    Only a flagship (`_own_vocabulary`: a name with no word that is not its maker's) and
    only a house named by two words or more: `Campari` under "Cutty Sark", the importer's
    other brand that TTB filed it beneath, is proven by no line of a Campari bottle.

    Two words of the house on one line, the label's own among them. The first cut asked
    for the house's name whole and in order, and the next scan never gave it: the small
    type under the wordmark came as "Davide Carpet MIL A", "Davide Cry M1 LA N", "Davide
    Copen" -- DAVIDE clean, CAMPARI and MILANO garbled past reading -- on every one of
    forty frames (2026-09-16). What those lines do carry is CAMPAR beside DAVIDE: the
    label's word and a word of its house that is not the label's, read together."""
    words = _identifying_tokens(resolved.product.name or "")
    if len(words) != 1 or len(words[0]) < _MIN_SELF_PROOF_CHARS:
        return False
    _, flagship = _own_vocabulary(resolved)
    if not flagship or not _read_as(words[0], read_toks):
        return False
    lines = [set(_tokens(d.text)) for d in detections]
    for m in _maker_names(resolved):
        house = _identifying_tokens(m)
        if len(house) < _MIN_SELF_PROOF_TOKENS:
            continue
        if any(sum(1 for w in house if _read_as(w, toks)) >= _MIN_SELF_PROOF_TOKENS
               for toks in lines):
            return True
    return False


# Two lines that read the same printed phrase are one piece of evidence, not two.
# Words a label prints that identify nothing: what is in the glass, what it is packaged in,
# and the suffixes companies carry. `_PRODUCER_SUFFIX` comes from dedup rather than a second
# list here, so "what counts as a company suffix" has one answer across ingest and resolve.
_SIGHTING_NOISE = (
    _PACKAGING
    | {w for words in _CATEGORY_WORDS.values() for w in words}
    | _PRODUCER_SUFFIX
    | {"the", "and", "with", "for", "from", "our"}
    # Strength and process qualifiers. These *can* distinguish two beers -- "Double Trouble"
    # is not "Trouble" -- so treating them as noise looks unsafe until you see where the veto
    # sits: the resolver has already picked the best row for this reading, and if the catalog
    # holds `Double Trouble` that is the row it returns. All this set decides is whether to
    # throw away `Trouble` when the catalog has nothing more specific, and a near row beats a
    # blank screen. The camera has spent this whole scan path returning nothing.
    | {"double", "imperial", "session", "unfiltered", "hazy", "juicy", "hopped",
       "barrel", "aged", "batch", "craft", "style", "premium", "classic", "natural"}
)
# Two letters is "oz", "by", "no" — a unit or a joiner, never the part of a label that
# distinguishes one beer from another.
_MIN_SIGHTING_TOKEN = 3


def _accounts_for_sighting(product: str, producer: str, sighting: str) -> bool:
    """Whether a catalog row is what was read off the label, or only part of it.

    `_accounts_for_the_line` cannot answer this, and the numbers say why. It measures the row
    against the whole reading — right for OCR, where the reading is the garbled thing and the
    row is the clean one. A model reading the picture inverts that: it returns the label the
    way the label is printed, maker and drink together, so `Heady Topper` scores 0.43 against
    "The Alchemist Heady Topper" while the fragment `Banger` scores 0.43 against "Focal
    Banger". One is the right answer and one is a different beer, and similarity cannot tell
    them apart at any threshold.

    Once the reading is clean the question is no longer how garbled it is but whether any of
    it is left unexplained. So: every identifying word read has to be accounted for by the
    product's name, by its producer's, or by being the sort of word every label prints.
    Nothing is left over from "The Alchemist Heady Topper". "Focal" is left over from "Focal
    Banger" against a row named `Banger` — and that leftover is the whole point.
    """
    in_name = set(_tokens(product))
    known = in_name | set(_tokens(producer))
    read = [t for t in _tokens(sighting) if len(t) >= _MIN_SIGHTING_TOKEN]
    # A reading made only of what every label prints names nothing, so nothing can account
    # for it. Without this a sighting of "IPA" was answered by a junk catalog row that is
    # literally named `Ipa Ipa` -- it shares the word, so every other check passed.
    if not [t for t in read if t not in _SIGHTING_NOISE]:
        return False
    # The row has to answer the drink, not just the brewery: without this a sighting of
    # "Sierra Nevada" would be accounted for by every beer they make.
    if not any(t in in_name for t in read):
        return False
    return not [t for t in read if t not in known and t not in _SIGHTING_NOISE]

_LINE_REREAD = 0.7
#: Shortest token a one-letter difference is allowed to reconcile in `_same_read`. Three
#: letters is where one substitution makes another real word of almost anything.
_SAME_READ_MIN = 4


def _same_read(a: str, b: str) -> bool:
    """Whether two tokens are one printed word read twice.

    Near-exact by trigram, as everywhere else -- or, for words of any length, one letter
    apart: substituted, dropped or added. Trigram similarity is harsh on short words, and
    the recognizer's mistakes are single letters. A can of Long Live Beerworks gave its
    script wordmark as "Fong files" and "Long fiRes" one tick apart; fong/long and
    files/fires are each a letter off and a 0.25 and 0.33 by trigram, so the two reads
    passed as independent lines, and between them they named `Long-fong` -- a Chinese
    spirit whose two words each happened to be one of the garbles (2026-09-15)."""
    if a == b or _trigram_sim(a, b) >= _TOKEN_SUPPORT_MIN:
        return True
    if len(a) < _SAME_READ_MIN or len(b) < _SAME_READ_MIN or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b, strict=True) if x != y) <= 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _independent_lines(line_tokens: list[list[str]]) -> list[list[str]]:
    """Collapse detections that are re-reads of one another.

    Corroboration is supposed to mean separate parts of the frame agreeing. A four-pack prints
    its brand once per can, so a single phrase arrives as three detections -- and whatever they
    happened to share got certified by its own echo. "LITTLE" read twice off a Little Willow
    pack proved six unrelated products with `little` in the name, and "DRINK FROM THE CAN!"
    read three times proved one called `Now & Then`, off the word THEN.

    Longest reading first, so the fullest version of a repeated phrase is the one kept: a line
    whose identifying tokens are nearly all already accounted for is an echo of it. Matching is
    fuzzy because each re-read is garbled differently -- "LITTLE WILLOW BREWING COMPANT" and
    "LITTLE / KEWING" are the same text off two cans -- and tolerant of a single letter, which
    is what the garble usually is (`_same_read`).
    """
    def _substance(toks: list[str]) -> int:
        return sum(len(t) for t in toks if len(t) >= _MIN_NAME_TOKEN_LEN)

    return [toks for toks, _ in _line_groups(line_tokens)]


def _line_groups(line_tokens: list[list[str]]) -> list[tuple[list[str], set[int]]]:
    """`_independent_lines` with its bookkeeping: each kept reading with the indices of the
    lines that are re-reads of it (a line with nothing identifying in it belongs to none)."""
    def _substance(toks: list[str]) -> int:
        return sum(len(t) for t in toks if len(t) >= _MIN_NAME_TOKEN_LEN)

    kept: list[tuple[list[str], set[int]]] = []
    # By identifying substance, not token count: "PDY TOPP" and "FADY-TOPP" both hold two
    # tokens, so counting them left the poorer reading first and the fuller one then looked
    # like new evidence rather than the same words read again. Two halves of one wordmark
    # certified `Snipes Mountain Lefty Topp's` off a Heady Topper can.
    order = sorted(range(len(line_tokens)), key=lambda i: _substance(line_tokens[i]),
                   reverse=True)
    for i in order:
        sig = [t for t in line_tokens[i] if len(t) >= _MIN_NAME_TOKEN_LEN]
        if not sig:
            continue
        echo = next((group for group in kept
                     if sum(any(_same_read(t, k) for k in group[0]) for t in sig) / len(sig)
                     >= _LINE_REREAD), None)
        if echo is None:
            kept.append((sig, {i}))
        else:
            # The echo's words join the line it re-reads, so the line keeps its best read of
            # each word. The fullest reading is not the cleanest: "WORMTOWNT" outranks
            # "WORMTOWN" on substance, and a kept line that only had the garble no longer
            # agreed with the beer at all; "DINO BREAN" was kept over "DINO BREAK" the
            # same way and lost the word that names it (2026-09-15).
            echo[0].extend(t for t in sig if t not in echo[0])
            echo[1].add(i)
    return kept


def _frame_support(vocab: list[str], line_tokens: list[list[str]], *,
                   category: str | None = None, hint: str | None = None) -> int:
    """How many distinct pieces of the frame agree with this candidate.

    Mostly that means detections naming it, using the same token-agreement test the per-line
    guard uses — so the answer is directly comparable and one threshold calibrates both.

    The label's category counts as one more, because that is what it is: another line of the
    frame agreeing. It is also the *reliably read* one. A stylized can whose brand OCRs as
    Cyrillic still prints "ALE / ALC. 8% BY VOL" in plain type, and that line is what separates
    `The Alchemist Heady Topper` (beer) from `Alchemist Amer` (other) when both come off the
    same brewery and the name match alone favours the wrong one.
    """
    n = 0
    if vocab:
        # Counted by *what* each line agrees on, not by line. Two lines that name the candidate
        # through the same words are the same printed phrase read twice, whatever the garble
        # around them: "ITHE CAN! DRINKER" and "SITHE CAN! DRINKER" differ only in how THE
        # misread, slipped past the re-read check on that difference, and certified `Day
        # Drinker` off a Heady Topper can by agreeing with each other about DRINKER. Keyed on
        # the candidate's words rather than the frame's so that differently garbled reads of
        # one word land on the same key. A line that reads a *new* word of the name is still
        # new evidence: HEADY beside HEADY TOPPER is two, DRINKER beside DRINKER is one.
        agreed: set[frozenset[str]] = set()
        for toks in line_tokens:
            words = frozenset(v for v in vocab
                              if any(_trigram_sim(v, t) >= _TOKEN_SUPPORT_MIN for t in toks))
            if words:
                agreed.add(words)
        # ...and a set that is part of another is the same phrase read worse, not a second one
        # agreeing. A Campari label read "MILANO BITTER" once and "MILANO TER" once, the
        # second with the last word truncated, and {milano, bitter} beside {milano} made two
        # lines naming `Gran Milano Bitter` -- a different maker's amaro (2026-09-14).
        n = sum(1 for a in agreed if not any(a < b for b in agreed))
    if hint and category and category == hint:
        n += 1
    return n


# A name too short to contain an identifying token has nothing for `_token_supported` to
# anchor on, so that guard waves it through and only the raised floor stands between it and any
# fragment that starts with the same letters. That is not enough, because the floor is measured
# with `word_similarity`, which asks whether the name appears *inside* the line — and a 3-letter
# name appears inside almost anything. A catalog row literally named `Ver` matched "VERMIKI",
# "VERMIL" and "VERM" off a Vermont can, all at ~1.0.
#
# Plain similarity is the right question for these, because it is the one measure that penalises
# what the name leaves out: 'ver' scores 1.00 against "VER" and 0.33 against "VERMONT".
_SHORT_NAME_SIM = 0.8


def _short_name_supported(query: str, name: str) -> bool:
    """Whether a very short name was actually *read*, rather than merely contained.

    A name with no letter tokens at all ("1664", "J&B") cannot be tested this way and defers to
    the raised floor, exactly as before — those are real products and must stay reachable."""
    name_toks = _tokens(name)
    if not name_toks:
        return True
    return any(_trigram_sim(nt, qt) >= _SHORT_NAME_SIM
               for nt in name_toks for qt in _tokens(query))


def _upc_variants(upc: str) -> list[str]:
    """A barcode's equivalent GTIN forms. A UPC-A (12 digits) and its EAN-13 form differ only by a
    leading zero and identify the *same* item, but a scanner and the catalog may store different
    forms — so a lookup tries both. EAN-8 and other lengths are used as-is."""
    u = (upc or "").strip()
    out = [u]
    if u.isdigit():
        if len(u) == 12:
            out.append("0" + u)             # UPC-A -> EAN-13
        elif len(u) == 13 and u.startswith("0"):
            out.append(u[1:])               # EAN-13 -> UPC-A
    return list(dict.fromkeys(out))         # de-dup, preserve order


_KEY_STRIP_RE = re.compile(r"[^a-z0-9]+")


def _norm_key(s: str) -> str:
    """Lowercase, strip accents, keep alphanumerics — digits included. Unlike `_tokens` (which
    keeps only letters, for word matching), the identity key must preserve numbers: "0.0%", "12
    ans", "Select 55" are real product distinctions, not noise. "Jupiler 0,0%" -> "jupiler 0 0"."""
    decomposed = unicodedata.normalize("NFKD", s or "")
    ascii_ = "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
    return _KEY_STRIP_RE.sub(" ", ascii_).strip()


def _identity_key(name: str, brand: str, pid: str) -> str:
    """One key per *real* product, so duplicate catalog records collapse into a single overlay.

    The catalog carries the same beer under multiple rows — different UPCs of one product (Lagunitas
    IPA ×2, Heineken ×7), or an OFF pull plus a TTB record. Keyed on the raw id they'd each draw
    their own overlay; keyed on normalized brand+name they merge. Brand is part of the key on
    purpose: OFF often names a product only by its class ("Blended Scotch Whisky" for eight
    different distilleries), and those must stay distinct — their brands (Johnnie Walker vs Queen
    Margot) are what separate them. A placeholder "Unknown" brand (or one that just echoes the name)
    carries no identity, so it drops out and the name alone keys — which is what lets the two
    Unknown-branded "Lagunitas IPA" rows collapse. Digits are kept, so an alcohol-free "0.0%" or an
    age-stated "12 ans" stays a distinct product from its sibling. A row with no alphanumeric name
    falls back to its id, so unnamed (e.g. non-Latin) rows never merge into one another."""
    n = _norm_key(name)
    if not n:
        return f"id:{pid}"
    b = _norm_key(brand)
    if not b or b == "unknown" or b == n:
        return n
    return f"{b}\x1f{n}"


@dataclass
class _Frame:
    """What `_resolve_lines` found: every supported candidate best-first, the ones the
    frame proves outright, and the lines nothing matched."""

    ranked: list[ScoredCandidate] = field(default_factory=list)
    proven: list[ScoredCandidate] = field(default_factory=list)
    unresolved: list[int] = field(default_factory=list)
    by_maker: set[str] = field(default_factory=set)     # proven by the wordmark contest


#: Lines each tracked object lends the scene when the objects are judged together, and how
#: many the scene may hold -- see `_resolve_scene`.
_SCENE_LINES_PER_OBJECT = 4
_SCENE_MAX_LINES = 16

#: How many rows an `ambiguous` verdict hands the client's fine stage.
_OBJECT_SHORTLIST = 5
# Lines of a tracked object judged. A tracked can accumulates every read of every line for
# as long as it stays in view, and the frame's work is lines times candidates: a Miller can
# twenty lines deep beside a fridge of stickers put one tick at 2.4 s, and a shelf of three
# bottles tracked as one object ran to eighty (2026-09-16). Twenty is a whole label and its
# neighbours; what a can has been read as eighty different ways is the same lines, garbled
# -- and `_object_lines` picks the ones that say something new.
_OBJECT_MAX_LINES = 20
#: Share of the reading's identifying words a row must explain to be worth a second look.
#: Strictly more than half: "NK FROM THEO BANGE" explains exactly half of `Theo P.` and
#: that row is a coincidence, not a shortlist.
_OBJECT_EXPLAINED = 0.5


def _object_lines(texts: list[str], cap: int) -> list[str]:
    """The lines of a tracked object worth judging, at most `cap` of them, in the order the
    client sent.

    The client sends a tracked object's lines most-seen first, and the most-seen lines are
    the ones the recognizer reads the same way every tick: the short words. A can's own
    line is long and stylized and never reads the same way twice, so each of its readings
    is seen once and sorts last -- the first cut kept the first twelve lines of a can of
    Dino Break and every one was WIDESCREEN, DUDE DUD or COLLECTIO off the fridge, with
    DINO BREAK and the brewery's line at seventeen and beyond, and the object was judged
    without the beer on it (2026-09-15, replayed 2026-09-16). Kept by what each line adds:
    greedily, the line with the most identifying words not yet kept, so the can's lines come
    before the fridge's echoes and a re-read that says nothing new comes last."""
    words = [set(_identifying_tokens(t)) for t in texts]
    seen: set[str] = set()
    chosen: list[int] = []
    left = list(range(len(texts)))
    while left and len(chosen) < cap:
        i = max(left, key=lambda j: (len(words[j] - seen), -j))
        chosen.append(i)
        left.remove(i)
        seen |= words[i]
    return [texts[i] for i in sorted(chosen)]


def _object_vocabulary(c: ScoredCandidate) -> tuple[str, str]:
    """(brand-qualified product name, producer name) — the words that would be printed on
    this product's label, the way `_accounts_for_sighting` wants them."""
    r = c.resolved
    return search_name(r.product.name, r.brand.name), r.producer.name


def _window_tokens(line: str, maker_tokens: list[str]) -> list[str]:
    """The words of a line a wordmark window may be made of: not chrome, not a category word,
    and not the maker's own name or a garble of it.

    Short fragments stay. A two-letter read is not a name and cannot make a window by itself
    (`_wordmark_score` refuses one), but it is part of the shape of the line it sits in: on a
    can of Heady Topper the wordmark arrived as "DY TOPP" five times in one session -- HEADY
    TOPPER with the first letters of each word lost -- and dropping the DY left one word
    against a two-word name, which is rightly no read at all. The window that reads that line
    is "dy topp", and it resembles `heady topper` at 0.31 and nothing else the maker brews."""
    return [t for t in _tokens(line)
            if t not in _SIGHTING_NOISE and not is_generic_token(t)
            and not any(_trigram_sim(t, m) >= _MAKER_TOKEN_SIM for m in maker_tokens)]


def _own_name_tokens(name: str, maker_tokens: list[str]) -> list[str]:
    """The words of a product's name that are the *beer's*: `_window_tokens` for a catalog
    name, where a fragment is not a word at all."""
    return [t for t in _window_tokens(name, maker_tokens) if len(t) >= _MAKER_PICK_MIN_TOKEN]


def _wordmark_score(line_tokens: list[list[str]], own: str, maker_tokens: list[str],
                    skip: frozenset[int] = frozenset(), min_width: int = 1,
                    whole_word: bool = True, prepared: bool = False) -> tuple[float, int, int]:
    """How much some window of the frame looks like this beer's own name: the similarity, the
    line it was on, and how many tokens the window had.

    Windows of one to three consecutive tokens, with the same exclusions as `_own_name_tokens`:
    the maker's line is evidence of the maker, and the fine print is evidence of nothing.
    Lines in `skip` are ones a product already accounts for in full -- on a shelf, BLUE MOON
    BELGIAN WHITE is Blue Moon's line, and the word WHITE in it is not evidence for a
    `Guinness White Ale`."""
    # A wordmark is stacked type, and the recognizer reads HEADY over TOPPER as one word when
    # the leading is tight: "OYTOPPER", "CYTOPPER", "ATOPPER" on nine frames of one scan. The
    # letters are the same either way, so a single token is compared to the name written as
    # one word -- but only a token with one of the name's words *whole* inside it. That is the
    # corroboration a second word would have given: "OYTOPPER" carries TOPPER, and is a 0.40
    # against "headytopper"; "FOCAILS" carries nothing of `focal banger` whole, and stays the
    # one-token-against-a-phrase garble the two-word window refuses.
    whole = [w for w in own.split() if len(w) >= _MIN_NAME_TOKEN_LEN]
    merged = own.replace(" ", "")
    best, at, width = 0.0, -1, 0
    for i, toks in enumerate(line_tokens):
        if i in skip:
            continue
        if not prepared:                 # `_pick_among` hands lines already stripped
            toks = _window_tokens(" ".join(toks), maker_tokens)
        for a in range(len(toks)):
            # A token that IS one of the name's words, and nothing more, is one word --
            # the corroboration it carries is the maker's line, so it counts only when
            # the maker was read outright (`whole_word`). "SAPPHIRE" beside ROMBAY picked
            # `Bombay sapphire murcian lemon` by shape off a bottle of the plain gin,
            # under the one-product importer that holds the stray row, which ROMBAY
            # merely resembled (2026-09-16); "TOPPER" under THE ALCHEMIST, read, is the
            # beer with its first word lost.
            if (min_width > 1 and len(toks[a]) > _MIN_NAME_TOKEN_LEN
                    and any(w in toks[a] and (whole_word or len(toks[a]) > len(w))
                            for w in whole)):
                sim = _trigram_sim(toks[a], merged)
                if sim > best:
                    best, at, width = sim, i, 1
            for b in range(a + min_width, min(len(toks), a + _MAKER_PICK_WINDOW) + 1):
                if not any(len(t) >= _MAKER_PICK_MIN_TOKEN for t in toks[a:b]):
                    continue        # "AL" off "ALC." is not a window; "DY TOPP" is
                if any(len(t) >= _MAKER_WINDOW_WORD_LEN
                       and max(_trigram_sim(t, w) for w in (*own.split(), merged))
                       < _MAKER_WINDOW_WORD_MIN for t in toks[a:b]):
                    continue        # a word of another name (see _MAKER_WINDOW_WORD_MIN)
                sim = _trigram_sim(" ".join(toks[a:b]), own)
                if sim > best:
                    best, at, width = sim, i, b - a
    return best, at, width


def _pick_among(items: list[dict], line_tokens: list[list[str]], maker_name: str,
                skip: frozenset[int] = frozenset(),
                other_makers: list[str] = (),
                maker_read: bool = True) -> tuple[dict, float, int] | None:
    """The one of a maker's beers whose name the wordmark garble resembles, by a margin.

    None when nothing in the frame looks like any of them, or when two look alike -- either
    is "the maker was read and the beer was not", which is what the caller already knew."""
    maker_tokens = [t for t in _tokens(maker_name) if len(t) >= _MAKER_PICK_MIN_TOKEN]
    # The frame's maker words are excluded from the *windows*; the beer's own name is stripped
    # only of its own maker's, so a sibling filed under a stray producer keeps its full name
    # and simply finds nothing left in the frame to match it.
    window_excl = list(maker_tokens) + list(other_makers)
    # The windows a line offers do not depend on the beer being scored: strip each line of
    # chrome and maker words once, not once per beer. A maker of forty beers over an
    # object of twenty lines was eight hundred passes of the same work (2026-09-16).
    windows = [_window_tokens(" ".join(toks), window_excl) for toks in line_tokens]
    scored: list[tuple[float, int, dict, bool]] = []
    for rec in items:
        own = _own_name_tokens(rec.get("name") or "", maker_tokens)
        if sum(len(t) for t in own) < _MIN_SELF_PROOF_CHARS:
            # A beer named after its maker -- "Bombay Sapphire London Dry Gin" is the maker
            # plus a style -- has no name of its own to be recognised by, and a frame that
            # read only the maker is consistent with it. The shape of a wordmark cannot
            # choose between it and its siblings, so it is not asked to: this was how East
            # displaced the plain gin on a bottle that never printed EAST.
            return None
        # A two-word name is read by a two-word window, literally: one word of "sapphire
        # murcian lemon" read exactly is a prefix, not a name, and scored 0.43 off a bottle
        # of the plain gin. A one-word name is read by whatever resembles it closely enough.
        sim, at, width = _wordmark_score(windows, " ".join(own), window_excl, skip,
                                         min_width=1 if len(own) == 1 else 2,
                                         whole_word=maker_read, prepared=True)
        scored.append((sim, at, rec, len(own) == 1))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best, at, rec, lone = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    # Two words agreeing on two words, or a near read of one: "ecan" off DRINK FROM THE CAN
    # resembled `pecan cream` at 0.21, and "FOCAILS" resembles `focal banger` at 0.24 -- one
    # token against a phrase, which the two-token window above no longer admits at all.
    floor = _MAKER_PICK_MIN_LONE if lone else _MAKER_PICK_MIN
    if best < floor or best - runner < _MAKER_PICK_MARGIN:
        return None
    return rec, best, at


def _accounts_for_object(c: ScoredCandidate, reading: str) -> bool:
    """`_accounts_for_sighting`, for a reading that is the camera's rather than a model's.

    A sighting is a name the model wrote down; an object reading is everything OCR saw on
    the can, fine print included — "INDIA PALE ALE", "AMERICAN DOUBLE IPA", "STOWE". The
    style vocabulary dedup shares with the store is therefore noise here too: a leftover
    "india" says nothing about which beer this is. The two checks that matter are kept
    exactly: a row must answer the drink, not just the brewery, and every identifying word
    read has to be the row's own or its maker's.
    """
    product, producer = _object_vocabulary(c)
    in_name = set(_tokens(product))
    known = in_name | set(_tokens(producer))
    read = [t for t in _tokens(reading)
            if len(t) >= _MIN_SIGHTING_TOKEN and t not in _SIGHTING_NOISE
            and not is_generic_token(t)]
    # The rule asks whether the row explains everything that was read, and a row explains one
    # word for free: "DRINK FROM THE CAN!" misread as DRINK FRONT is, once the chrome is gone,
    # the single word FRONT, and `Front Flips` accounted for it in full. A single word is not a
    # label here any more than it is in `_is_whole_label` -- it may match, it may not certify.
    if len(set(read)) < _MIN_SELF_PROOF_TOKENS:
        return False
    if not any(t in in_name for t in read):
        return False
    # And the reading has to account for the row: every word of the name that is the
    # drink's own read somewhere in it. The rule was written against a row that wins by
    # saying less than the label; a row that says *more* than the label won the same way --
    # "Long five", two words of a script wordmark, was explained in full by `Long Distance
    # High Five`, with DISTANCE and HIGH read nowhere (2026-09-15). A two-word reading and a
    # four-word name is not the label; it is a name the reading happens to fit inside. The
    # maker's words are not asked for: "Little Sip IPA" is the whole of what a Lawson's can
    # says about itself, and the row is filed as `Lawson's Finest Liquids Little Sip IPA`.
    if _unread(_own_vocabulary(c.resolved)[0], set(_tokens(reading))):
        return False
    return not [t for t in read if t not in known]


def _explains_enough(c: ScoredCandidate, reading: str) -> bool:
    """Whether a row explains enough of an object's reading to be shortlisted.

    Looser than `_accounts_for_sighting` — OCR leaves words over ("STOWE VERMONT" on a can
    whose maker is `The Alchemist`) — but not so loose that a one-word row can ride in on a
    single shared word: more than half of what was read has to be this label's, and one of
    the row's own name words of substance has to be among the words actually read.
    """
    product, producer = _object_vocabulary(c)
    read = [t for t in _tokens(reading)
            if len(t) >= _MIN_SIGHTING_TOKEN and t not in _SIGHTING_NOISE]
    if not read:
        return False
    name_words = _tokens(product)
    known = name_words + _tokens(producer)

    def seen(t: str, among: list[str]) -> bool:
        return t in among or any(_trigram_sim(t, k) >= _TOKEN_SUPPORT_MIN for k in among)

    read = [t for t in read if not is_generic_token(t)]
    # A shortlist is a choice, and one word gives the model nothing to choose by. The same
    # night `_accounts_for_object` learned that FRONT does not certify `Front Flips`, "DRINK
    # FRONT" shortlisted it alone and "MIST" shortlisted `Sno Mist` alone, and a model asked
    # to pick among one picked it -- the misfires back on the screen by the other door.
    if len(set(read)) < _MIN_SELF_PROOF_TOKENS:
        return False
    explained = [t for t in read if seen(t, known)]
    if len(explained) / len(read) <= _OBJECT_EXPLAINED:
        return False
    # And the label has to have shown most of the row's own name. The model chooses among
    # rows the label could be, and a name with two words the camera never saw is not one of
    # them: "Long five" shortlisted `Long Distance High Five` alone, and a model asked to
    # pick among one picks it (2026-09-15). One unread word is the garble the shortlist
    # exists for -- "FADY TOPPE" reads TOPPER and loses HEADY -- so one is allowed.
    if len(_unread(_own_vocabulary(c.resolved)[0], set(_tokens(reading)))) > 1:
        return False
    substantial = [t for t in _identifying_tokens(product)]
    return any(seen(t, substantial) for t in explained if len(t) >= _MIN_NAME_TOKEN_LEN)


class Resolver:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._producer_memo: dict[tuple[str, int], list[tuple[dict, float]]] | None = None
        self._hydrate_memo: dict[str, ResolvedProduct | None] | None = None
        self._catalog_memo: dict[tuple[str, int | None], list[dict]] | None = None

    # Matching is delegated to the store: token-overlap on the SQLite dev store,
    # real pg_trgm trigram similarity on Postgres — same signature either way.
    def _resolve_by_upc(self, upc: str) -> dict | None:
        for key in _upc_variants(upc):
            sku = self.store.get_gold(f"sku:{key}")
            if sku:
                return self.store.get_gold(sku["product_id"])
        return None

    def _hydrate(self, product_rec: dict) -> ResolvedProduct | None:
        # Remembered for the request (see `resolve`): a frame and the objects tracked over
        # it surface the same rows, and each hydration is two store reads.
        memo = self._hydrate_memo
        key = product_rec.get("id") if isinstance(product_rec, dict) else None
        if memo is not None and key is not None and key in memo:
            return memo[key]
        out = self._hydrate_uncached(product_rec)
        if memo is not None and key is not None:
            memo[key] = out
        return out

    def _hydrate_uncached(self, product_rec: dict) -> ResolvedProduct | None:
        # A merged-away row leaves a tombstone under its old id ({"id", "redirects_to"}), and
        # anything holding that id -- a SKU, a cached candidate, a client replaying an old
        # answer -- still hands it here. Follow it to the row that now holds the product
        # rather than validating a tombstone as a Product, which raises and 500s the scan.
        hops = 0
        while isinstance(product_rec, dict) and product_rec.get("redirects_to"):
            if hops >= 8:
                return None                      # cyclic or absurdly long chain
            nxt = self.store.get_gold(product_rec["redirects_to"])
            if not isinstance(nxt, dict) or nxt.get("id") == product_rec.get("id"):
                return None
            product_rec, hops = nxt, hops + 1
        if not isinstance(product_rec, dict) or not product_rec.get("name"):
            return None                          # not a product row; nothing to resolve
        producer = self.store.get_gold(product_rec.get("producer_id", ""))
        brand = self.store.get_gold(product_rec.get("brand_id", ""))
        if producer is None:
            producer = Producer(id="unknown", name="Unknown").model_dump(mode="json")
        if brand is None:
            brand = Brand(id="unknown", producer_id=producer["id"],
                          name=product_rec.get("name", "")).model_dump(mode="json")
        return ResolvedProduct(
            product=Product.model_validate(product_rec),
            producer=Producer.model_validate(producer),
            brand=Brand.model_validate(brand),
        )

    def _by_producer(self, lines: list[tuple[int, str]],
                     hint: str | None) -> list[tuple[int, dict, float]]:
        """Products inferred from the maker the label names, when it never named a drink.

        Reached only for a frame nothing corroborates, and discounted, so this competes with
        the nothing it would otherwise return — never with a product the label actually says.

        The category hint is what makes it safe. "A CHEMIST VER" off a Heady Topper can is a
        better trigram match for a distillery named `Chemist` (1.00) than for `Alchemist`
        (0.60), because OCR dropped the "AL" and word-similarity cannot recover a lost prefix.
        No name threshold separates those. What does is the fine print the same frame read
        perfectly: the can says ALE, `Chemist` makes 17 spirits, and `Alchemist` makes the beer.
        """
        match = getattr(self.store, "match_producers", None)
        products_of = getattr(self.store, "products_of", None)
        if match is None or products_of is None:
            return []                       # a store without the producer path; not an error
        out: list[tuple[int, dict, float]] = []
        seen: set[str] = set()
        for i, text in lines:
            # A line with nothing identifying in it cannot name a maker either. Without this
            # the two generic halves agree with each other: _token_supported accepts a
            # styleless name against a styleless line, so "DRINK FROM" -- now that both words
            # are known chrome -- reached a producer literally registered as "drink drink!"
            # and offered its beer at 0.60.
            if not _identifying_tokens(text) or len(_tokens(text)) > _MAKER_LINE_MAX_TOKENS:
                continue
            for prod, sc in self._producers(text):
                pid = prod.get("id") or ""
                if sc < _PRODUCER_MATCH_MIN or pid in seen:
                    continue
                # Same coincidental-window guards the product path uses: the producer's name
                # must actually be a word in the line, not a trigram accident — and a very
                # short one must have been read rather than merely contained. A producer
                # literally named `Ver` was reached from "VERMIKI" and "VERMONT" here after
                # the product path had already been taught not to.
                pname = prod.get("name") or ""
                if not _token_supported(text, pname):
                    continue
                if len(pname) < _SHORT_NAME_LEN and not _short_name_supported(text, pname):
                    continue
                seen.add(pid)
                items = self._products_of(pid)
                if hint:
                    items = [p for p in items if (p.get("category") or "") == hint]
                if not items or len(items) > _PRODUCER_MAX_PRODUCTS:
                    # A maker with a whole shelf has been identified; a drink has not. Saying
                    # which one would be inventing it.
                    continue
                for rec in items:
                    out.append((i, rec, round(sc * _PRODUCER_EVIDENCE, 3)))
        return out

    def _by_wordmark(self, lines: list[tuple[int, str]], hint: str | None,
                     line_tokens: list[list[str]], claimed: frozenset[int],
                     ) -> tuple[list[tuple[int, dict, float]], set[str]]:
        """The maker's beer, told from its siblings by the shape of an unreadable wordmark.

        Separate from `_by_producer` and additive to it. That path *enumerates* a small
        maker's products as discounted candidates for the frame to sort out; this one tries
        every maker a line so much as resembles -- the maker line is garbled too, THE
        ALCHEMIST arriving as "CHEMIST-VER" sixty times for every four clean reads, and the
        producer guards rightly refuse that as a read -- and keeps a maker only when the
        wordmark then names one of its beers by a margin (`_pick_among`). Two weak reads that
        agree are one strong read; a resemblance that agrees with nothing stays nothing. The
        beer picked is returned in the second value as *proven*: the maker line and the
        wordmark line are two parts of the frame agreeing, the second by shape rather than by
        letters. This is how a can whose name OCR cannot read resolves without a photo
        leaving the phone.
        """
        match = getattr(self.store, "match_producers", None)
        products_of = getattr(self.store, "products_of", None)
        picked: set[str] = set()
        if match is None or products_of is None:
            return [], picked
        # Every maker any line resembles, best hypothesis per maker across the frame's lines
        # (a maker read outright on one line and merely resembled on another is read).
        best: dict[str, tuple[int, str, dict, float, bool]] = {}
        for i, text in lines:
            ident = _identifying_tokens(text)
            if not ident:
                continue
            queries = [t for t in ident if len(t) >= _MAKER_HYPOTHESIS_WORD]
            if len(_tokens(text)) <= _MAKER_LINE_MAX_TOKENS:
                queries.insert(0, text)
            for q in queries:
                for prod, sc in self._producers(q, limit=_MAKER_HYPOTHESES):
                    pid = prod.get("id") or ""
                    pname = prod.get("name") or ""
                    if sc < _MAKER_HYPOTHESIS_MIN or not _affix_read(text, pname):
                        continue
                    # Whether the *line* reads as this maker, whichever query found it.
                    read = (sc >= _PRODUCER_MATCH_MIN and _token_supported(text, pname)
                            and (len(pname) >= _SHORT_NAME_LEN
                                 or _short_name_supported(text, pname)))
                    prev = best.get(pid)
                    if prev is None or (read, sc) > (prev[4], prev[3]):
                        best[pid] = (i, text, prod, sc, read)
        # The words of a maker the frame *read* are maker evidence for every pick, not only
        # that maker's. BOMBAY SAPPHIRE read the maker outright, and a one-product producer
        # registered as "Bombay spirits" holds a stray row for the Murcian Lemon -- whose own
        # name, under *that* maker, kept the word "sapphire". The window "sapphire santed"
        # then named it off a bottle of the plain gin, fifteen times.
        #
        # Read means read as a phrase: two or more of the maker's own words in the frame,
        # the same standard a label is held to (`_MIN_SELF_PROOF_TOKENS`). The catalog has
        # producers named after beers -- a permit filed as `Topper's`, another as `Heady
        # Topper` -- and on one word, "topper" off the wordmark DY TOPPER, they too were
        # "read", and excluding their words let a hypothesis that found no beer veto the one
        # that did: the can drew nothing under CHEMIST-VERMONT. One word is a hypothesis; a
        # phrase is a maker.
        frame_toks = {t for toks in line_tokens for t in toks}
        maker_words: list[str] = []
        for _, _, prod, _, read in best.values():
            if not read:
                continue
            words = [t for t in _tokens(prod.get("name") or "") if len(t) >= _MAKER_PICK_MIN_TOKEN]
            # Present the way a re-read is present (`_same_read`): a tracked bottle of the
            # plain gin gave BOMBAY as ROMBAY, GOMBAY and ÇOMBAY and never once whole, so
            # the maker counted as read on SAPPHIRE alone, its words stayed in the windows,
            # and "sapphire bor" picked the Murcian Lemon under the importer (2026-09-16).
            present = [w for w in _identifying_tokens(prod.get("name") or "")
                       if any(_same_read(w, r) for r in frame_toks)]
            if len(set(present)) >= _MIN_SELF_PROOF_TOKENS:
                maker_words += words
        out: list[tuple[int, dict, float]] = []
        for i, _, prod, _, read in best.values():
            items = self._products_of(prod.get("id") or "", limit=_MAKER_PICK_MAX_PRODUCTS + 1)
            if hint:
                items = [p for p in items if (p.get("category") or "") == hint]
            if not items or len(items) > _MAKER_PICK_MAX_PRODUCTS:
                continue
            pick = _pick_among(items, line_tokens, prod.get("name") or "", claimed | {i},
                               maker_words, maker_read=read)
            if pick is None:
                continue
            rec, shape, at = pick
            picked.add(rec.get("id") or "")
            # Anchored to the wordmark's line, which is where the beer's name is; scored as
            # maker evidence lifted by how well the shape was read.
            out.append((at if at >= 0 else i, rec,
                        round(_PRODUCER_EVIDENCE + (1 - _PRODUCER_EVIDENCE) * shape, 3)))
        return out, picked

    def _products_of(self, producer_id: str, limit: int | None = None) -> list[dict]:
        """`store.products_of`, remembered for the request like `_producers`: the same
        makers are hypothesised from every object tracked over a frame, and each catalog
        hydrated is a store read per row."""
        key = (producer_id, limit)
        memo = self._catalog_memo
        if memo is not None and key in memo:
            return memo[key]
        out = (self.store.products_of(producer_id) if limit is None
               else self.store.products_of(producer_id, limit=limit))
        if memo is not None:
            memo[key] = out
        return out

    def _producers(self, text: str, limit: int = 3) -> list[tuple[dict, float]]:
        """`store.match_producers`, remembered for the request: a frame's lines recur in
        every object tracked over them, and the maker paths ask about each line and each
        word of it. The memo lives for one `resolve()` (see there)."""
        key = (text, limit)
        memo = self._producer_memo
        if memo is not None and key in memo:
            return memo[key]
        out = self.store.match_producers(text, limit=limit)
        if memo is not None:
            memo[key] = out
        return out

    def _match_lines(self, texts: list[str]) -> list[list[tuple[dict, float]]]:
        """A frame's name matches, concurrently where the store can. The fallback keeps any
        store that only implements the single-line `match_products` working unchanged."""
        many = getattr(self.store, "match_products_many", None)
        if many is not None:
            return many(texts)
        return [self.store.match_products(t) for t in texts]

    def _qualified_name(self, rec: dict) -> str:
        """"<brand> <name>" for a product row, matching what `search_name` stores.

        The catalog splits a label in two, so half of it is invisible to any check that
        reads only `name`. Resolved through the store rather than a join because there are
        at most a handful of candidates per detection.
        """
        brand_id = rec.get("brand_id")
        brand = self.store.get_gold(brand_id) if brand_id else None
        return search_name(rec.get("name") or "", (brand or {}).get("name"))

    # ---- scoring ----
    def score(self, product: Product, profile: TasteProfile | None) -> tuple[float, str, bool]:
        """Predicted 0-1 enjoyment + a one-line reason + cold_start flag.

        Cold start = we scored it from chemistry/style alone, no reviews needed. That is
        the differentiator, so we flag and surface it.
        """
        sensory = product.sensory
        cold_start = sensory is not None and sensory.source.value in (
            "chemistry_prior", "style_prior", "llm_profile"
        )
        if profile is None or profile.sensory_ideal is None or sensory is None:
            # No personalization yet: fall back to a mild style-affinity prior.
            style = (product.style.value if product.style else "") or ""
            aff = (profile.style_affinities.get(style, 0.0) if profile else 0.0)
            return (0.5 + 0.5 * aff, "based on style", cold_start)

        sim = _cosine(sensory.to_array(), profile.sensory_ideal.to_array())
        score = max(0.0, min(1.0, 0.5 + 0.5 * sim))
        return (round(score, 3), _match_reason(score, sensory, profile.sensory_ideal), cold_start)

    # How many catalog rows one clean reading is allowed to consider. The first is usually
    # right; the rest matter when a shorter row wins by saying less — `word_similarity` is 1.0
    # for ANY name wholly inside the query, so a product literally named `Lawson's` scores a
    # perfect 1.00 against "Lawson's Sip of Sunshine", as `Tree House` does against Julius and
    # `Green` against Green City. Depth is what separates those: measured against the live
    # catalog, "Lawson's Sip of Sunshine" puts eleven fragments and near-misses -- `Lawson's`,
    # `Sunshine`, `Sunshiner`, `Laws`, `Sip Of Sunshine IPA` -- above the row it actually names,
    # which arrives twelfth. The rows are already fetched by one indexed query, so looking at
    # twenty-four of them costs a comparison each, not a lookup each.
    _READING_DEPTH = 24

    def resolve_reading(self, reading: str, *, index: int = 0,
                        profile: TasteProfile | None = None) -> ScoredCandidate | None:
        """The catalog row a *clean* reading of a label names, or None.

        Separate from `resolve` because the input is different in kind, not merely cleaner.
        `resolve` is built for OCR — fragmentary, garbled, several lines of one object — where
        containment scoring and one winner per line are the right instruments. A name read off
        the picture is a query, and the question is which row it is a reading *of*: the row has
        to account for what was read, not merely appear inside it.
        """
        for rec, raw in self.store.match_products(reading, limit=self._READING_DEPTH):
            resolved = self._hydrate(rec)
            if resolved is None:
                continue
            if not _accounts_for_sighting(resolved.product.name,
                                          resolved.producer.name, reading):
                continue
            personal, why, cold = self.score(resolved.product, profile)
            return ScoredCandidate(detection_index=index, resolved=resolved,
                                   match_score=round(min(1.0, float(raw)), 3),
                                   personal_score=personal, reason=why, cold_start=cold)
        return None

    def resolve(self, req: ScanResolveRequest,
                profile: TasteProfile | None = None) -> ScanResolveResponse:
        """Resolve a whole frame — its lines, and its tracked objects.

        Lines go through `_resolve_lines`, the frame-level corroboration that decides what a
        set of OCR lines photographed together actually names. Objects go through
        `resolve_object`, which runs the same judgement over one can's worth of lines and
        turns it into a verdict the HUD can act on without reading the candidates.
        """
        detections = [d.model_copy(update={"text": _latin(d.text)}) if d.kind != "barcode" else d
                      for d in req.detections]
        self._producer_memo, self._hydrate_memo, self._catalog_memo = {}, {}, {}
        try:
            return self._resolve(req, detections, profile)
        finally:
            self._producer_memo = self._hydrate_memo = self._catalog_memo = None

    def _resolve(self, req: ScanResolveRequest, detections: list[DetectedText],
                 profile: TasteProfile | None) -> ScanResolveResponse:
        frame = self._resolve_lines(detections, req.include_score, profile)
        corroborated = bool(frame.proven)
        # A frame nothing corroborates has no evidence to rank a list with, so offering one
        # implies a differentiation we cannot make. Measured over 78 such frames from a real
        # can: the right answer was first once, deeper never, and absent 77 times -- while the
        # frames carried two, three and five candidates each. They were not competing readings
        # of the label, they were the same wrong guess spelled five ways ("Chemist", "Chemist
        # 151", "Chemist Spirits", "Chemist Bierbrand"). One guess is as much as this frame has
        # earned the right to say, and the client is about to ask the model anyway.
        #
        # Corroboration is a property of a candidate, but it was only ever applied to the
        # frame -- so the unproven candidates rode in on the proven one's coat-tails. Three
        # four-packs in view is a frame that legitimately corroborates *something*, and that
        # opened the gate for every junk match beside it: reported from the camera as "a
        # number of answers stacked on top of each other", with the right answer behind
        # them. A shelf of real products still returns all of them -- each proves itself.
        candidates = list(frame.proven) if corroborated else frame.ranked[:1]

        objects = [self.resolve_object(o, profile, req.include_score, req.min_match_score)
                   for o in req.objects]
        if not corroborated and not any(r.status == "resolved" for r in objects):
            scene = self._resolve_scene(req.objects, profile, req.include_score)
            if scene is not None:
                objects = [scene if r.object_id == scene.object_id else r for r in objects]
        if frame.proven:
            objects = [r if r.status == "resolved" else self._inherited(o, r, frame.proven)
                       for o, r in zip(req.objects, objects, strict=True)]
        settled = [res.candidates[0] for res in objects if res.status == "resolved"]
        if settled:
            # An object's verdict makes the response corroborated, and the client draws a
            # corroborated response whole -- so the frame's one unproven guess, kept above
            # only because the model was about to be asked, would go up beside the verdict
            # as if it were one too. On a can of Heady Topper the tracked object settled on
            # the beer while the live lines guessed `Vermont Pale Lager` off VERMONT, and
            # both were drawn (2026-09-11). The verdict is the answer; the guess was for a
            # question that is no longer being asked.
            if not corroborated:
                candidates = []
            # And a verdict speaks for the lines it was reached over. The frame path is the
            # shelf's: every product that proves itself is drawn, because on a shelf each
            # label gets one line and nothing to agree with it. Run over one bottle's lines
            # it draws the bottle's siblings too. A bottle of Bombay Sapphire tracked as an
            # object settled on the gin, and the same tick's lines -- the same lines, the
            # object's own -- proved `East Vapour Infused London Dry Gin` off INFUSED and
            # `Bombay citron pressé` off the maker line, each by the words it shares with the
            # label and none by the word (EAST, CITRON) that would have told it apart. Three
            # names on one bottle (2026-09-15). The object judged those lines together and
            # chose; a line the object owns has been answered, and a second reading of it is
            # not a second product.
            #
            # Owns: the lines that carry a word of the verdict's label. The tracker follows
            # a screen region, and across a pan of the shelf one object gathered CAMPARI,
            # then RAMAZZOTTI, then BLACK SEAL BERMUDA BLACK RUM, and settled on Campari --
            # which was right for its lines and wrong for the Gosling's it then owned: the
            # rum's line was in the bag, so the frame's own proof of the rum was dropped
            # (2026-09-16). A verdict speaks for the lines that are its label's; a line that
            # shares no word with it is another bottle's, whatever box it was read in.
            by_id = {r.object_id: r for r in objects if r.status == "resolved"}
            owned: set[str] = set()
            for o in req.objects:
                res = by_id.get(o.id)
                if res is None:
                    continue
                vocab = _candidate_vocabulary(res.candidates[0].resolved)
                for t in o.texts:
                    if _agreeing_lines(vocab, [_tokens(_latin(t))]):
                        owned.add(_flatten(_latin(t)))
            owned -= {""}
            candidates = [c for c in candidates
                          if not (0 <= c.detection_index < len(detections)
                                  and _flatten(detections[c.detection_index].text) in owned)]
            candidates += settled
            corroborated = True
        return ScanResolveResponse(
            candidates=candidates,
            unresolved_indices=frame.unresolved,
            objects=objects,
            # Agreement across the frame, or — where there was no second line to agree with
            # — a strong read of the only line there was. Mirrors the penalty above: a lone
            # clean "BOMBAY SAPPHIRE LONDON DRY GIN" is not weak evidence, it is the whole
            # label, and asking the model about it would spend a second to confirm a 1.00.
            corroborated=corroborated,
        )

    def _resolve_lines(self, detections: list[DetectedText], include_score: bool,
                       profile: TasteProfile | None) -> _Frame:
        """What a set of lines photographed together names — every candidate the frame
        supports, best-first, and which of them the frame *proves*.

        A label is one object photographed once, so its lines are evidence about the *same*
        product and are strongest read together. Two passes: per line, every candidate that
        clears the existing guards; then per candidate, what the rest of the frame says about
        it. The guards are unchanged — they decide what may be evidence at all — and the frame
        decides which evidence wins.
        """
        line_tokens = [_tokens(d.text) for d in detections]
        # What corroboration is allowed to count: the frame's distinct readings, not its echoes.
        groups = _line_groups(line_tokens)
        independent = [toks for toks, _ in groups]
        # ...and which reading each line is: two re-reads of one line back a row once.
        reading_of = {i: g for g, (_, members) in enumerate(groups) for i in members}
        identity_lines = sum(1 for d in detections if _is_identity_text(d.text))
        hint = _category_hint(detections)
        frame_kinds = {k for d in detections for k in _kinds(d.text)}

        # ---- pass 1: every candidate any line supports, not just that line's best ----
        # Keeping only the top hit per line is what let chrome crowd out the beer: the real
        # product could be a line's second candidate and never be considered at all.
        hits: list[tuple[int, dict, float]] = []          # (line, record, that line's score)
        by_upc: set[str] = set()                         # records a barcode identified outright
        resolved_lines: set[int] = set()
        to_match: list[int] = []                          # lines worth a name query
        for i, det in enumerate(detections):
            if det.kind == "barcode":
                rec = self._resolve_by_upc(det.text)
                if rec is not None:
                    hits.append((i, rec, 1.0))
                    resolved_lines.add(i)
                    by_upc.add(rec.get("id") or "")
                continue
            if not _is_identity_text(det.text) or not _worth_matching(det.text):
                # A bare number/fragment is label chrome, not a name — and so is a line made
                # only of category and packaging words. Skipped rather than trigram-matched.
                continue
            to_match.append(i)

        # Ask for the frame's lines together. Each one costs a GIN scan sized by how common
        # its trigrams are, so in series a six-line can spends ~2s against a 700ms HUD tick;
        # the Postgres store runs them concurrently and the frame costs about its slowest
        # line instead of their sum.
        for i, found in zip(to_match, self._match_lines(
                [detections[i].text for i in to_match]), strict=True):
            det = detections[i]
            for rec, sc in found:
                # Judge the evidence on the brand-qualified name, because that is what the
                # label actually says. A row named "Irish Whiskey" is anonymous on its own;
                # as "Jameson Irish Whiskey" it is the product the line names.
                name = self._qualified_name(rec)
                # Low-information either way: too short to be distinctive, or built only from
                # category words. Both trigram-match label chrome far too easily, so they must
                # clear a near-exact bar rather than the normal floor.
                too_short = len(name) < _SHORT_NAME_LEN
                low_info = too_short or not _identifying_tokens(name)
                floor = _SHORT_MIN_MATCH if low_info else _MIN_MATCH
                if sc < floor or not _token_supported(det.text, name):
                    continue
                # A very short name additionally has to have been read, not just contained.
                if too_short and not _short_name_supported(det.text, name):
                    continue
                hits.append((i, rec, sc))
                resolved_lines.add(i)
        # ...and for what the lines name between them. A name printed across two lines is
        # on neither: `Goslings Black Seal` was fifth against BLACK SEAL 80 PROOF BERMUDA
        # BLACK RUM and sixth against "Goslings / Since 1806", and never a candidate; the
        # two-word filing of the same rum was, and once it was merged away the bottle drew
        # a gin called `Black Seal` (2026-09-15). The index ranks rows by the frame's
        # tokens together and hands each back with the line it reads best on, and that hit
        # is held to the same guards as any other.
        #
        # And to one more: every identifying word of the name read, somewhere in the frame.
        # A line's own match may carry an unread word, because one garbled line can still
        # prove a name; a candidate that exists only because the frame's words *together*
        # name it has no such excuse. Without this the wider net drew `Taft's Paint The Town
        # Hoppy` off a Wormtown can (TOWN and HOPPY read, TAFT'S nowhere), `Aslin Beer Co
        # This Shake Is Bananas` off two fragments, and a milkshake IPA off the words MILK
        # and VANILLA on a Miller can.
        match_frame = getattr(self.store, "match_frame", None)
        if match_frame is not None and to_match:
            read = {t for i in to_match for t in line_tokens[i]}
            for rec, j, sc in match_frame([detections[i].text for i in to_match],
                                          limit=_FRAME_CANDIDATES):
                i = to_match[j]
                name = self._qualified_name(rec)
                too_short = len(name) < _SHORT_NAME_LEN
                low_info = too_short or not _identifying_tokens(name)
                floor = _SHORT_MIN_MATCH if low_info else _MIN_MATCH
                if sc < floor or not _token_supported(detections[i].text, name):
                    continue
                if too_short and not _short_name_supported(detections[i].text, name):
                    continue
                if _unread(_identifying_tokens(name), read):
                    continue
                hits.append((i, rec, sc))
                resolved_lines.add(i)
        # Distinct lines backing each record, read straight off the hits — enough to tell
        # whether the frame agreed on anything, without paying to hydrate first.
        #
        # Distinct readings, not lines: "CHEMIS T-VER" and "CHEMIST-VERN" are one rim of one
        # can read twice, and a distillery named `Chemist` backed by both looked like the
        # frame agreeing on something -- so the maker was never asked, and the Heady Topper
        # under that rim was never found (2026-09-16).
        backing: dict[str, set[int]] = {}
        for i, rec, _ in hits:
            backing.setdefault(rec.get("id") or "", set()).add(reading_of.get(i, -1 - i))
        by_maker: set[str] = set()
        if not any(len(v) >= _MIN_FRAME_FOR_PENALTY for v in backing.values()):
            # Nothing the frame corroborates: the label has not named a product to us. Ask who
            # made it before giving up — on a stylized can the maker is the readable half.
            maker_lines = [(i, detections[i].text) for i in to_match]
            hits += self._by_producer(maker_lines, hint)
            # Lines some product already accounts for in full are that product's: on a
            # shelf, BLUE MOON BELGIAN WHITE is Blue Moon's line, and the word WHITE in it
            # is not evidence for a `Guinness White Ale`.
            claimed = frozenset(
                i for i, rec, _ in hits
                if _accounts_for_the_line(self._qualified_name(rec), detections[i].text,
                                          threshold=_SELF_PROOF_SIM))
            wordmark_hits, by_maker = self._by_wordmark(maker_lines, hint, line_tokens, claimed)
            hits += wordmark_hits
            resolved_lines.update(i for i, _, _ in hits)

        unresolved = [i for i in range(len(detections)) if i not in resolved_lines]

        # ---- pass 2: ask the whole frame about each distinct candidate ----
        best_hit: dict[str, tuple[int, float, dict]] = {}   # record id -> best (line, score)
        qualified_by_id: dict[str, str] = {}
        for i, rec, sc in hits:
            rid = rec.get("id") or ""
            prev = best_hit.get(rid)
            if prev is None or sc > prev[1]:
                best_hit[rid] = (i, sc, rec)
            qualified_by_id.setdefault(rid, self._qualified_name(rec))

        def _is_whole_label(resolved: ResolvedProduct, raw_score: float, line_i: int) -> bool:
            """True when the line this candidate matched is its name and essentially nothing
            else -- proof on its own, needing no second line to agree.

            Shared by the penalty and the corroboration test below because they are the same
            judgement, and when they were written separately they contradicted each other: the
            penalty pushed the score under the very bar the proof required.
            """
            rid, name = resolved.product.id, resolved.product.name
            qualified = qualified_by_id.get(rid, name)
            if sum(len(t) for t in _identifying_tokens(qualified)) < _MIN_SELF_PROOF_CHARS:
                return False
            # A single word is not a label. The character floor above and the similarity bar
            # below were each raised against the one-word coincidence -- `Bale`, `Mist`,
            # "CHEMIST-VE" at 0.73 -- and each time the next one read *exactly*: "CHEMIST"
            # off the tail of THE ALCHEMIST is a 1.00 against a distillery called `Chemist`,
            # and "DeadEye", the on-device model's tidying of a garbled HEADY, is a 1.00
            # against a rum called `Deadeye`. No threshold separates an exact read of a word
            # from an exact read of a word. What separates them is that a label is a phrase:
            # its words corroborate each other, and one word has nothing beside it to agree.
            # Measured over the 882 frames in the scan log, a one-word line proved a row 14
            # times and was right 0 -- and "STONE IPA", the least substantial label the
            # recogniser is meant to know, is two words and still proves itself.
            if len(_tokens(detections[line_i].text)) < _MIN_SELF_PROOF_TOKENS:
                return False
            # Nor is a piece of a line. A tracked object accumulates its reads, and among the
            # reads of BLACK SEAL 80 PROOF BERMUDA BLACK RUM was "BLACK SEA" -- the first line
            # with its last letter lost, two words, and a 1.00 against a spirit called `Black
            # Sea` (2026-09-14). A line whose text sits inside another line of the frame is
            # that line read short, and the whole label is the fuller read.
            flat = _flatten(detections[line_i].text)
            if any(j != line_i and flat and flat in _flatten(d.text) and flat != _flatten(d.text)
                   for j, d in enumerate(detections)):
                return False
            # And the line has to read a word that is the drink's own, not only its maker's.
            # A row named for its brewery and one word more -- `Toppling Goliath Brewing Co.
            # Mozee` -- is most of the brewery's line by similarity, and the brewery's line
            # alone, TOPPLING GOLIATH BREWING CO., proved it at 0.80 off a can of Dino Break
            # whose own line was out of view (2026-09-16). A flagship named for its house
            # has no such word to ask for and is held to the whole name as before.
            own, flagship = _own_vocabulary(resolved)
            if not flagship and not any(_read_as(w, set(line_tokens[line_i])) for w in own):
                return False
            if raw_score >= _STRONG_MATCH and _accounts_for_the_line(
                    qualified, detections[line_i].text, threshold=_SELF_PROOF_SIM):
                return True
            # Or the line reads the whole *name*, fine print and all. A label is one block of
            # type to the recognizer -- "BREWING COMPANY MILWAUKEE PREMIUM Miller BREWED HIGH
            # LIFE EST 1903 The Champagne of Beers 12 FLUID OUNCES" is one line -- and against
            # a line like that no name is ever most of the text. But every identifying word of
            # `Miller High Life` is in it. Two or more of the name's own words, all read on one
            # line, is the phrase rule met a second way; a name with one such word ("Colors",
            # `Black Sea` once "sea" is too short to count) is not helped by it.
            #
            # Read the way the recognizer reads, letters lost at one end allowed (`loose`):
            # RAMAZZOTTI arrived as RAMAZZON, RAMAZZOT, RAMAZZOI on thirty frames and never
            # once whole -- the TTI is where the wordmark's type thins -- and "1815 RAMAZZON
            # Aperiti Rosato" is the whole name in order, each word short a letter or two.
            # Nothing was drawn (2026-09-16). Five letters is the floor a lost-letter read
            # needs, so "BLACK SEA" still does not read `Black Seal`.
            return _reads_the_name(qualified, detections[line_i].text, loose=True)

        read_toks = {t for toks in line_tokens for t in toks}

        def _corroboration(resolved: ResolvedProduct, vocab: list[str],
                           readings: list[list[str]]) -> int:
            """How many of the frame's distinct readings name this product -- the count
            without the category's point (see the call)."""
            named = _frame_support(vocab, readings)
            # And lines that agree only on the maker's words have named the maker. A can of
            # Long Live Beerworks read LIVE on one line and LONG FIRES on another, and those
            # two named `Long Live Beerwoks Hola Fantasma` -- one of the brewery's beers, the
            # one the store's top three for "JONG LIVE" happened to hold, with HOLA and
            # FANTASMA read nowhere (2026-09-15). Evidence every sibling shares equally is
            # evidence for the maker, and one piece of it: the second line has to read a word
            # that is this beer's and not its siblings'.
            if named > 1:
                own, flagship = _own_vocabulary(resolved)
                # A row whose brand is its whole label has no maker to set its words apart
                # from, so any two of them agreeing would do -- and LONG and LIVE, the
                # brewery's name read on two lines, proved `Long Live Local Honey Brown
                # Lager`, a Pennsylvania beer that happens to start with the same two words,
                # LOCAL and HONEY read nowhere. Two lines make such a row's case only when,
                # between them and the rest of the frame, every word of it was read: that is
                # what MILLER beside HIGH LIFE has, and what GOSLINGS beside BLACK SEAL has
                # for `Goslings Black Seal` and not for `Goslings Gold Seal`.
                whole = not flagship or _reads_every_word(resolved.product.name or "", read_toks)
                if not whole or not _frame_support(own, readings):
                    named = 1
            return named

        scored: list[tuple[int, ScoredCandidate]] = []
        named_by_id: dict[str, int] = {}
        evidence: dict[str, tuple[ResolvedProduct, list[str]]] = {}
        whole_label: dict[str, bool] = {}
        by_house: set[str] = set()                        # one-word labels their house proved
        for line_i, sc, rec in best_hit.values():
            resolved = self._hydrate(rec)
            if (resolved is None or _is_business_name(resolved)
                    or _kind_contradicts(resolved, frame_kinds)):
                continue
            cat = resolved.product.category.value if resolved.product.category else None
            vocab = _candidate_vocabulary(resolved)
            support = _frame_support(vocab, line_tokens, category=cat, hint=hint)
            # The same count without the category's point. Agreeing on the category is real
            # evidence for *ranking* -- it is what separates `The Alchemist Heady Topper` from
            # `Alchemist Amer` -- but it cannot certify a frame, because on a can that prints
            # "ALE" every beer in the catalog earns it. Counting it here let a row named "Ache"
            # reach the corroboration bar off one mis-segmented fragment, and a certified frame
            # is precisely the one the client does not ask the model about.
            named = _corroboration(resolved, vocab, independent)
            evidence[resolved.product.id] = (resolved, vocab)
            if named <= 1 and _house_names_the_label(resolved, detections, read_toks):
                # The one-word label's second word is its house's line (see the rule).
                named = _MIN_FRAME_FOR_PENALTY
                by_house.add(resolved.product.id)
            named_by_id[resolved.product.id] = named
            # Report a score the frame actually justifies. One line naming a candidate while
            # several others sit there disagreeing is weaker evidence than the same number in
            # a frame that had nothing to corroborate with, and the overlay should say so.
            score = sc
            if hint and cat and cat != hint and cat in _CATEGORY_WORDS:
                score = sc * _CATEGORY_CONTRADICTS
            # `named`, not `support`, for the same reason corroboration uses it: the
            # category's point is not one of the frame's lines agreeing that this is the
            # product. A fragment match that only the category backs showed 1.00 in the HUD,
            # which is the number a user reads as certainty.
            # Whether the line *is* this product's name, judged on the raw similarity before
            # any markdown. Corroboration and confidence are different questions: the reported
            # score still says "one line, others disagreeing" -- a coincidence like `Chemist`
            # off "CHEMIST" beside two lines naming the Alchemist beer must still rank below it
            # -- while proof asks only whether some line is wholly this label, which is what a
            # shelf gives every product on it.
            whole_label[resolved.product.id] = _is_whole_label(resolved, sc, line_i)
            if named <= 1 and identity_lines >= _MIN_FRAME_FOR_PENALTY:
                score *= _UNCORROBORATED
            score = round(score, 3)
            personal, reason, cold = (
                (*self.score(resolved.product, profile),) if include_score
                else (None, None, False)
            )
            scored.append((
                support,
                ScoredCandidate(
                    detection_index=line_i,
                    resolved=resolved,
                    match_score=score,
                    personal_score=personal,
                    reason=reason,
                    cold_start=cold,
                ),
            ))

        # A line that is one product's whole label is that product's line, and its words
        # are that product's words. HIGH LIFE off a Miller can beside RIDGE FARM off a
        # sticker on the next shelf proved `High Ridge` -- a word from each, each on its own
        # line, the way two lines are meant to agree -- while the first line was Miller High
        # Life's whole label, proven as such (2026-09-17). A row that reads only SOME of the
        # label's words on that line, and nothing else there, has borrowed them, and is
        # counted again without the line. A row that reads a word there the label does not
        # is on a line the recognizer merged from two cans -- HOPPY IPA ran into the Miller
        # block on six frames, and WORMTOWN beside it is `Wormtown Be Hoppy` -- and keeps
        # it; so does `Miller High Life` over the importer's `High Life` on the HIGH LIFE
        # line, and so does `Goslings Black Seal` over the gin called `Black Seal` on BLACK
        # SEAL 80 PROOF BERMUDA BLACK RUM: a row that reads every word the label reads there
        # explains the line as well as the label does, and GOSLINGS beside it decides.
        def _reads_in(vocab: list[str], toks: list[str]) -> set[str]:
            read = set(toks)
            return {w for w in vocab if _read_as(w, read)}

        def _label_lines(rid: str) -> set[int]:
            """Every line that prints the whole label, not only the one that proved it: a
            tracked can of Modelo Negra held five reads of its label, and `Ruta Maya Negra`
            -- MAYA off the fine print, NEGRA off the label -- kept four of them when only
            the proving line was taken away (2026-09-17)."""
            words = _identifying_tokens(evidence[rid][0].product.name or "")
            lines = {best_hit[rid][0]}
            if words:
                lines |= {i for i, toks in enumerate(line_tokens)
                          if all(_read_as(w, set(toks)) for w in words)}
            return lines

        # By reading, not by line: the reading is the unit corroboration counts, and a can
        # of Wormtown Be Hoppy beside a Miller can gave HOPPY IPA six times as a line of
        # its own and once on the tail of the Miller block -- all one reading, echoes of
        # the block. Judged line by line, the block's own line (HOPP, not read) was owned
        # against Wormtown and took the whole reading with it (2026-09-17).
        label_readings = {rid: {reading_of[i] for i in _label_lines(rid) if i in reading_of}
                          for rid in evidence if whole_label.get(rid)}

        def _owned_against(pid: str, vocab: list[str]) -> set[int]:
            """The readings other whole labels own against this row: their readings, where
            this row reads a proper subset of the label's words."""
            return {g for rid, groups in label_readings.items() if rid != pid
                    for g in groups
                    if _reads_in(vocab, independent[g])
                    < _reads_in(evidence[rid][1], independent[g])}

        for pid, (resolved, vocab) in evidence.items():
            if (named_by_id.get(pid, 0) < _MIN_FRAME_FOR_PENALTY or whole_label.get(pid)
                    or pid in by_house):
                continue
            owned = _owned_against(pid, vocab)
            if not owned:
                continue
            readings = [toks for g, toks in enumerate(independent) if g not in owned]
            if _corroboration(resolved, vocab, readings) < _MIN_FRAME_FOR_PENALTY:
                named_by_id[pid] = 1
        # The wordmark contest is held to the same: a maker hypothesised off one word of the
        # fine print (MAYA, off "NAVA, MEXICO" misread) picked `Ruta Maya Negra` by the word
        # NEGRA -- the one word of its own -- read off the Modelo Negra label beside it
        # (2026-09-17). A window on a line another label owns, reading less of it than the
        # label does, is that label's word, not a wordmark.
        for pid in list(by_maker):
            if (pid in evidence and reading_of.get(best_hit[pid][0], -1)
                    in _owned_against(pid, evidence[pid][1])):
                by_maker.discard(pid)

        # A sibling that claims a word the frame never read yields to one that claims no more
        # than the frame read. A bottle of Bombay Sapphire prints VAPOUR INFUSED, and those
        # two words on two lines proved `East Vapour Infused London Dry Gin` -- the same
        # house's other gin, EAST read nowhere -- on every tick the wordmark came in as SOMBAA
        # (2026-09-17); the plain gin, its BOMBAY garbled, could not be proven whole, so
        # nothing shadowed East. The sibling rule needs no proof of the other row: `East`
        # has a word of its own name the plain gin's vocabulary lacks, EAST, and it was not
        # read, while every word the frame did read is a word the plain gin has too. The
        # frame is consistent with the plain gin and says nothing for East; East goes, and
        # the plain gin waits for BOMBAY. By the row's own name words, not its aliases: a
        # merge leaves a canon rows of alias vocabulary nobody printed, and the canon must
        # not yield to a variant over an alias's unread word. Only rows of one house: a
        # stranger's row that explains the same words is another product, and the shadow
        # rule below judges it on whether it was read whole. A claim is an identifying word:
        # the canon of Bombay Sapphire claims LONDON DRY GIN too, and on a frame that read
        # only BOMBAY and SAPPHIRE the first cut yielded those three to a junk sibling named
        # `Ginebra Bombay 0,70 CL.`, which yielded its one word back, and the bottle went
        # undrawn on seven frames it had been drawn on (2026-09-17).
        def _all_words(resolved: ResolvedProduct) -> set[str]:
            parts = [resolved.product.name, *(resolved.product.aliases or []),
                     *_maker_names(resolved)]
            return {w for part in parts for w in _name_words(part or "")}

        all_words = {pid: _all_words(resolved) for pid, (resolved, _) in evidence.items()}
        maker_ids = {pid: _maker_ids(resolved) for pid, (resolved, _) in evidence.items()}
        yields_to: dict[str, set[str]] = {}
        for pid, (resolved, _) in evidence.items():
            if pid in by_upc or pid in by_maker or pid in by_house:
                continue
            words = _identifying_tokens(resolved.product.name or "")
            mine = all_words[pid]
            makers = maker_ids[pid]
            for other in evidence:
                if other == pid or not (makers & maker_ids[other]):
                    continue
                theirs = all_words[other]
                extra = [w for w in words if w not in theirs]
                if not extra or any(_read_as(w, read_toks) for w in extra):
                    continue
                # ...and the row yielded to claims nothing unread of its own, of any kind:
                # `Goslings Gold Seal` is not the plainer row beside `Goslings Black Seal`
                # because GOLD is a colour to the category list, and BLACK unread does not
                # hand the bottle to the seal whose colour was not read either.
                unread_theirs = [w for w in theirs - mine
                                 if len(w) >= _MIN_SIGHTING_TOKEN and not _read_as(w, read_toks)]
                if not unread_theirs:
                    yields_to.setdefault(pid, set()).add(other)
        # The other row says everything this one's evidence says.
        scored = [entry for entry in scored if entry[1].resolved.product.id not in yields_to]

        # Collapse to one overlay per *real* product and cap the frame — the server-side
        # backstop against the crowding (and the duplicate-catalog-record double overlays) the
        # HUD showed. Keyed on canonical brand+name, not the raw id, so two rows for the same
        # beer merge. Corroboration outranks similarity: two independent lines naming a product
        # is stronger evidence than one perfect match on a fragment, which is exactly the
        # comparison a tie at 1.00 cannot make. On a tie the richer record (has ABV / sensory)
        # represents it, so the surviving overlay carries the most complete data — and that
        # also picks the better-linked of two duplicate rows.

        def _is_proven(c: ScoredCandidate) -> bool:
            # A barcode is an identifier, not a reading of one. Nothing in the frame needs to
            # agree with it, and a scan that succeeded must not be sent to the model to be
            # second-guessed -- nor capped below, since two barcodes legitimately name two
            # products.
            return (
                c.resolved.product.id in by_upc
                # The maker was read and, among its beers, the wordmark's shape picked this
                # one by a margin (`_pick_among`). Two parts of the frame agreeing, the second
                # by shape rather than by letters.
                or c.resolved.product.id in by_maker
                or named_by_id.get(c.resolved.product.id, 0) >= _MIN_FRAME_FOR_PENALTY
                # A line this candidate accounts for *entirely* proves it on its own, however
                # many other labels share the frame. This used to require the frame to hold
                # fewer than two identity lines -- which is to say, it only worked on a single
                # label photographed alone, and switched itself off on the one input the HUD
                # exists for. A shelf gives every product one line naming it and no second line
                # to agree, so nothing could corroborate, and the unproven-frame cap then threw
                # away all but one: three beers in view returned a single guess the client
                # withheld, and the shelf showed nothing at all.
                or whole_label.get(c.resolved.product.id, False)
            )

        def _rank(entry: tuple[int, ScoredCandidate]) -> tuple:
            support, c = entry
            p = c.resolved.product
            # The last two are a tie-break, and they are why the HUD stopped flickering. Two
            # rows can score identically on everything above, leaving the winner to whatever
            # order the store happened to return -- which varies between queries, so
            # consecutive frames of a motionless bottle named different rows.
            #
            # The tie is broken on how much of the row's OWN name the camera actually read.
            # This is the mirror of the leftover-word rule: that one rejects a row that fails
            # to explain the reading, this one prefers the row the reading explains. Length is
            # NOT the right proxy and was tried first -- it picks the longer name, which is how
            # "BOMBAY SAPPHIRE LOTTON DRY GIN" resolved to "East Vapour Infused London Dry
            # Gin" (a real, different bottle by the same maker) over "Bombay Sapphire London
            # Dry Gin": "east", "vapour" and "infused" are nowhere in the frame.
            unread = sum(1 for t in _identifying_tokens(p.name or "")
                         if not any(_trigram_sim(t, r) >= _TOKEN_SUPPORT_MIN for r in read_toks))
            # How much of what was read this row accounts for. `High Life` -- a permit filed
            # under the importer -- and `Miller High Life` both print whole on a Miller can,
            # and both are proven; the one that also explains MILLER is the one in view.
            # Every read word of substance counts, category words included: IPA is what tells
            # the row named "Dogfish Head 60 Minute IPA" from the one named "Dogfish Head 60
            # Minute", WHISKEY the two Stranahan's rows, and a brand row from the product
            # beside it. Only a producer's trade suffix is left out -- BREWING COMPANY is on
            # every can and is nobody's.
            printed = " ".join(filter(None, (p.name, c.resolved.brand.name,
                                             c.resolved.producer.name, *(p.aliases or []))))
            words = {_norm_token(w) for w in _NAME_WORD_RE.findall(_unapostrophed(printed).lower())
                     } - _PRODUCER_SUFFIX
            explained = sum(1 for r in read_toks if len(r) >= 3
                            and any(_trigram_sim(v, r) >= _TOKEN_SUPPORT_MIN for v in words))
            # Proof first. The one-candidate-per-line collapse below hands each line to its
            # best-ranked candidate, and ranked on resemblance alone, the wordmark line went
            # to whatever the garble happened to spell: "DY TOPP" is a 0.62 against `Lefty
            # Topp's`, with the word TOPP to back it, and the beer the maker's catalog had
            # just picked by the shape of that same line -- proven, and the only proven thing
            # in the frame -- was dropped as a second reading of text already spoken for. A
            # candidate the frame has proven represents its line ahead of one it merely
            # resembles; among the proven, and among the rest, nothing changes.
            # ...and, everything else equal, the shorter name: "Miller High Life High Life"
            # -- a permit filed as brand plus label -- explains the same three words as
            # `Miller High Life` with two words to spare, and the one with nothing to spare is
            # the one the line printed.
            return (_is_proven(c), support, explained, c.match_score,
                    bool(p.spec and p.spec.abv_pct), bool(p.sensory), -unread,
                    -len(_tokens(p.name or "")), p.id)

        best: dict[str, tuple[int, ScoredCandidate]] = {}
        for entry in scored:
            c = entry[1]
            key = _identity_key(c.resolved.product.name, c.resolved.brand.name,
                                c.resolved.product.id)
            if key not in best or _rank(entry) > _rank(best[key]):
                best[key] = entry
        ranked = sorted(best.values(), key=_rank, reverse=True)
        # One line names one product. Identity-keying collapses duplicate rows for the same
        # beer, but not a brand-level row sitting beside a product one -- `Lagunitas` next to
        # `Lagunitas IPA`, three Blue Moon variants next to each other. Each accounts for the
        # same line and so each proves itself against it, and a shelf of three beers drew five
        # overlays. Best candidate per line represents that line; the rest are readings of text
        # already spoken for.
        per_line: dict[int, tuple[int, ScoredCandidate]] = {}
        for entry in ranked:
            per_line.setdefault(entry[1].detection_index, entry)
        ranked = sorted(per_line.values(), key=_rank, reverse=True)[:_MAX_CANDIDATES]
        # One label, one product. Two proven candidates resting on the same lines are two
        # readings of one label, and a sibling whose name has a word the frame never read is
        # the worse one: GOSLINGS beside BLACK SEAL proved `Goslings Gold Seal` as surely as
        # it proved the Black Seal (2026-09-15), and INFUSED beside BOMBAY proved `East
        # Vapour Infused` on a bottle that printed EAST nowhere. The candidate whose every
        # identifying word was read speaks for those lines; one that needs a word the frame
        # did not read -- any word, GOLD is a colour to the category list and still the word
        # that tells the two seals apart -- is shadowed by it. On a shelf holding both
        # bottles the word is read, and nothing is shadowed.
        lines_of = {c.resolved.product.id: _agreeing_lines(_candidate_vocabulary(c.resolved),
                                                           line_tokens) for _, c in ranked}
        whole = [c for _, c in ranked if _is_proven(c) and lines_of[c.resolved.product.id]
                 and not _unread(_identifying_tokens(c.resolved.product.name or ""), read_toks)]
        shadowed = {
            c.resolved.product.id for _, c in ranked if _is_proven(c)
            and _unread([w for w in _name_words(c.resolved.product.name or "")
                         if len(w) >= _MIN_SIGHTING_TOKEN], read_toks)
            and any(a.resolved.product.id != c.resolved.product.id
                    and lines_of[c.resolved.product.id] <= lines_of[a.resolved.product.id]
                    for a in whole)
        }
        # And a label proven by its house's line yields to a proven product of the same
        # house: a bottle that reads BOMBAY SAPPHIRE proves the brand row `Bombay` that way
        # and the gin by its own two words, and the gin is the bottle.
        shadowed |= {
            c.resolved.product.id for _, c in ranked if c.resolved.product.id in by_house
            and any(_is_proven(a) and a.resolved.product.id != c.resolved.product.id
                    and _maker_ids(a.resolved) & _maker_ids(c.resolved) for _, a in ranked)
        }
        ranked = [entry for entry in ranked if entry[1].resolved.product.id not in shadowed]
        proven = [c for _, c in ranked if _is_proven(c)]
        return _Frame(ranked=[c for _, c in ranked], proven=proven, unresolved=unresolved,
                      by_maker=by_maker)

    # ---- the frame's proof, handed to the object that holds the line ----

    @staticmethod
    def _inherited(obj: DetectedObject, res: ObjectResolution,
                   proven: list[ScoredCandidate]) -> ObjectResolution:
        """An object holding a line the frame proved a product on is that product.

        The tracker follows regions of the screen, and a bottle of Gosling's was two: BLACK
        SEAL 80 PROOF BERMUDA BLACK RUM on one, GOSLINGS on the other. The frame, holding
        both lines, proved `Goslings Black Seal`; the first object, holding one, could not,
        and was handed a shortlist of one -- `Bermuda Brand Black Rum`, the row that reads
        most of its line -- which the fine stage, offered one name, took. Two names on one
        bottle (2026-09-17). A frame's proof covers the lines it rests on -- the ones that
        carry a word of the label, the same lines a verdict owns -- and the object holding
        one of them has been answered."""
        held = [_tokens(_latin(t)) for t in obj.texts if t and t.strip()]
        for c in proven:
            if _agreeing_lines(_candidate_vocabulary(c.resolved), held):
                return ObjectResolution(
                    object_id=obj.id, status="resolved", query=res.query,
                    candidates=[c.model_copy(update={"object_id": obj.id, "detection_index": -1})])
        return res

    # ---- the scene: the objects together, when none answered alone ----

    def _resolve_scene(self, objs: list[DetectedObject], profile: TasteProfile | None,
                       include_score: bool) -> ObjectResolution | None:
        """The maker's beer, when the maker is on one tracked object and the wordmark on
        another.

        The tracker follows regions of the screen, and on a can of Heady Topper the rim
        (THE ALCHEMIST, read as CHEMIST-VERNO) and the wordmark (HEADY TOPPER, read as
        RDY TOPP) are far enough apart to be two objects -- and each tick sends the three
        largest lines of the frame, which on a fridge of stickers were the stickers. So no
        request ever held both halves, and the maker path, which needs both, never ran:
        fifteen frames of the maker read and nothing drawn (2026-09-16). Judged together,
        the objects' lines are the frame the maker path was written for. Only the wordmark
        contest is trusted across objects -- a maker read on one and a beer of its own
        picked by shape on another is two parts of one can agreeing -- and the verdict lands
        on the object that holds the wordmark.
        """
        texts: list[str] = []
        owner: list[str] = []
        for o in objs:
            lines = [_latin(t) for t in o.texts if t and t.strip()]
            for lt in _object_lines(lines, _SCENE_LINES_PER_OBJECT):
                if lt not in texts:
                    texts.append(lt)
                    owner.append(o.id)
        if len(texts) < 2 or len(set(owner)) < 2:
            return None
        texts, owner = texts[:_SCENE_MAX_LINES], owner[:_SCENE_MAX_LINES]
        frame = self._resolve_lines([DetectedText(text=t) for t in texts], include_score, profile)
        picks = [c for c in frame.proven if c.resolved.product.id in frame.by_maker]
        if len(picks) != 1 or not (0 <= picks[0].detection_index < len(owner)):
            return None
        pick = picks[0]
        at = owner[pick.detection_index]
        return ObjectResolution(
            object_id=at, status="resolved", query=" | ".join(texts),
            candidates=[pick.model_copy(update={"object_id": at, "detection_index": -1})])

    # ---- objects: one can, one verdict ----

    def resolve_object(self, obj: DetectedObject, profile: TasteProfile | None = None,
                       include_score: bool = True,
                       min_score: float | None = None) -> ObjectResolution:
        """One tracked object's verdict.

        The object's lines are judged exactly as a frame is — same guards, same corroboration
        — and the outcome is folded into three states the client can act on:

          * `resolved`   the frame *proves* a row (a barcode, two independent lines naming
                         it, or one line that is wholly its label), or a row accounts for the
                         whole reading under the leftover-word rule: every identifying word
                         the camera read is explained by the product's name, its maker, or
                         label chrome. That rule is what stops a row from winning by saying
                         less: `Banger` leaves "focal" and "alchemist" unexplained on a
                         Focal Banger can and cannot resolve, however perfectly it matches
                         the one word it has.
          * `ambiguous`  rows the reading leans toward without any of them accounting for it
                         — the shortlist the client's fine stage (accurate OCR on the crop,
                         then the on-device model) chooses among. A row is on it only if it
                         explains more than half of what was read *and* one of its own name
                         words of substance was actually read: "FADY TOP" does not put
                         `Top's` here, "ACHE MIST-VERM" does not put `Ache` here.
          * `unresolved` nothing the reading supports. Show nothing; keep reading.
        """
        texts = _object_lines([_latin(t) for t in obj.texts if t and t.strip()], _OBJECT_MAX_LINES)
        query = " | ".join(texts + ([obj.barcode] if obj.barcode else []))

        def tagged(c: ScoredCandidate) -> ScoredCandidate:
            return c.model_copy(update={"object_id": obj.id, "detection_index": -1})

        # A barcode is an identifier, not a reading of one. It answers the object by itself
        # and nothing the text says can weaken it — the text beside a code is fine print
        # that reliably matches the wrong thing.
        if obj.barcode:
            rec = self._resolve_by_upc(obj.barcode)
            resolved = self._hydrate(rec) if rec is not None else None
            if resolved is not None:
                personal, why, cold = (
                    (*self.score(resolved.product, profile),) if include_score
                    else (None, None, False)
                )
                cand = ScoredCandidate(resolved=resolved, match_score=1.0,
                                       personal_score=personal, reason=why, cold_start=cold)
                return ObjectResolution(object_id=obj.id, status="resolved", query=query,
                                        candidates=[tagged(cand)])

        detections = [DetectedText(text=t) for t in texts]
        if not detections:
            return ObjectResolution(object_id=obj.id, status="unresolved", query=query)

        frame = self._resolve_lines(detections, include_score, profile)
        floor = 0.0 if min_score is None else min_score

        if frame.proven and frame.proven[0].match_score >= floor:
            return ObjectResolution(object_id=obj.id, status="resolved", query=query,
                                    candidates=[tagged(frame.proven[0])])
        reading = " ".join(texts)
        accounted = [c for c in frame.ranked if _accounts_for_object(c, reading)]
        if accounted and accounted[0].match_score >= floor:
            return ObjectResolution(object_id=obj.id, status="resolved", query=query,
                                    candidates=[tagged(accounted[0])])
        shortlist = [tagged(c) for c in frame.ranked
                     if _explains_enough(c, reading)][:_OBJECT_SHORTLIST]
        if shortlist:
            return ObjectResolution(object_id=obj.id, status="ambiguous", query=query,
                                    candidates=shortlist)
        return ObjectResolution(object_id=obj.id, status="unresolved", query=query)

    # ---- lexicon ----

    def lexicon(self, limit: int = 5000) -> list[str]:
        """The catalog's identifying vocabulary for the on-device recognizer's custom-words
        hint, commonest first. Served from the label index when the store carries one;
        otherwise a catalog walk, which is why the API caches it per process."""
        index = getattr(self.store, "index", None)
        if index is not None:
            return index.lexicon(limit)
        df: dict[str, int] = {}
        for kind in ("product", "brand", "producer"):
            for rec in self.store.iter_gold(kind):
                for tok in _identifying_tokens(rec.get("name") or ""):
                    if tok not in _PRODUCER_SUFFIX:
                        df[tok] = df.get(tok, 0) + 1
        return sorted(df, key=lambda t: (-df[t], t))[:limit]


def _top_axis(sv: SensoryVector) -> str | None:
    if not sv.axes:
        return None
    return max(sv.axes.items(), key=lambda kv: kv[1])[0].replace("_", " ")


def _match_reason(score: float, sensory: SensoryVector, ideal: SensoryVector) -> str:
    """Explain the score honestly.

    The axis we name is the one that actually drove the agreement — high on the product
    *and* high in the profile — not the product's loudest note. Naming the loudest note
    made a poor match still read "matches your smoky peat preference", which is the
    overlay telling the user something the score itself contradicts.
    """
    shared = _agreeing_axis(sensory, ideal)
    if score >= _STRONG_MATCH:
        return f"matches your {shared} preference" if shared else "matches your taste profile"
    if score >= _MILD_MATCH:
        return f"some {shared}, which you like" if shared else "a partial match"
    loud = _top_axis(sensory)
    return f"outside your usual — mostly {loud}" if loud else "outside your usual"


def _agreeing_axis(sensory: SensoryVector, ideal: SensoryVector) -> str | None:
    """The axis contributing most to the match: argmax of product·profile, per axis."""
    a, b = sensory.to_array(), ideal.to_array()
    weight, axis = max((a[i] * b[i], SENSORY_AXES[i]) for i in range(len(SENSORY_AXES)))
    return axis.replace("_", " ") if weight > 0 else None
