"""Which user contributions are ready to be catalog rows, and which are not.

`POST /v1/contribute` files a typed drink in bronze and stops. Nothing promoted one until now,
which was an open design call; these are the two gates the user set on 2026-10-12:

  1. **it must not already be in the catalog**, misspellings included;
  2. **two different accounts** must have told us about it.

The second is why "entries" has to mean accounts. The route is idempotent per account and
client id, so one phone can file the same drink twice under two ids in a minute, and two
entries from one person is not two people agreeing -- it is one person, which is the thing the
gate exists to improve on. Both contributions in bronze today are from a single account.

The first gate needs three answers rather than two, because a typed name is often not enough to
decide. A confident single match is already in the catalog and must not be promoted. SEVERAL
matches mean the name does not say which, and guessing is worse than leaving a duplicate: a
bare "Crusher" covers The Alchemist's, a different brewery's own bare "Crusher", and a dozen
Bone Crushers and Soul Crushers, and picking one hangs the wrong producer, the wrong flavour
profile and the wrong town off the can in somebody's hand. Those are held for a look. So are
names that identify a class rather than a drink ("Blended Scotch") and names with no word long
enough to pick anything off a shelf ("J&B", "1664") -- a promoted row needs a name the resolver
can find, and those three cases are exactly `dedup`'s existing guards.

Identity lives in `dedup.match_typed_name`; this module is only the policy on top of it.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .dedup import (
    _STYLE,
    _tokens,
    carries_no_identity,
    is_generic_name,
    match_typed_name,
)

#: How many different accounts have to have told us about a drink. The user's number.
PROMOTION_ACCOUNTS = 2

#: Verdicts. `NEW` is the only promotable one, and it is still not sufficient on its own.
IN_CATALOG = "in_catalog"
NEEDS_A_LOOK = "needs_a_look"
NEW = "new"


@dataclass(frozen=True)
class Contribution:
    """One thing a drinker typed, as filed in bronze."""

    doc_id: str
    account_id: str
    name: str
    category: str
    maker: str | None = None
    abv_pct: float | None = None
    received_at: str = ""
    sightings: tuple[str, ...] = ()

    @classmethod
    def from_bronze(cls, doc_id: str, payload: Mapping) -> Contribution:
        return cls(
            doc_id=doc_id,
            account_id=str(payload.get("account_id") or ""),
            name=str(payload.get("name") or ""),
            category=str(payload.get("category") or ""),
            maker=(payload.get("maker") or None),
            abv_pct=payload.get("abv_pct"),
            received_at=str(payload.get("received_at") or ""),
            sightings=tuple(payload.get("sightings") or ()),
        )

    def as_row(self) -> dict:
        """Shaped like a catalog row, so one contribution can be matched against another with
        the same function that matches it against the catalog."""
        return {"id": self.doc_id, "name": self.name, "brand": self.maker or ""}


@dataclass(frozen=True)
class Group:
    """One drink, as told by one or more people."""

    name: str
    category: str
    members: tuple[Contribution, ...]
    catalog: tuple[tuple[str, str], ...] = ()
    verdict: str = NEW
    reason: str = ""
    spellings: tuple[str, ...] = field(default=())

    @property
    def accounts(self) -> frozenset[str]:
        return frozenset(m.account_id for m in self.members if m.account_id)

    @property
    def promotable(self) -> bool:
        return self.verdict == NEW and len(self.accounts) >= PROMOTION_ACCOUNTS

    @property
    def held_because(self) -> str:
        """Why this is not being promoted, in the order the gates are applied."""
        if self.verdict != NEW:
            return self.reason
        short = PROMOTION_ACCOUNTS - len(self.accounts)
        if short > 0:
            n = len(self.accounts)
            return (f"only {n} account{'' if n == 1 else 's'}; "
                    f"{short} more needed")
        return ""

    @property
    def maker(self) -> str | None:
        """The maker somebody gave, if anybody did. The first in receipt order, so a later
        contributor who left it blank cannot erase it."""
        for m in sorted(self.members, key=lambda c: c.received_at):
            if m.maker:
                return m.maker
        return None

    @property
    def aliases(self) -> tuple[str, ...]:
        """Other spellings that deserve to be searchable names for this row.

        A shorter real name does: "Heady Topper" for "The Alchemist Heady Topper" is what the
        can actually says, and the resolver should land on it. A TYPO does not. "Petit Mutnat"
        is not written on anything, and indexing it would put a string no label carries into
        the thing the recognizer matches against -- which is the index getting worse, not
        better, in exchange for one person's thumb.

        The test is whether the spelling needed typo tolerance to get here: every word of an
        alternate name is a real word of the full one, and a typo is exactly the case where
        that fails.
        """
        kept = _tokens(self.name)
        return tuple(s for s in self.spellings
                     if s != self.name and _tokens(s) <= kept)

    @property
    def abv_pct(self) -> float | None:
        said = [m.abv_pct for m in self.members if m.abv_pct is not None]
        # Only when they agree. Two different numbers for one drink is not a fact about it,
        # and averaging them would invent a third nobody typed.
        return said[0] if said and len(set(said)) == 1 else None


def _best_spelling(members: Sequence[Contribution]) -> str:
    """Which typing of the name becomes the row's.

    Most accounts first, because agreement is the only evidence available about a spelling.
    Then the most identifying words, so "Sierra Nevada Hazy Little Thing" beats "Hazy Little
    Thing" -- the fuller name is the one the resolver can place. Then the longest, then
    alphabetical, so the answer does not depend on dict ordering.
    """
    by_name: dict[str, set[str]] = {}
    for m in members:
        by_name.setdefault(m.name, set()).add(m.account_id)
    return min(by_name,
               key=lambda n: (-len(by_name[n]),
                              -len([t for t in n.split() if len(t) >= 4]),
                              -len(n), n))


def group_contributions(contributions: Iterable[Contribution]) -> list[Group]:
    """Cluster what different people typed into one drink each.

    Grown one member at a time, and a contribution joins a cluster only if it is the same drink
    as EVERY member -- the rule `find_duplicate_merges` uses, and for the same reason: one
    permissive name ("Crusher") would otherwise chain two unrelated drinks together through it.
    """
    clusters: list[list[Contribution]] = []
    # Newest-last so the earliest contribution seeds its cluster and the order is stable.
    for c in sorted(contributions, key=lambda c: (c.received_at, c.doc_id)):
        for cl in clusters:
            if all(_same_drink(c, m) for m in cl):
                cl.append(c)
                break
        else:
            clusters.append([c])

    out: list[Group] = []
    for cl in clusters:
        members = tuple(cl)
        out.append(Group(name=_best_spelling(members),
                         category=members[0].category,
                         members=members,
                         spellings=tuple(sorted({m.name for m in members}))))
    return out


def _same_drink(a: Contribution, b: Contribution) -> bool:
    """Either name covering the other is enough. Neither typist is the authority: one may have
    typed the fuller name and one the shorter, and they are still the same can."""
    if a.category != b.category:
        return False
    return bool(match_typed_name(a.name, a.maker, [b.as_row()])
                or match_typed_name(b.name, b.maker, [a.as_row()]))


def judge(group: Group, catalog: Sequence[Mapping]) -> Group:
    """Decide a group against the catalog rows that could be it.

    `catalog` is the shortlist the caller retrieved -- this does not go looking, because the
    catalog is 534k rows and the retrieval is a trigram query per name.
    """
    # Both guards reject, and they overlap -- `carries_no_identity` is the broader of the two,
    # so the narrower one is asked first to get the more useful sentence. "Blended Scotch
    # Whisky" is every-word-a-category-word; "J&B" has no long word at all.
    if is_generic_name(group.name):
        return _verdict(group, NEEDS_A_LOOK, (),
                        "names a class of drink, not a drink")
    if carries_no_identity(group.name):
        return _verdict(group, NEEDS_A_LOOK, (),
                        "no word long enough to pick it off a shelf")

    hit_ids = set(match_typed_name(group.name, group.maker, catalog))
    # Every spelling anybody used, not just the winning one: one person's typo may be the only
    # one that reaches the row, and it is still the same evidence that the row exists.
    for spelling in group.spellings:
        if spelling != group.name:
            hit_ids |= set(match_typed_name(spelling, group.maker, catalog))

    hits = tuple(sorted(((str(r["id"]), str(r.get("name") or ""))
                         for r in catalog if str(r["id"]) in hit_ids),
                        key=lambda h: _closeness(group.name, h[1])))
    if len(hits) == 1:
        return _verdict(group, IN_CATALOG, hits, f"already in the catalog as {hits[0][1]!r}")
    if len(hits) > 1:
        return _verdict(group, NEEDS_A_LOOK, hits,
                        f"could be any of {len(hits)} catalog rows; the name does not say which")
    return _verdict(group, NEW, (), "")


def _closeness(typed: str, candidate: str) -> tuple:
    """How near a catalog name is to the typed one, nearest first.

    Ordering only, and deliberately not a likelihood: when a typed "Crusher" reaches 230 rows
    the honest answer is that the name does not decide, and a ranking that looked confident
    would undo the verdict. What this buys is that the rows a person is shown first are the ones
    closest in specificity -- the row named exactly "Crusher", then "The Alchemist Crusher" --
    rather than whichever eight the trigram search happened to return, and that reading the
    same queue twice shows the same thing.
    """
    # Every non-style word counts, however short. A four-letter floor is borrowed from
    # `carries_no_identity`, which answers a different question, and here it made the
    # distinguishing word invisible: "Da Crusher" and "Sol Crusher" came back as near as the
    # row named plain "Crusher".
    extra = len(_tokens(candidate) - _tokens(typed) - _STYLE)
    return (extra, len(candidate), candidate)


def _verdict(group: Group, verdict: str, hits: tuple, reason: str) -> Group:
    return Group(name=group.name, category=group.category, members=group.members,
                 catalog=hits, verdict=verdict, reason=reason, spellings=group.spellings)


def plan(contributions: Iterable[Contribution],
         shortlist: Mapping[str, Sequence[Mapping]]) -> list[Group]:
    """Every group, judged, promotable ones first then the ones closest to promoting.

    `shortlist` maps a group's name to the catalog rows retrieved for it. The caller does the
    retrieval because it owns the store; everything else here is a pure function of what it
    hands over, which is what makes the rule testable without a database.
    """
    judged = [judge(g, shortlist.get(g.name, ())) for g in group_contributions(contributions)]
    return sorted(judged, key=lambda g: (not g.promotable, -len(g.accounts),
                                         g.verdict, g.name))
