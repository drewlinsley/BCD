"""The label index — a scan's retrieval step, in memory, in about a millisecond.

Why this exists. Matching a label against the catalog used to be a trigram scan per OCR
line: each line walks a GIN posting list sized by how common its *trigrams* are, and
against 534k products a six-line can costs 1.6s run concurrently and 4.5s in series. Adding
per-object retrieval on top of that (whole-object query + every fragment + producer
expansion) measured over 7s. The HUD ticks at 350ms. No amount of re-tuning a trigram gate
gets a scan under a second, because the gate's cost is in the database, per line, per
probe.

So the retrieval step moves out of the database. At startup this walks the catalog once
and builds an inverted index from *identifying tokens* (the words that could pick a
product off a shelf — not "ale", not "brewing") to the products and producers that carry
them. A query is then a handful of dictionary lookups plus a trigram re-score of the few
dozen rows those lookups surface. Measured on a synthetic 534k-product catalog: ~1ms per
line, ~3ms per object, on one core, whichever store backs it (see `scripts/bench_index.py`).

What it deliberately keeps from the store it replaces:

  * **The same score.** The store's `match_products` returns a pg_trgm-style similarity
    (the greatest of `similarity` and both directions of `word_similarity`, on the name and
    on the brand-qualified name), and every threshold in the resolver was calibrated against
    that number on real frames. The index computes the same six terms in Python on the rows
    it surfaces, so the resolver's guards keep meaning what they were measured to mean.
  * **The same surface.** `IndexedStore` wraps any `Store` and answers only the four
    matching methods from memory; everything else passes through. `Resolver` does not know
    it is there.

Fuzziness. OCR misreads a letter or two ("TOPPFR"), or truncates ("ALCHEMIS", "FOCAL
BAN"). A query token therefore also reaches dictionary tokens one edit away (for words of
5+ letters) and dictionary tokens it is a prefix of. Both are generated on the query side
at lookup time — a few hundred dictionary probes per token, no second index — which is why
the index is small: ids, names, and postings, nothing else.

Refresh. The index is a snapshot; the store is the truth. It is rebuilt when the catalog's
row counts change and otherwise loaded from a pickle beside the data, because a 534k-row
walk is seconds and a scan cannot wait for it. A row promoted since the snapshot is
reachable by barcode (a keyed lookup that never touched the index) but not by name until
the next rebuild — the same lag `search_name` already has on Postgres.
"""

from __future__ import annotations

import bisect
import contextlib
import heapq
import math
import os
import pickle
import re
import time
import unicodedata
from array import array
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

from bcd_ingest.dedup import _PRODUCER_SUFFIX, is_generic_token, search_name
from bcd_ingest.store import Store

# ---- tokens ----------------------------------------------------------------------------

_LETTERS = re.compile(r"[^\W\d_]{2,}", re.UNICODE)   # runs of >=2 letters, as dedup._tokens
_LONG_DIGITS = re.compile(r"\d{4,}")                   # "1664": a number that IS the name
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")

#: Shortest token worth a posting. Two letters is "oz", "by", "no".
MIN_TOKEN = 3
#: Shortest token that may reach dictionary words one edit away. A four-letter token has
#: 26*4 one-letter substitutions and most of them are real words.
FUZZY_MIN = 5
#: Shortest token that may reach dictionary words it is a prefix of ("ALCHEMIS").
PREFIX_MIN = 5
PREFIX_MAX_EXPANSIONS = 8
#: Rows that survive the token stage into trigram scoring. The right row is nearly always
#: in the top few by token evidence; this is the margin for the cases where a common brand
#: word spreads its weight over many rows.
CANDIDATES = 48
EXACT_WEIGHT, FUZZY_WEIGHT, PREFIX_WEIGHT = 1.0, 0.7, 0.6
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _fold(s: str) -> str:
    """Casefold and strip diacritics, the way dedup and the resolver compare words."""
    d = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in d if not unicodedata.combining(c)).casefold()


def _words(s: str) -> list[str]:
    return _LETTERS.findall(_fold(s))


def _flat(s: str) -> str:
    return _NOT_ALNUM.sub(" ", _fold(s)).strip()


def identifying_tokens(s: str) -> list[str]:
    """The tokens a posting is worth: real words that are not category/chrome vocabulary,
    plus long digit runs. Order-preserving, deduped."""
    out: list[str] = []
    seen: set[str] = set()
    for w in _words(s):
        if len(w) >= MIN_TOKEN and not is_generic_token(w) and w not in seen:
            seen.add(w)
            out.append(w)
    for d in _LONG_DIGITS.findall(s or ""):
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _producer_tokens(name: str) -> list[str]:
    """Identifying tokens of a producer name, minus the corporate suffixes every other
    producer carries ("Brewing Co LLC")."""
    return [t for t in identifying_tokens(name) if t not in _PRODUCER_SUFFIX]


def _edits1(word: str) -> Iterable[str]:
    """Every string one edit away (delete, transpose, replace, insert), ASCII letters."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    for left, right in splits:
        if right:
            yield left + right[1:]
        if len(right) > 1:
            yield left + right[1] + right[0] + right[2:]
        for c in _ALPHABET:
            if right:
                yield left + c + right[1:]
            yield left + c + right


# ---- pg_trgm in Python --------------------------------------------------------------------
#
# Only the rows the token stage surfaces are ever scored, so this can afford to be exact
# about *what* it computes rather than fast about how.


@lru_cache(maxsize=200_000)
def _trigrams(s: str) -> frozenset[str]:
    """pg_trgm's trigram set: each word padded with two leading and one trailing space."""
    out: set[str] = set()
    for w in _flat(s).split():
        p = f"  {w} "
        out.update(p[i:i + 3] for i in range(len(p) - 2))
    return frozenset(out)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@lru_cache(maxsize=50_000)
def _windows(s: str) -> tuple[frozenset[str], ...]:
    """Trigram sets of every contiguous run of words in `s` — the "continuous extents"
    `word_similarity` searches. A word's trigrams are independent of its neighbours under
    pg_trgm's per-word padding, so an extent's set is the union of its words' sets."""
    words = _flat(s).split()
    per_word = [_trigrams(w) for w in words]
    out: list[frozenset[str]] = []
    for i in range(len(words)):
        acc: frozenset[str] = frozenset()
        for j in range(i, len(words)):
            acc = acc | per_word[j]
            out.append(acc)
    return tuple(out)


def similarity(a: str, b: str) -> float:
    return _jaccard(_trigrams(a), _trigrams(b))


def word_similarity(a: str, b: str) -> float:
    """Greatest similarity between `a`'s trigrams and any continuous extent of `b`."""
    ta = _trigrams(a)
    return max((_jaccard(ta, w) for w in _windows(b)), default=0.0)


def match_score(name: str, qualified: str, text: str) -> float:
    """The store's number: greatest of the six terms `PostgresStore.match_products` ranks
    by, so a threshold calibrated on Postgres holds here."""
    return max(
        similarity(name, text),
        word_similarity(name, text),
        word_similarity(text, name),
        similarity(qualified, text),
        word_similarity(qualified, text),
        word_similarity(text, qualified),
    )


# ---- the index -----------------------------------------------------------------------------


class LabelIndex:
    """Identifying-token postings over products and producers, plus what a match needs to
    be scored and hydrated: ids, names, brand-qualified names, and who makes what."""

    FORMAT = 1

    def __init__(self) -> None:
        self.signature: str = ""
        # products, by dense index
        self.ids: list[str] = []
        self.names: list[str] = []
        self.qualified: list[str] = []
        self.producer_of: array = array("i")
        # producers, by dense index
        self.producer_ids: list[str] = []
        self.producer_names: list[str] = []
        self.producer_index: dict[str, int] = {}
        self.products_by_producer: dict[int, array] = {}
        # token dictionary shared by both posting maps; `sorted_tokens` serves prefix lookups
        self.token_id: dict[str, int] = {}
        self.tokens: list[str] = []
        self.sorted_tokens: list[str] = []
        self.product_post: dict[int, array] = {}
        self.producer_post: dict[int, array] = {}
        # products whose name carries no identifying token, reachable only as a whole:
        # "IPA", "J&B", "1664" — keyed by flattened name and by their (generic) words
        self.by_flat_name: dict[str, array] = {}
        self.generic_post: dict[int, array] = {}
        self.built_at: float = 0.0
        self.build_seconds: float = 0.0

    # ---- build ----

    @classmethod
    def build(cls, store: Store) -> LabelIndex:
        t0 = time.perf_counter()
        ix = cls()
        ix.signature = cls.signature_of(store)

        brand_names: dict[str, str] = {}
        for b in store.iter_gold("brand"):
            brand_names[b["id"]] = b.get("name") or ""

        for rec in store.iter_gold("producer"):
            pid = rec.get("id") or ""
            if not pid or pid in ix.producer_index:
                continue
            ix.producer_index[pid] = len(ix.producer_ids)
            ix.producer_ids.append(pid)
            name = rec.get("name") or ""
            ix.producer_names.append(name)
            for tok in _producer_tokens(name):
                ix._post(ix.producer_post, tok, len(ix.producer_ids) - 1)

        for rec in store.iter_gold("product"):
            pid = rec.get("id") or ""
            if not pid:
                continue
            i = len(ix.ids)
            name = rec.get("name") or ""
            qualified = search_name(name, brand_names.get(rec.get("brand_id") or ""))
            ix.ids.append(pid)
            ix.names.append(name)
            ix.qualified.append(qualified)
            owner = ix.producer_index.get(rec.get("producer_id") or "", -1)
            ix.producer_of.append(owner)
            if owner >= 0:
                ix.products_by_producer.setdefault(owner, array("i")).append(i)
            toks = identifying_tokens(qualified)
            for tok in toks:
                ix._post(ix.product_post, tok, i)
            if not toks:
                flat = _flat(qualified)
                if flat:
                    ix.by_flat_name.setdefault(flat, array("i")).append(i)
                for w in _words(qualified):
                    ix._post(ix.generic_post, w, i)

        ix.sorted_tokens = sorted(ix.token_id)
        ix.built_at = time.time()
        ix.build_seconds = time.perf_counter() - t0
        return ix

    def _post(self, table: dict[int, array], tok: str, row: int) -> None:
        tid = self.token_id.get(tok)
        if tid is None:
            tid = len(self.tokens)
            self.token_id[tok] = tid
            self.tokens.append(tok)
        table.setdefault(tid, array("i")).append(row)

    @staticmethod
    def signature_of(store: Store) -> str:
        counts = store.counts()
        return f"v{LabelIndex.FORMAT}:" + ",".join(f"{k}={counts[k]}" for k in sorted(counts))

    # ---- persistence ----

    def save(self, path: str) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, signature: str | None = None) -> LabelIndex | None:
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
            return None
        if signature is not None and state.get("signature") != signature:
            return None
        ix = cls()
        ix.__dict__.update(state)
        return ix

    @classmethod
    def for_store(cls, store: Store, cache_path: str | None) -> LabelIndex:
        """The index for this catalog: loaded from `cache_path` when it still describes the
        store's row counts, otherwise rebuilt (and cached)."""
        sig = cls.signature_of(store)
        if cache_path:
            cached = cls.load(cache_path, sig)
            if cached is not None:
                return cached
        ix = cls.build(store)
        if cache_path:
            with contextlib.suppress(OSError):   # a cache that cannot be written is only slower
                ix.save(cache_path)
        return ix

    # ---- stats ----

    def __len__(self) -> int:
        return len(self.ids)

    def stats(self) -> dict[str, Any]:
        return {
            "products": len(self.ids),
            "producers": len(self.producer_ids),
            "tokens": len(self.tokens),
            "postings": sum(len(v) for v in self.product_post.values())
            + sum(len(v) for v in self.producer_post.values()),
            "build_seconds": round(self.build_seconds, 2),
        }

    # ---- lookup ----

    def _expand(self, q: str, table: dict[int, array]) -> list[tuple[int, float]]:
        """Dictionary tokens a query token reaches, with the weight each carries."""
        out: dict[int, float] = {}
        exact = self.token_id.get(q)
        if exact is not None and exact in table:
            out[exact] = EXACT_WEIGHT
        if len(q) >= FUZZY_MIN:
            for cand in _edits1(q):
                tid = self.token_id.get(cand)
                if tid is not None and tid in table and tid not in out:
                    out[tid] = FUZZY_WEIGHT
        if len(q) >= PREFIX_MIN:
            start = bisect.bisect_left(self.sorted_tokens, q)
            n = 0
            for tok in self.sorted_tokens[start:start + 64]:
                if not tok.startswith(q):
                    break
                tid = self.token_id[tok]
                if tid in table and tid not in out:
                    out[tid] = PREFIX_WEIGHT
                    n += 1
                    if n >= PREFIX_MAX_EXPANSIONS:
                        break
        return list(out.items())

    def _token_stage(self, text: str, table: dict[int, array], size: int) -> list[int]:
        """Rows with the most identifying-token evidence for `text`, by IDF-weighted sum."""
        ident = identifying_tokens(text)
        if not ident:
            return []
        acc: dict[int, float] = {}
        for q in ident:
            for tid, w in self._expand(q, table):
                rows = table[tid]
                weight = w * math.log(1.0 + size / len(rows))
                for r in rows:
                    acc[r] = acc.get(r, 0.0) + weight
        if len(acc) <= CANDIDATES:
            return sorted(acc, key=acc.__getitem__, reverse=True)
        return [r for r, _ in heapq.nlargest(CANDIDATES, acc.items(), key=lambda kv: kv[1])]

    def _generic_stage(self, text: str) -> list[int]:
        """A line with nothing identifying in it can only name a product whose name is
        equally generic — the resolver's own rule — so those are the rows it reaches."""
        rows: dict[int, int] = {}
        flat = _flat(text)
        for r in self.by_flat_name.get(flat, ()):
            rows[r] = rows.get(r, 0) + 3
        for w in _words(text):
            tid = self.token_id.get(w)
            if tid is None:
                continue
            for r in self.generic_post.get(tid, ()):
                rows[r] = rows.get(r, 0) + 1
        return sorted(rows, key=rows.__getitem__, reverse=True)[:CANDIDATES]

    def match_products(self, text: str, limit: int = 3) -> list[tuple[str, float]]:
        """Best-first (product id, similarity) for one OCR line — the store's answer, from
        memory. The similarity is pg_trgm's, computed on the rows the tokens surface."""
        text = (text or "").strip()
        if not text:
            return []
        rows = self._token_stage(text, self.product_post, len(self.ids))
        if not rows:
            rows = self._generic_stage(text)
        scored = []
        for r in rows:
            sim = match_score(self.names[r], self.qualified[r], text)
            if sim <= 0.0:
                continue
            # Ties at the top are the norm: `word_similarity` is 1.0 for ANY name wholly
            # inside the line. Plain similarity on the qualified name breaks them toward the
            # row that accounts for more of what was read — the store's own tiebreak.
            scored.append((-sim, -similarity(self.qualified[r], text), self.ids[r], r))
        scored.sort()
        return [(self.ids[r], round(-s, 3)) for s, _, _, r in scored[:limit]]

    def match_producers(self, text: str, limit: int = 3) -> list[tuple[str, float]]:
        text = (text or "").strip()
        if not text:
            return []
        rows = self._token_stage(text, self.producer_post, len(self.producer_ids))
        scored = []
        for r in rows:
            name = self.producer_names[r]
            sim = max(similarity(name, text), word_similarity(name, text),
                      word_similarity(text, name))
            if sim > 0.0:
                scored.append((-sim, -similarity(name, text), self.producer_ids[r], r))
        scored.sort()
        return [(self.producer_ids[r], round(-s, 3)) for s, _, _, r in scored[:limit]]

    def products_of(self, producer_id: str) -> list[str]:
        owner = self.producer_index.get(producer_id)
        if owner is None:
            return []
        return [self.ids[i] for i in self.products_by_producer.get(owner, ())]

    def lexicon(self, limit: int = 5000) -> list[str]:
        """The catalog's identifying vocabulary, commonest first: what the on-device text
        recognizer should prefer over dictionary words. It already knows "ale"; what it
        does not know is "alchemist"."""
        df: dict[int, int] = {}
        for table in (self.product_post, self.producer_post):
            for tid, rows in table.items():
                df[tid] = df.get(tid, 0) + len(rows)
        best = heapq.nlargest(limit, df.items(), key=lambda kv: (kv[1], -kv[0]))
        return [self.tokens[tid] for tid, _ in best]


# ---- the store front -------------------------------------------------------------------------


class IndexedStore:
    """A `Store` whose matching methods answer from a `LabelIndex`. Everything else — gold
    lookups, barcodes, iteration, writes — passes straight through to the wrapped store,
    so `Resolver` and the API see one object with the same surface they had."""

    def __init__(self, store: Store, index: LabelIndex) -> None:
        self._store = store
        self.index = index

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def _records(self, ids: Iterable[str]) -> list[dict]:
        out = []
        for pid in ids:
            rec = self._store.get_gold(pid)
            if rec is not None:
                out.append(rec)
        return out

    def match_products(self, text: str, limit: int = 3, **_: Any) -> list[tuple[dict, float]]:
        out = []
        for pid, sim in self.index.match_products(text, limit):
            rec = self._store.get_gold(pid)
            if rec is not None:
                out.append((rec, sim))
        return out

    def match_products_many(self, texts: Sequence[str],
                            limit: int = 3) -> list[list[tuple[dict, float]]]:
        return [self.match_products(t, limit) for t in texts]

    def match_producers(self, text: str, limit: int = 3) -> list[tuple[dict, float]]:
        out = []
        for pid, sim in self.index.match_producers(text, limit):
            rec = self._store.get_gold(pid)
            if rec is not None:
                out.append((rec, sim))
        return out

    def products_of(self, producer_id: str, limit: int = 8) -> list[dict]:
        """A producer's catalog, best-known first, as the stores order it."""
        recs = self._records(self.index.products_of(producer_id))
        recs.sort(key=lambda r: (r.get("sensory") is None,
                                 (r.get("spec") or {}).get("abv_pct") is None,
                                 r.get("name") or ""))
        return recs[:limit]


__all__ = ["IndexedStore", "LabelIndex", "identifying_tokens", "match_score"]
