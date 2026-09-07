"""Medallion store — bronze (raw) -> silver (normalized) -> gold (canonical).

Dev/local implementation on SQLite + JSONL so the whole pipeline runs on a laptop with
no server. The interface is what matters; a Postgres-backed implementation swaps in
behind the same methods for production. The invariant everywhere: never lose the raw
bytes, and every gold row traces back to a bronze document id.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any, Protocol, runtime_checkable

from bcd_schema import SENSORY_AXES


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _tokenize(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(t) > 2}


def _fuzzy_eq(a: str, b: str, min_ratio: float = 0.8, min_len: int = 4) -> bool:
    """Exact, or close enough to be an OCR misread of the same word. Short tokens must
    match exactly — there's no room in 'ipa' for a typo that isn't a different word."""
    if a == b:
        return True
    if len(a) < min_len or len(b) < min_len:
        return False
    return SequenceMatcher(None, a, b).ratio() >= min_ratio


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two dense vectors. Shared by the SQLite store's python-side
    nearest-neighbor and the resolver's scorer so there's one definition."""
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _sensory_array(record: dict[str, Any]) -> list[float] | None:
    """Dense 25-vector in SENSORY_AXES order from a product record, or None when there's
    no sensory / no signal — mirrors the same helper in pg_store so both backends agree."""
    sv = record.get("sensory")
    if not sv:
        return None
    axes = sv.get("axes") or {}
    arr = [float(axes.get(a, 0.0)) for a in SENSORY_AXES]
    return arr if any(arr) else None


def doc_id(source_id: str, natural_key: str) -> str:
    """Stable id for a source document, so re-ingests upsert instead of duplicating."""
    h = hashlib.sha256(f"{source_id}::{natural_key}".encode()).hexdigest()[:16]
    return f"{source_id}:{h}"


@dataclass
class BronzeDoc:
    id: str
    source_id: str
    natural_key: str
    fetched_at: str
    url: str | None
    payload: dict[str, Any]


class MedallionStore:
    def __init__(self, root: str = "./data") -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.db_path = os.path.join(root, "bcd.sqlite")
        # check_same_thread=False: FastAPI runs sync endpoints in a threadpool, so the
        # connection outlives its creating thread. A single lock serializes access —
        # fine for the local dev store; prod uses a Postgres pool instead.
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS bronze (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                natural_key TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                url TEXT,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS silver (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                bronze_id TEXT NOT NULL,
                record TEXT NOT NULL,
                FOREIGN KEY (bronze_id) REFERENCES bronze(id)
            );
            CREATE TABLE IF NOT EXISTS gold (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                record TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_silver_type ON silver(entity_type);
            CREATE INDEX IF NOT EXISTS ix_gold_type ON gold(entity_type);
            """
        )
        self._db.commit()

    # ---- bronze ----
    def put_bronze(self, doc: BronzeDoc) -> None:
        if not doc.fetched_at:
            doc.fetched_at = _now()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO bronze VALUES (?,?,?,?,?,?)",
                (doc.id, doc.source_id, doc.natural_key, doc.fetched_at, doc.url,
                 json.dumps(doc.payload)),
            )
            self._db.commit()

    def iter_bronze(self, source_id: str) -> Iterator[BronzeDoc]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM bronze WHERE source_id=?", (source_id,)
            ).fetchall()
        for r in rows:
            yield BronzeDoc(r["id"], r["source_id"], r["natural_key"],
                            r["fetched_at"], r["url"], json.loads(r["payload"]))

    # ---- silver ----
    def put_silver(self, sid: str, source_id: str, entity_type: str,
                   bronze_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO silver VALUES (?,?,?,?,?)",
                (sid, source_id, entity_type, bronze_id, json.dumps(record)),
            )
            self._db.commit()

    def iter_silver(self, entity_type: str) -> Iterator[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT record FROM silver WHERE entity_type=?", (entity_type,)
            ).fetchall()
        for r in rows:
            yield json.loads(r["record"])

    # ---- gold ----
    def put_gold(self, gid: str, entity_type: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO gold VALUES (?,?,?,?)",
                (gid, entity_type, json.dumps(record), _now()),
            )
            self._db.commit()

    def get_gold(self, gid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT record FROM gold WHERE id=?", (gid,)
            ).fetchone()
        return json.loads(row["record"]) if row else None

    def iter_gold(self, entity_type: str) -> Iterator[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT record FROM gold WHERE entity_type=?", (entity_type,)
            ).fetchall()
        for r in rows:
            yield json.loads(r["record"])

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for tbl in ("bronze", "silver", "gold"):
                out[tbl] = self._db.execute(f"SELECT COUNT(*) c FROM {tbl}").fetchone()["c"]
        return out

    def search_gold_products(self, q: str, limit: int = 20) -> list[dict[str, Any]]:
        """Cheap LIKE search over gold products — placeholder for pg trigram + pgvector."""
        like = f"%{q.lower()}%"
        with self._lock:
            rows = self._db.execute(
                "SELECT record FROM gold WHERE entity_type='product' "
                "AND lower(record) LIKE ? LIMIT ?",
                (like, limit),
            ).fetchall()
        return [json.loads(r["record"]) for r in rows]

    # ---- search (used by the resolver / recommend) ----
    def match_products(self, text: str, limit: int = 3) -> list[tuple[dict, float]]:
        """Fuzzy token-overlap name match, best-first — the laptop stand-in for pg_trgm.
        Tokens match exactly or, for 4+ letter words, within an OCR-typo distance
        ('toppfr' ~ 'topper'), so a mangled label read still *retrieves* the right
        product; the resolver's identity scorer decides whether it's confident. The
        Postgres store swaps in real trigram similarity behind this same signature."""
        want = _tokenize(text)
        if not want:
            return []
        scored: list[tuple[dict, float]] = []
        for p in self.iter_gold("product"):
            name_tokens = _tokenize(p.get("name", ""))
            if not name_tokens:
                continue
            hits = sum(1 for t in name_tokens if any(_fuzzy_eq(q, t) for q in want))
            if not hits:
                continue
            # Coverage of the product name — extra words in the query cost nothing.
            score = hits / max(len(name_tokens), 1)
            scored.append((p, round(score, 3)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def match_producers(self, text: str, limit: int = 3) -> list[tuple[dict, float]]:
        """Fuzzy name match over producers (and brands), best-first. Lets 'THE ALCHEMIST'
        on a can retrieve that brewery's products even when the beer's own name was
        unreadable — the resolver then ranks the siblings, or reports them ambiguous."""
        want = _tokenize(text)
        if not want:
            return []
        scored: list[tuple[dict, float]] = []
        for kind in ("producer", "brand"):
            for rec in self.iter_gold(kind):
                names = [rec.get("name", "")] + list(rec.get("aliases") or [])
                best = 0.0
                for n in names:
                    toks = _tokenize(n)
                    if not toks:
                        continue
                    hits = sum(1 for t in toks if any(_fuzzy_eq(q, t) for q in want))
                    best = max(best, hits / len(toks))
                if best > 0:
                    scored.append((rec, round(best, 3)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def products_by_producer(self, producer_id: str, limit: int = 25) -> list[dict[str, Any]]:
        """Every product a producer (or brand) owns. Python scan on the dev store; an
        indexed jsonb lookup on Postgres."""
        out: list[dict[str, Any]] = []
        for p in self.iter_gold("product"):
            if p.get("producer_id") == producer_id or p.get("brand_id") == producer_id:
                out.append(p)
                if len(out) >= limit:
                    break
        return out

    def nearest_by_sensory(self, vec: list[float], limit: int = 10) -> list[dict[str, Any]]:
        """Cosine nearest-neighbor over products that carry a sensory vector, computed in
        python. The Postgres store does this as a single pgvector `<=>` ANN query."""
        scored: list[tuple[dict, float]] = []
        for p in self.iter_gold("product"):
            arr = _sensory_array(p)
            if arr is None:
                continue
            scored.append((p, _cosine(vec, arr)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:limit]]

    def close(self) -> None:
        self._db.close()


@runtime_checkable
class Store(Protocol):
    """The storage contract both backends satisfy. Callers depend on this, not on a
    concrete class, so `open_store()` can hand back SQLite on a laptop and Postgres in
    prod without anything downstream changing."""

    db_path: str

    def put_bronze(self, doc: BronzeDoc) -> None: ...
    def iter_bronze(self, source_id: str) -> Iterator[BronzeDoc]: ...
    def put_silver(self, sid: str, source_id: str, entity_type: str,
                   bronze_id: str, record: dict[str, Any]) -> None: ...
    def iter_silver(self, entity_type: str) -> Iterator[dict[str, Any]]: ...
    def put_gold(self, gid: str, entity_type: str, record: dict[str, Any]) -> None: ...
    def get_gold(self, gid: str) -> dict[str, Any] | None: ...
    def iter_gold(self, entity_type: str) -> Iterator[dict[str, Any]]: ...
    def counts(self) -> dict[str, int]: ...
    def search_gold_products(self, q: str, limit: int = 20) -> list[dict[str, Any]]: ...
    def match_products(self, text: str, limit: int = 3) -> list[tuple[dict, float]]: ...
    def match_producers(self, text: str, limit: int = 3) -> list[tuple[dict, float]]: ...
    def products_by_producer(self, producer_id: str, limit: int = 25) -> list[dict[str, Any]]: ...
    def nearest_by_sensory(self, vec: list[float], limit: int = 10) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def open_store(root: str = "./data", url: str | None = None) -> Store:
    """Pick a backend. Explicit `BCD_STORE_BACKEND=sqlite|postgres` wins; otherwise the
    presence of a `BCD_DATABASE_URL` selects Postgres, and a bare laptop falls back to the
    SQLite dev store. The Postgres import is lazy so the SQLite path needs no psycopg."""
    backend = os.environ.get("BCD_STORE_BACKEND")
    url = url or os.environ.get("BCD_DATABASE_URL")
    if backend == "sqlite":
        return MedallionStore(root=root)
    if backend == "postgres" or (backend is None and url):
        from .pg_store import PostgresStore
        return PostgresStore(url or "postgresql://localhost:5432/bcd")
    return MedallionStore(root=root)
