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
from typing import Any, Protocol, runtime_checkable

from bcd_schema import SENSORY_AXES

from .dedup import is_generic_token, search_name


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _tokenize(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(t) > 2}


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

    def delete_gold(self, gid: str) -> None:
        """Remove a gold row outright. Used where a merge leaves nothing to redirect to —
        a producer is only ever reached through the rows that name it."""
        with self._lock:
            self._db.execute("DELETE FROM gold WHERE id=?", (gid,))
            self._db.commit()

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
        """Token-overlap name match, best-first — the laptop stand-in for pg_trgm. The
        Postgres store swaps in real trigram similarity behind this same signature."""
        want = _tokenize(text)
        if not want:
            return []
        # A product is scored against its own name *and* its brand-qualified name, because
        # the catalog splits a label across two rows — brand "Tito's" + name "Handmade
        # Vodka" for what a bottle simply calls Tito's Handmade Vodka.
        ident = {t for t in want if not is_generic_token(t)}
        brands = {b["id"]: b.get("name") or "" for b in self.iter_gold("brand")}
        scored: list[tuple[dict, float]] = []
        for p in self.iter_gold("product"):
            raw = p.get("name", "")
            qualified = search_name(raw, brands.get(p.get("brand_id") or ""))
            name_tokens = _tokenize(raw) | _tokenize(qualified)
            if not name_tokens:
                continue
            overlap = want & name_tokens
            if not overlap:
                continue
            # Best of both coverages, mirroring the two directions of pg_trgm's
            # `word_similarity` in the Postgres store. Dividing only by the name length
            # rewards stubby catalog entries: for "BOMBAY SAPPHIRE", "Gin Bombay" covers
            # half its own two tokens (0.5) while "Bombay Sapphire London Dry Gin" covers
            # only two of its five (0.4) — and the wrong one wins. Covering the *query*
            # instead gives the right answer 1.0.
            score = max(len(overlap) / max(len(name_tokens), 1),
                        len(overlap) / max(len(want), 1))
            # How much of the *label* this row accounts for. Ties at the top are the norm
            # — every name wholly inside the label scores 1.0 — so the row explaining more
            # of what was read wins, mirroring the plain-similarity tiebreak the Postgres
            # store uses. Only identifying tokens count: crediting category words would let
            # a row matching "extra stout" beat one matching the brand.
            covered = len(ident & _tokenize(qualified)) / max(len(ident), 1) if ident else 0.0
            scored.append((p, round(score, 3), covered))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(p, sim) for p, sim, _ in scored[:limit]]

    def refresh_search_names(self) -> int:
        """No-op: this store builds the brand-qualified name per query rather than storing
        it, so there is nothing to backfill. Present so callers need not know which store
        they hold."""
        return 0

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
    def delete_gold(self, gid: str) -> None: ...
    def iter_gold(self, entity_type: str) -> Iterator[dict[str, Any]]: ...
    def counts(self) -> dict[str, int]: ...
    def search_gold_products(self, q: str, limit: int = 20) -> list[dict[str, Any]]: ...
    def match_products(self, text: str, limit: int = 3) -> list[tuple[dict, float]]: ...
    def refresh_search_names(self) -> int: ...
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
