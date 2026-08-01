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
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


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

    def close(self) -> None:
        self._db.close()
