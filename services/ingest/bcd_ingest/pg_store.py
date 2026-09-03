"""Postgres-backed medallion store — the production sibling of MedallionStore.

Same interface as the SQLite dev store (so `open_store()` swaps them transparently), but
backed by Postgres 16 with two extensions doing the heavy lifting the SQLite path only
stubs:

  * **pg_trgm** — real trigram similarity for name matching (`match_products`), replacing
    the laptop store's token-overlap heuristic. GIN-indexed.
  * **pgvector** — cosine ANN over the 25-axis SensoryVector (`nearest_by_sensory`), the
    thing that makes "recommend something I'd like" a single indexed query. HNSW-indexed.

Every gold row still carries the full canonical `record` as jsonb; `name`, `sensory`,
`lat`, and `lon` are denormalized out of it into typed/indexed columns on write so the
search operators have something to bite on. The invariant is unchanged: never lose raw
bytes, every gold field traces back to a bronze document id.

Concurrency: one connection in autocommit, serialized by a lock — same pragmatic choice
as the SQLite store, since FastAPI runs sync endpoints in a threadpool. Production swaps
this for a psycopg_pool ConnectionPool; the method surface stays identical.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import psycopg
from bcd_schema import SENSORY_AXES
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb

from .dedup import is_generic_token, search_name
from .store import BronzeDoc  # reuse the shared bronze dataclass


def _normalize_dsn(url: str) -> str:
    """Accept SQLAlchemy-style URLs (`postgresql+psycopg://...`) and hand psycopg a plain
    `postgresql://...` — the `+driver` suffix is a SQLAlchemy convention psycopg rejects."""
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://"):]
    if url.startswith("postgres+psycopg://"):
        return "postgresql://" + url[len("postgres+psycopg://"):]
    return url


def _safe_label(dsn: str) -> str:
    """host/db label with no password, for logging."""
    try:
        d = conninfo_to_dict(dsn)
        host, port = d.get("host", "localhost"), d.get("port", 5432)
        return f"postgres://{host}:{port}/{d.get('dbname', '')}"
    except Exception:
        return "postgres"


def _vec_literal(arr: list[float]) -> str:
    """pgvector text input format: [a,b,c]. Cast to ::vector in SQL."""
    return "[" + ",".join(f"{float(x):.6g}" for x in arr) + "]"


def _sensory_array(record: dict[str, Any]) -> list[float] | None:
    """Dense 25-vector in SENSORY_AXES order from a product record's sensory block, or
    None when there's no sensory or it carries no signal (all-zero → NULL so cosine
    ordering stays well-defined and `sensory IS NOT NULL` filters it out)."""
    sv = record.get("sensory")
    if not sv:
        return None
    axes = sv.get("axes") or {}
    arr = [float(axes.get(a, 0.0)) for a in SENSORY_AXES]
    return arr if any(arr) else None


class PostgresStore:
    def __init__(self, url: str = "postgresql://localhost:5432/bcd", *,
                 search_path: str | None = None) -> None:
        """`search_path` (e.g. "bcd_test,public") scopes every table this store creates
        and reads to that schema — used for isolated test runs and future multi-tenancy.
        `public` must stay on the path so the vector/pg_trgm types + operators resolve."""
        self.dsn = _normalize_dsn(url)
        self.db_path = _safe_label(self.dsn)  # named db_path for parity with MedallionStore
        kwargs: dict[str, Any] = {"autocommit": True}
        if search_path:
            # applied at connect, before _init(), so DDL lands in the leading schema
            kwargs["options"] = f"-c search_path={search_path}"
        self._conn = psycopg.connect(self.dsn, **kwargs)
        self._lock = threading.Lock()
        if search_path:
            with self._conn.cursor() as cur:
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {search_path.split(',')[0].strip()}")
        self._init()
        self._set_trgm_thresholds()

    def _set_trgm_thresholds(self) -> None:
        """Thresholds for the GIN gate in `match_products`.

        Deliberately below the resolver's own 0.5 floor: the gate only decides what gets
        scored, so anything it lets through that scores badly is rejected downstream, while
        anything it wrongly excludes is invisible. Erring low costs a few hundred rows of
        scoring; erring high silently loses matches.
        """
        with self._lock, self._conn.cursor() as cur:
            cur.execute("SELECT set_limit(0.3)")            # similarity, and loads pg_trgm
            # 0.4 rather than the 0.5 default, measured: "Guinness Draught" scores 0.486
            # against a real canned label and would be excluded by a hair. Below 0.35 the
            # gate stops discriminating — "SIERRA NEVADA PALE ALE" goes from 1,204
            # candidate rows to 17,336 and the scan costs half a second.
            cur.execute("SET pg_trgm.word_similarity_threshold = 0.4")

    def _init(self) -> None:
        with self._lock, self._conn.cursor() as cur:
            # Extensions are pre-provisioned in prod; IF NOT EXISTS makes a fresh db work.
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            # Lower the trigram match floor so short OCR fragments still hit; the `%`
            # operator (and thus the GIN index) honors this session GUC.
            cur.execute("SELECT set_config('pg_trgm.similarity_threshold', '0.2', false)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bronze (
                    id          text PRIMARY KEY,
                    source_id   text NOT NULL,
                    natural_key text NOT NULL,
                    fetched_at  timestamptz NOT NULL,
                    url         text,
                    payload     jsonb NOT NULL
                );
                CREATE TABLE IF NOT EXISTS silver (
                    id          text PRIMARY KEY,
                    source_id   text NOT NULL,
                    entity_type text NOT NULL,
                    bronze_id   text NOT NULL REFERENCES bronze(id),
                    record      jsonb NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gold (
                    id          text PRIMARY KEY,
                    entity_type text NOT NULL,
                    record      jsonb NOT NULL,
                    name        text,
                    sensory     vector(25),
                    lat         double precision,
                    lon         double precision,
                    updated_at  timestamptz NOT NULL
                );
                ALTER TABLE gold ADD COLUMN IF NOT EXISTS search_name text;
                CREATE INDEX IF NOT EXISTS ix_silver_type ON silver(entity_type);
                CREATE INDEX IF NOT EXISTS ix_gold_type ON gold(entity_type);
                CREATE INDEX IF NOT EXISTS ix_gold_name_trgm
                    ON gold USING gin (name gin_trgm_ops);
                CREATE INDEX IF NOT EXISTS ix_gold_search_name_trgm
                    ON gold USING gin (search_name gin_trgm_ops);
                -- The gate `match_products` runs, and the only index it can use. Partial
                -- because scan resolution only ever asks about products: indexed over every
                -- entity type the planner BitmapANDs the trigram hit against ix_gold_type and
                -- walks all 457k entries, which is most of the 7.4s a resolve used to cost.
                -- Over the *coalesced* expression because `search_name` is filled by a batch
                -- refresh, so a freshly promoted row has none and must still be reachable by
                -- its plain name -- and because one probe of this costs half what separate
                -- `name` and `search_name` probes do, for provably identical rows.
                CREATE INDEX IF NOT EXISTS ix_gold_qualified_trgm_product
                    ON gold USING gin ((coalesce(search_name, name, '')) gin_trgm_ops)
                    WHERE entity_type='product';
                CREATE INDEX IF NOT EXISTS ix_gold_sensory_hnsw
                    ON gold USING hnsw (sensory vector_cosine_ops);
                """
            )

    # ---- bronze ----
    def put_bronze(self, doc: BronzeDoc) -> None:
        fetched = (
            datetime.fromisoformat(doc.fetched_at) if doc.fetched_at else datetime.now(UTC)
        )
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bronze (id, source_id, natural_key, fetched_at, url, payload)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    source_id=EXCLUDED.source_id, natural_key=EXCLUDED.natural_key,
                    fetched_at=EXCLUDED.fetched_at, url=EXCLUDED.url, payload=EXCLUDED.payload
                """,
                (doc.id, doc.source_id, doc.natural_key, fetched, doc.url, Jsonb(doc.payload)),
            )

    def iter_bronze(self, source_id: str) -> Iterator[BronzeDoc]:
        with self._lock, self._conn.cursor() as cur:
            rows = cur.execute(
                "SELECT id, source_id, natural_key, fetched_at, url, payload "
                "FROM bronze WHERE source_id=%s",
                (source_id,),
            ).fetchall()
        for r in rows:
            fetched = r[3].isoformat() if isinstance(r[3], datetime) else str(r[3])
            yield BronzeDoc(r[0], r[1], r[2], fetched, r[4], r[5])

    # ---- silver ----
    def put_silver(self, sid: str, source_id: str, entity_type: str,
                   bronze_id: str, record: dict[str, Any]) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO silver (id, source_id, entity_type, bronze_id, record)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    source_id=EXCLUDED.source_id, entity_type=EXCLUDED.entity_type,
                    bronze_id=EXCLUDED.bronze_id, record=EXCLUDED.record
                """,
                (sid, source_id, entity_type, bronze_id, Jsonb(record)),
            )

    def iter_silver(self, entity_type: str) -> Iterator[dict[str, Any]]:
        with self._lock, self._conn.cursor() as cur:
            rows = cur.execute(
                "SELECT record FROM silver WHERE entity_type=%s", (entity_type,)
            ).fetchall()
        for r in rows:
            yield r[0]

    # ---- gold ----
    def put_gold(self, gid: str, entity_type: str, record: dict[str, Any]) -> None:
        name = record.get("name")
        arr = _sensory_array(record) if entity_type == "product" else None
        sensory = _vec_literal(arr) if arr is not None else None
        lat = record.get("lat")
        lon = record.get("lon")
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO gold (id, entity_type, record, name, sensory, lat, lon, updated_at)
                VALUES (%s, %s, %s, %s, %s::vector, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    entity_type=EXCLUDED.entity_type, record=EXCLUDED.record,
                    name=EXCLUDED.name, sensory=EXCLUDED.sensory,
                    lat=EXCLUDED.lat, lon=EXCLUDED.lon, updated_at=now()
                """,
                (gid, entity_type, Jsonb(record), name, sensory, lat, lon),
            )

    def get_gold(self, gid: str) -> dict[str, Any] | None:
        with self._lock, self._conn.cursor() as cur:
            row = cur.execute("SELECT record FROM gold WHERE id=%s", (gid,)).fetchone()
        return row[0] if row else None

    def delete_gold(self, gid: str) -> None:
        """Remove a gold row outright. Used where a merge leaves nothing to redirect to —
        a producer is only ever reached through the rows that name it."""
        with self._lock, self._conn.cursor() as cur:
            cur.execute("DELETE FROM gold WHERE id=%s", (gid,))

    def iter_gold(self, entity_type: str) -> Iterator[dict[str, Any]]:
        with self._lock, self._conn.cursor() as cur:
            rows = cur.execute(
                "SELECT record FROM gold WHERE entity_type=%s", (entity_type,)
            ).fetchall()
        for r in rows:
            yield r[0]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock, self._conn.cursor() as cur:
            for tbl in ("bronze", "silver", "gold"):
                out[tbl] = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        return out

    def search_gold_products(self, q: str, limit: int = 20) -> list[dict[str, Any]]:
        """Trigram match on name, falling back to a substring scan of the record — the
        real version of MedallionStore's LIKE placeholder."""
        with self._lock, self._conn.cursor() as cur:
            rows = cur.execute(
                """
                SELECT record, similarity(coalesce(name,''), %s) AS sim
                FROM gold
                WHERE entity_type='product'
                  AND (coalesce(name,'') %% %s OR record::text ILIKE %s)
                ORDER BY sim DESC
                LIMIT %s
                """,
                (q, q, f"%{q}%", limit),
            ).fetchall()
        return [r[0] for r in rows]

    def refresh_search_names(self) -> int:
        """Denormalise the brand-qualified name onto every product, for matching only.

        The rule lives in `dedup.search_name` so this store and the dev store agree; see
        it for why a brand is sometimes withheld. Idempotent — run after any promote.
        """
        brands = {b["id"]: b.get("name") or "" for b in self.iter_gold("brand")}
        rows = [
            (search_name(p.get("name") or "", brands.get(p.get("brand_id") or "")), p["id"])
            for p in self.iter_gold("product")
        ]
        with self._lock, self._conn.cursor() as cur:
            cur.executemany("UPDATE gold SET search_name=%s WHERE id=%s", rows)
        return len(rows)

    # ---- search (used by the resolver / recommend) ----
    def match_products(self, text: str, limit: int = 3) -> list[tuple[dict, float]]:
        """Best-first name match for an OCR line, scored 0-1.

        `word_similarity` is directional — it finds its first argument inside a continuous
        extent of its second — and a label can be noisy in either direction, so both are
        taken:

          * `word_similarity(name, line)` finds a short catalog name inside a noisy OCR
            line: "GUINNESS DRAUGHT 440ML EXTRA STOUT" -> Guinness.
          * `word_similarity(line, name)` finds a short OCR line inside a longer catalog
            name: "BOMBAY SAPPHIRE" -> "Bombay Sapphire London Dry Gin".

        Only the first was measured for a long time, which quietly biased matching toward
        stubby catalog entries: every extra word in the *right* answer diluted its score
        while a short wrong one kept a high one. "BOMBAY SAPPHIRE" returned "Gin Bombay"
        (0.636) over "Bombay Sapphire London Dry Gin" (0.516), and "HEADY TOPPER" scored
        the product actually called that only 0.500. Both are 1.000 with the second
        direction included.

        Plain `similarity` stays for the case where the OCR simply *is* the name
        ("HEINEKEN"). The caller applies a confidence floor and token-support checks, so
        this returns the top few regardless and lets the resolver reject weak ones.

        Scoring every row was fine at a few thousand products and is not at 363k: the
        planner parallel-seq-scans the table and one resolve took 7.4 seconds, against a
        HUD that ticks every 700ms. So a GIN gate runs first and the six similarity terms
        are computed only on what survives it — 966 rows rather than 363,014, for the same
        answer.

        The gate cannot simply mirror the scoring. `gin_trgm_ops` indexes `%` and `%>` but
        NOT `<%`, so `word_similarity(name, line)` — the Guinness direction — has no
        index-usable operator at all. Gating on the whole line would therefore drop
        exactly the case that direction exists to catch. The line's own words are added to
        the gate to cover it: a catalog row named "Guinness" is reached from the token
        GUINNESS even though the full noisy line never comes close.
        """
        text = (text or "").strip()
        if not text:
            return []

        # Which words are worth an index probe of their own. Style and category words are
        # not: "PALE" matches 19,203 products and costs 300ms to find them, because
        # word-similarity is 1.0 against every name containing the word. Dropping them
        # takes the gate from 20,343 candidate rows to 803. `is_generic_token` is the same
        # vocabulary dedup and the resolver already judge identity with.
        toks = [t for t in re.findall(r"[^\W\d_]{4,}", text, re.UNICODE)
                if not is_generic_token(t)][:6]

        # The whole line, then its identifying words. The line alone is not enough: a
        # catalog row named just "Guinness" scores word_similarity(line, name) = 0.257
        # against "GUINNESS DRAUGHT 440ML EXTRA STOUT" and no workable threshold reaches
        # it, while the token GUINNESS finds it outright.
        # One probe per term, against the brand-qualified name the index is built on. Probing
        # `name` separately is redundant: search_name is "<brand> <name>", so every row the
        # name probe finds the qualified probe finds too — measured across the live catalog,
        # eight probes, zero rows lost — and each probe costs 100-600ms, because a GIN
        # trigram scan walks a posting list sized by how common the *trigrams* are, not by how
        # many rows come back ("stowe" costs 534ms to return 20 rows).
        qualified = "coalesce(search_name, name, '')"
        gate = [f"{qualified} %%> %s"]
        params: list[Any] = [text]
        for t in toks:
            gate += [f"{qualified} %% %s"]
            params += [t]

        sql = f"""
            WITH candidate AS (
                SELECT id, record, name, search_name
                FROM gold
                WHERE entity_type='product' AND ({" OR ".join(gate)})
            )
            SELECT record,
                   GREATEST(similarity(coalesce(name,''), %s),
                            word_similarity(coalesce(name,''), %s),
                            word_similarity(%s, coalesce(name,'')),
                            similarity(coalesce(search_name, name, ''), %s),
                            word_similarity(coalesce(search_name, name, ''), %s),
                            word_similarity(%s, coalesce(search_name, name, ''))) AS sim
            FROM candidate
            -- Ties at the top are the norm, not the exception: `word_similarity` scores
            -- 1.0 for ANY name wholly contained in the label, so "Handmade Vodka" and
            -- "Tito's Handmade Vodka" both max out on "TITOS HANDMADE VODKA". Plain
            -- `similarity` is the tiebreak because it is the only one of the three that
            -- penalises what the candidate *leaves out* — it drops for the row missing
            -- "Titos", and rises for the one that accounts for the whole label.
            ORDER BY sim DESC,
                     similarity(coalesce(search_name, name, ''), %s) DESC,
                     id
            LIMIT %s
        """
        params += [text] * 7 + [limit]
        with self._lock, self._conn.cursor() as cur:
            rows = cur.execute(sql, params).fetchall()
        return [(r[0], round(float(r[1]), 3)) for r in rows]

    def nearest_by_sensory(self, vec: list[float], limit: int = 10) -> list[dict[str, Any]]:
        """Cosine ANN over the sensory column — the pgvector core of recommendation."""
        with self._lock, self._conn.cursor() as cur:
            rows = cur.execute(
                """
                SELECT record
                FROM gold
                WHERE entity_type='product' AND sensory IS NOT NULL
                ORDER BY sensory <=> %s::vector
                LIMIT %s
                """,
                (_vec_literal(vec), limit),
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()


__all__ = ["PostgresStore"]
