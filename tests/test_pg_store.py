"""PostgresStore — the production backend, exercised only when a Postgres is reachable.

Skips cleanly (no failure) when there's no server, so the default `make test` stays green
on a bare laptop. When a DB IS present these assert the two things the SQLite dev store
can only fake: real pg_trgm fuzzy matching (typo-tolerant) and pgvector cosine ANN. Every
run is isolated in a throwaway schema that's dropped on teardown, so it never touches the
dev catalog.
"""

from __future__ import annotations

import os

import pytest

psycopg = pytest.importorskip("psycopg")

from bcd_ingest.pg_store import PostgresStore, _normalize_dsn  # noqa: E402
from bcd_ingest.store import BronzeDoc, Store, doc_id, open_store  # noqa: E402
from bcd_schema import (  # noqa: E402
    Brand,
    Category,
    Producer,
    Product,
    SensorySource,
    SensoryVector,
)

_URL = os.environ.get("BCD_DATABASE_URL", "postgresql://localhost:5432/bcd")
_SCHEMA = "bcd_test_pgstore"


@pytest.fixture()
def pg():
    """A PostgresStore scoped to a fresh throwaway schema, or skip if no server."""
    try:
        admin = psycopg.connect(_normalize_dsn(_URL), autocommit=True, connect_timeout=3)
    except Exception as exc:  # noqa: BLE001 — any connect failure -> skip, not fail
        pytest.skip(f"Postgres not reachable at {_URL}: {exc}")
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
    admin.close()
    store = PostgresStore(_URL, search_path=f"{_SCHEMA},public")
    try:
        yield store
    finally:
        with store._conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
        store.close()


def _seed_product(store: PostgresStore, pid: str, name: str,
                  axes: dict[str, float] | None) -> None:
    store.put_gold(f"prod:{pid}", "producer",
                   Producer(id=f"prod:{pid}", name="P", lat=40.0, lon=-74.0)
                   .model_dump(mode="json"))
    store.put_gold(f"brand:{pid}", "brand",
                   Brand(id=f"brand:{pid}", producer_id=f"prod:{pid}", name=name)
                   .model_dump(mode="json"))
    sensory = (SensoryVector(source=SensorySource.CHEMISTRY_PRIOR, confidence=0.6, axes=axes)
               if axes else None)
    p = Product(id=pid, brand_id=f"brand:{pid}", producer_id=f"prod:{pid}",
                category=Category.BEER, name=name, sensory=sensory)
    store.put_gold(pid, "product", p.model_dump(mode="json"))


def test_medallion_roundtrip(pg: PostgresStore):
    doc = BronzeDoc(id=doc_id("t", "k1"), source_id="t", natural_key="k1",
                    fetched_at="", url="https://example.com/x", payload={"a": 1, "b": [2, 3]})
    pg.put_bronze(doc)
    got = list(pg.iter_bronze("t"))
    assert len(got) == 1 and got[0].payload == {"a": 1, "b": [2, 3]}
    assert got[0].fetched_at  # server-stamped when blank

    pg.put_silver("s1", "t", "thing", doc.id, {"name": "x"})
    assert list(pg.iter_silver("thing")) == [{"name": "x"}]

    _seed_product(pg, "p1", "Test Beer", {"citrus": 0.5})
    assert pg.get_gold("p1")["name"] == "Test Beer"
    assert {r["name"] for r in pg.iter_gold("product")} == {"Test Beer"}
    counts = pg.counts()
    assert counts["bronze"] == 1 and counts["gold"] >= 3  # producer+brand+product


def test_trigram_match_is_typo_tolerant(pg: PostgresStore):
    _seed_product(pg, "gh", "Galaxy Haze", {"tropical": 1.0})
    _seed_product(pg, "ms", "Midnight Roast Stout", {"roasted_coffee_choc": 1.0})
    _seed_product(pg, "hw", "Bavarian Hefeweizen", {"banana_ester": 0.8})

    # exact-ish
    top = pg.match_products("galaxy haze")
    assert top and top[0][0]["name"] == "Galaxy Haze"
    # real trigram value: a misspelling the SQLite token-overlap store would miss
    fuzzy = pg.match_products("midnite rost stowt")
    assert fuzzy and fuzzy[0][0]["name"] == "Midnight Roast Stout"
    assert 0.0 < fuzzy[0][1] <= 1.0  # a similarity score comes back


def test_nearest_by_sensory_cosine(pg: PostgresStore):
    _seed_product(pg, "trop", "Tropical IPA", {"tropical": 1.0, "citrus": 0.9})
    _seed_product(pg, "roast", "Roasty Stout", {"roasted_coffee_choc": 1.0, "bitterness": 0.4})
    _seed_product(pg, "none", "No Sensory Lager", None)  # must be excluded

    from bcd_schema import SENSORY_AXES
    ax = {a: i for i, a in enumerate(SENSORY_AXES)}
    ideal = [0.0] * len(SENSORY_AXES)
    ideal[ax["tropical"]] = 0.9
    ideal[ax["citrus"]] = 0.8

    near = pg.nearest_by_sensory(ideal, limit=10)
    names = [p["name"] for p in near]
    assert names[0] == "Tropical IPA"          # cosine puts the tropical one first
    assert "No Sensory Lager" not in names     # NULL vector rows never surface
    assert "Roasty Stout" in names


def test_open_store_selects_postgres_from_url():
    """Factory picks Postgres when a database URL is supplied, SQLite otherwise."""
    try:
        s = open_store(url=_URL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable: {exc}")
    assert isinstance(s, PostgresStore)
    assert isinstance(s, Store)
    s.close()
