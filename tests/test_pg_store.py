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


def test_a_short_catalog_name_is_found_inside_a_noisy_label(pg: PostgresStore):
    """The case the index gate cannot see, and the reason it probes the line's words.

    `gin_trgm_ops` indexes `%` and `%>` but not `<%`, so `word_similarity(name, line)` —
    a short catalog name found inside a long label — has no index-usable operator. Gating
    on the whole line alone drops it: a row named "Guinness" scores word_similarity(line,
    name) = 0.257 against a real canned label, below any threshold that still
    discriminates. The identifying word GUINNESS reaches it outright.
    """
    _seed_product(pg, "gd", "Guinness", {"roasted_coffee_choc": 0.9})
    _seed_product(pg, "sn", "Sierra Nevada Pale Ale", {"citrus": 0.6})

    top = pg.match_products("GUINNESS DRAUGHT 440ML EXTRA STOUT")
    assert top and top[0][0]["name"] == "Guinness"


def test_a_label_of_only_style_words_still_answers(pg: PostgresStore):
    """Every word here is a category word, so the gate probes none of them individually
    and falls back to the whole-line term. It must still return something rather than
    error or scan the table."""
    _seed_product(pg, "ipa", "Hazy IPA", {"citrus": 0.7})
    assert isinstance(pg.match_products("HAZY IPA"), list)


def test_concurrent_frame_matching_agrees_with_one_line_at_a_time(pg: PostgresStore):
    """`match_products_many` fans a frame out over its own connections, so the guarantee that
    matters is that concurrency changes nothing but the clock.

    The trigram thresholds are per-*session* GUCs. A reader that was handed out without them
    would gate differently and quietly return fewer matches for the same label — the kind of
    bug that shows up as an intermittently unrecognised beer, not as an error.
    """
    for pid, name in (("p1", "Heady Topper"), ("p2", "Sierra Nevada Pale Ale"),
                      ("p3", "Guinness Draught"), ("p4", "Bombay Sapphire London Dry Gin"),
                      ("p5", "Pint Cake"), ("p6", "Tootsie Topper")):
        _seed_product(pg, pid, name, None)
    pg.refresh_search_names()

    frame = ["HEADY TOPPER", "SIERRA NEVADA PALE ALE", "GUINNESS DRAUGHT EXTRA STOUT",
             "BOMBAY SAPPHIRE", "PINT", "TOPPER", "NOTHING LIKE THIS IN THE CATALOG"]
    one_at_a_time = [pg.match_products(t) for t in frame]
    together = pg.match_products_many(frame)

    assert len(together) == len(frame)          # positionally aligned with the input
    for line, seq, con in zip(frame, one_at_a_time, together, strict=True):
        assert [r["id"] for r, _ in con] == [r["id"] for r, _ in seq], line
        assert [s for _, s in con] == [s for _, s in seq], line


def test_frame_matching_handles_the_degenerate_frames(pg: PostgresStore):
    _seed_product(pg, "p1", "Heady Topper", None)
    pg.refresh_search_names()
    assert pg.match_products_many([]) == []
    assert len(pg.match_products_many(["HEADY TOPPER"])) == 1
    # More lines than the pool has workers: every line still gets its own answer slot.
    many = pg.match_products_many(["HEADY TOPPER"] * 9)
    assert len(many) == 9
    assert all(m and m[0][0]["name"] == "Heady Topper" for m in many)


def test_frame_readers_are_closed_with_the_store(pg: PostgresStore):
    """The pool holds real connections; leaking them exhausts the server's slots.

    Uses a second store over the same schema so the fixture's own connection survives to
    tear the schema down.
    """
    _seed_product(pg, "p1", "Heady Topper", None)
    pg.refresh_search_names()
    other = PostgresStore(_URL, search_path=f"{_SCHEMA},public")
    other.match_products_many(
        ["HEADY TOPPER", "SOMETHING ELSE", "A THIRD LINE", "AND A FOURTH"])
    readers = list(other._readers)
    assert readers, "expected the frame match to have opened reader connections"
    other.close()
    assert all(r.closed for r in readers)
    assert other._readers == []

def test_a_nul_byte_does_not_end_the_run(pg: PostgresStore):
    # A 2008 TTB label carried a NUL in its fanciful name. jsonb refuses it outright, so the
    # write raised UntranslatableCharacter and took a backfill down 194,101 rows in. The byte
    # is export padding, never content: drop it and keep the rest of the record.
    doc = BronzeDoc(id=doc_id("t", "nul"), source_id="t", natural_key="nul", fetched_at="",
                    url=None,
                    payload={"fanciful_name": "\x00\x00", "brand_name": "BRAND\x00X",
                             "nested": [{"k\x00": "v\x00"}]})
    pg.put_bronze(doc)
    (got,) = list(pg.iter_bronze("t"))
    assert got.payload["fanciful_name"] == ""
    assert got.payload["brand_name"] == "BRANDX", "the rest of the field survives"
    assert got.payload["nested"] == [{"k": "v"}], "keys and nested values too"

    pg.put_silver("s-nul", "t", "thing", doc.id, {"name": "A\x00B"})
    assert list(pg.iter_silver("thing")) == [{"name": "AB"}]

    _seed_product(pg, "p-nul", "Nul\x00Beer", {"citrus": 0.4})
    assert pg.get_gold("p-nul")["name"] == "NulBeer", "the text column rejects it too"


def test_a_line_naming_only_a_style_reaches_only_styleless_names(pg: PostgresStore):
    # _token_supported accepts a candidate whose name carries no identity only when the
    # line carries none either, so for such a line every identifying-named row is scored
    # and then discarded. On the live catalog "INDIA PALE ALE" gated 21,606 rows to keep
    # almost none; the partial index answers the same question against 1.8% of the table.
    _seed_product(pg, "p-generic", "India Pale Ale", {"citrus": 0.4})
    _seed_product(pg, "p-named", "Sierra Nevada India Pale Ale", {"citrus": 0.5})

    got = {r["name"] for r, _ in pg.match_products("INDIA PALE ALE", limit=10)}
    assert got == {"India Pale Ale"}, "the identifying-named row is not a candidate at all"

    # A line that does carry identity still reaches it, by the ordinary path.
    got = {r["name"] for r, _ in pg.match_products("SIERRA NEVADA INDIA PALE ALE", limit=10)}
    assert "Sierra Nevada India Pale Ale" in got


def test_the_generic_flag_is_written_for_products(pg: PostgresStore):
    _seed_product(pg, "p-a", "India Pale Ale", {"citrus": 0.4})
    _seed_product(pg, "p-b", "Focal Banger", {"citrus": 0.5})
    with pg._conn.cursor() as cur:
        rows = dict(cur.execute(
            "SELECT id, generic FROM gold WHERE id IN ('p-a','p-b')").fetchall())
    assert rows["p-a"] is True
    assert rows["p-b"] is False
