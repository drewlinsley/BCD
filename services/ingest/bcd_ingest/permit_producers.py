"""Rekey TTB producers onto the permit holder, not the brand on the label.

A COLA filing names a brand, not a brewery. TTB's `permittee` field would name the company,
but it is populated on **5 of 1,049,670 rows**, so the connector fell through to `brand_name`
and every beer became its own producer: 219,313 producers for 534,145 products, and 164,222 of
those producers sharing a name with their own single product. Nothing could be answered at
producer level, because no producer had a range.

`permit_no` is the fix and it was there all along — present on 98.3% of rows, 27,830 distinct
values. It is the licence number of the actual bonded brewery or distillery, so it groups
Focal Banger with Alena, Beelzebub, Holy Cow and the rest of one Vermont brewery's range.

Naming a permit is the part TTB cannot fully answer. Where a row carries a `fanciful_name`,
its `brand_name` is acting as the *brand* with the beer in the other field — the one place the
registry names something larger than a single product, and good for 64% of permits. Otherwise
the best available name is the permit's most-filed product standing in for its maker, recorded
with `name_is_a_brand: false` so a caller can tell the two apart and decline to show a stand-in
as a brewery.

Idempotent, and safe to re-run after any TTB backfill.
"""

from __future__ import annotations

import psycopg

from .connectors.ttb_cola import _titlecase

# The same permit is filed both ways -- "BR CO AVE 1" and "BR-CO-AVE-1" are one brewery, and
# there are a dozen such pairs. Normalising is the point, not a collision to be avoided.
_PRODUCER_ID = (
    "'prod:ttb-cola-registry:permit:' || "
    "left(regexp_replace(lower({col}), '[^a-z0-9]+', '-', 'g'), 60)"
)

_REKEY = f"""
CREATE TEMP TABLE permit_pick ON COMMIT DROP AS
WITH rows AS (
  SELECT {_PRODUCER_ID.format(col="payload->>'permit_no'")} AS producer_id,
         payload->>'brand_name'  AS brand,
         payload->>'origin_desc' AS region,
         (coalesce(payload->>'fanciful_name','') <> '') AS brand_acts_as_brand
  FROM bronze
  WHERE source_id = 'ttb-cola-registry'
    AND coalesce(payload->>'permit_no','') <> ''
    AND coalesce(payload->>'brand_name','') <> ''
),
tally AS (
  SELECT producer_id, brand, bool_or(brand_acts_as_brand) AS branded, count(*) AS n
  FROM rows GROUP BY producer_id, brand
),
ranked AS (
  SELECT producer_id, brand, branded,
         row_number() OVER (PARTITION BY producer_id
                            ORDER BY branded DESC, n DESC, brand) AS rn
  FROM tally
),
regions AS (
  SELECT producer_id, mode() WITHIN GROUP (ORDER BY region) AS region
  FROM rows WHERE region IS NOT NULL GROUP BY producer_id
)
SELECT r.producer_id, r.brand AS name, r.branded, g.region
FROM ranked r LEFT JOIN regions g USING (producer_id)
WHERE r.rn = 1;

INSERT INTO gold (id, entity_type, record, name, updated_at)
SELECT producer_id, 'producer',
       jsonb_build_object(
         'id', producer_id, 'name', name, 'kind', 'permittee',
         -- Title Case, to match how the connector writes regions from silver and how
         -- OpenBreweryDB writes its own -- region is the join key for location linking.
         -- (Names get the same treatment in `_titlecase_names`, which needs the connector's
         -- apostrophe-aware helper: initcap turns "STRANAHAN'S" into "Stranahan'S".)
         'region', initcap(region),
         'ttb_permit', replace(producer_id, 'prod:ttb-cola-registry:permit:', ''),
         'name_is_a_brand', branded,
         'city', NULL, 'lat', NULL, 'lon', NULL,
         'aliases', '[]'::jsonb, 'country', NULL, 'website', NULL, 'parent_company', NULL),
       name, now()
FROM permit_pick
ON CONFLICT (id) DO UPDATE
  SET record = EXCLUDED.record, name = EXCLUDED.name, updated_at = now();

UPDATE gold g
SET record = jsonb_set(g.record, '{{producer_id}}', to_jsonb(m.producer_id)), updated_at = now()
FROM (
  SELECT DISTINCT ON (b.payload->>'ttb_id')
         'ttb:' || (b.payload->>'ttb_id') AS gid,
         {_PRODUCER_ID.format(col="b.payload->>'permit_no'")} AS producer_id
  FROM bronze b
  WHERE b.source_id = 'ttb-cola-registry' AND coalesce(b.payload->>'permit_no','') <> ''
) m
WHERE g.entity_type = 'product' AND g.id = m.gid;
"""

# A brand and its products must agree about who makes them. Joined through the products, the
# only link that survives the rekey.
_REPOINT_BRANDS = """
UPDATE gold b
SET record = jsonb_set(b.record, '{producer_id}', to_jsonb(m.pid)), updated_at = now()
FROM (
  SELECT DISTINCT ON (p.record->>'brand_id')
         p.record->>'brand_id' AS bid, p.record->>'producer_id' AS pid
  FROM gold p
  WHERE p.entity_type = 'product' AND p.record->>'brand_id' IS NOT NULL
    AND p.record->>'producer_id' LIKE 'prod:ttb-cola-registry:permit:%'
) m
WHERE b.entity_type = 'brand' AND b.id = m.bid;
"""

# TTB only. An OpenBreweryDB producer with no products is a real brewery carrying the city and
# coordinates that name-linking needs -- "Alchemist Cannery, Stowe VT" is exactly one of those.
_DROP_EMPTIED = """
DELETE FROM gold pr
WHERE pr.entity_type = 'producer'
  AND pr.id LIKE 'prod:ttb-cola-registry:%'
  AND pr.id NOT LIKE 'prod:ttb-cola-registry:permit:%'
  AND NOT EXISTS (SELECT 1 FROM gold p WHERE p.entity_type = 'product'
                  AND p.record->>'producer_id' = pr.id);
"""


def _titlecase_names(conn: psycopg.Connection) -> int:
    """Un-SHOUT the names the registry stores in caps.

    Done here rather than in SQL because `initcap` has no idea what an apostrophe is --
    "STRANAHAN'S" comes back as "Stranahan'S". The connector already title-cases brand names on
    the way into silver, so a name minted by either path has to read the same.
    """
    rows = conn.execute(
        "SELECT id, name FROM gold WHERE entity_type='producer'"
        " AND id LIKE 'prod:ttb-cola-registry:permit:%'").fetchall()
    fixed = [(t, t, i) for i, n in rows if n and (t := _titlecase(n)) != n]
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE gold SET name=%s, record=jsonb_set(record,'{name}',to_jsonb(%s::text)),"
            " updated_at=now() WHERE id=%s", fixed)
    return len(fixed)


def rekey_to_permits(dsn: str) -> dict[str, int]:
    """Point every TTB product at its permit holder. Returns row counts per step."""
    counts: dict[str, int] = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(_REKEY)
            counts["products_relinked"] = cur.rowcount
            cur.execute(_REPOINT_BRANDS)
            counts["brands_repointed"] = cur.rowcount
            cur.execute(_DROP_EMPTIED)
            counts["producers_dropped"] = cur.rowcount
        counts["names_titlecased"] = _titlecase_names(conn)
        conn.commit()
    return counts
