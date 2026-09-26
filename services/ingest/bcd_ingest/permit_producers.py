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
from .dedup import _PRODUCER_SUFFIX

# A brand that is nothing but an incorporation word cannot name anything. Filers do enter them:
# one Alabama permit carries a brand literally recorded as "LLC", and because those three
# filings happened to have fanciful names they outranked every real beer on the permit and
# 2,891 products ended up under a producer called "LLC". The vocabulary is dedup's, so the two
# passes agree on what a company suffix is.
_DEGENERATE = "', '".join(sorted(_PRODUCER_SUFFIX))

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
         coalesce(payload->>'fanciful_name','') AS fanciful,
         (coalesce(payload->>'fanciful_name','') <> '') AS brand_acts_as_brand
  FROM bronze
  WHERE source_id = 'ttb-cola-registry'
    AND coalesce(payload->>'permit_no','') <> ''
    AND coalesce(payload->>'brand_name','') <> ''
    -- ...and is not purely an incorporation word, nor too short to identify anyone ("4b").
    AND lower(regexp_replace(payload->>'brand_name', '[^A-Za-z0-9]', '', 'g'))
          NOT IN ('{_DEGENERATE}')
    AND length(regexp_replace(payload->>'brand_name', '[^A-Za-z0-9]', '', 'g')) >= 3
),
tally AS (
  SELECT producer_id, brand,
         bool_or(brand_acts_as_brand) AS branded,
         -- How many *distinct* products file under this brand. This is what separates a brand
         -- from a product, and filing count is not: on Coors's permit BR-CO-CBC-1, COORS LIGHT
         -- has 567 filings and 2 distinct fanciful names -- it is a beer -- while BLUE MOON has
         -- 287 filings and 112, which is a range. Ranking on volume picked the beer.
         count(DISTINCT fanciful) FILTER (WHERE fanciful <> '') AS products,
         count(*) AS n
  FROM rows GROUP BY producer_id, brand
),
ranked AS (
  SELECT producer_id, brand, branded,
         row_number() OVER (PARTITION BY producer_id
                            ORDER BY branded DESC, products DESC, n DESC, brand) AS rn
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


def _options(search_path: str | None) -> dict[str, str]:
    """Scope a pass to one schema, the way `PostgresStore` does — an isolated test run against
    a real Postgres is the only way to exercise SQL this long."""
    return {"options": f"-c search_path={search_path}"} if search_path else {}


def _titlecase_names(conn: psycopg.Connection,
                     prefix: str = "prod:ttb-cola-registry:permit:") -> int:
    """Un-SHOUT the names the registry stores in caps.

    Done here rather than in SQL because `initcap` has no idea what an apostrophe is --
    "STRANAHAN'S" comes back as "Stranahan'S". The connector already title-cases brand names on
    the way into silver, so a name minted by either path has to read the same.
    """
    rows = conn.execute(
        "SELECT id, name FROM gold WHERE entity_type='producer' AND id LIKE %s",
        (prefix + "%",)).fetchall()
    fixed = [(t, t, i) for i, n in rows if n and (t := _titlecase(n)) != n]
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE gold SET name=%s, record=jsonb_set(record,'{name}',to_jsonb(%s::text)),"
            " updated_at=now() WHERE id=%s", fixed)
    return len(fixed)


def rekey_to_permits(dsn: str, *, search_path: str | None = None) -> dict[str, int]:
    """Point every TTB product at its permit holder. Returns row counts per step."""
    counts: dict[str, int] = {}
    with psycopg.connect(dsn, **_options(search_path)) as conn:
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


# ---- importers are not producers -----------------------------------------------------------

# A permit number says what the permit *authorises*. `CT-I-15074` is a basic importer permit:
# it licences bringing a bottle into the country, which is not a claim about who made it. The
# rekey above treats every permit as a maker and names it after its best-filed brand, which is
# right for a brewery's notice and false for an importer's -- one Diageo import permit holds
# Lagavulin, Talisker, Crown Royal and Johnnie Walker, so `Lagavulin 12 Year Old` was filed
# under a producer called "Johnnie Walker", `Lagavulin Aged 16 Years` under "The Macallan", and
# `CT-I-325` put "Scotland" on Crown Royal, which is Canadian. 113,662 rows are filed on an
# import permit and 89,201 of them carried another company's name, 52,887 of those spirits --
# a third of every spirit in the catalog.
#
# There is no way to name the importer: `permittee` is populated on 5 of 1,049,670 rows and the
# payload carries nothing else that names a company (`permit_no`, `brand_name`, `class_type`,
# `origin_desc`, dates). So the brand is the best maker signal the row has, and the producer is
# keyed on the brand rather than the permit -- which also gathers a brand's range back together
# across the seven importers who file Lagavulin instead of splitting it between them.
#
# Measured on the replay log before it was run against the catalog (`scripts/replay_scans.py`
# over a `createdb -T bcd` copy, deterministic index both sides): 11 of 1,927 frames change,
# +6 drawn overall. Gained: Campari +2, `Colonel E.h. Taylor Small Batch` +2, Modelo +3,
# `Gosling's Bermuda Rum`. Lost: `The Alchemist Heady Topper` 170 -> 168.
#
# Those two frames are the honest price and they are attributable. Both are object `1C80836F`
# reading the back of a Heady can, and the object's producer path turns on a garbled line,
# "ECHEMSTU N SHOULD N DURING PREGNA". A new brand producer, `Big Shoulders`, matches it at
# 0.400 against Monkey Shoulder's 0.353, takes the top slot, and the object stops resolving.
# Nothing is wrong with `Big Shoulders` -- it is a real brand that this pass correctly gives a
# producer. 21,633 more names in the table is simply a denser table, and on warning-label OCR a
# right name will occasionally lose a slot to another right name. Not tuned around: one frame
# of back-label text is exactly the corpus the harness exists to stop us fitting to.
#
# Not applied to `DSP-`/`BR-` permits even where one holds a thousand unrelated brands, because
# those are real plants and nothing here can tell a co-packer from a house: `DSP-OH-22` is named
# "Colonial Club" and bottles 1,891 brands including Jefferson's, while `DSP-MN-22` is named
# "Phillips" and bottles 1,770 that genuinely are Phillips's. Both name themselves on about 4%
# of their own products, so the obvious discriminator does not discriminate. Left alone rather
# than guessed at.

# `<state>-I-####`, and the 71 permits filed without the state ("I 1214"). The state is
# sometimes a city instead ("PHI-I-4477"), which is why the class letter is read from the
# second position rather than the first. `-W-` (wholesaler) is 441 filings and left alone: it
# is the same argument, but too few rows to be worth a second rule.
_PERMIT_N = "lower(regexp_replace(payload->>'permit_no', '[^A-Za-z0-9]+', '-', 'g'))"
#: A bronze filing on an import permit. Public because `scripts/fix_importer_producers.py`
#: reports what this pass will do, and a dry run that read the rule differently could promise
#: something the pass then declines.
IMPORTER_PERMIT = f"(split_part({_PERMIT_N}, '-', 2) = 'i' OR {_PERMIT_N} ~ '^i-[0-9]')"

#: ...whose brand can name a producer. The same two guards the permit pass uses: an
#: incorporation word names nobody, and a two-character brand identifies nobody.
NAMEABLE_BRAND = f"""(coalesce(payload->>'brand_name','') <> ''
  AND lower(regexp_replace(payload->>'brand_name', '[^A-Za-z0-9]', '', 'g'))
        NOT IN ('{_DEGENERATE}')
  AND length(regexp_replace(payload->>'brand_name', '[^A-Za-z0-9]', '', 'g')) >= 3)"""

# Runs of punctuation collapse, so "MR. BOSTON" and "MR BOSTON" are one brand -- the same
# normalising `_PRODUCER_ID` does to permit numbers, and for the same reason.
_BRAND_PRODUCER_ID = (
    "'prod:ttb-cola-registry:brand:' || "
    "left(trim(both '-' from regexp_replace(lower({col}), '[^a-z0-9]+', '-', 'g')), 60)"
)

# ...and whose brand is a brand rather than a product wearing the brand field. This is the gate
# that stops the fix recreating the disease the permit rekey cured. Keyed on the brand alone,
# 29,893 of 55,154 new producers would have held exactly one product and shared its name -- and
# the reason is the same as it was upstream: where a filer leaves `fanciful_name` empty, the
# whole product name goes in `brand_name`, so "LAGAVULIN (21YO)" and "LAGAVULIN 12 YR" are filed
# as brands. Naming a producer after one of those says nothing a row's own name did not already.
#
# A brand earns the field by appearing, anywhere in the registry, with its product named
# separately from it -- the signal the rank above already trusts first (`branded DESC`). Read
# across all permits, not just import ones, because whether a string is a brand is a fact about
# the string rather than about who filed it. That names 21,633 producers over 82,060 rows,
# including every Lagavulin row whose label was filed the ordinary way, and leaves 31,602 rows
# with no producer -- see `_UNNAME_REFUSED` for why that is the better answer.
_BRANDS = f"""
CREATE TEMP TABLE brands ON COMMIT DROP AS
SELECT DISTINCT {_BRAND_PRODUCER_ID.format(col="payload->>'brand_name'")} AS producer_id
FROM bronze
WHERE source_id = 'ttb-cola-registry'
  AND coalesce(payload->>'fanciful_name','') <> ''
  AND {NAMEABLE_BRAND};

CREATE INDEX ON brands (producer_id);
"""


_REKEY_IMPORTERS = f"""
CREATE TEMP TABLE filings ON COMMIT DROP AS
SELECT DISTINCT 'ttb:' || (payload->>'ttb_id') AS gid,
       {_BRAND_PRODUCER_ID.format(col="payload->>'brand_name'")} AS producer_id,
       payload->>'brand_name' AS brand,
       nullif(upper(payload->>'origin_desc'), '') AS origin
FROM bronze
WHERE source_id = 'ttb-cola-registry'
  AND {IMPORTER_PERMIT}
  AND {NAMEABLE_BRAND}
  AND {_BRAND_PRODUCER_ID.format(col="payload->>'brand_name'")}
        IN (SELECT producer_id FROM brands);

-- One producer per product. A COLA is filed per label, so the same TTB id can arrive twice
-- under two spellings of its brand, and a product may have only one maker: ordered so a re-run
-- makes the same choice rather than whichever row the scan reached first.
CREATE TEMP TABLE imported ON COMMIT DROP AS
SELECT DISTINCT ON (gid) gid, producer_id FROM filings ORDER BY gid, producer_id;

CREATE INDEX ON imported (gid);

-- The commonest country among a brand's own products, and how far it carries. Read per BRAND,
-- which is what makes it trustworthy: the old bug was the mode across an importer's whole
-- portfolio, so Crown Royal took "Scotland" from a Diageo permit that mostly carries Islay. A
-- brand's own filings do not have that problem -- Crown Royal is Canada on 270 of its 271.
CREATE TEMP TABLE origins ON COMMIT DROP AS
SELECT DISTINCT ON (producer_id) producer_id, origin, n,
       sum(n) OVER (PARTITION BY producer_id) AS tot
FROM (SELECT producer_id, origin, count(*) AS n FROM filings
      WHERE origin IS NOT NULL GROUP BY 1, 2) s
ORDER BY producer_id, n DESC, origin;

CREATE INDEX ON origins (producer_id);

-- Every other row on an import permit: its brand field holds a product name, or a word that
-- names nobody ("LLC"). The permit is still no claim about who made it, so there is no producer
-- to name -- see `_UNNAME_REFUSED`.
CREATE TEMP TABLE refused ON COMMIT DROP AS
SELECT DISTINCT 'ttb:' || (payload->>'ttb_id') AS gid
FROM bronze
WHERE source_id = 'ttb-cola-registry' AND {IMPORTER_PERMIT}
EXCEPT SELECT gid FROM imported;

CREATE INDEX ON refused (gid);

INSERT INTO gold (id, entity_type, record, name, updated_at)
SELECT producer_id, 'producer',
       jsonb_build_object(
         'id', producer_id, 'name', name, 'kind', 'brand',
         -- A country when four fifths of the brand's products agree on one. Not unanimity:
         -- that refused Lagavulin (Scotland on 84 of 91, the rest a filer's "Virginia") and
         -- Talisker (77 of 80), and 1,194 brands like Balblair, Kasteel and Martin Miller's
         -- that the registry is perfectly clear about. Not a bare mode either -- below four
         -- fifths the disagreement is real, and the band holds names like "arak" and "me" that
         -- several unrelated drinks share. Measured over the 21,633: 86.9% carry a country at
         -- unanimity, 91.5% at this line, and Crown Royal is still Canada.
         'region', initcap(region),
         -- Deliberately null: there is no one permit. The brand is filed by every importer who
         -- carries it, and recording one of them would read as the maker.
         'ttb_permit', NULL,
         'name_is_a_brand', true,
         'city', NULL, 'lat', NULL, 'lon', NULL,
         'aliases', '[]'::jsonb, 'country', NULL, 'website', NULL, 'parent_company', NULL),
       name, now()
FROM (
  SELECT f.producer_id,
         mode() WITHIN GROUP (ORDER BY f.brand) AS name,
         min(o.origin) FILTER (WHERE o.n::numeric / o.tot >= 0.8) AS region
  FROM filings f LEFT JOIN origins o USING (producer_id)
  GROUP BY f.producer_id
) b
ON CONFLICT (id) DO UPDATE
  SET record = EXCLUDED.record, name = EXCLUDED.name, updated_at = now();
"""

# Its own statement so `rowcount` is this UPDATE's and not whatever the batch above ended on --
# the count is how the pass reports what it moved, and a re-run must be able to say "nothing".
#
# Only a row that still points at an import permit. This pass takes rows AWAY from importers; it
# is not entitled to overrule anything else, and a filing on an import permit is no reason to.
# Without this condition it reached one row -- `Ramazzotti Aperitivo Rosato`, hand-linked to the
# curated house `prod:bcd:ramazzotti` in an earlier merge -- pulled it onto a brand producer and
# emptied the house. One row in 534,103, and it cost 22 frames of the replay log: the house is
# what the resolver's producer path corroborates a Ramazzotti label against.
_REPOINT_IMPORTED_PRODUCTS = """
UPDATE gold g
SET record = jsonb_set(g.record, '{producer_id}', to_jsonb(i.producer_id)), updated_at = now()
FROM imported i
WHERE g.entity_type = 'product' AND g.id = i.gid
  AND g.record->>'producer_id' IN (SELECT 'prod:ttb-cola-registry:permit:' || permit
                                   FROM importer_permits)
  AND g.record->>'producer_id' IS DISTINCT FROM i.producer_id;
"""

# The rows with no nameable brand. Emptied rather than left holding a company that did not make
# the bottle: `_hydrate` reads a producer it cannot find as "Unknown", the recommender omits the
# line, and the label index already keeps -1 for a product whose producer it has no row for. It
# also takes the importer's name out of the vocabulary the resolver counts as explaining a label,
# which is the part that matters to a scan.
#
# The empty string, not null, and not by dropping the key: `Product.producer_id` is a required
# `str`, so both of those make the row fail validation and disappear from the app entirely. No
# product in the catalog has been in this state before -- 31,602 will be after this pass.
#
# Only where the row still points at an import permit. A product filed twice, once by its
# distillery and once by an importer, keeps the distillery: the permit pass picked one of the two
# filings and if it picked the real plant, that is a better answer than nothing.
_UNNAME_REFUSED = """
UPDATE gold g
SET record = jsonb_set(g.record, '{producer_id}', '""'::jsonb), updated_at = now()
FROM refused r
WHERE g.entity_type = 'product' AND g.id = r.gid
  AND g.record->>'producer_id' IN (SELECT 'prod:ttb-cola-registry:permit:' || permit
                                   FROM importer_permits);
"""

# The permits the rule disqualifies, as producer ids -- so `_UNNAME_REFUSED` empties a field
# holding an importer and leaves one holding a distillery.
_IMPORTER_PERMITS = f"""
CREATE TEMP TABLE importer_permits ON COMMIT DROP AS
SELECT DISTINCT left(regexp_replace(lower(payload->>'permit_no'), '[^a-z0-9]+', '-', 'g'), 60)
       AS permit
FROM bronze
WHERE source_id = 'ttb-cola-registry' AND {IMPORTER_PERMIT};
"""

# A brand belongs to one producer, and after the rekey above that producer is the brand itself
# for everything an importer filed. Joined through the products, the same way `_REPOINT_BRANDS`
# does it -- and only where the products agree, because a brand filed both by an importer and by
# a distillery keeps the distillery.
_REPOINT_IMPORTED_BRANDS = """
UPDATE gold b
SET record = jsonb_set(b.record, '{producer_id}', to_jsonb(m.pid)), updated_at = now()
FROM (
  SELECT record->>'brand_id' AS bid, min(record->>'producer_id') AS pid
  FROM gold
  WHERE entity_type = 'product' AND record->>'brand_id' IS NOT NULL
    AND record->>'producer_id' LIKE 'prod:ttb-cola-registry:brand:%'
  GROUP BY 1
  HAVING count(DISTINCT record->>'producer_id') = 1
) m
WHERE b.entity_type = 'brand' AND b.id = m.bid
  AND b.record->>'producer_id' IS DISTINCT FROM m.pid;
"""

# An import permit whose rows all moved to their brands, or were emptied, now holds nothing. It
# must not survive in the producer table: the resolver matches label text against every name in
# it, so "Johnnie Walker" would keep reaching a permit that names no maker.
#
# Only the permits this pass emptied. A `DSP-` or `BR-` producer standing empty is somebody
# else's business -- an OpenBreweryDB row with no products is a real brewery carrying the city
# that location linking needs, and this is not the pass that decides about those.
_DROP_EMPTIED_PERMITS = """
DELETE FROM gold pr
WHERE pr.entity_type = 'producer'
  AND pr.id IN (SELECT 'prod:ttb-cola-registry:permit:' || permit FROM importer_permits)
  AND NOT EXISTS (SELECT 1 FROM gold p WHERE p.entity_type = 'product'
                  AND p.record->>'producer_id' = pr.id);
"""


def rekey_importers_to_brands(dsn: str, *, search_path: str | None = None) -> dict[str, int]:
    """Point every row an importer filed at its brand, or at nothing.

    Run after `rekey_to_permits`, which claims every TTB row for its permit; this hands back the
    ones the permit had no right to. Idempotent, and safe to re-run after any backfill.
    """
    counts: dict[str, int] = {}
    with psycopg.connect(dsn, **_options(search_path)) as conn:
        with conn.cursor() as cur:
            cur.execute(_BRANDS)
            cur.execute(_IMPORTER_PERMITS)
            cur.execute(_REKEY_IMPORTERS)
            cur.execute(_REPOINT_IMPORTED_PRODUCTS)
            counts["products_relinked"] = cur.rowcount
            cur.execute(_UNNAME_REFUSED)
            counts["products_unnamed"] = cur.rowcount
            cur.execute("SELECT count(DISTINCT producer_id) FROM filings")
            counts["brand_producers"] = cur.fetchone()[0]
            cur.execute(_REPOINT_IMPORTED_BRANDS)
            counts["brands_repointed"] = cur.rowcount
            cur.execute(_DROP_EMPTIED_PERMITS)
            counts["permits_dropped"] = cur.rowcount
        counts["names_titlecased"] = _titlecase_names(
            conn, "prod:ttb-cola-registry:brand:")
        conn.commit()
    return counts
