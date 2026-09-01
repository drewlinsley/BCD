"""TTB COLA connector — Tier A, US-gov public domain, the US SKU universe.

Every US alcohol label approval since 1999: brand, fanciful name, class/type, ABV,
permittee, label image. This is the regulatory-provenance backbone — facts here are
`REGULATORY_FILING`, the highest-authority method.

Access reality: TTB's Public COLA Registry is a form-driven search at ttbonline.gov with
no clean bulk endpoint. Two real paths in prod:
  1. License COLA Cloud (2.6M+ records pre-parsed, barcodes + ABV OCR'd) — preferred.
  2. Drive the public search form via PoliteFetcher, parse result pages, OCR label
     images for ABV. Slower, same public data.

`fetch()` now drives the public search form. The result table alone carries brand name,
fanciful name, class/type and origin state — which is product name, style and producer
region without opening a single detail page. That matters: details are one request per
COLA, and the registry holds ~2M of them.

The bundled fixture (data/fixtures/ttb_cola_sample.json) is still available via
`fixture_path=` or BCD_TTB_FIXTURE=1, for offline work. It is never a silent fallback —
a failed live run raises rather than quietly serving five hand-made records as if they
were a pull.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

from bcd_schema import (
    SKU,
    Brand,
    Category,
    ContainerType,
    ExtractionMethod,
    Producer,
    Product,
    ProductSpec,
    Provenance,
    Sourced,
)

from ..base import Connector
from ..store import BronzeDoc, doc_id

_BASE = "https://www.ttbonline.gov/colasonline"
_SEARCH = f"{_BASE}/publicSearchColasBasicProcess.do?action=search"
_NEXT = f"{_BASE}/publicPageBasicCola.do?action=page&pgfcn=nextset"
_DETAILS = f"{_BASE}/viewColaDetails.do?action=publicDisplaySearchBasic&ttbid={{}}"

#: Identify honestly. ttbonline.gov serves no robots.txt, the data is CC0, and TTB
#: publishes a "save your search results" guide — so extraction is an intended use. That
#: is not licence to hammer a government server: one request per second, in series.
_UA = "BCDBot/0.1 (+https://github.com/drewlinsley/BCD)"
_DELAY_S = 1.0
_PAGE_ROWS = 20

#: A day of one class range has never come near this (a busy day is ~10 pages). It exists
#: so that no failure of the two end-of-results checks can turn into an unbounded loop.
_MAX_PAGES = 500

#: Class/type code ranges to pull. TTB numbers distilled spirits 100-699 and malt
#: beverages 900-999. Wine falls outside both and is left there deliberately: the sensory
#: model behind recommendation is built on beer and spirit axes, and wine would outnumber
#: everything else. Two searches per day rather than one is the price of that filter.
_CLASS_RANGES = (("100", "699"), ("900", "999"))

#: The result table, in column order: TTB ID, Permit No., Serial, Completed Date,
#: Fanciful Name, Brand Name, Origin, Origin Desc, Class/Type, Class/Type Desc.
_ROW_FIELDS = ("ttb_id", "permit_no", "serial", "completed_date", "fanciful_name",
               "brand_name", "origin_code", "origin_desc", "class_type_code", "class_type")

#: "1 to 20 of 140" in the result header. This is the only trustworthy end-of-results
#: signal the registry gives: the paginator clamps at the last page and serves it again
#: forever rather than returning an empty set, so "a short page is the last page" is wrong
#: whenever the total is an exact multiple of the page size.
_RANGE = re.compile(r"(\d+)\s+to\s+(\d+)\s+of\s+(\d+)", re.I)

_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


#: ttbonline.gov presents only its leaf certificate — the chain stops at depth 0 and
#: OpenSSL reports "unable to verify the first certificate". Browsers and macOS `curl`
#: paper over this by fetching the missing issuer from the leaf's Authority Information
#: Access extension; Python's ssl module does not. So we do it ourselves.
#:
#: This tightens verification rather than loosening it. The downloaded intermediate is only
#: useful if it chains to a root already in certifi (it does — Sectigo Public Server
#: Authentication Root R46), and a certificate is signed, so fetching it over plain HTTP
#: cannot forge trust. Nothing here disables a check.
_AIA_INTERMEDIATE = "http://crt.sectigo.com/EntrustOVTLSIssuingRSACA2.crt"


def _ssl_context():
    """certifi's roots plus the intermediate TTB omits."""
    import ssl

    import certifi
    import httpx

    ctx = ssl.create_default_context(cafile=certifi.where())
    resp = httpx.get(_AIA_INTERMEDIATE, timeout=30.0, headers={"User-Agent": _UA})
    resp.raise_for_status()
    ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(resp.content))
    return ctx

def _cell(raw: str) -> str:
    """Cell text: strip tags, decode entities (TTB emits &#x2f; for "/"), collapse space."""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", raw))).strip()


def parse_results(page: str) -> list[dict[str, str]]:
    """Every COLA row on one result page. A row without the full column set is skipped
    rather than guessed at — a layout change should lose records loudly, not silently
    shift every field by one."""
    out: list[dict[str, str]] = []
    for row in _TR.findall(page):
        if "ttbid=" not in row:
            continue
        cells = [_cell(c) for c in _TD.findall(row)]
        if len(cells) < len(_ROW_FIELDS):
            continue
        out.append(dict(zip(_ROW_FIELDS, cells, strict=False)))
    return out


def parse_range(page: str) -> tuple[int, int, int] | None:
    """(first, last, total) from the result header, or None if it is not there.

    Returned as read: the registry counts from 1, so `last == total` means this page is
    the end of the results.
    """
    m = _RANGE.search(html.unescape(_TAG.sub(" ", page)))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "data", "fixtures", "ttb_cola_sample.json",
)

# TTB class/type text -> our Category. Matched as substrings against the class/type
# description, first hit wins, so the order is load-bearing: anything specific has to come
# before the general word it contains. TTB's own vocabulary is the source — these are the
# descriptions the registry actually emits, not a guess at what it might say.
_CLASS_TO_CATEGORY = {
    # Specific before general: "single malt" is a whisky, "malt beverage" is a beer, and
    # a bare "malt" key would swallow both.
    "single malt": Category.SPIRIT,

    # Beer
    "ale": Category.BEER,
    "lager": Category.BEER,
    "stout": Category.BEER,
    "porter": Category.BEER,
    "malt beverage": Category.BEER,
    "beer": Category.BEER,

    # Whisky and friends
    "whisky": Category.SPIRIT,
    "whiskey": Category.SPIRIT,
    "bourbon": Category.SPIRIT,
    "rye": Category.SPIRIT,

    # Agave
    "tequila": Category.SPIRIT,
    "mezcal": Category.SPIRIT,
    "agave": Category.SPIRIT,

    # Brandy family
    "brandy": Category.SPIRIT,
    "cognac": Category.SPIRIT,
    "armagnac": Category.SPIRIT,
    "calvados": Category.SPIRIT,
    "pisco": Category.SPIRIT,
    "grappa": Category.SPIRIT,
    "slivovitz": Category.SPIRIT,

    # Liqueurs, cordials and the flavoured shelf
    "liqueur": Category.SPIRIT,
    "cordial": Category.SPIRIT,
    "schnapps": Category.SPIRIT,
    "amaretto": Category.SPIRIT,
    "triple sec": Category.SPIRIT,
    "curacao": Category.SPIRIT,
    "anisette": Category.SPIRIT,
    "ouzo": Category.SPIRIT,
    "absinthe": Category.SPIRIT,
    "bitters": Category.SPIRIT,
    "arack": Category.SPIRIT,
    "raki": Category.SPIRIT,
    "aquavit": Category.SPIRIT,

    # Base spirits
    "vodka": Category.SPIRIT,
    "gin": Category.SPIRIT,
    "rum": Category.SPIRIT,
    "neutral spirits": Category.SPIRIT,
    "spirits": Category.SPIRIT,

    # Neither beer nor spirit, and it has its own Category
    "sake": Category.SAKE,
    "cider": Category.CIDER,
    "mead": Category.MEAD,
    "wine": Category.WINE,
}


#: Words that are wrong in title case. TTB writes everything in capitals, so casing has to
#: be inferred, and an acronym is the one place inference reliably fails: "IPA" must not
#: become "Ipa" on a label card.
_KEEP_UPPER = {
    "ipa", "dipa", "neipa", "apa", "xpa", "esb", "ipl", "pbr", "abv", "ibu", "usa", "us",
    "uk", "nz", "vsop", "vs", "xo", "bba", "bbl", "ii", "iii", "iv", "vi", "vii", "viii",
    "ix", "xi", "xii", "llc", "inc", "lp", "dc", "nyc", "la", "sf", "pa", "ny", "ca",
}


def _titlecase(text: str | None) -> str | None:
    """Make a SHOUTED registry name readable, and leave anything else alone.

    Only strings that are entirely uppercase are touched — a name already carrying its own
    casing ("McSorley's", "goodBeer") knows better than this function does.
    """
    if not text:
        return text
    stripped = text.strip()
    if not stripped or stripped != stripped.upper():
        return text
    out = []
    for word in stripped.split():
        core = word.strip(".,()&-/'\"").lower()
        out.append(word if core in _KEEP_UPPER else _cap(word))
    return " ".join(out)


def _cap(word: str) -> str:
    """Capitalise a word without str.title()'s apostrophe bug, which turns "STRANAHAN'S"
    into "Stranahan'S". A segment after an apostrophe is capitalised only when it is more
    than one letter — that separates the Irish "O'BRIEN" from a possessive "'S"."""
    head, *rest = word.split("'")
    parts = [head[:1].upper() + head[1:].lower()]
    parts += [(seg[:1].upper() + seg[1:].lower()) if len(seg) > 1 else seg.lower()
              for seg in rest]
    return "'".join(parts)


class TTBColaConnector(Connector):
    source_id = "ttb-cola-registry"
    provides = ("product", "producer", "sku", "abv", "label_image")

    def __init__(self, store, fixture_path: str | None = None,
                 days: int | None = None, until: date | None = None,
                 use_fixture: bool | None = None) -> None:
        super().__init__(store)
        self.fixture_path = fixture_path or os.path.normpath(_FIXTURE)
        # Explicit fixture_path, or the env flag, means offline. Otherwise: live.
        self.use_fixture = (
            use_fixture if use_fixture is not None
            else bool(fixture_path) or os.environ.get("BCD_TTB_FIXTURE") == "1"
        )
        self.days = days or int(os.environ.get("BCD_TTB_DAYS", "30"))
        self.until = until or date.today()

    def fetch(self, limit: int | None = None) -> Iterator[BronzeDoc]:
        rows = self._fixture_rows(limit) if self.use_fixture else self._live_rows(limit)
        for row in rows:
            yield BronzeDoc(
                id=doc_id(self.source_id, row["ttb_id"]),
                source_id=self.source_id,
                natural_key=row["ttb_id"],
                fetched_at="",
                url=_DETAILS.format(row["ttb_id"]),
                payload=row,
            )

    def _fixture_rows(self, limit: int | None) -> Iterator[dict]:
        with open(self.fixture_path, encoding="utf-8") as f:
            rows = json.load(f)
        yield from (rows[:limit] if limit is not None else rows)

    def _live_rows(self, limit: int | None) -> Iterator[dict]:
        """Walk the public registry a day at a time, newest first.

        One day per search rather than a wide range, because the result set is paged and
        session-held: a narrow window keeps each walk short, and a day that fails costs
        only that day. Newest first so a run cut short still leaves the most current
        labels rather than an arbitrary slice of history.
        """
        import httpx

        seen = 0
        with httpx.Client(timeout=60.0, headers={"User-Agent": _UA},
                          verify=_ssl_context(), follow_redirects=True) as client:
            for offset in range(self.days):
                day = self.until - timedelta(days=offset)
                for lo, hi in _CLASS_RANGES:
                    for row in self._walk_day(client, day, lo, hi):
                        yield row
                        seen += 1
                        if limit is not None and seen >= limit:
                            return

    def _walk_day(self, client, day: date, lo: str, hi: str) -> Iterator[dict]:
        stamp = day.strftime("%m/%d/%Y")
        form = {
            "searchCriteria.dateCompletedFrom": stamp,
            "searchCriteria.dateCompletedTo": stamp,
            "searchCriteria.productNameSearchType": "U",
            "searchCriteria.classTypeFrom": lo,
            "searchCriteria.classTypeTo": hi,
        }
        try:
            page = self._get(client, "POST", _SEARCH, data=form)
        except Exception as exc:  # noqa: BLE001 - a bad day must not end the run
            print(f"  ! ttb {stamp} class {lo}-{hi}: {exc}")
            return

        last_first_id = None
        for _ in range(_MAX_PAGES):
            rows = parse_results(page)
            span = parse_range(page)

            # The registry's paginator clamps: ask for a page past the end and it serves
            # the last one again, indefinitely. Two independent stops, because spinning
            # here is silent — the rows keep arriving, they are just ones we already have.
            if last_first_id is not None and rows and rows[0]["ttb_id"] == last_first_id:
                print(f"  ! ttb {stamp} class {lo}-{hi}: page repeated, stopping")
                return
            last_first_id = rows[0]["ttb_id"] if rows else None

            yield from rows

            if span is not None:
                _, last, total = span
                if last >= total:
                    return
            elif len(rows) < _PAGE_ROWS:
                # No header to read: fall back to the short-page heuristic, which is right
                # except when the total is an exact multiple of the page size.
                return

            try:
                page = self._get(client, "GET", _NEXT)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! ttb {stamp} paging stopped: {exc}")
                return
        print(f"  ! ttb {stamp} class {lo}-{hi}: hit the {_MAX_PAGES}-page cap")

    def _get(self, client, method: str, url: str, data: dict | None = None) -> str:
        """One request, then wait. The sleep is before the return rather than at the call
        sites so no future caller can forget it."""
        resp = client.request(method, url, data=data)
        resp.raise_for_status()
        time.sleep(_DELAY_S)
        return resp.text

    def normalize(self, doc: BronzeDoc) -> list[dict[str, Any]]:
        r = doc.payload
        return [
            {
                "entity_type": "cola",
                "bronze_id": doc.id,
                "ttb_id": r.get("ttb_id"),
                # TTB shouts. The label card sets product names in a serif display face,
                # where all-caps reads as an error rather than a style.
                "brand_name": _titlecase(r.get("brand_name")),
                "fanciful_name": _titlecase(r.get("fanciful_name")),
                "class_type": _titlecase(r.get("class_type")),
                "permittee": r.get("permittee"),
                # The state the COLA was filed from. Free with every search row, and the
                # thing producer records are shortest of — OpenBreweryDB name-matching only
                # ever reached 116 of them.
                "origin_desc": r.get("origin_desc"),
                "abv": r.get("abv"),
                "net_contents": r.get("net_contents"),
                "upc": r.get("upc"),
                "label_image_url": r.get("label_image_url"),
                "url": doc.url,
            }
        ]

    def promote(self) -> dict[str, int]:
        n_prod = n_producer = n_sku = 0
        for rec in self.store.iter_silver("cola"):
            brand_name = rec.get("brand_name")
            if not brand_name:
                continue
            reg_prov = Provenance(
                source_id=self.source_id,
                url=rec.get("url"),
                method=ExtractionMethod.REGULATORY_FILING,
                confidence=1.0,
                quote=f"{brand_name} / {rec.get('class_type')}",
            )

            permittee = rec.get("permittee") or brand_name
            producer_id = f"prod:{self.source_id}:{_slug(permittee)}"
            # TTB origin is a US state (or a country, for imports) in caps. Title-case it
            # so it reads on a label card and joins with the OpenBreweryDB regions already
            # in gold, which are written that way.
            origin = (rec.get("origin_desc") or "").strip()
            region = origin.title() if origin else None
            self.store.put_gold(
                producer_id, "producer",
                Producer(id=producer_id, name=permittee, kind="permittee",
                         region=region, ttb_permit=rec.get("ttb_id")).model_dump(mode="json"),
            )
            n_producer += 1

            brand_id = f"brand:{self.source_id}:{_slug(brand_name)}"
            self.store.put_gold(
                brand_id, "brand",
                Brand(id=brand_id, producer_id=producer_id, name=brand_name)
                .model_dump(mode="json"),
            )

            category = _category_of(rec.get("class_type", ""))
            product_name = _display_name(brand_name, rec.get("fanciful_name"))
            pid = f"ttb:{rec['ttb_id']}"
            spec = ProductSpec(
                abv_pct=Sourced[float](value=float(rec["abv"]), provenance=reg_prov)
                if rec.get("abv") is not None else None
            )
            product = Product(
                id=pid,
                brand_id=brand_id,
                producer_id=producer_id,
                category=category,
                name=product_name,
                style=Sourced[str](value=rec["class_type"], provenance=reg_prov)
                if rec.get("class_type") else None,
                spec=spec,
            )
            self.store.put_gold(pid, "product", product.model_dump(mode="json"))
            n_prod += 1

            if rec.get("upc"):
                sku_id = f"sku:{rec['upc']}"
                self.store.put_gold(
                    sku_id, "sku",
                    SKU(id=sku_id, product_id=pid, container=ContainerType.BOTTLE,
                        upc=rec["upc"]).model_dump(mode="json"),
                )
                n_sku += 1
        return {"product": n_prod, "producer": n_producer, "sku": n_sku}


def _display_name(brand: str, fanciful: str | None) -> str:
    """Brand-forward product name. TTB's fanciful_name is often just the class/type
    ("Pale Ale", "Kentucky Straight Bourbon Whiskey") — useless as an overlay label and a weak
    trigram target — so lead with the brand the COLA always carries: "Sierra Nevada" + "Pale Ale"
    -> "Sierra Nevada Pale Ale". Skip the prepend only when the fanciful already names the brand
    (avoids "Sierra Nevada Sierra Nevada ..."). No fanciful -> the brand stands alone."""
    brand = (brand or "").strip()
    fanciful = (fanciful or "").strip()
    if not fanciful:
        return brand
    if brand and brand.lower() not in fanciful.lower():
        return f"{brand} {fanciful}"
    return fanciful


def _category_of(class_type: str) -> Category:
    c = (class_type or "").lower()
    for key, cat in _CLASS_TO_CATEGORY.items():
        if key in c:
            return cat
    return Category.OTHER


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-")[:60] or "x"
