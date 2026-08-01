"""TTB COLA connector — Tier A, US-gov public domain, the US SKU universe.

Every US alcohol label approval since 1999: brand, fanciful name, class/type, ABV,
permittee, label image. This is the regulatory-provenance backbone — facts here are
`REGULATORY_FILING`, the highest-authority method.

Access reality: TTB's Public COLA Registry is a form-driven search at ttbonline.gov with
no clean bulk endpoint. Two real paths in prod:
  1. License COLA Cloud (2.6M+ records pre-parsed, barcodes + ABV OCR'd) — preferred.
  2. Drive the public search form via PoliteFetcher, parse result pages, OCR label
     images for ABV. Slower, same public data.

For a laptop-runnable pipeline this connector reads a small bundled fixture of real-shape
COLA records (data/fixtures/ttb_cola_sample.json). Swap `fetch()` for either path above
without touching normalize()/promote(). The fixture is clearly labeled, not passed off
as a live pull.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
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

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "data", "fixtures", "ttb_cola_sample.json",
)

# TTB class/type code -> our Category (abbreviated; full map lives in the enrich service)
_CLASS_TO_CATEGORY = {
    "ale": Category.BEER,
    "lager": Category.BEER,
    "malt beverage": Category.BEER,
    "beer": Category.BEER,
    "whisky": Category.SPIRIT,
    "whiskey": Category.SPIRIT,
    "bourbon": Category.SPIRIT,
    "vodka": Category.SPIRIT,
    "gin": Category.SPIRIT,
    "rum": Category.SPIRIT,
    "tequila": Category.SPIRIT,
    "wine": Category.WINE,
}


class TTBColaConnector(Connector):
    source_id = "ttb-cola-registry"
    provides = ("product", "producer", "sku", "abv", "label_image")

    def __init__(self, store, fixture_path: str | None = None) -> None:
        super().__init__(store)
        self.fixture_path = fixture_path or os.path.normpath(_FIXTURE)

    def fetch(self, limit: int | None = None) -> Iterator[BronzeDoc]:
        with open(self.fixture_path, encoding="utf-8") as f:
            rows = json.load(f)
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                break
            yield BronzeDoc(
                id=doc_id(self.source_id, row["ttb_id"]),
                source_id=self.source_id,
                natural_key=row["ttb_id"],
                fetched_at="",
                url=f"https://ttbonline.gov/colasonline/viewColaDetails.do?action=publicDisplaySearchBasic&ttbid={row['ttb_id']}",
                payload=row,
            )

    def normalize(self, doc: BronzeDoc) -> list[dict[str, Any]]:
        r = doc.payload
        return [
            {
                "entity_type": "cola",
                "bronze_id": doc.id,
                "ttb_id": r.get("ttb_id"),
                "brand_name": r.get("brand_name"),
                "fanciful_name": r.get("fanciful_name"),
                "class_type": r.get("class_type"),
                "permittee": r.get("permittee"),
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
            self.store.put_gold(
                producer_id, "producer",
                Producer(id=producer_id, name=permittee, kind="permittee",
                         ttb_permit=rec.get("ttb_id")).model_dump(mode="json"),
            )
            n_producer += 1

            brand_id = f"brand:{self.source_id}:{_slug(brand_name)}"
            self.store.put_gold(
                brand_id, "brand",
                Brand(id=brand_id, producer_id=producer_id, name=brand_name)
                .model_dump(mode="json"),
            )

            category = _category_of(rec.get("class_type", ""))
            product_name = rec.get("fanciful_name") or brand_name
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


def _category_of(class_type: str) -> Category:
    c = (class_type or "").lower()
    for key, cat in _CLASS_TO_CATEGORY.items():
        if key in c:
            return cat
    return Category.OTHER


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-")[:60] or "x"
