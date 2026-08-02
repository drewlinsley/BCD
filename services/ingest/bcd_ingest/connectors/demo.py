"""Demo seed connector — a few canonical products WITH published hop/grain bills.

Not a live source. It reads a small, clearly-labeled fixture
(data/fixtures/demo_products.json) so the whole pipeline has something carrying a real
`RecipeGraph` to work on. This is what makes the ingest → enrich → vector-search chain
demonstrable end to end: TTB label approvals and barcode corpora almost never include a
hop bill, and the cold-start moat (scoring from chemistry with zero reviews) only shows
up when there IS one. Ingredient facts here are `STATED_BY_PRODUCER` at 0.9 confidence —
illustrative seed data, capped below regulatory certainty, never passed off as scraped.

The four products are deliberately spread across sensory space (tropical NEIPA, piney
West Coast IPA, roasty stout, banana/clove hefeweizen) so `nearest_by_sensory` returns a
visibly sensible ranking after enrichment.
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
    IngredientRole,
    Producer,
    Product,
    ProductSpec,
    Provenance,
    RecipeGraph,
    RecipeIngredient,
    Sourced,
)

from ..base import Connector
from ..store import BronzeDoc, doc_id

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "data", "fixtures", "demo_products.json",
)


class DemoConnector(Connector):
    source_id = "bcd-demo"
    provides = ("product", "producer", "brand", "sku", "recipe")

    def __init__(self, store, fixture_path: str | None = None) -> None:
        super().__init__(store)
        self.fixture_path = fixture_path or os.path.normpath(_FIXTURE)

    def fetch(self, limit: int | None = None) -> Iterator[BronzeDoc]:
        with open(self.fixture_path, encoding="utf-8") as f:
            rows = json.load(f)
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                break
            key = row["product"]["key"]
            yield BronzeDoc(
                id=doc_id(self.source_id, key),
                source_id=self.source_id,
                natural_key=key,
                fetched_at="",
                url=f"https://example.com/demo/{key}",
                payload=row,
            )

    def normalize(self, doc: BronzeDoc) -> list[dict[str, Any]]:
        return [{"entity_type": "demo_product", "bronze_id": doc.id,
                 "url": doc.url, **doc.payload}]

    def promote(self) -> dict[str, int]:
        n_prod = n_producer = n_sku = 0
        seen_producers: set[str] = set()
        for rec in self.store.iter_silver("demo_product"):
            prod_meta = rec["producer"]
            pinfo = rec["product"]
            prov = Provenance(
                source_id=self.source_id,
                url=rec.get("url"),
                method=ExtractionMethod.STATED_BY_PRODUCER,
                confidence=0.9,
                quote=f"{pinfo['name']} — published recipe (demo)",
            )

            producer_id = f"prod:{self.source_id}:{prod_meta['slug']}"
            if producer_id not in seen_producers:
                self.store.put_gold(
                    producer_id, "producer",
                    Producer(id=producer_id, name=prod_meta["name"],
                             kind=prod_meta.get("kind"), country=prod_meta.get("country"),
                             region=prod_meta.get("region"), lat=prod_meta.get("lat"),
                             lon=prod_meta.get("lon"), website=prod_meta.get("website"))
                    .model_dump(mode="json"),
                )
                seen_producers.add(producer_id)
                n_producer += 1

            brand_id = f"brand:{self.source_id}:{pinfo['key']}"
            self.store.put_gold(
                brand_id, "brand",
                Brand(id=brand_id, producer_id=producer_id, name=pinfo["name"])
                .model_dump(mode="json"),
            )

            ingredients = [
                RecipeIngredient(role=IngredientRole(ing["role"]),
                                 entity_kind=ing["entity_kind"], raw_name=ing["raw_name"],
                                 provenance=prov)
                for ing in pinfo.get("ingredients", [])
            ]
            spec = ProductSpec(
                abv_pct=Sourced[float](value=float(pinfo["abv"]), provenance=prov)
                if pinfo.get("abv") is not None else None,
                ibu=Sourced[float](value=float(pinfo["ibu"]), provenance=prov)
                if pinfo.get("ibu") is not None else None,
            )
            pid = f"{self.source_id}:{pinfo['key']}"
            product = Product(
                id=pid, brand_id=brand_id, producer_id=producer_id,
                category=Category(pinfo["category"]), name=pinfo["name"],
                style=Sourced[str](value=pinfo["style"], provenance=prov)
                if pinfo.get("style") else None,
                spec=spec,
                recipe=RecipeGraph(ingredients=ingredients),
                description=Sourced[str](value=pinfo["description"], provenance=prov)
                if pinfo.get("description") else None,
            )
            self.store.put_gold(pid, "product", product.model_dump(mode="json"))
            n_prod += 1

            if pinfo.get("upc"):
                sku_id = f"sku:{pinfo['upc']}"
                self.store.put_gold(
                    sku_id, "sku",
                    SKU(id=sku_id, product_id=pid,
                        container=ContainerType(pinfo.get("container", "can")),
                        volume_ml=pinfo.get("volume_ml"), upc=pinfo["upc"])
                    .model_dump(mode="json"),
                )
                n_sku += 1
        return {"product": n_prod, "producer": n_producer, "sku": n_sku}
