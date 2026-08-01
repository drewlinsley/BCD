"""Open Food Facts connector — Tier A, ODbL.

The only large *open* barcode -> ingredient corpus, so it directly serves scan-a-can:
a UPC resolves to a real ingredient list with provenance. We query the beer/spirits
categories. Ingredient text becomes RecipeIngredient parts tagged as producer-stated
(OFF transcribes the physical label).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from bcd_schema import (
    Brand,
    Category,
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
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..base import Connector
from ..store import BronzeDoc, doc_id

# v2 search API — the legacy cgi/search.pl is deprecated and frequently 503s.
SEARCH = "https://world.openfoodfacts.org/api/v2/search"
# OFF blocks default client UAs; identify honestly (their docs require this).
_UA = "BCDBot/0.1 (+https://github.com/drewlinsley/BCD)"
_FIELDS = (
    "code,product_name,brands,ingredients_text,categories,"
    "generic_name,countries,labels,alcohol_value,serving_size"
)


class OpenFoodFactsConnector(Connector):
    source_id = "openfoodfacts"
    provides = ("product", "producer", "sku", "upc", "abv", "ingredients")

    def __init__(self, store, categories: tuple[str, ...] = ("beers", "whiskies")) -> None:
        super().__init__(store)
        self.categories = categories

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _get_page(self, client: httpx.AsyncClient, category: str, page: int) -> dict:
        resp = await client.get(
            SEARCH,
            params={
                "categories_tags_en": category,
                "fields": _FIELDS,
                "page_size": 50,
                "page": page,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def fetch(self, limit: int | None = None) -> AsyncIterator[BronzeDoc]:
        fetched = 0
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": _UA}) as client:
            for category in self.categories:
                page = 1
                while True:
                    try:
                        data = await self._get_page(client, category, page)
                    except httpx.HTTPError as exc:
                        # Degrade gracefully: OFF being down shouldn't abort a run that
                        # may already have landed the primary (TTB/OBDB) sources.
                        print(f"  ! openfoodfacts '{category}' page {page} failed: {exc}")
                        break
                    products = data.get("products", [])
                    if not products:
                        break
                    for row in products:
                        code = row.get("code")
                        if not code:
                            continue
                        yield BronzeDoc(
                            id=doc_id(self.source_id, code),
                            source_id=self.source_id,
                            natural_key=code,
                            fetched_at="",
                            url=f"https://world.openfoodfacts.org/product/{code}",
                            payload=row,
                        )
                        fetched += 1
                        if limit is not None and fetched >= limit:
                            return
                    page += 1

    def normalize(self, doc: BronzeDoc) -> list[dict[str, Any]]:
        r = doc.payload
        return [
            {
                "entity_type": "off_product",
                "bronze_id": doc.id,
                "upc": r.get("code"),
                "name": (r.get("product_name") or "").strip(),
                "brand": (r.get("brands") or "").split(",")[0].strip(),
                "ingredients_text": r.get("ingredients_text"),
                "abv": _to_float(r.get("alcohol_value") or r.get("abv")),
                "category": _category_of(r.get("categories", "")),
                "url": doc.url,
            }
        ]

    def promote(self) -> dict[str, int]:
        n_prod = n_producer = 0
        for rec in self.store.iter_silver("off_product"):
            if not rec.get("name"):
                continue
            base = f"off:{rec['upc']}"
            prov = Provenance(
                source_id=self.source_id,
                url=rec.get("url"),
                method=ExtractionMethod.LABEL_OCR,  # OFF transcribes the physical label
                confidence=0.85,
                quote=rec.get("ingredients_text"),
            )

            producer_name = rec.get("brand") or "Unknown"
            producer_id = f"prod:{self.source_id}:{_slug(producer_name)}"
            self.store.put_gold(
                producer_id, "producer",
                Producer(id=producer_id, name=producer_name).model_dump(mode="json"),
            )
            n_producer += 1

            brand_id = f"brand:{self.source_id}:{_slug(producer_name)}"
            self.store.put_gold(
                brand_id, "brand",
                Brand(id=brand_id, producer_id=producer_id, name=producer_name)
                .model_dump(mode="json"),
            )

            spec = ProductSpec(
                abv_pct=Sourced[float](value=rec["abv"], provenance=prov)
                if rec.get("abv") is not None else None
            )
            recipe = _ingredients_to_recipe(rec.get("ingredients_text"), prov)
            product = Product(
                id=base,
                brand_id=brand_id,
                producer_id=producer_id,
                category=rec.get("category") or Category.BEER,
                name=rec["name"],
                spec=spec,
                recipe=recipe,
            )
            self.store.put_gold(base, "product", product.model_dump(mode="json"))
            n_prod += 1
        return {"product": n_prod, "producer": n_producer}


def _ingredients_to_recipe(text: str | None, prov: Provenance) -> RecipeGraph:
    rg = RecipeGraph()
    if not text:
        return rg
    for raw in [t.strip() for t in text.replace(";", ",").split(",") if t.strip()]:
        rg.ingredients.append(
            RecipeIngredient(
                role=_guess_role(raw),
                entity_kind=_guess_kind(raw),
                raw_name=raw,
                provenance=prov,
            )
        )
    return rg


def _guess_role(name: str) -> IngredientRole:
    n = name.lower()
    if any(w in n for w in ("malt", "barley", "wheat", "oat", "rye")):
        return IngredientRole.BASE_MALT
    if "hop" in n:
        return IngredientRole.AROMA_HOP
    if "yeast" in n:
        return IngredientRole.YEAST
    if "water" in n:
        return IngredientRole.WATER_SALT
    return IngredientRole.OTHER


def _guess_kind(name: str) -> str:
    n = name.lower()
    if "hop" in n:
        return "hop"
    if any(w in n for w in ("malt", "barley")):
        return "malt"
    if "yeast" in n:
        return "yeast"
    return "other"


def _category_of(categories: str) -> Category:
    c = categories.lower()
    if any(w in c for w in ("whisk", "spirit", "vodka", "gin", "rum", "tequila")):
        return Category.SPIRIT
    if "wine" in c:
        return Category.WINE
    if "cider" in c:
        return Category.CIDER
    return Category.BEER


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-")[:60] or "x"


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
