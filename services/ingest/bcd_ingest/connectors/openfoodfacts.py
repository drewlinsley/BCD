"""Open Food Facts connector — Tier A, ODbL.

The only large *open* barcode -> ingredient corpus, so it directly serves scan-a-can:
a UPC resolves to a real ingredient list with provenance. We query the beer/spirits
categories. Ingredient text becomes RecipeIngredient parts tagged as producer-stated
(OFF transcribes the physical label).
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
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

    def __init__(self, store,
                 categories: tuple[str, ...] = ("beers", "whiskies", "spirits"),
                 country: str | None = None) -> None:
        super().__init__(store)
        self.categories = categories
        # Optional OFF country filter (e.g. "united-states") to target a market instead of OFF's
        # Euro-first ordering. Env-driven so `BCD_OFF_COUNTRY=united-states make ingest SOURCE=off`
        # works without threading a flag through the generic registry/CLI.
        self.country = country or os.environ.get("BCD_OFF_COUNTRY") or None

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _get_page(self, client: httpx.AsyncClient, category: str, page: int) -> dict:
        params: dict[str, Any] = {
            "categories_tags_en": category,
            "fields": _FIELDS,
            "page_size": 50,
            "page": page,
        }
        if self.country:
            params["countries_tags_en"] = self.country
        resp = await client.get(SEARCH, params=params)
        resp.raise_for_status()
        # OFF intermittently answers 200 with an HTML "temporarily unavailable" page.
        # raise_for_status() passes it, but resp.json() would then throw JSONDecodeError
        # — which is neither an httpx.HTTPError (so tenacity wouldn't retry) nor caught by
        # fetch()'s degrade path. Convert both bad-payload modes into httpx.HTTPError so a
        # blip gets retried and, if it persists, the run degrades gracefully.
        if "json" not in resp.headers.get("content-type", "").lower():
            raise httpx.HTTPError("OFF returned a non-JSON page (transient outage)")
        try:
            return resp.json()
        except ValueError as exc:  # JSONDecodeError subclasses ValueError
            raise httpx.HTTPError(f"OFF returned unparseable JSON: {exc}") from exc

    async def fetch(self, limit: int | None = None) -> AsyncIterator[BronzeDoc]:
        fetched = 0
        # Round-robin one page from each category per pass, so a global --limit is spread
        # across beer *and* spirits instead of being exhausted by whichever category sorts
        # first. A category drops out of rotation when it errors or runs dry.
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": _UA}) as client:
            page = {c: 1 for c in self.categories}
            active = list(self.categories)
            while active:
                for category in list(active):
                    try:
                        data = await self._get_page(client, category, page[category])
                    except httpx.HTTPError as exc:
                        # Degrade gracefully: OFF being down shouldn't abort a run that
                        # may already have landed the primary (TTB/OBDB) sources.
                        print(f"  ! openfoodfacts '{category}' page {page[category]} failed: {exc}")
                        active.remove(category)
                        continue
                    products = data.get("products", [])
                    if not products:
                        active.remove(category)
                        continue
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
                    page[category] += 1

    def normalize(self, doc: BronzeDoc) -> list[dict[str, Any]]:
        r = doc.payload
        categories = r.get("categories", "")
        category = _category_of(categories)
        brand = (r.get("brands") or "").split(",")[0].strip()
        # Recover a usable display name: OFF's product_name is often just "Gin"/"Rum"/"15",
        # which both misreads in the HUD and trigram-matches any stray OCR token. Lead with
        # the brand OFF also carries when the name is that thin; None when nothing's salvageable.
        name = _display_name(r.get("product_name"), brand)
        # Quality gate. OFF is a food database, so its alcohol categories are noisy: some
        # rows are unnamed, some name no drink at all (a gas station that slipped in), and a
        # few are plainly food mistagged as liquor (a drinking yogurt). Drop those — an empty
        # list skips the document — so the catalog stays beverages-only.
        if (not name or _is_placeholder(name) or category is None
                or _is_nonbeverage(name, categories)):
            return []
        return [
            {
                "entity_type": "off_product",
                "bronze_id": doc.id,
                "upc": r.get("code"),
                "name": name,
                "brand": brand,
                "ingredients_text": r.get("ingredients_text"),
                "abv": _to_float(r.get("alcohol_value") or r.get("abv")),
                "category": category,
                "url": doc.url,
            }
        ]

    def promote(self) -> dict[str, int]:
        n_prod = n_producer = n_sku = 0
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

            # Emit the barcode->product SKU so the scan barcode path (`sku:<upc>`) resolves this
            # product. OFF is a barcode database, so every row carries one — without this the 380+
            # OFF products are invisible to the most reliable scan path (only the hand-seeded demo
            # SKUs resolved). Container isn't in OFF's search fields; default to bottle.
            upc = rec.get("upc")
            if upc:
                sku_id = f"sku:{upc}"
                self.store.put_gold(
                    sku_id, "sku",
                    SKU(id=sku_id, product_id=base, container=ContainerType.BOTTLE, upc=upc)
                    .model_dump(mode="json"),
                )
                n_sku += 1
        return {"product": n_prod, "producer": n_producer, "sku": n_sku}


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


def _category_of(categories: str) -> Category | None:
    """Map OFF's free-text category list to our enum, or None when nothing in it names a
    drink — the caller drops those (OFF is a food DB; non-beverages leak in). Order matters:
    match the specific alcohol families before the generic 'beer' fallback. The old code
    knew only whisk/spirit/vodka/gin/rum/tequila, so cognac, brandy, liqueurs, eaux-de-vie,
    amaro, grappa, ouzo … all fell through and were mislabeled beer."""
    c = categories.lower()

    # Patterns, not bare substrings: `\b` word boundaries keep short tokens honest — `gin`
    # matches "Gins"/"London Dry Gin" but not "ginger", `ale` matches "Pale Ales" but not
    # "céréales". Prefixes (\bwhisk) absorb plurals/spellings (whisky/whiskey/whiskies).
    def has(*patterns: str) -> bool:
        return any(re.search(p, c) for p in patterns)

    if has(r"\bwhisk", r"\bbourbon", r"\bscotch\b", r"\bspirit", r"\bliqueur", r"\bliquor",
           r"\bvodka", r"\bgins?\b", r"\brums?\b", r"\brhums?\b", r"\btequila", r"\bmezcal",
           r"\bcognac", r"\bbrandy", r"\bbrandies", r"\barmagnac", r"\bcalvados",
           r"\beaux?[\s-]de[\s-]vie", r"\bcacha[çc]a", r"\bgrappa", r"\bouzo", r"\bamaro",
           r"\bvermouth",
           r"\bap[eé]ritif", r"\babsinthe", r"\bschnapps", r"\baquavit", r"\bpastis",
           r"\bsambuca", r"\btriple sec", r"\bbitters", r"\bsake\b", r"\bsoju"):
        return Category.SPIRIT
    if has(r"\bcidres?\b", r"\bcider"):
        return Category.CIDER
    if has(r"\bwines?\b", r"\bchampagne", r"\bprosecco", r"\bsherry", r"\bsparkling"):
        return Category.WINE
    if has(r"\bbeers?\b", r"\bbiers?\b", r"\bbirra", r"\bcerveza", r"\bcerveja",
           r"\bales?\b", r"\blager", r"\bpils", r"\bstout", r"\bipas?\b", r"\bporter",
           r"\bsaison", r"\bweizen", r"\bweiss", r"\bweiß"):
        return Category.BEER
    if has(r"\bbeverage", r"\bdrink", r"\balcohol"):  # a drink, no family named → spine
        return Category.BEER
    return None  # nothing here names a drink


# Terms that never occur in a real beer/spirit name but do show up on food rows OFF has
# mistagged with an alcohol category. Deliberately tiny and safe — nothing here collides
# with a legitimate style (a "Chocolate Stout" or "Coffee Porter" is untouched).
_NONBEVERAGE_TERMS = ("yaourt", "yogurt", "yoghurt", "vinegar", "vinaigre",
                      "pasta", "pâtes", "pates",
                      # A soft drink OFF double-tagged with a beer category (a "Beers, Sodas"
                      # Coca-Cola). "sodas"/"soft drink" are OFF's own category tokens, not beer
                      # styles — a hard seltzer is tagged "flavored malt beverage", not a soda.
                      "sodas", "soft drink")


def _is_nonbeverage(name: str, categories: str) -> bool:
    hay = f"{name} {categories}".lower()
    return any(t in hay for t in _NONBEVERAGE_TERMS)


# Scraper leftovers that land in product_name when a page had not finished rendering.
_PLACEHOLDER_NAMES = {"chargement", "loading", "unknown", "unknown product",
                      "sans nom", "no name", "n/a", ""}


def _is_placeholder(name: str) -> bool:
    # "Chargement…" / "Loading..." etc.; strip trailing dots/ellipsis before matching.
    return name.strip().lower().rstrip(" .…") in _PLACEHOLDER_NAMES


# Bare style/category words OFF sometimes drops into product_name. Alone they name no product
# and trigram-match anything sharing the letters, so they count as "weak" and get the brand
# prepended (or the row dropped). Kept lowercase for a case-folded lookup.
_BARE_CATEGORY = {
    "gin", "rum", "rhum", "beer", "beers", "ale", "ales", "ipa", "lager", "lagers",
    "pils", "pilsner", "stout", "porter", "vodka", "whisky", "whiskey", "wine", "cider",
    "eau", "spirit", "spirits", "liqueur", "mini", "biere", "bière", "cerveza", "birra",
    # Spirit classes OFF also drops in bare — a product named just "Tequila" is no more a
    # product than one named "Gin", and trigram-matches any stray "tequila" OCR.
    "tequila", "tequilas", "mezcal", "bourbon", "scotch", "brandy", "cognac", "sake",
    "ouzo", "grappa", "absinthe", "schnapps", "sherry", "vermouth", "mead", "soju", "shochu",
}


def _has_latin(s: str) -> bool:
    """Whether a string carries any A–Z letter — the alphabet the HUD renders and OCR emits."""
    return bool(re.search(r"[A-Za-z]", s or ""))


def _is_weak_name(name: str) -> bool:
    """A product_name too thin to stand on its own: a 1-3 digit number ("15", "40"), a bare
    category word ("Gin", "Pils", "Tequila"), a name with no Latin letters at all (a Cyrillic /
    Hebrew / Arabic foreign-market entry like "Виски … Джемесон" or "הוגרדן פחית", which can't
    render in the HUD or trigram-match Latin OCR), or a tiny all-lowercase fragment ("fen"). A
    4-digit number ("1664") or a self-identifying short brand ("J&B", "OB") is *not* weak."""
    n = (name or "").strip()
    if not n:
        return True
    if re.fullmatch(r"[0-9]{1,3}", n):
        return True
    if n.lower() in _BARE_CATEGORY:
        return True
    if not _has_latin(n) and not n.isdigit():
        # No Latin letters and not a pure number ("1664" stays real via the 4-digit rule below).
        return True
    return len(n) <= 3 and n.isalpha() and n.islower()


def _display_name(product_name: str | None, brand: str | None) -> str | None:
    """Best display name for an OFF row, or None when nothing usable remains. A strong
    product_name stands as-is; a weak one is anchored on the brand OFF also carries
    ("Glenfiddich" + "15" -> "Glenfiddich 15", "Hendrick's" + "Gin" -> "Hendrick's Gin");
    a weak name with no brand to rescue it ("fen") is dropped."""
    pn = (product_name or "").strip()
    brand = (brand or "").strip()
    if not _is_weak_name(pn):
        return pn
    if not brand or _is_weak_name(brand):
        # Brand can't rescue it — keep a combo only if it turns out non-weak (rare), else drop.
        combo = f"{brand} {pn}".strip() if (brand and pn) else (brand or pn).strip()
        return combo if (combo and not _is_weak_name(combo)) else None
    # Strong brand + weak qualifier: append the qualifier only when it's pure ASCII — a readable
    # age/proof ("Glenfiddich" + "15") or style ("Hendrick's" + "Gin") the HUD can show — and it
    # adds something the brand lacks. A non-Latin fragment ("Бира 5%") is dropped rather than
    # mixed into a "Stella Artois Бира 5%" label; there the Latin brand stands alone.
    if pn and pn.isascii() and pn.lower() not in brand.lower():
        return f"{brand} {pn}"
    return brand


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-")[:60] or "x"


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None
