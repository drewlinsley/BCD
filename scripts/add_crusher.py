"""Put The Alchemist's Crusher in the catalog, by hand.

A scan on 2026-10-01 read `Crusher`, `AMERICAN DOUBLE INDIA PALE ALE` and `ALC. 8% BY VOL.`
cleanly, 33 frames out of 49, and resolved to nothing. The recognizer was not at fault: the
beer is not in the catalog at all. The Alchemist has 21 rows -- Heady Topper, Focal Banger,
Skadoosh, Rapture, Petit Mutant -- and no Crusher, and no BRONZE row anywhere in the system
holds both "crusher" and "alchemist", so it was never ingested rather than lost on the way.

The catalog is built from TTB COLA filings, which a brewery needs only to ship across a state
line. A can sold mostly in Vermont can live its whole life on state approval and never appear
federally. That is a gap no amount of resolver work closes.

Everything written here has a source, and the source is the user's own scan:

    style  "Double IPA"   label_ocr, "AMERICAN DOUBLE INDIA PALE ALE"  (23 frames)
    abv     8.0%          label_ocr, "ALC. 8% BY VOL."                 (20 frames)
    name   "The Alchemist Crusher", alias "Crusher"

No description and no tasting note: nobody has given us one, and a flavour sentence invented
here would read on the detail screen exactly like the ones that came from somewhere. The
sensory vector is NOT hand-written either -- it is `sensory_from_style`, the same call the
enrichment pass makes, so the row carries the Double IPA centroid and says `style_prior` about
where it came from. That also keeps it off the Discover shelves, which take known vectors only.
It is scannable, searchable and rateable, which is what was asked for.

What this does NOT fix: a frame that reads `Crusher` and no brand still cannot resolve, because
`Crusher` alone is 182 beers in this catalog and the right answer is still "I don't know".
Get the words "The Alchemist" in frame and it lands.

    .venv/bin/python scripts/add_crusher.py            # dry run: writes nothing
    .venv/bin/python scripts/add_crusher.py --apply    # write it

Then restart the API: the label index is built at startup, and until it is rebuilt the new row
is in the catalog but not yet scannable.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest", "services/enrich"):
    sys.path.insert(0, str(ROOT / p))

from bcd_api.app import _load_dotenv  # noqa: E402
from bcd_enrich.style_prior import sensory_from_style  # noqa: E402
from bcd_ingest.store import open_store  # noqa: E402
from bcd_schema import Product  # noqa: E402

PID = "bcd:the-alchemist-crusher"
PRODUCER = "prod:ttb-cola-registry:permit:br-vt-21045"      # The Alchemist, Stowe VT
SCAN = "data/scans10.jsonl"


def _ocr(quote: str) -> dict:
    """Provenance for something the camera actually read off the can."""
    return {"source_id": "label_ocr", "url": None, "quote": quote, "method": "label_ocr",
            "confidence": 0.9, "extracted_at": datetime.now(UTC).isoformat(),
            "extractor_version": f"scan/{SCAN}"}


def row() -> dict:
    record = {
        "id": PID,
        "name": "The Alchemist Crusher",
        "aliases": ["Crusher"],
        "category": "beer",
        "producer_id": PRODUCER,
        "brand_id": "brand:ttb-cola-registry:the-alchemist",
        "style": {"value": "Double IPA",
                  "provenance": _ocr("AMERICAN DOUBLE INDIA PALE ALE")},
        "spec": {"abv_pct": {"value": 8.0, "provenance": _ocr("ALC. 8% BY VOL.")}},
    }
    # The same call the enrichment pass makes, so this is the style's centroid and not a
    # vector somebody typed. `source: style_prior` is the row saying so out loud.
    sv = sensory_from_style(record["name"], record["category"],
                            style_hint=record["style"]["value"])
    if sv and sv.axes:
        record["sensory"] = sv.model_dump(mode="json")
    return Product.model_validate(record).model_dump(mode="json")


def main(argv: list[str]) -> int:
    write = "--apply" in argv
    _load_dotenv()
    store = open_store(root=str(ROOT / "data"))
    try:
        if store.get_gold(PID) is not None:
            print(f"{PID} is already in the catalog; nothing to do")
            return 0
        rec = row()
        sensory = rec.get("sensory") or {}
        print(json.dumps(rec, indent=1, ensure_ascii=False))
        print(f"\nproducer : {PRODUCER}")
        maker = store.get_gold(PRODUCER)
        print(f"           {'FOUND: ' + (maker or {}).get('name', '?') if maker else 'MISSING'}")
        print(f"sensory  : {sensory.get('source')} / {len(sensory.get('axes') or {})} axes")
        if not write:
            print("\n(dry run: nothing written; add --apply to write)")
            return 0
        store.put_gold(PID, "product", rec)
        store.refresh_search_names([PID])
        back = store.get_gold(PID)
        print(f"\nwrote {PID}: {(back or {}).get('name')!r}")
        print("\nNow restart the API so the label index picks it up -- until then it is in the\n"
              "catalog but not yet scannable.")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
