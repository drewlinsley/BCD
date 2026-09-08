#!/usr/bin/env python3
"""Latency benchmark for scan resolution against a catalog the size of production.

Builds a synthetic catalog (default 534k products, 38k producers — the live catalog's
shape after the TTB permit-holder rekey) in a throwaway SQLite store, builds the label
index over it, then times what the HUD actually asks for: single OCR lines, six-line
frames, and tracked objects, including the reviewer's real device readings.

    .venv/bin/python scripts/bench_index.py --products 534000

Numbers are printed per stage so a regression is attributable. The synthetic names are
drawn from a vocabulary sized like the real one (tens of thousands of distinct words, a
Zipf tail of common brand words), so posting-list sizes are realistic; the real catalog's
names are of course different, and the reviewer's rows are seeded verbatim so the
correctness table can be reproduced on the same run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
import time

from bcd_api.index import IndexedStore, LabelIndex
from bcd_api.resolver import Resolver
from bcd_ingest.store import MedallionStore
from bcd_schema import DetectedObject, DetectedText, ScanResolveRequest

STYLES = ["IPA", "Pale Ale", "Lager", "Stout", "Double IPA", "Hazy IPA", "Pilsner", "Porter",
          "Saison", "Bourbon", "Vodka", "Gin", "Rye Whiskey", "Sour", "Amber Ale"]
SUFFIXES = ["Brewing Co", "Brewery", "Brewing Company", "Distillery", "Beer Co", "Brewers"]

# The rows jps4444 measured against real frames, plus the products they were read off.
SEED = [
    ("prod:alchemist", "The Alchemist LLC", [("Heady Topper", "Heady"), ("Focal Banger", None)]),
    ("prod:dogfish", "Dogfish Head Craft Brewery", [("60 Minute IPA", "Dogfish Head")]),
    ("prod:tops", "Top's Brewing", [("Top's", None)]),
    ("prod:ache", "Ache Brewing", [("Ache", None)]),
    ("prod:theo", "Theo P. Brewing", [("Theo P.", None)]),
    ("prod:banger", "Banger", [("Banger", None)]),
    ("prod:chemist", "Chemist Spirits", [("Chemist", None), ("Chemist 151", None)]),
    ("prod:mist", "Mist Brewing", [("Mist", None)]),
]


def _vocab(rng: random.Random, n: int) -> list[str]:
    syll = ["ka", "to", "ri", "mo", "sa", "ne", "lu", "ve", "bra", "sto", "hop", "pin", "gra",
            "wil", "low", "ber", "hil", "ran", "dor", "mel", "tan", "cor", "fer", "ash", "oak",
            "fox", "elk", "bay", "sun", "moon", "lake", "peak", "iron", "gold", "wolf", "bear"]
    words = set()
    while len(words) < n:
        k = rng.randint(2, 4)
        words.add("".join(rng.choice(syll) for _ in range(k)).capitalize())
    return sorted(words)


def make_catalog(root: str, n_products: int, n_producers: int, seed: int = 7) -> MedallionStore:
    rng = random.Random(seed)
    store = MedallionStore(root=root)
    vocab = _vocab(rng, 40_000)
    # Zipf-ish: a few hundred words appear on thousands of labels (the "Boston"s and
    # "Sierra"s), the long tail once or twice.
    weights = [1.0 / (i + 1) ** 0.8 for i in range(len(vocab))]
    cum: list[float] = []
    for w in weights:                    # cumulative once; `choices` is O(n) per call otherwise
        cum.append((cum[-1] if cum else 0.0) + w)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[tuple[str, str, str, str]] = []
    producer_ids: list[str] = []
    for i in range(n_producers):
        pid = f"prod:{i}"
        producer_ids.append(pid)
        name = " ".join(rng.choices(vocab, cum_weights=cum, k=rng.randint(1, 2)))
        name += " " + rng.choice(SUFFIXES)
        rows.append((pid, "producer", json.dumps({"id": pid, "name": name}), now))
    for i in range(n_products):
        pid = f"ttb:{i}"
        k = rng.random()
        words = rng.choices(vocab, cum_weights=cum, k=1 if k < 0.35 else (2 if k < 0.8 else 3))
        name = " ".join(words)
        if rng.random() < 0.6:
            name += " " + rng.choice(STYLES)
        rec = {"id": pid, "name": name, "producer_id": rng.choice(producer_ids),
               "brand_id": "brand:none", "category": "beer"}
        rows.append((pid, "product", json.dumps(rec), now))
    for pid, pname, products in SEED:
        rows.append((pid, "producer", json.dumps({"id": pid, "name": pname}), now))
        for j, (name, brand) in enumerate(products):
            bid = None
            if brand:
                bid = f"brand:{pid}:{j}"
                rows.append((bid, "brand", json.dumps({"id": bid, "producer_id": pid,
                                                       "name": brand}), now))
            rid = f"{pid}:{j}"
            rec = {"id": rid, "name": name, "producer_id": pid,
                   "brand_id": bid or "brand:none", "category": "beer"}
            rows.append((rid, "product", json.dumps(rec), now))
    with store._lock:
        store._db.executemany("INSERT OR REPLACE INTO gold VALUES (?,?,?,?)", rows)
        store._db.commit()
    return store


def _timed(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def _report(label: str, ms: list[float]) -> None:
    ms = sorted(ms)
    p50 = statistics.median(ms)
    p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))]
    print(f"  {label:<46} p50 {p50:8.2f} ms   p95 {p95:8.2f} ms   max {ms[-1]:8.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=int, default=534_000)
    ap.add_argument("--producers", type=int, default=38_845)
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()

    root = tempfile.mkdtemp(prefix="bcd-bench-")
    t0 = time.perf_counter()
    store = make_catalog(root, args.products, args.producers)
    print(f"catalog: {store.counts()} in {time.perf_counter() - t0:.1f}s at {root}")

    cache = os.path.join(root, "label_index.pkl")
    t0 = time.perf_counter()
    index = LabelIndex.for_store(store, cache)
    print(f"index build: {index.stats()} ({time.perf_counter() - t0:.1f}s wall)")
    t0 = time.perf_counter()
    LabelIndex.for_store(store, cache)
    print(f"index load from cache: {time.perf_counter() - t0:.1f}s, "
          f"{os.path.getsize(cache) / 1e6:.0f} MB on disk")

    resolver = Resolver(IndexedStore(store, index))

    lines = ["HEADY TOPPER", "THE ALCHEMIST", "STOWE VERMONT", "FADY TOP", "ACHE MIST-VERM",
             "NK FROM THEO BANGE", "FOCAL BANGER THE ALCHEMIST INDIA PALE ALE",
             "DOGFISH HEAD 60 MINUTE IPA", "DRINK FROM THE CAN", "16 FL OZ"]
    frame = [DetectedText(text=t) for t in ["HEADY TOPPER", "THE ALCHEMIST", "STOWE VERMONT",
                                             "DRINK FROM THE CAN", "AMERICAN DOUBLE IPA",
                                             "ALC 8% BY VOL"]]
    heady = DetectedObject(id="o1", texts=["HEADY TOPPER", "THE ALCHEMIST", "STOWE VERMONT",
                                           "DRINK FROM THE CAN"], frames_seen=3)
    focal = DetectedObject(id="o2", texts=["FOCAL BAN", "THE ALCHEMIST", "INDIA PALE ALE"],
                           frames_seen=3)
    garbage = DetectedObject(id="o3", texts=["FADY TOP", "ACHE MIST-VERM", "NK FROM THEO BANGE"],
                             frames_seen=3)

    print("\nretrieval (index only):")
    for line in lines:
        _report(f"match_products {line!r}",
                _timed(lambda line=line: index.match_products(line, 24), args.rounds))
    print("\nend to end (Resolver.resolve, hydrated + scored):")
    _report("one line", _timed(lambda: resolver.resolve(
        ScanResolveRequest(detections=frame[:1])), args.rounds))
    _report("six-line frame", _timed(lambda: resolver.resolve(
        ScanResolveRequest(detections=frame)), args.rounds))
    _report("one object (4 lines)", _timed(lambda: resolver.resolve(
        ScanResolveRequest(objects=[heady])), args.rounds))
    _report("three objects", _timed(lambda: resolver.resolve(
        ScanResolveRequest(objects=[heady, focal, garbage])), args.rounds))

    print("\nverdicts:")
    resp = resolver.resolve(ScanResolveRequest(objects=[heady, focal, garbage]))
    for o in resp.objects:
        names = [f"{c.resolved.product.name} ({c.match_score})" for c in o.candidates]
        print(f"  {o.object_id:<4} {o.status:<11} {names}")
    for text, expect in [("FADY TOP", "unresolved"), ("ACHE MIST-VERM", "unresolved"),
                         ("NK FROM THEO BANGE", "unresolved"),
                         ("FOCAL BANGER THE ALCHEMIST INDIA PALE ALE", "resolved")]:
        o = resolver.resolve(ScanResolveRequest(
            objects=[DetectedObject(id="x", texts=[text])])).objects[0]
        names = [f"{c.resolved.product.name} ({c.match_score})" for c in o.candidates]
        mark = "ok " if o.status == expect else "!! "
        print(f"  {mark}{text!r:<48} {o.status:<11} {names}")
    store.close()


if __name__ == "__main__":
    main()
