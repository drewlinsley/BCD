"""Replay logged scan frames through the resolver, and diff what two code states would draw.

The API appends every /v1/scan/resolve request to `BCD_SCAN_LOG` (a .jsonl of the frame's
OCR lines and tracked objects). Replaying that log through a `Resolver` built exactly as
`app.lifespan` builds it, before and after a change, and diffing the DRAWN set -- corroborated
frame candidates plus resolved object verdicts -- is how every resolver change is measured.
It has rejected several designs that read well: a ranking that preferred fewer unread name
tokens cost 36 frames their right answer, and one-word proofs were 14 for 14 wrong.

    .venv/bin/python scripts/replay_scans.py out.json            # replay data/scans*.jsonl
    .venv/bin/python scripts/replay_scans.py before.json after.json --diff [-v]

Frames are keyed by timestamp in the diff, so a log that grew between the two runs only adds
frames to the "new" line; a frame present in both is compared. The store and index come from
the API's environment (.env is read), the same way the API gets them.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("services/api", "packages/schema", "services/ingest"):
    sys.path.insert(0, str(ROOT / p))

SCAN_GLOB = str(ROOT / "data" / "scans*.jsonl")


def _is_code(s: str) -> bool:
    return s.isdigit() and len(s) in (8, 12, 13, 14)


def replay(out_path: str) -> None:
    from bcd_api.app import _load_dotenv
    from bcd_api.index import IndexedStore, LabelIndex
    from bcd_api.resolver import Resolver
    from bcd_ingest.store import open_store
    from bcd_schema.api import DetectedObject, DetectedText, ScanResolveRequest

    os.chdir(ROOT)
    _load_dotenv()
    store = open_store(root="./data")
    index = LabelIndex.for_store(store, os.path.join("./data", "label_index.pkl"))
    resolver = Resolver(IndexedStore(store, index))

    rows = []
    for f in sorted(glob.glob(SCAN_GLOB)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    r["_file"] = os.path.basename(f)
                    rows.append(r)

    out = []
    t0 = time.time()
    for r in rows:
        dets = [DetectedText(text=t, kind="barcode" if _is_code(t) else "text")
                for t in (r.get("ocr") or [])]
        objs = [DetectedObject(id=o["id"], texts=o.get("texts") or [], barcode=o.get("barcode"))
                for o in (r.get("objects") or [])]
        if not dets and not objs:
            continue
        resp = resolver.resolve(ScanResolveRequest(detections=dets, objects=objs,
                                                   include_score=False))
        out.append({
            "file": r["_file"], "ts": r["ts"],
            "ocr": r.get("ocr") or [],
            "obj_texts": [o.get("texts") or [] for o in (r.get("objects") or [])],
            "corroborated": resp.corroborated,
            "cands": [(c.resolved.product.name, c.resolved.producer.name, c.match_score)
                      for c in resp.candidates],
            "verdicts": [(o.object_id, o.status,
                          o.candidates[0].resolved.product.name if o.candidates else None)
                         for o in resp.objects],
        })
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=0)
    print(f"replayed {len(out)} frames in {time.time() - t0:.1f}s -> {out_path}")


def _drawn(r: dict) -> set[str]:
    s = {c[0] for c in r["cands"]} if r["corroborated"] else set()
    return s | {v[2] for v in r["verdicts"] if v[1] == "resolved" and v[2]}


def _shortlisted(r: dict) -> set[str]:
    return {v[2] for v in r["verdicts"] if v[1] == "ambiguous" and v[2]}


def diff(before_path: str, after_path: str, verbose: bool) -> None:
    with open(before_path) as fh:
        before = {r["ts"]: r for r in json.load(fh)}
    with open(after_path) as fh:
        after = {r["ts"]: r for r in json.load(fh)}
    shared = sorted(set(before) & set(after))
    gained, lost, sl_gained, sl_lost = Counter(), Counter(), Counter(), Counter()
    changed = []
    for ts in shared:
        a, b = before[ts], after[ts]
        da, db = _drawn(a), _drawn(b)
        gained.update(db - da)
        lost.update(da - db)
        sl_gained.update(_shortlisted(b) - _shortlisted(a))
        sl_lost.update(_shortlisted(a) - _shortlisted(b))
        if da != db:
            changed.append((ts, a["ocr"], sorted(da), sorted(db)))
    print(f"{len(shared)} frames compared; drawn set changed on {len(changed)}")
    print("GAINED:", dict(gained))
    print("LOST:", dict(lost))
    print("shortlist gained:", dict(sl_gained))
    print("shortlist lost:", dict(sl_lost))
    new = sorted(set(after) - set(before))
    if new:
        drew = Counter(n for ts in new for n in _drawn(after[ts]))
        print(f"{len(new)} frames only in the after-run; they draw: {dict(drew)}")
    if verbose:
        for ts, ocr, da, db in changed:
            print(ts[5:19], ocr, da, "->", db)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--diff" in sys.argv:
        diff(args[0], args[1], "-v" in sys.argv)
    else:
        replay(args[0])
