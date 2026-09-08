# 08 — Build & verify: the coarse-to-fine scan pipeline

Step-by-step for a human or a coding agent picking up PR #8 on a fresh checkout. Run in
order; each step says what "good" looks like and what to do if it isn't.

PR #8 is stacked on `feat/image-capture-scan` (PR #6): it contains that branch plus the
object path, the label index and the object verdicts. Everything below was run on the
branch as pushed — the Swift package was compiled and its tests executed on a Linux
toolchain (Swift 6.1), the Python suite and the latency benchmark on Python 3.12.

## Requirements

| Need | Version | Why |
|---|---|---|
| A Swift toolchain | **6.0+** (`swift --version`) — macOS Command Line Tools, or `swiftlang` on Ubuntu 25.10+ / any `swift.org` Linux toolchain | `make test-swift` builds and tests `BCDKit` on the host — no Xcode, no Apple SDK |
| Python | **3.12** (`python3.12`) | backend + tests (`make venv`) |
| Xcode | **26.0** (iOS 26 SDK) | app build; the Vision / VisionKit / Foundation Models files only compile here. The reference machine caps at 26.0 — do **not** reach for iOS 27 / WWDC26 APIs (image input to Foundation Models, tap-to-segment, Core AI) |
| xcodegen | any recent (`brew install xcodegen`) | `make ios-gen` generates `ios/BCDApp.xcodeproj` (never committed) |
| A physical iPhone, iOS 18+ | iOS 26 on A17 Pro / M-series for the on-device model | the camera pipeline; the Simulator has no camera and no Apple Intelligence |
| `ios/Local.xcconfig` | copied from `Local.xcconfig.example` by `make ios-gen` | `DEVELOPMENT_TEAM`, `BCD_BUNDLE_ID`, `BCD_API_HOSTPORT` (your Mac's LAN IP), `BCD_SCAN_ENGINE` |
| Postgres | optional | the `test_pg_store` tests skip without one; nothing in this PR needs it |

No new third-party dependencies, Swift or Python. The Swift package is `ios/BCDKit/Package.swift`
(tools 6.0, iOS 18 / macOS 13 platforms). On Linux it adds a two-name stand-in for Combine
(`Sources/CombineShim`) that Apple platforms never compile.

## Step 1 — backend

```bash
make venv          # once: python3.12 venv + editable install
make test-py       # expect: 261 passed, 12 skipped (Postgres)
```

`make lint` reports three pre-existing `I001` import-order findings in `tests/test_connectors.py`
and `tests/test_vision.py`; they come from `feat/image-capture-scan`, not this PR (`ruff --fix`
clears them). Everything else is clean.

The scorer the review measured against real frames is reproduced as tests: run
`.venv/bin/pytest tests/test_resolver.py -k "garbled_fragment or clean_full_label"` and read
the fixture — `Top's`, `Ache`, `Theo P.`, `Banger` are seeded as the junk rows, and the four
device readings from the review table must come back `unresolved` / `resolved: Focal Banger`.

### Latency

```bash
.venv/bin/python scripts/bench_index.py                 # 534k products, 38.8k producers
```

Builds a synthetic catalog the size of production in a throwaway SQLite store, builds the
label index over it and times the HUD's requests. Expect, on one laptop core:

| stage | expect |
|---|---|
| index build | ~10 s, ~40 MB pickled; later starts load it in < 1 s (rebuilt when row counts change) |
| `match_products`, one line | < 1 ms |
| `Resolver.resolve`, six-line frame | ~2–3 ms |
| `Resolver.resolve`, three objects | ~5 ms |

The old path (trigram scan per line in the store) was 1.6 s per frame concurrently and 7 s
per object; the budget is a 350 ms HUD tick. The index answers `match_products`,
`match_products_many`, `match_producers` and `products_of` from memory and passes everything
else to the wrapped store (`IndexedStore`), so `Resolver` is unchanged. `BCD_LABEL_INDEX=0`
serves from the store's own matching again, for an A/B on the same catalog. On Postgres the
first start walks the catalog once (`iter_gold` of 534k rows) to build the cache at
`data/label_index.pkl`; that is the one slow start.

Optional end-to-end check against the demo catalog:

```bash
make demo && make api      # seeds 4 products, serves on :8000
curl -s -X POST localhost:8000/v1/scan/resolve -H 'content-type: application/json' \
  -d '{"objects":[{"id":"a","texts":["GALAXY","HAZE","BCD DEMO BREWING CO"],"frames_seen":4},
                  {"id":"b","texts":["Mist","hop"],"frames_seen":4},
                  {"id":"c","barcode":"000000000031","texts":[]}]}' | python3 -m json.tool
# a -> "resolved" Galaxy Haze · b -> "unresolved", no candidates · c -> "resolved" by UPC
curl -s localhost:8000/v1/lexicon | head -c 300     # catalog vocabulary, no "ipa"/"brewing"
curl -s localhost:8000/healthz                       # includes the index's stats
```

## Step 2 — BCDKit on the host

```bash
make test-swift    # = cd ios/BCDKit && swift build && swift test
```

Expect a clean build and `Test run with 88 tests passed`: the 65 suites/tests from
`feat/image-capture-scan` plus, from this PR, `ObjectContract`, `Geometry`, `TextClustering`,
`Tracking`, `ConstrainedPick`, `ObjectStageBehaviour` and `ObjectsOnTheLiveTick`
(`Tests/BCDKitTests/ObjectStageTests.swift`).

The host build compiles the pure-Swift core — `ScanContract`, `Geometry`, `SceneObjects`,
`ObjectStage`, `ScanEngine`, `ScanCoordinator`, `LLMProvider`, `APIClient`, `Telemetry`. The
iOS-only files (`VisionKitScanEngine`, `VisionTextReader`, `VisionFrameScanEngine`,
`FoundationModelsProvider`) are behind `#if canImport(...) && os(iOS)` and are skipped here.

**If it doesn't compile on your toolchain** (it did on Swift 6.1 / Linux): the tests define
the intended behaviour. Fix the code to satisfy them; do not loosen assertions, the tracker's
thresholds (`ObjectTracker.Config`) or the mock adjudicator's (0.6 coverage, 0.2 margin, a
substantial name word read), and do not delete tests to get green. Keep the public names the
app uses (`ObjectStage`, `ScanCoordinator.objectStage`, `RegionProvider`, `LexiconConsumer`,
`FineTextReader`, `ScanEngine.contentAspect`).

## Step 3 — the app under the iOS 26 SDK (compiles the Vision files)

```bash
make ios-gen                                   # xcodegen → ios/BCDApp.xcodeproj (+ Local.xcconfig if missing)
cd ios && xcodebuild -project BCDApp.xcodeproj -scheme BCDApp \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build   # compile-only, no team needed
```

(`make ios-build` is the same without `CODE_SIGNING_ALLOWED=NO`; it needs `DEVELOPMENT_TEAM` set.)
App-target tests run in the Simulator: `xcodebuild test -project BCDApp.xcodeproj -scheme BCDApp -destination 'platform=iOS Simulator,name=iPhone 16'`.

This is the one step nobody has run yet: the iOS-only files cannot be compiled without the
SDK. **If they don't compile**, these are the APIs the code assumes (all iOS 18 Vision
Swift API, all present in the iOS 26 SDK) and the fallback if a name differs:

| Used | If it doesn't exist as written |
|---|---|
| `RecognizeTextRequest.perform(on: CGImage/CVPixelBuffer)` | `try await ImageRequestHandler(image).perform(request)` |
| `NormalizedRect.toImageCoordinates(CGSize(width:1,height:1), origin: .upperLeft)` | use `.cgRect` (origin lower-left) and flip: `y = 1 - (rect.minY + rect.height)` |
| `RecognizedTextObservation.topCandidates(1).first?.string / .confidence` | unchanged from `VNRecognizedTextObservation` |
| `RecognizeTextRequest.customWords`, `.usesLanguageCorrection`, `.recognitionLevel` | same names as the `VN*` request |
| `DetectBarcodesRequest` → `BarcodeObservation.payloadString`, `.symbology` | `String(describing:)` on the symbology is intentional |
| `GenerateForegroundInstanceMaskRequest.perform(on:)` → `InstanceMaskObservation?` | typed as optional on purpose; works if non-optional too |
| `InstanceMaskObservation.allInstances`, `.instanceMask`, `.generateMaskedImage(for:imageFrom:croppedToInstancesExtent:)` | check the exact label spelling in the SDK header |
| `ClassifyImageRequest` → `ClassificationObservation.identifier / .confidence` | unchanged |
| `AVCaptureConnection.isVideoRotationAngleSupported(90)` / `.videoRotationAngle` | iOS 17+; if unavailable pass `orientation: .right` to Vision instead |
| `DataScannerViewController.capturePhoto()` (VisionKit, iOS 16+) | unchanged |
| `LanguageModelSession(instructions: String)` (Foundation Models, iOS 26) | `LanguageModelSession()` and prepend the instructions to the prompt |

Same rules as Step 2: fix names, keep behaviour. Do not switch `usesLanguageCorrection`
back on for the coarse pass or drop `customWords` — those are the point.

## Step 4 — on a device

```bash
make api-lan                      # backend on 0.0.0.0:8000; put this Mac's LAN IP in Local.xcconfig
# ios/Local.xcconfig: DEVELOPMENT_TEAM, BCD_BUNDLE_ID, BCD_API_HOSTPORT = <lan-ip>:8000, BCD_SCAN_ENGINE = visionkit
make ios-gen && open ios/BCDApp.xcodeproj  # run on the phone
BCD_SCAN_LOG=scan.jsonl make api-lan       # optional: every request's lines, objects and verdicts, one row each
```

1. **`visionkit` (default) first.** Point at a shelf. The HUD is the one from
   `feat/image-capture-scan` — chips only on what the frame corroborates — plus one chip per
   *resolved object*, pinned to the can and following it. An ambiguous or unresolved object
   draws nothing. No chip should ever carry a name that isn't on the can; if one does, the
   `BCD_SCAN_LOG` row for that request has the object's texts and the server's verdict.
2. **Then `BCD_SCAN_ENGINE = vision`** (regenerate: `make ios-gen`). Same HUD, but boxes come
   from Vision instance masks and OCR is uncorrected. This engine has not run on a device
   yet; things to tune in `VisionFrameScanEngine.Config` are `segmentEveryNFrames`,
   `minRegionArea`, and the `containerWords` classifier gate. If overlays sit off the can,
   the buffer→view mapping (`AspectFillMapper`, driven by `contentAspect`) is the suspect.
3. Watch telemetry: `data/telemetry_events.jsonl` on the Mac gets `scan_object_resolution`
   rows (`status`, `stage` ∈ barcode/coarse/fine_ocr/llm_pick/user). That tells you which
   stage is earning its keep.

## Step 5 — before merging

```bash
make codegen && git diff --exit-code   # telemetry bindings are generated, must be in sync
make verify                            # registry + python + swift
```
