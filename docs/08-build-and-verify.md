# 08 — Build & verify: the coarse-to-fine scan pipeline

Step-by-step for a human or a coding agent picking up PR #8 on a fresh checkout. Run in
order; each step says what "good" looks like and what to do if it isn't.

## Requirements

| Need | Version | Why |
|---|---|---|
| macOS + Xcode Command Line Tools | Swift **6.0** toolchain (`swift --version`) | `make test-swift` builds and tests `BCDKit` on the host — no Xcode needed |
| Python | **3.12** (`python3.12`) | backend + tests (`make venv`) |
| Xcode | **26.0** (iOS 26 SDK) | app build; the Vision / VisionKit / Foundation Models files only compile here. The reference machine caps at 26.0 — do **not** reach for iOS 27 / WWDC26 APIs (image input to Foundation Models, tap-to-segment, Core AI) |
| xcodegen | any recent (`brew install xcodegen`) | `make ios-gen` generates `ios/BCDApp.xcodeproj` (never committed) |
| A physical iPhone, iOS 18+ | iOS 26 on A17 Pro / M-series for the on-device model | the camera pipeline; the Simulator has no camera and no Apple Intelligence |
| `ios/Local.xcconfig` | copied from `Local.xcconfig.example` by `make ios-gen` | `DEVELOPMENT_TEAM`, `BCD_BUNDLE_ID`, `BCD_API_HOSTPORT` (your Mac's LAN IP), `BCD_SCAN_ENGINE` |
| Postgres | optional | the 4 `test_pg_store` tests skip without one; nothing in this PR needs it |

No new third-party dependencies, Swift or Python. The Swift package is `ios/BCDKit/Package.swift` (tools 6.0, iOS 18 / macOS 13 platforms).

## Step 1 — backend

```bash
make venv          # once: python3.12 venv + editable install
make test-py       # expect: 36 passed, 4 skipped (Postgres)
```

`make lint` reports one pre-existing E501 in `services/ingest/bcd_ingest/connectors/openfoodfacts.py:48`; it is not from this PR. Everything else is clean.

Optional end-to-end check of the new object path against the demo catalog:

```bash
make demo && make api      # seeds 4 products, serves on :8000
curl -s -X POST localhost:8000/v1/scan/resolve -H 'content-type: application/json' \
  -d '{"objects":[{"id":"a","texts":["GALAXY","HAZE","BCD DEMO BREWING CO"],"frames_seen":4},
                  {"id":"b","texts":["Mist","hop"],"frames_seen":4},
                  {"id":"c","barcode":"000000000031","texts":[]}]}' | python3 -m json.tool
# a -> "resolved" Galaxy Haze · b -> "unresolved", no candidates · c -> "resolved" by UPC
curl -s localhost:8000/v1/lexicon | head -c 300     # product/brand/producer names, no "IPA"/"Brewery"
```

## Step 2 — BCDKit on the host (the part that was NOT compiled before this PR was opened)

```bash
make test-swift    # = cd ios/BCDKit && swift build && swift test
```

Expect a clean build and all suites green: `ModelDecoding`, `Geometry`, `TextClustering`,
`Tracking`, `LLMParsing`, `TelemetryConsent`, `ScanCoordination` (~26 tests).

On macOS the host build compiles only the pure-Swift core — `ScanContract`, `Geometry`,
`SceneObjects`, `ScanEngine`, `ScanCoordinator`, `LLMProvider`, `APIClient`, `Telemetry`.
The iOS-only files (`VisionKitScanEngine`, `VisionTextReader`, `VisionFrameScanEngine`,
`FoundationModelsProvider`) are behind `#if canImport(...) && os(iOS)` and are skipped here.

**If it doesn't compile.** The code was written without a Swift toolchain available, so
expect only small, local fixes. Rules for fixing:

- The tests in `ios/BCDKit/Tests/BCDKitTests/BCDKitTests.swift` define the intended
  behavior. Fix the code to satisfy them; do not loosen assertions, thresholds
  (`ObjectTracker.Config`, `MockLLMProvider.pickProduct`'s 0.6 / 0.2), or delete tests to get green.
- Likely spots: Swift 6 strict-concurrency complaints (`Sendable` on closures/captures),
  tuple comparison in `Track.stableTexts`, multi-statement closure type inference in
  `ScanViewModel.rebuildOverlays` / `ScriptedAPI.resolveScan` (add an explicit return
  type or split the closure), optional enum pattern matching (add `?` after the pattern).
- Keep public API names used by the app (`ScanCoordinator.SceneObject`, `ObjectStatus`,
  `Stage`, `HUDOverlay` inputs, `LexiconConsumer`, `FineTextReader`, `ScanEngine.contentAspect`).

## Step 3 — the app under the iOS 26 SDK (compiles the Vision files)

```bash
make ios-gen                                   # xcodegen → ios/BCDApp.xcodeproj (+ Local.xcconfig if missing)
cd ios && xcodebuild -project BCDApp.xcodeproj -scheme BCDApp \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build   # compile-only, no team needed
```

(`make ios-build` is the same without `CODE_SIGNING_ALLOWED=NO`; it needs `DEVELOPMENT_TEAM` set.)
App-target tests run in the Simulator: `xcodebuild test -project BCDApp.xcodeproj -scheme BCDApp -destination 'platform=iOS Simulator,name=iPhone 16'`.

**If the Vision files don't compile**, these are the APIs the code assumes (all iOS 18
Vision Swift API, all present in the iOS 26 SDK) and the fallback if a name differs:

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

Same rules as Step 2: fix names, keep behavior. Do not switch `usesLanguageCorrection`
back on for the coarse pass or drop `customWords` — those are the point.

## Step 4 — on a device

```bash
make api-lan                      # backend on 0.0.0.0:8000; put this Mac's LAN IP in Local.xcconfig
# ios/Local.xcconfig: DEVELOPMENT_TEAM, BCD_BUNDLE_ID, BCD_API_HOSTPORT = <lan-ip>:8000, BCD_SCAN_ENGINE = visionkit
make ios-gen && open ios/BCDApp.xcodeproj  # run on the phone
```

1. **`visionkit` (default) first.** Point at a shelf. Expect: chips only on cans the server
   resolved; a "N possible · tap" chip on ambiguous ones; faint dashed outlines on the
   rest; a status line "N named · M in view · Xms". No chip should ever carry a name that
   isn't on the can. Tapping an ambiguous chip → shortlist → pick → chip with a person icon.
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
