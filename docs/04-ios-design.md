# 04 — iOS design

Target **iPhone**, deployment floor **iOS 18**, built against the **iOS 26 SDK** (Xcode 26). iOS 26+ features are `#if canImport` / `@available` gated so the install base isn't cut off.

## Package split

```
ios/BCDKit/   SwiftPM core — builds & tests on the macOS host (no Xcode needed)
ios/BCDApp/   SwiftUI app — generated into an .xcodeproj by XcodeGen
ios/project.yml
```

`BCDKit` is deliberately host-buildable so `swift build && swift test` verifies real code on any machine with only Command Line Tools (Swift Testing, not XCTest, which needs full Xcode) — the model contract, geometry, text clustering, object tracking, the coarse-to-fine coordinator and the rule-based adjudicator all run there. The app target's own tests run under Xcode.

## Screens

| Screen | Role |
|---|---|
| **Scan** ([ScanView](../ios/BCDApp/Sources/ScanView.swift)) | the camera HUD — the whole thesis in one view |
| **Product detail** ([ProductDetailView](../ios/BCDApp/Sources/ProductDetailView.swift)) | "the receipt" — full ingredient/process tree, a provenance chip on every fact |
| **Search** | catalog lookup |
| **Alerts** | sentinel hits; Tier-1 agent = deep link to buy |
| **You** ([ProfileView](../ios/BCDApp/Sources/ProfileView.swift)) | weekly-evolution card + consent controls |

## The HUD

- Tracked **objects** (cans, bottles) become overlays anchored to their boxes, color-coded by predicted enjoyment (green > 0.75 > yellow > 0.5 > orange). Only a *resolved* object gets a name.
- An **ambiguous** object shows a "N possible · tap" chip; tapping opens the server's shortlist and the user's pick is logged as `scan_corrected_by_user` — the highest-value label we collect.
- Anything else that has proven it's really there is a faint dashed outline, never a name.
- A **flask icon** marks cold-start scores (from chemistry, no reviews) — a visible signal of the moat. A **sparkles** icon marks a name the on-device model adjudicated; a **person** icon, one the user chose.
- Tap an overlay → the receipt. Pinch to zoom.
- A **persistent chat bar** takes natural-language asks ("cheapest hazy here", "nothing over 6%") and routes them, via `LLMProvider.rerank`, against **the items currently in frame** — non-matching overlays dim.

## Coarse-to-fine scan pipeline

The first HUD queried every OCR line and showed the first hit. Stylized label type (Heady Topper, hazy-IPA cans, Dogfish Head) OCRs into fragments — "Chemist", "hop chemist", "Mist" — and every fragment found *some* product to match, so wrong names fired constantly. The pipeline now reasons about **objects**, escalates only when it has to, and never shows a name nothing cleared.

```
 frame ──► coarse ─────────────► server ─────────► fine 1 ────────► fine 2 ────────► HUD
 OCR +     TextClusterer /       /v1/scan/resolve   careful OCR on    on-device model
 barcode   segmenter regions     per *object*:      the object crop   picks among the
 (+regions)  → ObjectTracker     resolved /         (accurate, no     server's shortlist
             2 frames of         ambiguous /        lang. correction, — or says none
             agreement           unresolved         catalog lexicon)
```

| Stage | Where | What |
|---|---|---|
| **coarse** | [SceneObjects.swift](../ios/BCDKit/Sources/BCDKit/SceneObjects.swift) | `TextClusterer` groups lines on one label; with `VisionFrameScanEngine`, Vision's foreground-instance masks give real can/bottle boxes and lines are assigned to the region they fall in. `ObjectTracker` matches across frames (IoU / center distance) and counts how many frames each line was seen on each object. An object is queried only after **two frames of agreement**, once per change in evidence, with a cooldown. |
| **server** | [resolver.py](../services/api/bcd_api/resolver.py), [matching.py](../services/api/bcd_api/matching.py) | One `DetectedObject` (all its lines + barcode + box) per object. Candidates are retrieved by product *and* producer name, scored against the full identity with generic label words down-weighted and OCR-typo-tolerant token matching, and returned with a verdict: `resolved` needs a floor **and** a margin over the runner-up. |
| **fine 1** | [VisionTextReader.swift](../ios/BCDKit/Sources/BCDKit/VisionTextReader.swift) | Not resolved? Read that object's crop again at `.accurate`, both with language correction **off** (raw glyphs; correction is what turns ALCHEMIST into "Chemist") and with it on plus the catalog **lexicon** (`/v1/lexicon`) as `customWords`. New text re-queries. Once per object. |
| **fine 2** | `LLMProvider.pickProduct` | Still ambiguous? The on-device model is shown the OCR fragments and the shortlist and must answer with a number or NONE. It cannot invent a beer. Once per object. |
| **user** | `ScanCoordinator.confirm` | Still ambiguous? The chip says so; the user's pick trains the resolver. |

`ScanCoordinator` ([BCDKit](../ios/BCDKit/Sources/BCDKit/ScanCoordinator.swift)) drives it and publishes one `SceneObject` per track with its status and box. Every stage is host-testable: the tracker, clusterer and coordinator run under `swift test` with `MockScanEngine` (which also scripts fine reads) and a scripted server.

Two engines implement the coarse stage; `BCD_SCAN_ENGINE` in `Local.xcconfig` picks:

- **`visionkit`** (default) — [VisionKitScanEngine](../ios/BCDKit/Sources/BCDKit/VisionKitScanEngine.swift): `DataScannerViewController`, text + barcode, view-space boxes. Can't segment or disable language correction; the fine pass captures a still (`capturePhoto`) and reads the object crop.
- **`vision`** — [VisionFrameScanEngine](../ios/BCDKit/Sources/BCDKit/VisionFrameScanEngine.swift): our own `AVCaptureSession` into Vision. Raw OCR every 2nd frame; `GenerateForegroundInstanceMaskRequest` + `ClassifyImageRequest` every 12th frame to lift cans/bottles and drop hands, faces and menus; fine reads crop the last frame directly. Buffer-space boxes, mapped onto the aspect-fill preview by `AspectFillMapper`. **Not yet exercised on a device** — try it once `visionkit` is confirmed working.

## LLM routing

`LLMProvider` protocol, three implementations:

- **`FoundationModelsProvider`** (iOS 26+, on-device) — free, private, fast. Used for **intent parsing, reranking and constrained label adjudication, never facts** — the 3B model hallucinates world knowledge confidently, so product facts always come from the backend, and in the scan path it may only choose among catalog candidates. Guarded by `#if canImport(FoundationModels)`; inert on the Intel Simulator.
- **`CloudLLMProvider`** — Claude / Gemini Flash, server-side, for the cold path.
- **`MockLLMProvider`** — deterministic rule-based parser and adjudicator for tests, previews, and offline fallback.

The composition root ([AppEnvironment](../ios/BCDApp/Sources/BCDApp.swift)) picks the best available at launch.

**On image input.** The iOS 26 SDK's on-device model is text-only. WWDC26's multimodal prompts (an image alongside text), the point-prompted segmentation API and Core AI need the iOS 27 SDK / Xcode 27 — out of reach on the reference machine, which caps at Xcode 26.0. So the design keeps *understanding* on text the on-device model can handle (OCR fragments → constrained choice) and does *seeing* with the iOS 18 Vision APIs that are in the iOS 26 SDK today (instance masks, classification, accurate OCR with custom words). When the floor rises, an image-prompted reader slots in behind the same `FineTextReader` / `pickProduct` seams — it becomes fine stage 1½, not a rewrite. iOS 27's `LanguageModelExecutor` would likewise unify the providers behind one `LanguageModelSession`.

## System integration

- **Visual Intelligence** (iOS 26+): `IntentValueQuery` + `SemanticContentDescriptor` registers BCD as a provider — point the system camera at a beer, BCD is offered.
- **App Intents / Shortcuts / Spotlight**, a **Live Activity** for an active bar session, and a home-screen **widget** for alerts.

## Camera & privacy

`NSCameraUsageDescription` and `NSLocationWhenInUseUsageDescription` are set in [project.yml](../ios/project.yml). **Raw camera frames are never uploaded** — only derived OCR strings, and only under the personalization consent tier. Age gate: alcohol content requires a 17+ rating and an age check ([06-legal.md](06-legal.md)).
