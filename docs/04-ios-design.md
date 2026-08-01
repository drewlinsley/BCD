# 04 — iOS design

Target **iPhone**, deployment floor **iOS 18**, built against the **iOS 26 SDK** (Xcode 26). iOS 26+ features are `#if canImport` / `@available` gated so the install base isn't cut off.

## Package split

```
ios/BCDKit/   SwiftPM core — builds & tests on the macOS host (no Xcode needed)
ios/BCDApp/   SwiftUI app — generated into an .xcodeproj by XcodeGen
ios/project.yml
```

`BCDKit` is deliberately host-buildable so `swift build && swift test` verifies real code on any machine — **8 tests pass on the Intel reference laptop with only Command Line Tools** (Swift Testing, not XCTest, which needs full Xcode). The app target's own tests run under Xcode.

## Screens

| Screen | Role |
|---|---|
| **Scan** ([ScanView](../ios/BCDApp/Sources/ScanView.swift)) | the camera HUD — the whole thesis in one view |
| **Product detail** ([ProductDetailView](../ios/BCDApp/Sources/ProductDetailView.swift)) | "the receipt" — full ingredient/process tree, a provenance chip on every fact |
| **Search** | catalog lookup |
| **Alerts** | sentinel hits; Tier-1 agent = deep link to buy |
| **You** ([ProfileView](../ios/BCDApp/Sources/ProfileView.swift)) | weekly-evolution card + consent controls |

## The HUD

- Live detections from `DataScannerViewController` (`recognizesMultipleItems: true`, `qualityLevel: .fast`) become **overlays anchored to their bounding boxes**, color-coded by predicted enjoyment (green > 0.75 > yellow > 0.5 > orange).
- A **flask icon** marks cold-start scores (from chemistry, no reviews) — a visible signal of the moat.
- Tap an overlay → the receipt. Pinch to zoom.
- A **persistent chat bar** takes natural-language asks ("cheapest hazy here", "nothing over 6%") and routes them, via `LLMProvider.rerank`, against **the items currently in frame** — not a global search.

`ScanCoordinator` ([BCDKit](../ios/BCDKit/Sources/BCDKit/ScanCoordinator.swift)) is the client half of the latency path: it dedupes stable text so we don't re-query it, batches fresh detections to `/v1/scan/resolve`, and publishes ranked candidates the HUD renders. That dedupe is what keeps us inside the <400ms budget.

## LLM routing

`LLMProvider` protocol, three implementations:

- **`FoundationModelsProvider`** (iOS 26+, on-device) — free, private, fast. Used for **intent parsing and reranking, never facts** — the 3B model hallucinates world knowledge confidently, so product facts always come from the backend. Guarded by `#if canImport(FoundationModels)`; inert on the Intel Simulator.
- **`CloudLLMProvider`** — Claude / Gemini Flash, server-side, for the cold path.
- **`MockLLMProvider`** — deterministic rule-based parser for tests, previews, and offline fallback.

The composition root ([AppEnvironment](../ios/BCDApp/Sources/BCDApp.swift)) picks the best available at launch. iOS 27's `LanguageModelExecutor` would unify these behind one `LanguageModelSession` — adopt when the floor rises (out of reach on this machine; post-v1).

## System integration

- **Visual Intelligence** (iOS 26+): `IntentValueQuery` + `SemanticContentDescriptor` registers BCD as a provider — point the system camera at a beer, BCD is offered.
- **App Intents / Shortcuts / Spotlight**, a **Live Activity** for an active bar session, and a home-screen **widget** for alerts.

## Camera & privacy

`NSCameraUsageDescription` and `NSLocationWhenInUseUsageDescription` are set in [project.yml](../ios/project.yml). **Raw camera frames are never uploaded** — only derived OCR strings, and only under the personalization consent tier. Age gate: alcohol content requires a 17+ rating and an age check ([06-legal.md](06-legal.md)).
