import Testing
import Foundation
@testable import BCDKit

// Swift Testing (not XCTest) so `swift test` runs on a machine with only the Command Line
// Tools — no full Xcode required. The same suite runs under Xcode/CI unchanged.

@Suite struct ModelDecoding {
    // Exact JSON the FastAPI /v1/scan/resolve emits — proves the Codable contract holds.
    @Test func decodesScanResponseFromServerJSON() throws {
        let json = """
        {
          "candidates": [{
            "detection_index": 0,
            "resolved": {
              "product": {
                "id": "ttb:1", "brand_id": "brand:x", "producer_id": "prod:x",
                "category": "beer", "name": "Heady Topper",
                "style": {"value": "NEIPA", "provenance": {
                    "source_id": "ttb", "url": null, "quote": "ale",
                    "method": "regulatory_filing", "confidence": 1.0}},
                "spec": {"abv_pct": {"value": 8.0, "provenance": {
                    "source_id": "ttb", "url": null, "quote": null,
                    "method": "regulatory_filing", "confidence": 1.0}}},
                "recipe": {"ingredients": [{
                    "role": "aroma_hop", "entity_kind": "hop", "entity_ref": null,
                    "raw_name": "Citra", "provenance": {
                        "source_id": "producer", "url": null, "quote": "Citra",
                        "method": "stated_by_producer", "confidence": 1.0}}]}
              },
              "producer": {"id": "prod:x", "name": "The Alchemist", "kind": null,
                "country": null, "region": null, "lat": null, "lon": null, "website": null},
              "brand": {"id": "brand:x", "producer_id": "prod:x", "name": "Heady"}
            },
            "match_score": 1.0, "personal_score": 0.86,
            "reason": "matches your tropical preference", "cold_start": true
          }],
          "unresolved_indices": [],
          "latency_ms": 1.2
        }
        """.data(using: .utf8)!

        let resp = try JSONDecoder().decode(ScanResolveResponse.self, from: json)
        try #require(resp.candidates.count == 1)
        let c = resp.candidates[0]
        #expect(c.resolved.product.name == "Heady Topper")
        #expect(c.resolved.product.spec.abvPct?.value == 8.0)
        #expect(c.resolved.product.spec.abvPct?.provenance.method == .regulatoryFiling)
        #expect(c.coldStart)
        #expect(c.resolved.product.recipe.ingredients.first?.rawName == "Citra")
    }

    @Test func provenanceTrustRankOrders() {
        #expect(ExtractionMethod.regulatoryFiling.trustRank >
                ExtractionMethod.llmInferredFromStylePrior.trustRank)
        #expect(ExtractionMethod.statedByProducer.trustRank >
                ExtractionMethod.reviewConsensus.trustRank)
    }
}

@Suite struct LLMParsing {
    @Test func parsesAbvCeiling() async throws {
        let intent = try await MockLLMProvider().parseQuery("nothing over 6%")
        #expect(intent.maxAbv == 6.0)
    }

    @Test func parsesCheapAndStyle() async throws {
        let intent = try await MockLLMProvider().parseQuery("cheapest hazy IPA here")
        #expect(intent.sortBy == .price)
        #expect(intent.styleContains == "hazy")
    }

    @Test func rerankFiltersByAbv() async throws {
        let cands = [
            makeCandidate(id: "a", name: "Big DIPA", abv: 8.5, personal: 0.9),
            makeCandidate(id: "b", name: "Light Lager", abv: 4.2, personal: 0.4),
        ]
        let ranked = try await MockLLMProvider().rerank(cands, for: "nothing over 6%")
        #expect(ranked == ["b"])  // the 8.5% is filtered out
    }
}

@Suite struct TelemetryConsent {
    @Test func consentGatesEvents() async {
        let q = TelemetryQueue(consent: ConsentState(analytics: true, personalization: false))
        let ok = await q.log("session_start", tier: .analytics)
        let blocked = await q.log("scan_frame_batch", tier: .personalization)
        #expect(ok)
        #expect(!blocked)  // personalization not granted -> dropped
        let count = await q.pendingCount
        #expect(count == 1)
    }

    @Test func flushClearsOnlyAfterSinkAccepts() async throws {
        let sink = CountingSink()
        let q = TelemetryQueue(consent: ConsentState(analytics: true), sink: sink)
        _ = await q.log("session_start", tier: .analytics)
        try await q.flush()
        let remaining = await q.pendingCount
        #expect(remaining == 0)
        #expect(sink.batches == 1)
    }
}

@Suite struct ScanCoordination {
    @MainActor
    @Test func liveModeResolvesLatestFrameAndSkipsUnchanged() async throws {
        // Fixed-rate live mode resolves the latest frame, pins each overlay to its detection's box
        // center, and skips the re-resolve when the OCR is unchanged (camera held still) — so a
        // held viewfinder stays cheap instead of firehosing the backend every tick. No shutter.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "Krombacher", kind: "text", x: 0.3, y: 0.4, w: 0.2, h: 0.1)],
        ])
        let api = StubAPI()
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)  // let the frame buffer
        #expect(api.resolveCallCount == 0)              // the viewfinder alone never resolves
        await coord.resolveLatest()                     // one live tick
        #expect(api.resolveCallCount == 1)
        #expect(coord.overlays.count == 1)
        let overlay = try #require(coord.overlays.first)
        #expect(overlay.candidate.resolved.product.name == "Krombacher")
        #expect(abs(overlay.x - 0.4) < 0.001)           // box center x = 0.3 + 0.2/2
        #expect(abs(overlay.y - 0.45) < 0.001)          // box center y = 0.4 + 0.1/2
        await coord.resolveLatest()                     // same frame → deduped, no round-trip
        #expect(api.resolveCallCount == 1)
    }

    @MainActor
    @Test func liveAutoInterpretsWhenNothingResolves() async throws {
        // A stylized label OCRs as garbage that matches nothing. With no shutter, the live tick
        // itself triggers the on-device fallback: it names the product, and *that* clean name
        // resolves and anchors — zero clicking.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "FADY TOPP", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)],
        ])
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let llm = StubLLM(guess: "Heady Topper")
        let coord = ScanCoordinator(engine: engine, api: api, llm: llm)
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()                     // one live tick, no clicking
        await coord.interpretation?.value               // fallback is detached; await it here
        #expect(api.resolveCallCount == 2)              // raw OCR (miss) then the LLM guess (hit)
        #expect(llm.calls == 1)
        #expect(coord.overlays.count == 1)
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")
    }

    @MainActor
    @Test func liveAutoInterpretRunsOncePerFrame() async throws {
        // The fallback is debounced by OCR signature: a held-still garbled label runs the on-device
        // model once, not on every tick.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "FADY TOPP", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)],
        ])
        let api = CatalogStubAPI(known: [])             // nothing ever resolves
        let llm = StubLLM(guess: "Still Unmatched")     // guess doesn't resolve either
        let coord = ScanCoordinator(engine: engine, api: api, llm: llm)
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()
        await coord.resolveLatest()
        await coord.resolveLatest()
        await coord.interpretation?.value                // detached fallback; await it here
        #expect(llm.calls == 1)                          // same frame → interpreted exactly once
    }

    @MainActor
    @Test func liveAutoInterpretDoesNotBlockTheTick() async throws {
        // The on-device call takes ~1s. Awaiting it inside the tick froze the HUD for that
        // long; it must run off the critical path instead. The stub records whether the tick
        // had already returned by the time it was invoked — inline, it could not have.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "FADY TOPP", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)],
        ])
        let llm = OrderRecordingLLM(guess: "Heady Topper")
        let coord = ScanCoordinator(engine: engine, api: CatalogStubAPI(known: ["Heady Topper"]),
                                    llm: llm)
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)

        await coord.resolveLatest()
        llm.tickReturned = true          // set before the detached task can be scheduled
        await coord.interpretation?.value

        #expect(llm.calls == 1)
        #expect(llm.sawTickReturned == true)   // false ⇒ it ran inside the tick again
    }

    @MainActor
    @Test func interpretedOverlaySurvivesTheNextEmptyTick() async throws {
        // The reported bug: once the on-device model named the beer, the box vanished before it
        // could be tapped. `interpret` sets the overlay but leaves `lastResolvedKey` on the
        // garbled frame, so the very next tick re-resolved the raw OCR, matched nothing, and
        // blanked the HUD — one tick of visibility, ~350ms.
        let engine = PushEngine()
        let coord = ScanCoordinator(engine: engine, api: CatalogStubAPI(known: ["Heady Topper"]),
                                    llm: StubLLM(guess: "Heady Topper"))
        coord.start()

        engine.push([DetectedText(text: "FADY TOPP", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 50_000_000)
        await coord.resolveLatest()
        await coord.interpretation?.value
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")

        // Same can, jittered garble — still matches nothing. The earned result must stand.
        engine.push([DetectedText(text: "FADY T0PP", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 50_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")

        // ...but pointing away from the shelf still clears it at once, so nothing goes stale.
        engine.push([])
        try await Task.sleep(nanoseconds: 50_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.isEmpty)
    }

    @MainActor
    @Test func filterHidingEverythingIsNotUndoneByTheHold() async throws {
        // The hold keys off what the catalog returned, not what survives the filter — otherwise
        // a filter that legitimately hides every candidate would look like an empty resolve and
        // the hidden overlays would be held on screen.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "SHELF", kind: "text", x: 0.2, y: 0.3, w: 0.4, h: 0.1)],
        ])
        let coord = ScanCoordinator(engine: engine, api: TwoCandidateAPI(), llm: MockLLMProvider())
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 2)
        await coord.setFilter("nothing over 3%")   // excludes the 4.2% lager and the 8.5% DIPA
        #expect(coord.overlays.isEmpty)
    }

    @MainActor
    @Test func liveFilterHidesOutOfSpecOverlaysAndRestores() async throws {
        // The persistent chat-bar filter is parsed once and applied to each tick's candidates:
        // "nothing over 6%" hides the 8.5% DIPA and keeps the 4.2% lager; clearing restores both.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "SHELF", kind: "text", x: 0.2, y: 0.3, w: 0.4, h: 0.1)],
        ])
        let api = TwoCandidateAPI()
        let coord = ScanCoordinator(engine: engine, api: api, llm: MockLLMProvider())
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 2)
        await coord.setFilter("nothing over 6%")
        #expect(coord.overlays.count == 1)
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Light Lager")
        #expect(coord.filterText == "nothing over 6%")
        await coord.clearFilter()
        #expect(coord.overlays.count == 2)
        #expect(coord.filterText == nil)
    }
}

// MARK: - test doubles

/// An engine whose frames the test pushes one at a time, so successive ticks can see
/// *different* OCR — the jitter a real camera produces, which `MockScanEngine` (it yields its
/// whole script at once) cannot express.
private final class PushEngine: ScanEngine, @unchecked Sendable {
    private var cont: AsyncStream<[DetectedText]>.Continuation?
    let frames: AsyncStream<[DetectedText]>
    init() {
        var c: AsyncStream<[DetectedText]>.Continuation!
        frames = AsyncStream { c = $0 }
        cont = c
    }
    func start() async {}
    func stop() { cont?.finish() }
    func push(_ frame: [DetectedText]) { cont?.yield(frame) }
}

private func makeCandidate(id: String, name: String, abv: Double,
                          personal: Double, index: Int = 0) -> ScoredCandidate {
    let prov = Provenance(sourceId: "t", url: nil, quote: nil,
                          method: .regulatoryFiling, confidence: 1)
    let product = Product(
        id: id, brandId: "b", producerId: "p", category: .beer, name: name,
        style: nil, spec: ProductSpec(abvPct: Sourced(value: abv, provenance: prov),
                                      ibu: nil, proof: nil, ageStatementYears: nil),
        recipe: RecipeGraph())
    let resolved = ResolvedProduct(
        product: product,
        producer: Producer(id: "p", name: "P", kind: nil, country: nil, region: nil,
                           city: nil, lat: nil, lon: nil, website: nil),
        brand: Brand(id: "b", producerId: "p", name: "B"))
    return ScoredCandidate(detectionIndex: index, resolved: resolved, matchScore: 1,
                           personalScore: personal, reason: nil, coldStart: true)
}

private final class CountingSink: APIClientProtocol, @unchecked Sendable {
    var batches = 0
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        ScanResolveResponse(candidates: [], unresolvedIndices: [], latencyMs: nil)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws { batches += 1 }
}

private final class StubAPI: APIClientProtocol, @unchecked Sendable {
    var resolveCallCount = 0
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        resolveCallCount += 1
        let cands = req.detections.map { d in
            makeCandidate(id: d.text, name: d.text, abv: 8.0, personal: 0.8)
        }
        return ScanResolveResponse(candidates: cands, unresolvedIndices: [], latencyMs: 0.5)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

/// Resolves a detection only when its text is a known catalog name — so garbled OCR misses,
/// the way the real trigram store does. Lets the LLM-fallback path be exercised deterministically.
private final class CatalogStubAPI: APIClientProtocol, @unchecked Sendable {
    let known: Set<String>
    var resolveCallCount = 0
    init(known: Set<String>) { self.known = known }
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        resolveCallCount += 1
        var candidates: [ScoredCandidate] = []
        var unresolved: [Int] = []
        for (i, d) in req.detections.enumerated() {
            if known.contains(d.text) {
                candidates.append(makeCandidate(id: d.text, name: d.text, abv: 8, personal: 0.8, index: i))
            } else {
                unresolved.append(i)
            }
        }
        return ScanResolveResponse(candidates: candidates, unresolvedIndices: unresolved, latencyMs: 0.5)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

/// Returns two candidates of different ABV for any single detection — an 8.5% DIPA and a 4.2%
/// lager pinned to the same box — so the persistent chat-bar filter can be exercised.
private final class TwoCandidateAPI: APIClientProtocol, @unchecked Sendable {
    var resolveCallCount = 0
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        resolveCallCount += 1
        return ScanResolveResponse(candidates: [
            makeCandidate(id: "dipa", name: "Big DIPA", abv: 8.5, personal: 0.9, index: 0),
            makeCandidate(id: "lager", name: "Light Lager", abv: 4.2, personal: 0.4, index: 0),
        ], unresolvedIndices: [], latencyMs: 0.5)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

/// LLM double: returns a fixed product-name guess and counts how many times it was asked.
/// Records whether the live tick had already returned when the fallback fired. Both actors
/// are MainActor-isolated, so the ordering is deterministic rather than timing-dependent.
private final class OrderRecordingLLM: LLMProvider, @unchecked Sendable {
    let guess: String
    var calls = 0
    var tickReturned = false
    var sawTickReturned: Bool?
    init(guess: String) { self.guess = guess }
    func parseQuery(_ text: String) async throws -> QueryIntent { QueryIntent(freeText: text) }
    func rerank(_ candidates: [ScoredCandidate], for ask: String) async throws -> [String] {
        candidates.map { $0.resolved.product.id }
    }
    func interpretLabels(_ ocrLines: [String]) async throws -> [String] {
        calls += 1
        sawTickReturned = tickReturned
        return [guess]
    }
}

private final class StubLLM: LLMProvider, @unchecked Sendable {
    let guess: String
    var calls = 0
    init(guess: String) { self.guess = guess }
    func parseQuery(_ text: String) async throws -> QueryIntent { QueryIntent(freeText: text) }
    func rerank(_ candidates: [ScoredCandidate], for ask: String) async throws -> [String] {
        candidates.map { $0.resolved.product.id }
    }
    func interpretLabels(_ ocrLines: [String]) async throws -> [String] {
        calls += 1
        return [guess]
    }
}
