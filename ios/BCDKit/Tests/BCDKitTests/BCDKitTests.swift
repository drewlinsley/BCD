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
        #expect(ExtractionMethod.llmRecalled.trustRank <
                ExtractionMethod.communityClone.trustRank)
    }

    /// A server that has learned a new provenance method or sensory source must not blank
    /// the HUD on a phone that has not: the unknown value decodes as the weakest known one.
    @Test func unknownEnumValuesDecodeAsTheWeakest() throws {
        let method = try JSONDecoder().decode(ExtractionMethod.self,
                                              from: "\"psychic_reading\"".data(using: .utf8)!)
        #expect(method == .llmInferredFromStylePrior)
        let known = try JSONDecoder().decode(ExtractionMethod.self,
                                             from: "\"llm_recalled\"".data(using: .utf8)!)
        #expect(known == .llmRecalled)
        let source = try JSONDecoder().decode(SensorySource.self,
                                              from: "\"tea_leaves\"".data(using: .utf8)!)
        #expect(source == .stylePrior)
        let profile = try JSONDecoder().decode(SensorySource.self,
                                               from: "\"llm_profile\"".data(using: .utf8)!)
        #expect(profile == .llmProfile)
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

    @Test func sayingYesLaterTakesEffectOnTheNextEvent() async {
        // The queue is built at launch, when personalization is off. Holding that answer as
        // a constant, it went on refusing the tier for the rest of the run -- so the first
        // rating a user ever gave, the one that turns the loop on, was dropped on its way
        // out of the app and only a relaunch fixed it (2026-09-24).
        let q = TelemetryQueue(consent: ConsentState(analytics: true, personalization: false))
        #expect(!(await q.log("rating_submitted", tier: .personalization)))
        await q.setConsent(ConsentState(analytics: true, personalization: true))
        #expect(await q.log("rating_submitted", tier: .personalization))
        #expect(await q.pendingCount == 1)
    }

    @Test func takingItBackStopsTheNextEvent() async {
        // And it reads both ways, or "turn this off any time under You" would be a lie.
        let q = TelemetryQueue(consent: ConsentState(analytics: true, personalization: true))
        #expect(await q.log("rating_submitted", tier: .personalization))
        await q.setConsent(ConsentState(analytics: true, personalization: false))
        #expect(!(await q.log("rating_submitted", tier: .personalization)))
        #expect(await q.pendingCount == 1)   // the one already accepted keeps its place
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
    @Test func aDeadServerIsSaidAfterThreeTicksAndUnsaidOnTheFirstAnswer() async throws {
        // The API lives on a laptop. With the lid closed the phone scanned for a minute and the
        // HUD stayed blank -- indistinguishable from a can it could not read. One failed tick
        // is a dropped packet; three is a server that is not there, and the HUD must say so.
        let engine = PushEngine()
        let api = DeadThenAliveAPI()
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()
        for i in 0..<3 {
            engine.push([DetectedText(text: "FRAME \(i)", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
            try await Task.sleep(nanoseconds: 60_000_000)
            #expect(!coord.isServerUnreachable)
            await coord.resolveLatest()
        }
        #expect(coord.isServerUnreachable)
        api.alive = true
        engine.push([DetectedText(text: "FRAME 3", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(!coord.isServerUnreachable)
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

        // ...and pointing away no longer erases it on the spot. That behaviour was deliberate
        // once -- nothing should linger over a bare shelf -- but it made the hold worthless for
        // a result the frame had actually proven. A barcode lives in a single frame and is gone
        // the moment the can tilts, and lowering the phone to *tap* the answer empties the frame
        // too, so clearing on empty cleared exactly when someone was reaching for it. A proven
        // answer keeps its window; an unproven one still goes at once.
        engine.push([])
        try await Task.sleep(nanoseconds: 50_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")
    }

    @MainActor
    @Test func anUnprovenOverlayStillGoesTheMomentTheViewEmpties() async throws {
        // The other half of the rule: only a corroborated answer earns the grace.
        let engine = ManualScanEngine()
        let coord = ScanCoordinator(engine: engine, api: UncorroboratedAPI(known: ["Heady Topper"]))
        coord.start()

        engine.push([DetectedText(text: "CHEMIST-VER", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Chemist")

        engine.push([])
        try await Task.sleep(nanoseconds: 60_000_000)
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

@Suite("SeenLog")
struct SeenLogTests {
    /// Its own defaults suite per test, so the queue under test is never the simulator's.
    private func fresh() -> SeenLog {
        let name = "bcd.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return SeenLog(defaults: defaults, limit: 3)
    }

    private func stub(_ id: String) -> SeenProduct {
        SeenProduct(id: id, name: "Beer \(id)", producer: "Brewery", abvPct: 5)
    }

    @Test func newestOpenedComesFirst() {
        let log = fresh()
        log.record(stub("a"))
        log.record(stub("b"))
        #expect(log.all().map(\.id) == ["b", "a"])
    }

    @Test func reopeningMovesToTopWithoutDuplicating() {
        // The queue is a worklist: looking at something again should surface it, not add a
        // second copy you would then have to rate twice.
        let log = fresh()
        log.record(stub("a"))
        log.record(stub("b"))
        log.record(stub("a"))
        #expect(log.all().map(\.id) == ["a", "b"])
    }

    @Test func oldestFallsOffTheEnd() {
        let log = fresh()   // limit 3
        for id in ["a", "b", "c", "d"] { log.record(stub(id)) }
        #expect(log.all().map(\.id) == ["d", "c", "b"])
    }

    @Test func dismissingDropsOnlyThatEntry() {
        let log = fresh()
        log.record(stub("a"))
        log.record(stub("b"))
        log.remove("a")
        #expect(log.all().map(\.id) == ["b"])
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
        // A hit here is an exact match on a name the catalog knows, which is the case the
        // server corroborates. The stub predated the flag.
        return ScanResolveResponse(candidates: candidates, unresolvedIndices: unresolved,
                                   latencyMs: 0.5, corroborated: !candidates.isEmpty)
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
        ], unresolvedIndices: [], latencyMs: 0.5, corroborated: true)
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

// MARK: - taste copy

/// The vectors here are the exact ones in the gold table, so a change to the copy rules
/// shows up against real products rather than convenient ones.
@Suite struct TasteSummaryTests {
    private let headyTopper = SensoryVector(source: .stylePrior, confidence: 0.25, axes: [
        "grassy": 0.25, "bitterness": 0.35, "carbonation": 0.55,
        "malty_bready": 0.45, "body_fullness": 0.4, "dryness_finish": 0.4,
    ])
    private let ouzo = SensoryVector(source: .stylePrior, confidence: 0.35, axes: [
        "sweet": 0.4, "herbal": 0.75, "alcohol_warmth": 0.65,
        "dryness_finish": 0.4, "spicy_phenolic": 0.6,
    ])

    @Test func namesOnlyNotesThatClearTheFloor() {
        // grassy sits at 0.25 — real in the vector, too faint to claim in a sentence.
        #expect(TasteSummary.notes(headyTopper) == "Bready malt.")
    }

    @Test func ordersNotesByStrength() {
        #expect(TasteSummary.notes(ouzo) == "Herbal, peppery spice and sweetness.")
    }

    @Test func capsTheNoteListAtThree() {
        let busy = SensoryVector(source: .reconciled, confidence: 0.9, axes: [
            "citrus": 0.9, "tropical": 0.85, "honey": 0.8, "floral": 0.75, "berry": 0.7,
        ])
        #expect(TasteSummary.notes(busy) == "Citrus, tropical fruit and honey.")
    }

    @Test func structureBecomesItsOwnSentence() {
        #expect(TasteSummary.structure(headyTopper) == "Medium-bodied and mildly bitter.")
        #expect(TasteSummary.structure(ouzo) == "Warming.")
    }

    @Test func finishGetsTheLastClause() {
        let dry = SensoryVector(source: .stylePrior, confidence: 0.4,
                                axes: ["body_fullness": 0.7, "dryness_finish": 0.8])
        #expect(TasteSummary.structure(dry) == "Full-bodied, with a dry finish.")
    }

    @Test func joinsNotesAndStructure() {
        #expect(TasteSummary.sentence(for: ouzo)
                == "Herbal, peppery spice and sweetness. Warming.")
    }

    @Test func withholdsTheCategoryFallbackTier() {
        // Heady Topper's own vector sits at 0.25 — no style keyword matched its name, so
        // enrich handed it the generic "beer" centroid. The parts still assemble, but the
        // sentence they assemble into describes every beer in the catalog, so the screen
        // must show nothing rather than tell someone a double IPA is mildly bitter.
        #expect(headyTopper.confidence == 0.25)
        #expect(TasteSummary.notes(headyTopper) == "Bready malt.")
        #expect(TasteSummary.sentence(for: headyTopper) == nil)
    }

    @Test func keepsTheNamedStyleTier() {
        // 0.35 is a style that actually matched; that tier is the bulk of the catalog and
        // has to survive the gate.
        #expect(ouzo.confidence == 0.35)
        #expect(TasteSummary.sentence(for: ouzo) != nil)
    }

    @Test func saysNothingRatherThanGuessing() {
        // No axes at all: the screen must drop the section, not print an empty flourish.
        #expect(TasteSummary.sentence(for: SensoryVector(source: .stylePrior)) == nil)
    }

    @Test func ignoresAxesThisBuildHasNeverHeardOf() {
        // The server is allowed to append axes ahead of the app; an unknown one is dropped,
        // not fatal, and must not take a slot from a note we can actually name.
        let future = SensoryVector(source: .reconciled, confidence: 1.0,
                                   axes: ["umami_seaweed": 0.99, "citrus": 0.8])
        #expect(future.ranked.count == 1)
        #expect(TasteSummary.notes(future) == "Citrus.")
    }
}

// MARK: - names fit to read

@Suite struct DisplayNameTests {
    @Test func stripsTheLegalWrapper() {
        #expect(DisplayName.producer("The Alchemist LLC") == "The Alchemist")
        #expect(DisplayName.producer("Lidl US LLC") == "Lidl US")
    }

    @Test func keepsTradeWordsThatAreActuallyTheName() {
        // "Brewing" is part of what the business is called; "Co" is paperwork.
        #expect(DisplayName.producer("Sierra Nevada Brewing Co") == "Sierra Nevada Brewing")
    }

    @Test func neverStripsANameToNothing() {
        #expect(DisplayName.producer("Co") == "Co")
    }

    @Test func dropsTheBrandRepeatedInTheProductName() {
        #expect(DisplayName.product("The Alchemist Heady Topper",
                                    producer: "The Alchemist LLC") == "Heady Topper")
    }

    @Test func aLabelPrintsTheBrandTheNameLeftOut() {
        // Open Food Facts files the brand apart from the name: a bottle of Tito's came up
        // as "Handmade Vodka" (2026-09-17).
        #expect(DisplayName.label("Handmade Vodka", brand: "Tito’s") == "Tito’s Handmade Vodka")
        #expect(DisplayName.label("Tito's Vodka", brand: "Tito’s") == "Tito's Vodka")
        #expect(DisplayName.label("Titos Vodka", brand: "Tito's") == "Titos Vodka")
        #expect(DisplayName.label("Bitter Campari", brand: "Campari") == "Bitter Campari")
        #expect(DisplayName.label("The Alchemist Heady Topper", brand: "The Alchemist") == "The Alchemist Heady Topper")
        #expect(DisplayName.label("Heady Topper", brand: "Heady Topper") == "Heady Topper")
        #expect(DisplayName.label("Heady Topper", brand: "unknown") == "Heady Topper")
        #expect(DisplayName.label("Heady Topper", brand: "") == "Heady Topper")
        #expect(DisplayName.label("East Vapour Infused London Dry Gin", brand: "Bombay Sapphire")
                == "Bombay Sapphire East Vapour Infused London Dry Gin")
    }

    @Test func keepsTheBrandWhenAllThatIsLeftIsACategory() {
        // "Ouzo" and "Vodka" under a brand line identify nothing — the repetition is worth
        // less than the loss.
        #expect(DisplayName.product("Plomari Ouzo", producer: "Plomari") == "Plomari Ouzo")
        #expect(DisplayName.product("Titos Vodka", producer: "Titos") == "Titos Vodka")
    }

    @Test func leavesAnUnrelatedNameAlone() {
        #expect(DisplayName.product("Heady Topper",
                                    producer: "The Alchemist LLC") == "Heady Topper")
        #expect(DisplayName.product("Pliny the Elder",
                                    producer: "Russian River Brewing Co") == "Pliny the Elder")
    }

    @Test func keepsTheBrandWhenAllThatIsLeftIsAStyle() {
        // "London Dry Gin" is what the bottle is, not what it is called. Reported from the
        // camera as the detail screen showing the wrong name (2026-09-16).
        #expect(DisplayName.product("Bombay Sapphire London Dry Gin",
                                    producer: "Bombay Sapphire") == "Bombay Sapphire London Dry Gin")
        #expect(DisplayName.product("Guinness Extra Stout", producer: "Guinness") == "Guinness Extra Stout")
        // ...while a name of its own still sheds the brand.
        #expect(DisplayName.product("Goslings Black Seal", producer: "Goslings") == "Black Seal")
    }
}

@Suite struct FramePrioritisation {
    private func line(_ text: String, area: Double, confidence: Double = 0.9) -> DetectedText {
        // square box of the requested area, so ordering is by size alone
        let side = area.squareRoot()
        return DetectedText(text: text, kind: "text", x: 0.5, y: 0.5, w: side, h: side,
                            confidence: confidence)
    }

    @MainActor
    @Test func sendsOnlyTheLargestTextLines() {
        // A Heady Topper can: the brand and the beer are the big print, the rest is chrome.
        let frame = [
            line("DRINK FROM THE CAN", area: 0.01),
            line("HEADY TOPPER", area: 0.20),
            line("STOWE VERMONT", area: 0.02),
            line("THE ALCHEMIST", area: 0.10),
            line("AMERICAN DOUBLE IPA", area: 0.03),
            line("PINT", area: 0.005),
        ]
        let sent = ScanCoordinator.prioritised(frame).map(\.text)
        #expect(sent == ["HEADY TOPPER", "THE ALCHEMIST", "AMERICAN DOUBLE IPA"])
    }

    @MainActor
    @Test func keepsEveryBarcodeRegardlessOfSize() {
        // A barcode is a definitive answer and costs a keyed lookup, not a trigram scan, so it
        // must never be dropped for being small — it is usually the smallest thing on a can.
        var frame = (1...5).map { line("LINE \($0)", area: Double($0) / 10.0) }
        frame.append(DetectedText(text: "854416001019", kind: "barcode", symbology: "ean13",
                                  x: 0.5, y: 0.9, w: 0.01, h: 0.01, confidence: 1.0))
        let sent = ScanCoordinator.prioritised(frame)
        #expect(sent.filter { $0.kind == "barcode" }.count == 1)
        // ...and once there is one, the text goes: an exact identifier cannot be improved on,
        // and scanning the label beside it only costs time and offers wrong answers.
        #expect(sent.filter { $0.kind == "text" }.isEmpty)
    }

    @MainActor
    @Test func ordersDeterministicallyWhenNoBoxesAreReported() {
        // A detector that reports no box gives every line an area of zero; the frame must still
        // send the same three lines every tick rather than whatever order OCR happened to emit.
        let frame = ["ZEBRA", "APPLE", "MANGO", "CHERRY"].map {
            DetectedText(text: $0, kind: "text", confidence: 0.5)
        }
        let once = ScanCoordinator.prioritised(frame).map(\.text)
        let twice = ScanCoordinator.prioritised(frame.reversed()).map(\.text)
        #expect(once == twice)
        #expect(once.count == 3)
    }
}

/// Answers *something* for any frame — a confident-looking guess off a single fragment, marked
/// uncorroborated the way the server marks it. Reproduces the real failure: a Heady Topper can
/// whose garbled rim print matched a distillery named `Chemist` at a plausible score.
/// An API that is not there, until it is.
private final class DeadThenAliveAPI: APIClientProtocol, @unchecked Sendable {
    var alive = false
    var resolveCallCount = 0
    struct Unreachable: Error {}
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        resolveCallCount += 1
        guard alive else { throw Unreachable() }
        return ScanResolveResponse(candidates: [], unresolvedIndices: [], latencyMs: 1)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

private final class UncorroboratedAPI: APIClientProtocol, @unchecked Sendable {
    let known: Set<String>
    var resolveCallCount = 0
    init(known: Set<String>) { self.known = known }
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        resolveCallCount += 1
        if let hit = req.detections.first(where: { known.contains($0.text) }) {
            return ScanResolveResponse(
                candidates: [makeCandidate(id: hit.text, name: hit.text, abv: 8, personal: 0.8)],
                unresolvedIndices: [], latencyMs: 0.5, corroborated: true)
        }
        return ScanResolveResponse(
            candidates: [makeCandidate(id: "chemist", name: "Chemist", abv: 40, personal: 0.3)],
            unresolvedIndices: [], latencyMs: 0.5, corroborated: false)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

@Suite struct UncorroboratedFallback {
    @MainActor
    @Test func aConfidentLookingGuessDoesNotSuppressTheModel() async throws {
        // The bug, exactly: the catalog answered *something*, so `candidates.isEmpty` was false
        // and the fallback never ran — for eleven frames of a can it could not read.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "FADY TOPPE", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)],
        ])
        let api = UncorroboratedAPI(known: ["Heady Topper"])
        let llm = StubLLM(guess: "Heady Topper")
        let coord = ScanCoordinator(engine: engine, api: api, llm: llm)
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()
        await coord.interpretation?.value
        #expect(llm.calls == 1)                          // it asked, despite having an answer
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")
    }

    @MainActor
    @Test func realAgreementAcrossTheFrameLeavesTheModelAlone() async throws {
        // The other half: the model costs ~1s and must not run whenever the catalog is merely
        // unsure. Corroborated means the label named the same thing twice — that is an answer.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "Heady Topper", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)],
        ])
        let api = UncorroboratedAPI(known: ["Heady Topper"])
        let llm = StubLLM(guess: "Something Else")
        let coord = ScanCoordinator(engine: engine, api: api, llm: llm)
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()
        await coord.interpretation?.value
        #expect(llm.calls == 0)
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")
    }
}

/// A model that blocks until released, so a call can be held in flight across several ticks.
private final class SlowLLM: LLMProvider, @unchecked Sendable {
    let guess: String
    var calls = 0
    private let gate = AsyncStream<Void>.makeStream()
    init(guess: String) { self.guess = guess }
    func release() { gate.continuation.yield(); gate.continuation.finish() }
    func parseQuery(_ text: String) async throws -> QueryIntent { QueryIntent(freeText: text) }
    func rerank(_ candidates: [ScoredCandidate], for ask: String) async throws -> [String] {
        candidates.map { $0.resolved.product.id }
    }
    func interpretLabels(_ ocrLines: [String]) async throws -> [String] {
        calls += 1
        for await _ in gate.stream { break }          // wait for release()
        return [guess]
    }
}

/// A scan engine the test drives one frame at a time, so a model call can be held in flight
/// across several ticks the way the live ticker does. `MockScanEngine` yields its whole script
/// at start and finishes, which cannot express "the OCR changed while we were thinking".
private final class ManualScanEngine: ScanEngine, @unchecked Sendable {
    private var continuation: AsyncStream<[DetectedText]>.Continuation?
    let frames: AsyncStream<[DetectedText]>
    init() {
        var cont: AsyncStream<[DetectedText]>.Continuation!
        self.frames = AsyncStream { cont = $0 }
        self.continuation = cont
    }
    func start() async {}
    func stop() { continuation?.finish() }
    func push(_ frame: [DetectedText]) { continuation?.yield(frame) }
}

/// An engine whose stream is finished by `stop()` and reopened by `start()`, the way the
/// camera engines' are (see `VisionKitScanEngine.frames`).
private final class RestartableEngine: ScanEngine, @unchecked Sendable {
    private var continuation: AsyncStream<[DetectedText]>.Continuation?
    private var stream: AsyncStream<[DetectedText]>
    private let lock = NSLock()
    var starts = 0
    var frames: AsyncStream<[DetectedText]> { lock.withLock { stream } }
    init() {
        var cont: AsyncStream<[DetectedText]>.Continuation!
        stream = AsyncStream { cont = $0 }
        continuation = cont
    }
    func start() async {
        lock.withLock {
            starts += 1
            if continuation == nil {
                var cont: AsyncStream<[DetectedText]>.Continuation!
                stream = AsyncStream { cont = $0 }
                continuation = cont
            }
        }
    }
    func stop() { lock.withLock { continuation?.finish(); continuation = nil } }
    func push(_ frame: [DetectedText]) { lock.withLock { continuation }?.yield(frame) }
}

@Suite struct TheScanComesBack {
    /// The Scan tab left and returned: `stop()` finished the engine's stream and the next
    /// `start()` read the finished one, so the camera ran and nothing reached the HUD until
    /// the app was relaunched. Reported as "it wasn't doing anything" (2026-09-17).
    @MainActor
    @Test func aStoppedScanStartedAgainSeesFrames() async throws {
        let engine = RestartableEngine()
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        let can = [DetectedText(text: "Heady Topper", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)]
        coord.start()
        engine.push(can)
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 1)
        coord.stop()
        #expect(coord.overlays.isEmpty)

        coord.start()
        try await Task.sleep(nanoseconds: 30_000_000)
        engine.push(can)
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 1, "the second run's frames reach the HUD")
        coord.stop()
    }

    /// Back from the background, a scanning coordinator wakes the engine and keeps its
    /// stream; a stopped one starts over.
    @MainActor
    @Test func resumingWakesTheEngineWithoutLosingTheStream() async throws {
        let engine = RestartableEngine()
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        let can = [DetectedText(text: "Heady Topper", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)]
        coord.startLive(intervalMs: 10_000)
        try await Task.sleep(nanoseconds: 30_000_000)
        coord.resume(intervalMs: 10_000)
        try await Task.sleep(nanoseconds: 30_000_000)
        #expect(engine.starts == 2, "the engine is started again")
        engine.push(can)
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 1, "and the frames still arrive")
        coord.stop()
        coord.resume(intervalMs: 10_000)
        try await Task.sleep(nanoseconds: 30_000_000)
        #expect(coord.isScanning, "a stopped coordinator resumes by starting over")
        coord.stop()
    }
}

@Suite struct FallbackSurvivesTheTick {
    @MainActor
    @Test func aChangingFrameDoesNotKillTheModelMidThought() async throws {
        // The regression this exists for. The fallback used to be cancelled and restarted on
        // every new uncorroborated frame; at a 350ms tick against a ~1s call that meant it was
        // killed by the next tick every time. On a real Focal Banger can it completed once in
        // 19 frames, because garbled OCR is never identical three ticks running.
        let engine = ManualScanEngine()
        let api = UncorroboratedAPI(known: ["Focal Banger"])
        let llm = SlowLLM(guess: "Focal Banger")
        let coord = ScanCoordinator(engine: engine, api: api, llm: llm)
        coord.start()
        // Real garble off the can, not a placeholder: the model's answer is now checked
        // against the frame that produced it, so a stub the guess cannot be grounded in would
        // be rejected before this test got to the thing it is about.
        engine.push([DetectedText(text: "THE CAN! DRINKF FOCAL BAN", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()                    // tick 1 — starts the model
        let started = coord.interpretation
        engine.push([DetectedText(text: "HAN! DRINK FRO FOCALB 3I", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 50_000_000)
        await coord.resolveLatest()                    // tick 2 — a different garble
        #expect(started?.isCancelled == false)         // the first call is still thinking
        #expect(llm.calls == 1)                        // and no second call piled on
        llm.release()
        await coord.interpretation?.value
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Focal Banger")
    }

    @MainActor
    @Test func aGuessIsDroppedOnceTheCatalogRecognisesSomethingItself() async throws {
        // The other direction: if the catalog corroborated a real answer while the model was
        // thinking, the model's guess is stale and must not overwrite it.
        let engine = ManualScanEngine()
        let api = UncorroboratedAPI(known: ["Heady Topper"])
        let llm = SlowLLM(guess: "Heady Topper")
        let coord = ScanCoordinator(engine: engine, api: api, llm: llm)
        coord.start()
        engine.push([DetectedText(text: "DRINK FRO", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()                    // weak frame — model starts
        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 50_000_000)
        await coord.resolveLatest()                    // the catalog now recognises it outright
        llm.release()
        await coord.interpretation?.value
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")
        #expect(coord.isInterpreting == false)         // flag comes back down on the dropped path
    }
}

@Suite struct CorroboratedOverlayHoldsTheScreen {
    @MainActor
    @Test func anUncorroboratedTickDoesNotEvictACorroboratedOne() async throws {
        // Reported from the camera: "the right answer popped up for a second but was behind a
        // bunch of other incorrect things". At a 350ms tick a garbled frame lands between every
        // good pair, and every tick with any candidate at all replaced the overlays outright —
        // so a correct answer held the screen for one tick and was overwritten by the next
        // fragment's guess. Measured server-side over 78 uncorroborated frames off a real can,
        // the answer was wrong on 77.
        let engine = ManualScanEngine()
        let api = UncorroboratedAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()

        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")

        // the next tick reads a fragment and the catalog offers "Chemist", uncorroborated
        engine.push([DetectedText(text: "CHEMIST-VER", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()

        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper",
                "the earned answer stays up; the guess does not take the screen from it")
    }

    @MainActor
    @Test func anUnprovenGuessIsNotShownWhileTheModelCanStillAnswer() async throws {
        // Capping unproven frames to one candidate was not enough: the one guess still took
        // the screen, and a different wrong one took it 350ms later. Reported from the camera
        // as "seven or eight different answers". Across two live sessions off a real can, 120
        // unproven frames returned a candidate and none was the product in front of it.
        let engine = ManualScanEngine()
        let api = UncorroboratedAPI(known: ["Heady Topper"])
        let llm = StubLLM(guess: "Heady Topper")
        let coord = ScanCoordinator(engine: engine, api: api, llm: llm)
        coord.start()

        engine.push([DetectedText(text: "FADY TOPPE", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.isEmpty,
                "an unproven guess does not take the screen while the model is still reading")

        // and the model's answer, which is the one that reads a stylized can, does show
        await coord.interpretation?.value
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")
    }

    @MainActor
    @Test func theModelsGuessAlsoHasToBeCorroboratedToShow() async throws {
        // The hole the live-path rule left open: this path wrote to the screen without the
        // check. The model read a Heady Topper can as "Alchemist Vermont Ale", the catalog
        // matched a product literally called `Vermont` at a plausible score, and it went up
        // with full confidence -- reported from the camera as "I got VERMONT and BRINK".
        let engine = ManualScanEngine()
        let api = UncorroboratedAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api,
                                    llm: StubLLM(guess: "Alchemist Vermont Ale"))
        coord.start()

        engine.push([DetectedText(text: "CHEMIST-VERMONT", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        await coord.interpretation?.value

        #expect(coord.overlays.isEmpty,
                "the model named it, but the catalog did not corroborate the name it gave")
    }

    @MainActor
    @Test func anUnprovenGuessDoesNotEraseTheAnswerAlreadyEarned() async throws {
        // Withholding it must not clear what is already up: the frame after a good one is
        // usually garbled, and blanking on it would flicker the earned answer away.
        let engine = ManualScanEngine()
        let api = UncorroboratedAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api, llm: StubLLM(guess: "Heady Topper"))
        coord.start()

        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")

        engine.push([DetectedText(text: "CHEMIST-VER", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Heady Topper")
    }

    @MainActor
    @Test func anUncorroboratedTickStillShowsWhenNothingBetterIsUp() async throws {
        // The rule must not become a refusal to ever answer: with no model configured, an
        // uncorroborated guess is the best there is, so it still shows.
        let engine = ManualScanEngine()
        let api = UncorroboratedAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()

        engine.push([DetectedText(text: "CHEMIST-VER", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()

        #expect(coord.overlays.first?.candidate.resolved.product.name == "Chemist")
    }
}

@Suite struct ModelGuessIsCheckedAgainstTheFrame {
    /// Real OCR off a Focal Banger can, from the scan log.
    let focal = ["HECAN! DRINK FRO TALB", "THE CAN! DRINKF FOCAL BAN", "IL CHEMIST VERNA"]
    /// Real OCR off a Heady Topper can, from the same log.
    let heady = ["FADY TOPPE", "CHEMIST-VER", "CAN! DRINK FROM THE"]

    @Test func aParrotedPromptExampleIsRejected() {
        // Both of these are examples out of the app's own prompt, and both were displayed over
        // a can of Focal Banger. A model asked to name a label it cannot read hands back the
        // example it was shown; the guess then reached the catalog as the only line in its own
        // frame and matched itself at 1.00, so nothing downstream could tell.
        #expect(!ScanCoordinator.frameSupports(guess: "Bombay Sapphire", ocr: focal))
        #expect(!ScanCoordinator.frameSupports(guess: "Sierra Nevada Pale Ale", ocr: focal))
    }

    @Test func anInventedAnswerIsRejected() {
        #expect(!ScanCoordinator.frameSupports(guess: "Heineken", ocr: focal))
        #expect(!ScanCoordinator.frameSupports(guess: "Matcha Omoi", ocr: heady))
    }

    @Test func theRightAnswerOffAGarbledCanSurvives() {
        // The whole point: the model is here to read what OCR cannot, so a guess whose words
        // the camera never spelled correctly must still pass. "FADY TOPPE" is Heady Topper.
        #expect(ScanCoordinator.frameSupports(guess: "Heady Topper", ocr: heady))
        #expect(ScanCoordinator.frameSupports(guess: "The Alchemist Heady Topper", ocr: heady))
        #expect(ScanCoordinator.frameSupports(guess: "Focal Banger", ocr: focal))
    }

    @Test func theOtherBeerFromTheSameBreweryIsRejected() {
        // The failure that started this: pointed at Focal Banger, the model answered Heady
        // Topper. Both cans print ALCHEMIST, so agreeing on that one word is not evidence --
        // which is why two of the name's own words have to be in the frame, not just any one.
        #expect(!ScanCoordinator.frameSupports(guess: "The Alchemist Heady Topper", ocr: focal))
    }
}

@Suite struct ModelEchoIsNotAnAnswer {
    @Test func handingTheFragmentsBackIsNotANameAndIsRejected() {
        // Straight from the scan log. An echo defeats every check that asks whether the frame
        // supports the answer, because it *is* the frame -- so it has to be caught on shape.
        let echo = "ECAN! DRINKER | CAN! DRINK FROM FICALSE DIN | THE ALCHEMIST | THE ALEH ASTAVER"
        #expect(!ScanCoordinator.looksLikeAName(echo))
        #expect(!ScanCoordinator.frameSupports(guess: echo, ocr: ["ECAN! DRINKER",
                                                                 "THE ALCHEMIST"]))
        #expect(!ScanCoordinator.looksLikeAName("ECAN! DE"))
    }

    @Test func realNamesAreStillNames() {
        #expect(ScanCoordinator.looksLikeAName("Heady Topper"))
        #expect(ScanCoordinator.looksLikeAName("The Alchemist Heady Topper"))
        #expect(ScanCoordinator.looksLikeAName("Bombay Sapphire London Dry Gin"))
    }
}

@Suite struct BarcodeAnswerSurvivesLongEnoughToTap {
    /// The exact frame from the scan log: one barcode, resolved at 1.00 in 22ms, corroborated
    /// -- and invisible on the phone, because the next tick had nothing in view.
    @MainActor
    @Test func aBarcodeReadInOneFrameIsStillOnScreenAfterItLeavesView() async throws {
        let engine = ManualScanEngine()
        let api = UncorroboratedAPI(known: ["0793573117267"])
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()

        engine.push([DetectedText(text: "0793573117267", kind: "barcode",
                                  x: 0.3, y: 0.6, w: 0.3, h: 0.08)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 1)

        // the can tilts and the code is gone -- which is every frame after the one that read it
        engine.push([])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 1, "the one exact answer the app can give stays tappable")
    }
}

@Suite struct AChipStaysWhereItAppeared {
    /// Reported from the camera as "the HUD text box jumps all over the place when it pops
    /// up" (2026-09-15): the overlay re-anchors to whichever line named the product this
    /// tick, and a label's lines are a few percent of the screen apart.
    @MainActor
    @Test func theSameProductNamedOffANearbyLineDoesNotMove() async throws {
        let engine = ManualScanEngine()
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()

        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        let first = coord.overlays.first
        #expect(first?.candidate.resolved.product.name == "Heady Topper")
        #expect(first?.x == 0.45 && first?.y == 0.35)

        // the next tick reads the same label a little lower, with the maker's line under it
        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.37, w: 0.5, h: 0.1),
                     DetectedText(text: "THE ALCHEMIST", kind: "text",
                                  x: 0.2, y: 0.5, w: 0.5, h: 0.06)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(api.resolveCallCount == 2, "a different frame, so the catalog was asked again")
        let second = coord.overlays.first
        #expect(second?.candidate.resolved.product.name == "Heady Topper")
        #expect(second?.x == 0.45 && second?.y == 0.35, "the chip stays where it first appeared")
    }

    @MainActor
    @Test func aChipFollowsTheBottleOnceTheCameraHasPanned() async throws {
        // The first cut moved the chip a third of the way toward its anchor on every frame
        // once the anchor left the dead zone -- thirty frames a second on a live shelf,
        // toward a target the tracker re-blends each frame. Reported as "the boxes are
        // still jumpy and kind of swimming across the screen" (2026-09-17). Now the chip
        // stays put until the anchor has been away for `HUDLayout.settle`, then moves once.
        let engine = ManualScanEngine()
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        var clock = Date(timeIntervalSince1970: 1_000)
        coord.now = { clock }
        coord.start()

        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.y == 0.35)

        // the can is now at the bottom of the screen, well past the dead zone
        let panned = [DetectedText(text: "Heady Topper", kind: "text",
                                   x: 0.2, y: 0.7, w: 0.5, h: 0.1),
                      DetectedText(text: "ALE", kind: "text", x: 0.2, y: 0.85, w: 0.2, h: 0.05)]
        engine.push(panned)
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.y == 0.35, "a moment away is jitter: the chip stays")

        clock = clock.addingTimeInterval(HUDLayout.settle + 0.1)
        engine.push(panned)
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.first?.y == 0.75, "away for good: the chip moves, once, to the bottle")
        #expect(coord.overlays.first?.isDisplaced == false)
    }
}

@Suite struct TheChipsKeepTheirDistance {
    private func chip(_ id: String, _ name: String, x: Double, y: Double,
                      box: BoundingBox? = nil) -> ResolvedOverlay {
        let c = makeCandidate(id: id, name: name, abv: 5, personal: 0.6)
        return ResolvedOverlay(id: id, candidate: c, x: x, y: y, box: box)
    }

    @Test func twoChipsOnOneSpotAreSpreadApartAndTiedToTheirBottles() {
        // Two bottles side by side, chips wanting the same place: the second drops below
        // the first, keeps its anchor on its own bottle, and the layout is a function of
        // the input alone.
        let a = chip("a", "Ramazzotti Aperitivo Rosato", x: 0.5, y: 0.3)
        let b = chip("b", "Campari", x: 0.52, y: 0.31)
        let laid = HUDLayout.spread([a, b])
        #expect(laid[0].y == 0.3)
        #expect(laid[1].y > laid[0].y + HUDLayout.chipHeight)
        let ra = HUDLayout.footprint(x: laid[0].x, y: laid[0].y, name: HUDLayout.title(of: a.candidate), hasReason: false)
        let rb = HUDLayout.footprint(x: laid[1].x, y: laid[1].y, name: HUDLayout.title(of: b.candidate), hasReason: false)
        #expect(ra.intersection(rb) == nil)
        #expect(laid[1].anchorX == 0.52 && laid[1].anchorY == 0.31 && laid[1].isDisplaced)
        #expect(HUDLayout.spread([a, b]).map(\.y) == laid.map(\.y), "deterministic")
        // Chips that already sit apart are left exactly where they are.
        let apart = HUDLayout.spread([a, chip("c", "Aperol", x: 0.5, y: 0.7)])
        #expect(apart.map(\.y) == [0.3, 0.7])
    }

    @Test func anObjectsChipPerchesAboveItsBoxAndTiesToItsTop() {
        let box = BoundingBox(x: 0.3, y: 0.4, w: 0.4, h: 0.3)
        let p = HUDLayout.perch(for: box, name: "Goslings Black Seal", hasReason: false)
        #expect(p.x == 0.5 && p.anchorX == 0.5 && p.anchorY == 0.4)
        #expect(p.y < box.minY && p.y + HUDLayout.chipHeight / 2 <= box.minY)
        // A box up against the top of the screen puts its chip below instead.
        let high = BoundingBox(x: 0.3, y: 0.02, w: 0.4, h: 0.1)
        let q = HUDLayout.perch(for: high, name: "Goslings Black Seal", hasReason: false)
        #expect(q.y > high.maxY && q.anchorY == high.maxY)
        // ...and one at the edge keeps the chip on screen.
        let edge = BoundingBox(x: 0.9, y: 0.5, w: 0.1, h: 0.2)
        let r = HUDLayout.perch(for: edge, name: "Goslings Black Seal", hasReason: false)
        #expect(r.x + HUDLayout.chipMaxWidth / 2 <= 1 && abs(r.anchorX - 0.95) < 1e-9)
    }

    @Test func aPinHoldsThroughJitterAndMovesOnceAfterSettling() {
        let t0 = Date(timeIntervalSince1970: 0)
        var pin = HUDLayout.steadied(nil, anchorX: 0.5, anchorY: 0.5, now: t0)
        #expect(pin.x == 0.5 && pin.y == 0.5)
        pin = HUDLayout.steadied(pin, anchorX: 0.55, anchorY: 0.53, now: t0.addingTimeInterval(0.1))
        #expect(pin.x == 0.5 && pin.y == 0.5 && pin.driftingSince == nil, "inside the dead zone")
        pin = HUDLayout.steadied(pin, anchorX: 0.8, anchorY: 0.5, now: t0.addingTimeInterval(0.2))
        #expect(pin.x == 0.5 && pin.driftingSince != nil, "out, but only just")
        pin = HUDLayout.steadied(pin, anchorX: 0.52, anchorY: 0.5, now: t0.addingTimeInterval(0.3))
        #expect(pin.x == 0.5 && pin.driftingSince == nil, "back inside: the clock resets")
        pin = HUDLayout.steadied(pin, anchorX: 0.8, anchorY: 0.5, now: t0.addingTimeInterval(0.4))
        pin = HUDLayout.steadied(pin, anchorX: 0.8, anchorY: 0.5, now: t0.addingTimeInterval(0.4 + HUDLayout.settle))
        #expect(pin.x == 0.8 && pin.driftingSince == nil, "away for `settle`: moved, in one step")
    }
}

@Suite struct TheHUDClearsWhenTheCameraMovesOn {
    /// The hold window keeps an earned answer up through garbled ticks so it can be tapped,
    /// and through empty frames so lowering the phone does not wipe it. It was also keeping
    /// it up for ten seconds over the *next* shelf -- reported as "it lingers too long"
    /// (2026-09-15). A frame that shares nothing with the scene the answer came from, twice
    /// running, is the camera having moved on.
    @MainActor
    @Test func aShelfThatSharesNoWordsWithTheAnswerClearsIt() async throws {
        let engine = ManualScanEngine()
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()

        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 1)

        // panned to a shelf of gin the catalog does not know; one such frame is not enough
        engine.push([DetectedText(text: "BOMBAY SAPPHIRE", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 1, "one strange frame is glare or a hand; the answer holds")

        engine.push([DetectedText(text: "BOMBAY SAPPHIRE", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1),
                     DetectedText(text: "LONDON DRY GIN", kind: "text",
                                  x: 0.2, y: 0.45, w: 0.5, h: 0.08)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.isEmpty, "two frames of another shelf, and the old answer is gone")
    }

    @MainActor
    @Test func theSameLabelReadWorseIsNotANewShelf() async throws {
        // A stylized can garbles differently every tick; while any word of it still reads
        // like the scene the answer came from, the answer stays.
        let engine = ManualScanEngine()
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()

        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        #expect(coord.overlays.count == 1)

        for garble in ["HEADY TOPPE", "FADY TOPPER", "HEAOY TOPPR"] {
            engine.push([DetectedText(text: garble, kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
            try await Task.sleep(nanoseconds: 60_000_000)
            await coord.resolveLatest()
            #expect(coord.overlays.count == 1, "\(garble) is the same can, read worse")
        }
    }

    @MainActor
    @Test func anEmptyFrameStillHoldsTheAnswer() async throws {
        // Lowering the phone to tap empties the frame; that is not a new shelf.
        let engine = ManualScanEngine()
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()

        engine.push([DetectedText(text: "Heady Topper", kind: "text",
                                  x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 60_000_000)
        await coord.resolveLatest()
        for _ in 0..<3 {
            engine.push([])
            try await Task.sleep(nanoseconds: 60_000_000)
            await coord.resolveLatest()
        }
        #expect(coord.overlays.count == 1)
    }
}

@Suite struct ABarcodeFrameIsResolvedOnTheBarcodeAlone {
    @MainActor
    @Test func theFinePrintBesideACodeIsNotSent() {
        // The real frame from the scan log. The warning paragraph cannot identify a product --
        // it is the same text that once matched Bacardi off "...drive A CAR OR..." -- and on
        // device it turned a 51ms answer into 1032-3078ms.
        let frame = [
            DetectedText(text: "0793573117267", kind: "barcode", x: 0.3, y: 0.6, w: 0.3, h: 0.08),
            DetectedText(text: "THIS CAN!\nSTETHE SURGEON\nMA OF THE RISK OF ACCIDENTS",
                         kind: "text", x: 0.1, y: 0.2, w: 0.8, h: 0.3),
        ]
        let sent = ScanCoordinator.prioritised(frame)
        #expect(sent.count == 1)
        #expect(sent.first?.kind == "barcode")
    }

    @MainActor
    @Test func aFrameWithNoBarcodeStillSendsItsText() {
        let frame = (1...5).map {
            DetectedText(text: "LINE \($0)", kind: "text",
                         x: 0.1, y: 0.1 * Double($0), w: 0.5, h: 0.08)
        }
        #expect(ScanCoordinator.prioritised(frame).count == ScanCoordinator.maxTextLines)
    }
}

// MARK: - naming a label from the picture

/// A camera that can be re-pointed between ticks and can hand over a still.
private final class PhotoCamera: ScanEngine, @unchecked Sendable {
    let frames: AsyncStream<[DetectedText]>
    private let cont: AsyncStream<[DetectedText]>.Continuation
    private let still: Data?
    var photosTaken = 0

    init(still: Data? = Data([0xFF, 0xD8, 0xFF, 0xE0])) {
        var c: AsyncStream<[DetectedText]>.Continuation!
        frames = AsyncStream { c = $0 }
        cont = c
        self.still = still
    }

    func point(at frame: [DetectedText]) { cont.yield(frame) }
    func start() async {}
    func stop() { cont.finish() }
    func captureFrame() async -> Data? { photosTaken += 1; return still }
}

/// The garbled-can case end to end: text never resolves to anything the frame agrees on, and
/// the picture names the beer. Records what the picture path was actually sent.
private final class VisionAPI: APIClientProtocol, @unchecked Sendable {
    let sightings: [String]
    let placeable: Bool
    var visionCalls = 0
    var lastVisionRequest: ScanVisionRequest?

    init(sightings: [String] = ["The Alchemist Heady Topper"], placeable: Bool = true) {
        self.sightings = sightings
        self.placeable = placeable
    }

    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        ScanResolveResponse(candidates: [], unresolvedIndices: [0], latencyMs: 0.5,
                            corroborated: false)
    }

    func resolveVision(_ req: ScanVisionRequest) async throws -> ScanVisionResponse {
        visionCalls += 1
        lastVisionRequest = req
        let frame = sightings.map {
            DetectedText(text: $0, kind: "text", x: 0.2, y: 0.3, w: 0.4, h: 0.2)
        }
        guard placeable else {
            // The model read the can; the catalog does not have it. Not the same as reading
            // nothing, and the two look identical from the candidates alone.
            return ScanVisionResponse(candidates: [], unresolvedIndices: Array(sightings.indices),
                                      latencyMs: 900, corroborated: false,
                                      sightings: sightings, detections: frame,
                                      provider: "stub")
        }
        let cands = sightings.enumerated().map { i, name in
            makeCandidate(id: name, name: name, abv: 8, personal: 0.8, index: i)
        }
        return ScanVisionResponse(candidates: cands, unresolvedIndices: [], latencyMs: 900,
                                  corroborated: true, sightings: sightings,
                                  detections: frame, provider: "stub")
    }

    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

@MainActor
private func garble(_ camera: PhotoCamera, _ coord: ScanCoordinator, ticks: Int) async throws {
    // A label the camera cannot read never OCRs the same way twice — which is exactly why the
    // resolve path's held-still shortcut does not fire here, and why the counter can climb.
    for i in 0..<ticks {
        camera.point(at: [DetectedText(text: "FADY TOPPE \(i)", kind: "text",
                                       x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 20_000_000)
        await coord.resolveLatest()
    }
    await coord.visionTask?.value
}

@Suite struct NamingALabelFromThePicture {
    @MainActor
    @Test func aLabelNothingCanReadEscalatesToThePicture() async throws {
        // The failure this path exists for: OCR reads a drawing as "FADY TOPPE", so the
        // information is gone before any query runs and no threshold recovers it.
        let camera = PhotoCamera()
        let api = VisionAPI()
        let coord = ScanCoordinator(engine: camera, api: api, sendsFrames: true)
        coord.start()
        try await garble(camera, coord, ticks: ScanCoordinator.visionAfterTicks)
        #expect(api.visionCalls == 1)
        #expect(camera.photosTaken == 1)
        #expect(coord.overlays.first?.candidate.resolved.product.name
                == "The Alchemist Heady Topper")
        // Anchored to the frame the *server* built — the client never had those boxes.
        #expect(coord.overlays.first?.x == 0.4)
    }

    @MainActor
    @Test func onePassingTickIsNotEnoughToSpendAPicture() async throws {
        // A round-trip with a real cost must not fire while OCR is still settling on a can.
        let camera = PhotoCamera()
        let api = VisionAPI()
        let coord = ScanCoordinator(engine: camera, api: api, sendsFrames: true)
        coord.start()
        try await garble(camera, coord, ticks: ScanCoordinator.visionAfterTicks - 1)
        #expect(api.visionCalls == 0)
        #expect(camera.photosTaken == 0)
    }

    @MainActor
    @Test func withoutConsentThePictureNeverLeavesTheDevice() async throws {
        // `sendsFrames` is the whole switch. Text still goes to the server; the frame does not.
        let camera = PhotoCamera()
        let api = VisionAPI()
        let coord = ScanCoordinator(engine: camera, api: api)      // default: off
        coord.start()
        try await garble(camera, coord, ticks: ScanCoordinator.visionAfterTicks + 4)
        #expect(api.visionCalls == 0)
        #expect(camera.photosTaken == 0)
    }

    @MainActor
    @Test func aCameraThatCannotTakeAPictureSimplyDoesNotEscalate() async throws {
        let camera = PhotoCamera(still: nil)
        let api = VisionAPI()
        let coord = ScanCoordinator(engine: camera, api: api, sendsFrames: true)
        coord.start()
        try await garble(camera, coord, ticks: ScanCoordinator.visionAfterTicks)
        #expect(camera.photosTaken == 1)       // asked
        #expect(api.visionCalls == 0)          // got nothing, sent nothing
    }

    @MainActor
    @Test func theSecondPictureWaitsForTheCooldown() async throws {
        // At a 350ms tick a shelf would otherwise fire this continuously.
        let camera = PhotoCamera()
        let api = VisionAPI(placeable: false)   // never answers, so the counter keeps climbing
        let coord = ScanCoordinator(engine: camera, api: api, sendsFrames: true)
        coord.start()
        try await garble(camera, coord, ticks: ScanCoordinator.visionAfterTicks * 3)
        #expect(api.visionCalls == 1)
    }

    @MainActor
    @Test func aSightingTheCatalogCannotPlaceIsReportedNotDrawn() async throws {
        // "The model read nothing" and "the catalog has nothing" are different problems.
        let camera = PhotoCamera()
        let api = VisionAPI(sightings: ["Pliny The Elder"], placeable: false)
        let coord = ScanCoordinator(engine: camera, api: api, sendsFrames: true)
        coord.start()
        try await garble(camera, coord, ticks: ScanCoordinator.visionAfterTicks)
        #expect(coord.overlays.isEmpty)
        #expect(coord.lastSightings == ["Pliny The Elder"])
    }

    @MainActor
    @Test func theCamerasOwnReadingRidesAlongToCorroborate() async throws {
        let camera = PhotoCamera()
        let api = VisionAPI()
        let coord = ScanCoordinator(engine: camera, api: api, sendsFrames: true)
        coord.start()
        try await garble(camera, coord, ticks: ScanCoordinator.visionAfterTicks)
        let sent = try #require(api.lastVisionRequest)
        #expect(!sent.imageB64.isEmpty)
        #expect(sent.detections.contains { $0.text.hasPrefix("FADY TOPPE") })
        #expect(sent.detections.count <= ScanCoordinator.maxVisionOCRLines)
    }

    @MainActor
    @Test func aFrameTheCatalogRecognisesResetsTheCount() async throws {
        // Only an unread label escalates. One recognised tick means the text path is working
        // and the picture is not needed.
        let camera = PhotoCamera()
        let api = CatalogStubAPI(known: ["Heady Topper"])
        let coord = ScanCoordinator(engine: camera, api: api, sendsFrames: true)
        coord.start()
        for i in 0..<(ScanCoordinator.visionAfterTicks * 2) {
            let text = i % 3 == 0 ? "Heady Topper" : "FADY TOPPE \(i)"
            camera.point(at: [DetectedText(text: text, kind: "text",
                                           x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
            try await Task.sleep(nanoseconds: 20_000_000)
            await coord.resolveLatest()
        }
        await coord.visionTask?.value
        #expect(camera.photosTaken == 0)
    }

    @Test func answersWithoutABoxSpreadInsteadOfStacking() {
        // A vision model volunteers a box or it doesn't. Without a layout every boxless answer
        // lands on (0,0), stacked in the corner.
        let alone = ScanCoordinator.fallbackAnchor(index: 0, of: 1)
        #expect(alone.x == 0.5 && alone.y == 0.5)
        let ys = (0..<3).map { ScanCoordinator.fallbackAnchor(index: $0, of: 3).y }
        #expect(ys == ys.sorted())
        #expect(Set(ys).count == 3)
        #expect(ys.allSatisfy { $0 > 0 && $0 < 1 })
    }
}

@Suite struct RecommendContract {
    /// Exact JSON `POST /v1/recommend` emits, copied from the live service.
    @Test func decodesRecommendResponseFromServerJSON() throws {
        let json = """
        {
          "user_id": "probe",
          "results": [
            {
              "product_id": "ttb:22235001000253",
              "name": "Other Half Brewing Ddh Citra + Nelson",
              "producer": "Other Half Brewing",
              "score": 0.908,
              "reason": "matches your tropical preference",
              "cold_start": true,
              "evidence": "known"
            },
            {
              "product_id": "ttb:1", "name": "Registry IPA", "producer": null,
              "score": 0.92, "reason": "based on style",
              "cold_start": true, "evidence": "guessed"
            }
          ]
        }
        """.data(using: .utf8)!

        let resp = try JSONDecoder().decode(RecommendResponse.self, from: json)
        try #require(resp.results.count == 2)
        #expect(resp.userId == "probe")
        let first = resp.results[0]
        #expect(first.productId == "ttb:22235001000253")
        #expect(first.producer == "Other Half Brewing")
        #expect(first.score == 0.908)
        #expect(first.evidence == .known)
        #expect(first.coldStart)
        // The registry files rows with no producer of their own, so it has to be optional.
        #expect(resp.results[1].producer == nil)
        #expect(resp.results[1].evidence == .guessed)
    }

    /// A tier the client has not heard of must not lose the whole list.
    @Test func anUnknownEvidenceTierReadsAsAGuess() throws {
        let json = """
        {"user_id": "u", "results": [
          {"product_id": "p", "name": "N", "evidence": "sommelier_verified"}]}
        """.data(using: .utf8)!
        let resp = try JSONDecoder().decode(RecommendResponse.self, from: json)
        #expect(resp.results.first?.evidence == .guessed)
        #expect(resp.results.first?.score == 0)
        #expect(resp.results.first?.reason == "")
    }

    @Test func evidenceSaysWhereTheAnswerCameFrom() {
        #expect(Recommendation.Evidence.rated.blurb.contains("rated"))
        #expect(Recommendation.Evidence.known.blurb != Recommendation.Evidence.guessed.blurb)
    }

    /// The list is shown in the server's order, which weighs evidence as well as score, so a
    /// client that re-sorted on `score` would undo the ranking.
    @Test func theServerOrderIsNotTheScoreOrder() throws {
        let json = """
        {"user_id": "u", "results": [
          {"product_id": "a", "name": "Rated", "score": 0.85, "evidence": "rated"},
          {"product_id": "b", "name": "Guessed", "score": 0.92, "evidence": "guessed"}]}
        """.data(using: .utf8)!
        let got = try JSONDecoder().decode(RecommendResponse.self, from: json).results
        #expect(got.map(\.productId) == ["a", "b"])
        #expect(got[0].score < got[1].score)
    }

    /// A stub that does not implement the route says so, rather than answering "nothing".
    @Test func anUnimplementedStubReportsTheRoute() async {
        struct Stub: APIClientProtocol {
            func resolveScan(_: ScanResolveRequest) async throws -> ScanResolveResponse {
                ScanResolveResponse(candidates: [], unresolvedIndices: [], latencyMs: 0)
            }
            func searchProducts(_: String) async throws -> [ResolvedProduct] { [] }
            func sendTelemetry(_: TelemetryBatch) async throws {}
        }
        await #expect(throws: APIError.self) { _ = try await Stub().recommend(limit: 5) }
    }
}

@Suite struct RecentSearchHistory {
    private func store() -> RecentSearches {
        let d = UserDefaults(suiteName: "recents.\(UUID().uuidString)")!
        return RecentSearches(defaults: d)
    }

    @Test func mostRecentFirstAndNoRepeats() {
        let r = store()
        r.record("heady topper")
        r.record("lagavulin")
        r.record("Heady Topper")
        // One entry, at the top, spelled the way it was last typed -- the history is a list of
        // things looked for, not of keystrokes, and the same search twice is one thing.
        #expect(r.all() == ["Heady Topper", "lagavulin"])
    }

    @Test func blankSearchesAreNotHistory() {
        let r = store()
        r.record("   ")
        r.record("")
        #expect(r.all().isEmpty)
        r.record("  gray whale gin  ")
        #expect(r.all() == ["gray whale gin"])   // stored trimmed, so it matches next time
    }

    @Test func theListStaysAShortcut() {
        let r = store()
        for i in 0..<(RecentSearches.limit + 6) { r.record("search \(i)") }
        #expect(r.all().count == RecentSearches.limit)
        #expect(r.all().first == "search \(RecentSearches.limit + 5)")
    }

    @Test func oneCanBeForgottenAndAllCanBe() {
        let r = store()
        ["a", "b", "c"].forEach(r.record)
        r.remove("B")                       // case-insensitively, as it was matched going in
        #expect(r.all() == ["c", "a"])
        r.clear()
        #expect(r.all().isEmpty)
    }
}

@Suite struct SimilarProfileRoute {
    /// The bug this exists for: ids carry colons, and percent-encoding one before handing it to
    /// `appendingPathComponent` gets the `%` escaped in turn. The app sent
    /// `ttb%253A94033604`, the server returned 404, and the view's `try?` turned that into "no
    /// similar products" -- indistinguishable on screen from a row that genuinely has none.
    @Test func anIdWithAColonSurvivesTheRoundTripThroughTheURL() throws {
        let base = URL(string: "http://127.0.0.1:8000")!
        let url = base.appendingPathComponent("v1/product")
            .appendingPathComponent("ttb:94033604")
            .appendingPathComponent("similar")
        let comps = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        #expect(comps.path == "/v1/product/ttb:94033604/similar")
        #expect(try #require(comps.url).absoluteString.contains("%253A") == false)
    }

    @Test func aStyleOnlyAnswerDecodesToAnEmptySection() throws {
        let json = #"{"product_id":"x","basis":"style_only","results":[]}"#
        let out = try JSONDecoder().decode(SimilarResponse.self, from: Data(json.utf8))
        #expect(out.basis == .styleOnly)
        #expect(out.results.isEmpty)
    }

    @Test func aNeighbourCarriesItsEvidenceAndHowManyTasteIdentical() throws {
        let json = """
        {"product_id":"a","basis":"profile","results":[
          {"product_id":"b","name":"Big Peat","producer":"Monarch","evidence":"known","also":10}]}
        """
        let out = try JSONDecoder().decode(SimilarResponse.self, from: Data(json.utf8))
        #expect(out.basis == .profile)
        let row = try #require(out.results.first)
        #expect(row.name == "Big Peat")
        #expect(row.evidence == .known)
        #expect(row.also == 10)
    }
}
