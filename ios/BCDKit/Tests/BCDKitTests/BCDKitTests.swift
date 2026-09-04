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
        #expect(sent.filter { $0.kind == "text" }.count == ScanCoordinator.maxTextLines)
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
            [DetectedText(text: "A CHEMIST VER", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)],
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
        engine.push([DetectedText(text: "DRINK FRO", kind: "text", x: 0.2, y: 0.3, w: 0.5, h: 0.1)])
        try await Task.sleep(nanoseconds: 100_000_000)
        await coord.resolveLatest()                    // tick 1 — starts the model
        let started = coord.interpretation
        engine.push([DetectedText(text: "HAN! DRINK FRO MAL31", kind: "text",
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

        engine.push([DetectedText(text: "CHEMIST-VER", kind: "text",
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
