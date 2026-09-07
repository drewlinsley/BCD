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
          "candidates": [\(candidateJSON(objectId: nil, detectionIndex: 0))],
          "unresolved_indices": [],
          "objects": [{
            "object_id": "obj-1", "status": "ambiguous", "query": "the alchemist",
            "candidates": [\(candidateJSON(objectId: "obj-1", detectionIndex: nil))]
          }],
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
        #expect(c.detectionIndex == 0 && c.objectId == nil)
        try #require(resp.objects.count == 1)
        #expect(resp.objects[0].status == .ambiguous)
        #expect(resp.objects[0].candidates[0].objectId == "obj-1")
    }

    @Test func toleratesServerWithoutObjectPath() throws {
        let json = #"{"candidates": [], "unresolved_indices": [0]}"#.data(using: .utf8)!
        let resp = try JSONDecoder().decode(ScanResolveResponse.self, from: json)
        #expect(resp.objects.isEmpty && resp.unresolvedIndices == [0] && resp.latencyMs == nil)
    }

    @Test func encodesObjectRequestWithSnakeCase() throws {
        let obj = DetectedObject(id: "t1", label: "can", texts: ["HEADY", "TOPPER"],
                                 box: BoundingBox(x: 0.1, y: 0.2, w: 0.3, h: 0.4), framesSeen: 5)
        let req = ScanResolveRequest(objects: [obj], minMatchScore: 0.8)
        let data = try JSONEncoder().encode(req)
        let s = String(decoding: data, as: UTF8.self)
        #expect(s.contains(#""frames_seen":5"#))
        #expect(s.contains(#""min_match_score":0.8"#))
        #expect(s.contains(#""objects":[{"#))
    }

    @Test func provenanceTrustRankOrders() {
        #expect(ExtractionMethod.regulatoryFiling.trustRank >
                ExtractionMethod.llmInferredFromStylePrior.trustRank)
        #expect(ExtractionMethod.statedByProducer.trustRank >
                ExtractionMethod.reviewConsensus.trustRank)
    }
}

@Suite struct Geometry {
    @Test func iouAndUnion() {
        let a = BoundingBox(x: 0, y: 0, w: 0.5, h: 0.5)
        let b = BoundingBox(x: 0.25, y: 0.25, w: 0.5, h: 0.5)
        #expect(abs(a.iou(b) - (0.0625 / 0.4375)) < 1e-9)
        #expect(a.union(b) == BoundingBox(x: 0, y: 0, w: 0.75, h: 0.75))
        #expect(a.iou(BoundingBox(x: 0.6, y: 0.6, w: 0.1, h: 0.1)) == 0)
    }

    @Test func aspectFillMapsContentOntoTallerView() {
        // 1080x1920 buffer (0.5625) shown on a 9:19.5 screen (0.4615): sides get cropped.
        let m = AspectFillMapper(contentAspect: 1080.0 / 1920.0)
        let view = 9.0 / 19.5
        let frame = m.contentFrame(inViewAspect: view)
        #expect(frame.w > 1 && frame.h == 1 && frame.x < 0)
        let center = BoundingBox(x: 0.45, y: 0.45, w: 0.1, h: 0.1)
        let v = m.toView(center, viewAspect: view)
        #expect(abs(v.midX - 0.5) < 1e-9 && abs(v.midY - 0.5) < 1e-9)
        let back = m.toContent(v, viewAspect: view)
        #expect(abs(back.x - center.x) < 1e-9 && abs(back.w - center.w) < 1e-9)
    }
}

@Suite struct TextClustering {
    @Test func groupsLinesOnOneLabelAndSeparatesNeighbours() {
        // Two cans side by side, three stacked lines each.
        let left = [
            DetectedText(text: "THE ALCHEMIST", x: 0.10, y: 0.30, w: 0.25, h: 0.04),
            DetectedText(text: "HEADY", x: 0.12, y: 0.36, w: 0.20, h: 0.05),
            DetectedText(text: "TOPPER", x: 0.12, y: 0.43, w: 0.22, h: 0.05),
        ]
        let right = [
            DetectedText(text: "DOGFISH HEAD", x: 0.60, y: 0.31, w: 0.25, h: 0.04),
            DetectedText(text: "60 MINUTE IPA", x: 0.60, y: 0.37, w: 0.28, h: 0.05),
        ]
        let clusters = TextClusterer.cluster(left + right)
        #expect(clusters.count == 2)
        let sizes = clusters.map(\.members.count).sorted()
        #expect(sizes == [2, 3])
        let big = clusters.first { $0.members.count == 3 }!
        #expect(approx(big.box, BoundingBox(x: 0.10, y: 0.30, w: 0.25, h: 0.18)))
    }

    @Test func farApartLinesInOneColumnStaySeparate() {
        let a = DetectedText(text: "TOP SHELF", x: 0.1, y: 0.05, w: 0.2, h: 0.03)
        let b = DetectedText(text: "BOTTOM SHELF", x: 0.1, y: 0.80, w: 0.2, h: 0.03)
        #expect(TextClusterer.cluster([a, b]).count == 2)
    }

    @Test func boxlessTextsAreSingletons() {
        let clusters = TextClusterer.cluster([DetectedText(text: "A"), DetectedText(text: "B")])
        #expect(clusters.count == 2 && clusters.allSatisfy { $0.box == nil })
    }
}

@Suite struct Tracking {
    private func frame(_ lines: [(String, Double)], x: Double = 0.1,
                       regions: [ObjectRegion]? = nil) -> ScanFrame {
        ScanFrame(texts: lines.map { DetectedText(text: $0.0, x: x, y: $0.1, w: 0.2, h: 0.04) },
                  regions: regions)
    }

    @Test func oneObjectAccumulatesEvidenceAcrossFrames() {
        let tracker = ObjectTracker()
        tracker.update(with: frame([("HEADY", 0.30), ("Chemist", 0.36)]))
        // Same label a frame later: slight jitter, one line misread differently.
        var tracks = tracker.update(with: frame([("HEADY", 0.305), ("TOPPER", 0.365)], x: 0.11))
        #expect(tracks.count == 1)
        #expect(tracks[0].framesSeen == 2)
        // Only lines seen twice count; "Chemist" and "TOPPER" were seen once each.
        #expect(tracks[0].stableTexts(minCount: 2) == ["HEADY"])
        #expect(tracker.isReady(tracks[0]))
        tracks = tracker.update(with: frame([("HEADY", 0.30), ("TOPPER", 0.36)]))
        #expect(tracks[0].stableTexts(minCount: 2) == ["HEADY", "TOPPER"])
    }

    @Test func requeryOnlyWhenEvidenceChangesAndAfterCooldown() {
        var cfg = ObjectTracker.Config()
        cfg.requeryCooldown = 4
        let tracker = ObjectTracker(config: cfg)
        tracker.update(with: frame([("HEADY", 0.30)]))
        var t = tracker.update(with: frame([("HEADY", 0.30)]))[0]
        #expect(tracker.isReady(t))
        tracker.markQueried(t.id)
        t = tracker.update(with: frame([("HEADY", 0.30)]))[0]
        #expect(!tracker.isReady(t))  // same evidence: nothing new to ask
        t = tracker.update(with: frame([("HEADY", 0.30), ("TOPPER", 0.36)]))[0]
        t = tracker.update(with: frame([("HEADY", 0.30), ("TOPPER", 0.36)]))[0]
        #expect(!tracker.isReady(t))  // new evidence, but inside the cooldown
        t = tracker.update(with: frame([("HEADY", 0.30), ("TOPPER", 0.36)]))[0]
        #expect(tracker.isReady(t))   // cooldown elapsed
    }

    @Test func fineReaderTextIsPinnedAndJumpsCooldown() {
        let tracker = ObjectTracker()
        tracker.update(with: frame([("Chemist", 0.30)]))
        let t = tracker.update(with: frame([("Chemist", 0.30)]))[0]
        tracker.markQueried(t.id)
        tracker.addTexts(["HEADY TOPPER"], to: t.id)
        let after = tracker.update(with: frame([("Chemist", 0.30)]))[0]
        #expect(after.stableTexts(minCount: 2).contains("HEADY TOPPER"))
        #expect(tracker.isReady(after))
    }

    @Test func textsInsideARegionBelongToIt() {
        let can = ObjectRegion(id: "r1", box: BoundingBox(x: 0.05, y: 0.1, w: 0.3, h: 0.7),
                               label: "can")
        let other = ObjectRegion(id: "r2", box: BoundingBox(x: 0.55, y: 0.1, w: 0.3, h: 0.7),
                                 label: "bottle")
        let f = ScanFrame(texts: [
            DetectedText(text: "HEADY", x: 0.10, y: 0.30, w: 0.2, h: 0.04),
            DetectedText(text: "TOPPER", x: 0.10, y: 0.60, w: 0.2, h: 0.04),  // far from HEADY
            DetectedText(text: "PLINY", x: 0.60, y: 0.30, w: 0.2, h: 0.04),
        ], regions: [can, other])
        let tracker = ObjectTracker()
        tracker.update(with: f)
        let tracks = tracker.update(with: f)
        #expect(tracks.count == 2)
        let canTrack = tracks.first { $0.label == "can" }!
        #expect(Set(canTrack.stableTexts(minCount: 2)) == ["HEADY", "TOPPER"])
        #expect(approx(canTrack.box, can.box))
        #expect(tracks.first { $0.label == "bottle" }!.stableTexts(minCount: 2) == ["PLINY"])
    }

    @Test func barcodeIsEvidenceImmediately() {
        let tracker = ObjectTracker()
        let f = ScanFrame(texts: [DetectedText(text: "854416001019", kind: "barcode",
                                               x: 0.3, y: 0.5, w: 0.2, h: 0.1)])
        tracker.update(with: f)
        let t = tracker.update(with: f)[0]
        #expect(t.barcode == "854416001019")
        #expect(tracker.isReady(t))
        #expect(t.detectedObject(minCount: 2).barcode == "854416001019")
    }

    @Test func boxlessTextsTrackByIdentity() {
        let tracker = ObjectTracker()
        tracker.update(with: ScanFrame(texts: [DetectedText(text: "Heady Topper")]))
        let tracks = tracker.update(with: ScanFrame(texts: [DetectedText(text: "Heady Topper")]))
        #expect(tracks.count == 1 && tracks[0].framesSeen == 2 && !tracks[0].anchored)
        #expect(tracks[0].detectedObject(minCount: 2).box == nil)
    }

    @Test func missingObjectsAreDroppedAfterGrace() {
        var cfg = ObjectTracker.Config()
        cfg.maxMissing = 2
        let tracker = ObjectTracker(config: cfg)
        tracker.update(with: frame([("HEADY", 0.3)]))
        for _ in 0..<2 { #expect(tracker.update(with: ScanFrame(texts: [])).count == 1) }
        #expect(tracker.update(with: ScanFrame(texts: [])).isEmpty)
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

    @Test func constrainedPickToleratesOneMisreadLetter() async throws {
        let cands = [
            makeCandidate(id: "heady", name: "Heady Topper", abv: 8, personal: 0.9),
            makeCandidate(id: "focal", name: "Focal Banger", abv: 7, personal: 0.8),
        ]
        let pick = try await MockLLMProvider().pickProduct(ocr: ["HEADY", "TOPPFR"], candidates: cands)
        #expect(pick?.productId == "heady")
    }

    @Test func constrainedPickRefusesWithoutNameEvidence() async throws {
        let cands = [
            makeCandidate(id: "heady", name: "Heady Topper", abv: 8, personal: 0.9),
            makeCandidate(id: "focal", name: "Focal Banger", abv: 7, personal: 0.8),
        ]
        let pick = try await MockLLMProvider().pickProduct(ocr: ["THE ALCHEMIST", "IPA"],
                                                          candidates: cands)
        #expect(pick == nil)
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
    private static func headyFrames(_ n: Int, text: String = "Heady Topper") -> [ScanFrame] {
        Array(repeating: ScanFrame(texts: [
            DetectedText(text: text, kind: "text", x: 0.2, y: 0.3, w: 0.4, h: 0.06),
        ]), count: n)
    }

    @MainActor
    @Test func queriesOncePerObjectAfterTwoFramesOfAgreement() async throws {
        // Five identical frames — the coordinator must query the object exactly once.
        let engine = MockScanEngine(frames: Self.headyFrames(5))
        let api = ScriptedAPI { obj in
            obj.texts.contains("Heady Topper")
                ? .resolved(obj.id, makeCandidate(id: "heady", name: "Heady Topper", abv: 8, personal: 0.8))
                : .unresolved(obj.id)
        }
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()
        try await Task.sleep(nanoseconds: 300_000_000)  // let the stream drain
        #expect(api.resolveCallCount == 1)
        #expect(coord.objects.count == 1)
        #expect(coord.candidates.map { $0.resolved.product.id } == ["heady"])
        if case .resolved(_, let stage) = coord.objects[0].status { #expect(stage == .coarse) }
        else { Issue.record("expected a coarse resolution") }
    }

    @MainActor
    @Test func unresolvedObjectGetsAFineReadThenResolves() async throws {
        // The coarse pass reads "Chemist"; the careful pass reads the real name.
        let engine = MockScanEngine(frames: Self.headyFrames(4, text: "Chemist"))
        engine.fineReads = ["THE ALCHEMIST", "HEADY TOPPER"]
        let api = ScriptedAPI { obj in
            obj.texts.contains("HEADY TOPPER")
                ? .resolved(obj.id, makeCandidate(id: "heady", name: "Heady Topper", abv: 8, personal: 0.8))
                : .unresolved(obj.id)
        }
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()
        try await Task.sleep(nanoseconds: 300_000_000)
        #expect(engine.fineReadCount == 1)
        #expect(api.resolveCallCount == 2)  // coarse (unresolved) + re-query with fine text
        if case .resolved(let c, let stage)? = coord.objects.first?.status {
            #expect(c.resolved.product.id == "heady" && stage == .fineOCR)
        } else { Issue.record("expected fine-OCR resolution, got \(String(describing: coord.objects.first?.status))") }
    }

    @MainActor
    @Test func ambiguousShortlistIsAdjudicatedByTheModel() async throws {
        let engine = MockScanEngine(frames: Self.headyFrames(3, text: "HEADY TOPPFR"))
        let heady = makeCandidate(id: "heady", name: "Heady Topper", abv: 8, personal: 0.8)
        let focal = makeCandidate(id: "focal", name: "Focal Banger", abv: 7, personal: 0.7)
        let api = ScriptedAPI { obj in .ambiguous(obj.id, [heady, focal]) }
        let coord = ScanCoordinator(engine: engine, api: api, llm: MockLLMProvider())
        coord.start()
        try await Task.sleep(nanoseconds: 300_000_000)
        if case .resolved(let c, let stage)? = coord.objects.first?.status {
            #expect(c.resolved.product.id == "heady" && stage == .llmPick)
        } else { Issue.record("expected an llm pick") }
        #expect(coord.candidates.count == 1)
    }

    @MainActor
    @Test func ambiguousWithoutEvidenceShowsNoName() async throws {
        let engine = MockScanEngine(frames: Self.headyFrames(3, text: "THE ALCHEMIST"))
        let heady = makeCandidate(id: "heady", name: "Heady Topper", abv: 8, personal: 0.8)
        let focal = makeCandidate(id: "focal", name: "Focal Banger", abv: 7, personal: 0.7)
        let api = ScriptedAPI { obj in .ambiguous(obj.id, [heady, focal]) }
        let coord = ScanCoordinator(engine: engine, api: api, llm: MockLLMProvider())
        coord.start()
        try await Task.sleep(nanoseconds: 300_000_000)
        if case .ambiguous(let cands)? = coord.objects.first?.status { #expect(cands.count == 2) }
        else { Issue.record("expected the shortlist to remain") }
        #expect(coord.candidates.isEmpty)  // nothing fires on the HUD

        // ...until the user picks — the highest-value label we get.
        coord.confirm(objectId: coord.objects[0].id, candidate: focal)
        #expect(coord.candidates.map { $0.resolved.product.id } == ["focal"])
    }

    @MainActor
    @Test func networkFailureRetriesLater() async throws {
        let engine = MockScanEngine(frames: Self.headyFrames(3))
        let api = ScriptedAPI(failFirst: 1) { obj in
            .resolved(obj.id, makeCandidate(id: "heady", name: "Heady Topper", abv: 8, personal: 0.8))
        }
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()
        try await Task.sleep(nanoseconds: 300_000_000)
        #expect(api.resolveCallCount == 2)
        #expect(coord.candidates.count == 1)
    }
}

// MARK: - test doubles

private func approx(_ a: BoundingBox?, _ b: BoundingBox, tol: Double = 1e-9) -> Bool {
    guard let a else { return false }
    return abs(a.x - b.x) < tol && abs(a.y - b.y) < tol && abs(a.w - b.w) < tol && abs(a.h - b.h) < tol
}

private func makeCandidate(id: String, name: String, abv: Double,
                          personal: Double) -> ScoredCandidate {
    let prov = Provenance(sourceId: "t", url: nil, quote: nil,
                          method: .regulatoryFiling, confidence: 1)
    let product = Product(
        id: id, brandId: "b", producerId: "p", category: .beer, name: name,
        style: nil, spec: ProductSpec(abvPct: Sourced(value: abv, provenance: prov),
                                      ibu: nil, proof: nil, ageStatementYears: nil),
        recipe: RecipeGraph())
    let resolved = ResolvedProduct(
        product: product,
        producer: Producer(id: "p", name: "The Alchemist", kind: nil, country: nil, region: nil,
                           lat: nil, lon: nil, website: nil),
        brand: Brand(id: "b", producerId: "p", name: "B"))
    return ScoredCandidate(detectionIndex: 0, resolved: resolved, matchScore: 1,
                           personalScore: personal, reason: nil, coldStart: true)
}

private func candidateJSON(objectId: String?, detectionIndex: Int?) -> String {
    """
    {
      "detection_index": \(detectionIndex.map { String($0) } ?? "null"),
      "object_id": \(objectId.map { "\"\($0)\"" } ?? "null"),
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
    }
    """
}

private final class CountingSink: APIClientProtocol, @unchecked Sendable {
    var batches = 0
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        ScanResolveResponse(candidates: [], unresolvedIndices: [], latencyMs: nil)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws { batches += 1 }
}

/// Server stand-in that answers per object from a script.
private final class ScriptedAPI: APIClientProtocol, @unchecked Sendable {
    enum Verdict {
        case resolved(String, ScoredCandidate)
        case ambiguous(String, [ScoredCandidate])
        case unresolved(String)
    }
    var resolveCallCount = 0
    private var failuresLeft: Int
    private let script: @Sendable (DetectedObject) -> Verdict

    init(failFirst: Int = 0, _ script: @escaping @Sendable (DetectedObject) -> Verdict) {
        self.failuresLeft = failFirst
        self.script = script
    }

    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        resolveCallCount += 1
        if failuresLeft > 0 { failuresLeft -= 1; throw APIError.http(503) }
        var flat: [ScoredCandidate] = []
        let objects: [ObjectResolution] = req.objects.map { obj in
            switch script(obj) {
            case .resolved(let id, let c):
                let cand = ScoredCandidate(objectId: id, resolved: c.resolved, matchScore: 0.9,
                                           personalScore: c.personalScore, reason: c.reason,
                                           coldStart: c.coldStart)
                flat.append(cand)
                return ObjectResolution(objectId: id, status: .resolved, candidates: [cand])
            case .ambiguous(let id, let cs):
                return ObjectResolution(objectId: id, status: .ambiguous, candidates: cs.map {
                    ScoredCandidate(objectId: id, resolved: $0.resolved, matchScore: 0.4,
                                    personalScore: $0.personalScore, reason: $0.reason,
                                    coldStart: $0.coldStart)
                })
            case .unresolved(let id):
                return ObjectResolution(objectId: id, status: .unresolved)
            }
        }
        return ScanResolveResponse(candidates: flat, unresolvedIndices: [], objects: objects,
                                   latencyMs: 0.5)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}
