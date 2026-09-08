import Foundation
import Testing
@testable import BCDKit

// The coarse-to-fine object path: geometry, clustering, tracking, the object stage, and
// the stage riding along on the live tick. Everything here runs on the host.

private func approx(_ a: BoundingBox?, _ b: BoundingBox, tol: Double = 1e-9) -> Bool {
    guard let a else { return false }
    return abs(a.x - b.x) < tol && abs(a.y - b.y) < tol
        && abs(a.w - b.w) < tol && abs(a.h - b.h) < tol
}

@Suite struct ObjectContract {
    @Test func decodesObjectVerdictsAndToleratesTheirAbsence() throws {
        let json = """
        {"candidates": [], "unresolved_indices": [0], "latency_ms": 3.1, "corroborated": true,
         "objects": [{"object_id": "t1", "status": "ambiguous", "query": "focal ban",
                      "candidates": []},
                     {"object_id": "t2", "status": "unresolved"}]}
        """.data(using: .utf8)!
        let resp = try JSONDecoder().decode(ScanResolveResponse.self, from: json)
        #expect(resp.objects.map(\.status) == [.ambiguous, .unresolved])
        #expect(resp.objects[1].query == "" && resp.corroborated)

        let old = #"{"candidates": [], "unresolved_indices": []}"#.data(using: .utf8)!
        let legacy = try JSONDecoder().decode(ScanResolveResponse.self, from: old)
        #expect(legacy.objects.isEmpty && legacy.corroborated == false)
    }

    @Test func encodesObjectsWithSnakeCase() throws {
        let obj = DetectedObject(id: "t1", label: "can", texts: ["HEADY TOPPER"],
                                 box: BoundingBox(x: 0.1, y: 0.2, w: 0.3, h: 0.4), framesSeen: 3)
        let req = ScanResolveRequest(objects: [obj], minMatchScore: 0.7)
        let s = String(decoding: try JSONEncoder().encode(req), as: UTF8.self)
        #expect(s.contains("\"frames_seen\":3") && s.contains("\"min_match_score\":0.7"))
        #expect(s.contains("\"objects\":[{") && s.contains("\"detections\":[]"))
    }
}

@Suite struct Geometry {
    @Test func iouAndUnion() {
        let a = BoundingBox(x: 0, y: 0, w: 0.5, h: 0.5)
        let b = BoundingBox(x: 0.25, y: 0.25, w: 0.5, h: 0.5)
        #expect(abs(a.iou(b) - (0.0625 / 0.4375)) < 1e-9)
        #expect(approx(a.union(b), BoundingBox(x: 0, y: 0, w: 0.75, h: 0.75)))
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
        #expect(clusters.map(\.members.count).sorted() == [2, 3])
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

@Suite struct ConstrainedPick {
    @Test func toleratesOneMisreadLetterAndPicksTheCoveredEntry() async throws {
        let cands = [objCandidate("p1", "Heady Topper"), objCandidate("p2", "Focal Banger")]
        let pick = try await MockLLMProvider().pickProduct(ocr: ["HEADY TOPPFR"], candidates: cands)
        #expect(pick?.productId == "p1")
    }

    @Test func refusesWithoutNameEvidence() async throws {
        let cands = [objCandidate("p1", "Heady Topper"), objCandidate("p2", "Focal Banger")]
        let pick = try await MockLLMProvider().pickProduct(ocr: ["DRINK FROM THE CAN", "16 OZ"],
                                                            candidates: cands)
        #expect(pick == nil)
    }
}

// MARK: - the object stage

private let canBox = BoundingBox(x: 0.10, y: 0.30, w: 0.25, h: 0.11)   // two lines, 0.05 tall, 0.06 apart

private func label(_ lines: [String], y: Double = 0.30) -> [DetectedText] {
    lines.enumerated().map { i, t in
        DetectedText(text: t, x: 0.10, y: y + Double(i) * 0.06, w: 0.25, h: 0.05)
    }
}

func objCandidate(_ id: String, _ name: String, score: Double = 1.0) -> ScoredCandidate {
    let prov = Provenance(sourceId: "t", url: nil, quote: nil, method: .regulatoryFiling,
                          confidence: 1)
    let product = Product(id: id, brandId: "b", producerId: "pr", category: .beer, name: name,
                          style: nil,
                          spec: ProductSpec(abvPct: Sourced(value: 8, provenance: prov),
                                            ibu: nil, proof: nil, ageStatementYears: nil),
                          recipe: RecipeGraph())
    let resolved = ResolvedProduct(
        product: product,
        producer: Producer(id: "pr", name: "The Alchemist", kind: nil, country: nil,
                           region: nil, city: nil, lat: nil, lon: nil, website: nil),
        brand: Brand(id: "b", producerId: "pr", name: name))
    return ScoredCandidate(objectId: nil, resolved: resolved, matchScore: score,
                           personalScore: 0.7, reason: nil, coldStart: false)
}

@MainActor
@Suite struct ObjectStageBehaviour {
    @Test func asksAfterTwoFramesOfAgreementAndNotAgainForTheSameEvidence() {
        let stage = ObjectStage()
        stage.ingest(label(["HEADY TOPPER", "THE ALCHEMIST"]))
        #expect(stage.pending().isEmpty)                 // one frame is not evidence yet
        stage.ingest(label(["HEADY TOPPER", "THE ALCHEMIST"]))
        let sent = stage.pending()
        #expect(sent.count == 1)
        #expect(Set(sent[0].texts) == ["HEADY TOPPER", "THE ALCHEMIST"])
        #expect(sent[0].framesSeen == 2 && approx(sent[0].box, canBox))
        stage.ingest(label(["HEADY TOPPER", "THE ALCHEMIST"]))
        #expect(stage.pending().isEmpty)                 // in flight; nothing new to ask
    }

    @Test func aResolvedVerdictPinsAnOverlayToTheObject() async {
        let stage = ObjectStage()
        stage.ingest(label(["HEADY TOPPER"])); stage.ingest(label(["HEADY TOPPER"]))
        let sent = stage.pending()
        let verdict = ObjectResolution(objectId: sent[0].id, status: .resolved,
                                       candidates: [objCandidate("p1", "Heady Topper")])
        await stage.apply([verdict], sent: sent, fineReader: nil, llm: nil, telemetry: nil)
        #expect(stage.overlays.map(\.id) == ["p1"])
        #expect(abs(stage.overlays[0].x - canBox.midX) < 1e-9)
        let before = stage.overlays[0].y
        // The can drifts a little; the overlay follows it, without another query.
        stage.ingest(label(["HEADY TOPPER"], y: 0.33))
        #expect(stage.pending().isEmpty)
        #expect(stage.overlays[0].y > before)
    }

    @Test func anUnresolvedObjectGetsOneFineReadThenReQueries() async {
        let stage = ObjectStage()
        let reader = MockScanEngine(scripted: [])
        reader.fineReads = ["HEADY TOPPER"]
        stage.ingest(label(["Chemist"])); stage.ingest(label(["Chemist"]))
        let sent = stage.pending()
        let unresolved = ObjectResolution(objectId: sent[0].id, status: .unresolved)
        let again = await stage.apply([unresolved], sent: sent, fineReader: reader, llm: nil,
                                      telemetry: nil)
        #expect(again && reader.fineReadCount == 1)
        #expect(stage.overlays.isEmpty)
        // The careful read is pinned evidence: the next frame re-queries with it.
        stage.ingest(label(["Chemist"]))
        let resent = stage.pending()
        #expect(resent.count == 1 && resent[0].texts.contains("HEADY TOPPER"))
        // ...and a second unresolved verdict does not read again.
        await stage.apply([ObjectResolution(objectId: sent[0].id, status: .unresolved)],
                          sent: resent, fineReader: reader, llm: nil, telemetry: nil)
        #expect(reader.fineReadCount == 1)
    }

    @Test func anAmbiguousShortlistIsAdjudicatedByTheModelOnce() async {
        let stage = ObjectStage()
        stage.ingest(label(["HEADY TOPPFR"])); stage.ingest(label(["HEADY TOPPFR"]))
        let sent = stage.pending()
        let shortlist = [objCandidate("p2", "Focal Banger", score: 0.5),
                         objCandidate("p1", "Heady Topper", score: 0.5)]
        let ambiguous = ObjectResolution(objectId: sent[0].id, status: .ambiguous,
                                         candidates: shortlist)
        await stage.apply([ambiguous], sent: sent, fineReader: nil, llm: MockLLMProvider(),
                          telemetry: nil)
        #expect(stage.overlays.map(\.id) == ["p1"])
        if case .resolved(_, let how)? = stage.objects.first?.status { #expect(how == .llmPick) }
        else { Issue.record("expected a model-adjudicated resolution") }
    }

    @Test func anAmbiguousObjectWithoutEvidenceShowsNothing() async {
        let stage = ObjectStage()
        stage.ingest(label(["FADY TOP"])); stage.ingest(label(["FADY TOP"]))
        let sent = stage.pending()
        let ambiguous = ObjectResolution(objectId: sent[0].id, status: .ambiguous,
                                         candidates: [objCandidate("p9", "Top's", score: 0.4)])
        await stage.apply([ambiguous], sent: sent, fineReader: nil, llm: MockLLMProvider(),
                          telemetry: nil)
        #expect(stage.overlays.isEmpty)
        if case .ambiguous(let list)? = stage.objects.first?.status { #expect(list.count == 1) }
        else { Issue.record("expected the shortlist to be kept, not shown") }
    }

    @Test func aFailedRequestLetsTheSameEvidenceAskAgain() {
        let stage = ObjectStage()
        stage.ingest(label(["HEADY TOPPER"])); stage.ingest(label(["HEADY TOPPER"]))
        let sent = stage.pending()
        stage.retry(sent)
        stage.ingest(label(["HEADY TOPPER"]))
        #expect(stage.pending().count == 1)
    }
}

// MARK: - riding along on the live tick

private final class ObjectEngine: ScanEngine, @unchecked Sendable {
    let frames: AsyncStream<[DetectedText]>
    private var cont: AsyncStream<[DetectedText]>.Continuation?
    init() {
        var c: AsyncStream<[DetectedText]>.Continuation!
        frames = AsyncStream { c = $0 }
        cont = c
    }
    func start() async {}
    func stop() { cont?.finish() }
    func push(_ frame: [DetectedText]) { cont?.yield(frame) }
}

private final class ObjectAPI: APIClientProtocol, @unchecked Sendable {
    var requests: [ScanResolveRequest] = []
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        requests.append(req)
        let objects = req.objects.map {
            ObjectResolution(objectId: $0.id, status: .resolved,
                             candidates: [objCandidate("p1", "Heady Topper")])
        }
        // The server answers the lines with nothing the frame corroborates; the object
        // is what resolves.
        return ScanResolveResponse(candidates: objects.flatMap(\.candidates),
                                   unresolvedIndices: Array(req.detections.indices),
                                   objects: objects, latencyMs: 2, corroborated: !objects.isEmpty)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

@MainActor
@Suite struct ObjectsOnTheLiveTick {
    @Test func theTickSendsReadyObjectsAndDrawsTheirVerdicts() async throws {
        let engine = ObjectEngine()
        let api = ObjectAPI()
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()
        engine.push(label(["HEADY TOPPER", "THE ALCHEMIST"]))
        try await Task.sleep(nanoseconds: 30_000_000)
        await coord.resolveLatest()
        // One frame of evidence: the lines went, the object did not.
        #expect(api.requests.count == 1 && api.requests[0].objects.isEmpty)
        engine.push(label(["HEADY TOPPER", "THE ALCHEMIST"]))
        try await Task.sleep(nanoseconds: 30_000_000)
        // The same lines as last tick would normally skip the round-trip; an object that
        // has just earned its query is new evidence.
        await coord.resolveLatest()
        #expect(api.requests.count == 2 && api.requests[1].objects.count == 1)
        for _ in 0..<20 where coord.overlays.isEmpty {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        #expect(coord.overlays.map(\.id) == ["p1"])
        #expect(abs(coord.overlays[0].x - canBox.midX) < 1e-9)
        #expect(coord.objectStage.queryCount == 1)
        coord.stop()
        #expect(coord.overlays.isEmpty)
    }
}
