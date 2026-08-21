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
    @Test func captureResolvesCurrentFrameOnceAndAnchors() async throws {
        // The live viewfinder buffers frames but never resolves; the shutter resolves the
        // current frame exactly once and pins each overlay to its detection's box center.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "Heady Topper", kind: "text", x: 0.1, y: 0.2, w: 0.2, h: 0.1)],
        ])
        let api = StubAPI()
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()
        try await Task.sleep(nanoseconds: 200_000_000)  // let the frame buffer
        #expect(api.resolveCallCount == 0)              // live viewfinder never resolves
        await coord.capture()
        #expect(api.resolveCallCount == 1)              // the shutter resolves exactly once
        #expect(coord.captured)
        let overlay = try #require(coord.overlays.first)
        #expect(abs(overlay.x - 0.2) < 0.001)           // box center x = 0.1 + 0.2/2
        #expect(abs(overlay.y - 0.25) < 0.001)          // box center y = 0.2 + 0.1/2
    }

    @MainActor
    @Test func liveModeResolvesLatestFrameAndSkipsUnchanged() async throws {
        // Fixed-rate mode resolves the latest frame and swaps overlays in place *without*
        // freezing, and skips the re-resolve when the OCR is unchanged (camera held still) —
        // so a held viewfinder stays cheap instead of firehosing the backend every tick.
        let engine = MockScanEngine(scripted: [
            [DetectedText(text: "Krombacher", kind: "text", x: 0.3, y: 0.4, w: 0.2, h: 0.1)],
        ])
        let api = StubAPI()
        let coord = ScanCoordinator(engine: engine, api: api)
        coord.start()
        try await Task.sleep(nanoseconds: 100_000_000)  // let the frame buffer
        await coord.resolveLatest()                     // one live tick
        #expect(api.resolveCallCount == 1)
        #expect(!coord.captured)                         // live never freezes
        #expect(coord.overlays.count == 1)
        #expect(coord.overlays.first?.candidate.resolved.product.name == "Krombacher")
        await coord.resolveLatest()                     // same frame → deduped, no round-trip
        #expect(api.resolveCallCount == 1)
    }
}

// MARK: - test doubles

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
        producer: Producer(id: "p", name: "P", kind: nil, country: nil, region: nil,
                           lat: nil, lon: nil, website: nil),
        brand: Brand(id: "b", producerId: "p", name: "B"))
    return ScoredCandidate(detectionIndex: 0, resolved: resolved, matchScore: 1,
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
