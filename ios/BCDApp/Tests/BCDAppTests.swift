import XCTest
@testable import BCDApp
import BCDKit

// App-layer smoke tests. The heavy logic lives in BCDKit (tested there with Swift
// Testing); these just prove the composition root and view models wire up. Runs under
// Xcode once installed.
final class BCDAppTests: XCTestCase {
    @MainActor
    func testScanViewModelBuildsOverlaysFromCandidates() async throws {
        let env = AppEnvironment(
            api: PreviewAPI(), llm: MockLLMProvider(),
            telemetry: TelemetryQueue(consent: ConsentState(analytics: true)),
            makeScanEngine: {
                MockScanEngine(scripted: [[DetectedText(text: "Heady Topper", kind: "text")]])
            })
        let model = ScanViewModel()
        model.configure(env: env)
        model.startLive()
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertFalse(model.overlays.isEmpty)
    }
}

private final class PreviewAPI: APIClientProtocol, @unchecked Sendable {
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        let prov = Provenance(sourceId: "t", url: nil, quote: nil,
                              method: .regulatoryFiling, confidence: 1)
        let product = Product(id: "p", brandId: "b", producerId: "pr", category: .beer,
                              name: "Heady Topper", style: nil,
                              spec: ProductSpec(abvPct: Sourced(value: 8, provenance: prov),
                                                ibu: nil, proof: nil, ageStatementYears: nil),
                              recipe: RecipeGraph())
        let resolved = ResolvedProduct(
            product: product,
            producer: Producer(id: "pr", name: "Alchemist", kind: nil, country: nil,
                               region: nil, city: nil, lat: nil, lon: nil, website: nil),
            brand: Brand(id: "b", producerId: "pr", name: "Heady"))
        return ScanResolveResponse(
            candidates: [ScoredCandidate(detectionIndex: 0, resolved: resolved, matchScore: 1,
                                         personalScore: 0.8, reason: "tropical", coldStart: true)],
            unresolvedIndices: [], latencyMs: 1)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}
