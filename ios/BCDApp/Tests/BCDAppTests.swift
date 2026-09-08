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
                // Two frames of agreement before the coordinator asks the server.
                MockScanEngine(scripted: [[DetectedText(text: "Heady Topper", kind: "text")],
                                          [DetectedText(text: "Heady Topper", kind: "text")]])
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
        // Answer per object, the way the real server does on the coarse-to-fine path.
        let objects = req.objects.map { obj in
            ObjectResolution(objectId: obj.id, status: .resolved, query: "heady topper", candidates: [
                ScoredCandidate(objectId: obj.id, resolved: resolved, matchScore: 1,
                                personalScore: 0.8, reason: "tropical", coldStart: true),
            ])
        }
        return ScanResolveResponse(
            candidates: objects.flatMap(\.candidates), unresolvedIndices: [],
            objects: objects, latencyMs: 1)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}
