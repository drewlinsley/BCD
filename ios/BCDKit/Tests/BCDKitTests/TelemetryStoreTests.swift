import Testing
import Foundation
@testable import BCDKit

// The queue says it is disk-backed and that events are never dropped. Until 2026-10-16 no
// test had ever passed it a `storeURL`, and on iOS it wrote into `Library/Application
// Support`, which does not exist until something creates it -- so the write failed through a
// `try?` on every event and the queue was in memory only. Everything queued died with the
// process, including the rating withdrawals the withdraw path hands it to deliver.

/// A sink that accepts whatever it is given, so `flush` can succeed.
private actor AcceptingSink: APIClientProtocol {
    private(set) var batches: [TelemetryBatch] = []
    func sendTelemetry(_ batch: TelemetryBatch) async throws { batches.append(batch) }
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        throw APIError.http(501)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func profile() async throws -> TasteProfile { TasteProfile(userId: "", version: 0) }
}

/// A sink that refuses, so a failed drain can be told from a successful one.
private actor RefusingSink: APIClientProtocol {
    func sendTelemetry(_ batch: TelemetryBatch) async throws { throw APIError.http(503) }
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        throw APIError.http(501)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func profile() async throws -> TasteProfile { TasteProfile(userId: "", version: 0) }
}

@Suite struct TheQueueIsActuallyOnDisk {
    /// A path inside a directory that does not exist, which is what the app hands it.
    private func freshStore() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("bcd-telemetry-\(UUID().uuidString)")
            .appendingPathComponent("Application Support")
            .appendingPathComponent("bcd_telemetry_queue.json")
    }

    private func everything(_ consent: Bool = true) -> ConsentState {
        ConsentState(analytics: true, personalization: consent)
    }

    @Test func aQueuedEventSurvivesTheProcess() async {
        let url = freshStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let first = TelemetryQueue(consent: everything(), storeURL: url)
        #expect(await first.log("rating_withdrawn", tier: .personalization,
                                ["product_id": .string("ttb:05200000000175")]))

        // The whole claim: a different queue reading the same file finds it. This failed
        // before the directory was created -- the file was never written at all.
        #expect(FileManager.default.fileExists(atPath: url.path), "the file is on disk")
        let next = TelemetryQueue(consent: everything(), storeURL: url)
        #expect(await next.pendingCount == 1)
    }

    @Test func theWithdrawalKeepsItsProductAndTier() async {
        let url = freshStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let queue = TelemetryQueue(consent: everything(), storeURL: url)
        await queue.log("rating_withdrawn", tier: .personalization,
                        ["product_id": .string("ttb:05200000000175")])

        let saved = try! JSONDecoder().decode([TelemetryEnvelope].self,
                                              from: Data(contentsOf: url))
        #expect(saved.count == 1)
        #expect(saved[0].name == "rating_withdrawn")
        #expect(saved[0].tier == .personalization)
        #expect(saved[0].properties["product_id"] == .string("ttb:05200000000175"))
    }

    @Test func aDrainTheCollectorRefusesKeepsTheEvent() async {
        let url = freshStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let queue = TelemetryQueue(consent: everything(), sink: RefusingSink(), storeURL: url)
        await queue.log("rating_withdrawn", tier: .personalization,
                        ["product_id": .string("ttb:1")])
        try? await queue.flush()

        #expect(await queue.pendingCount == 1, "a refused batch is still owed")
        let next = TelemetryQueue(consent: everything(), storeURL: url)
        #expect(await next.pendingCount == 1, "and still owed after a relaunch")
    }

    @Test func anAcceptedDrainLeavesNothingToSendTwice() async throws {
        let url = freshStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let sink = AcceptingSink()
        let queue = TelemetryQueue(consent: everything(), sink: sink, storeURL: url)
        await queue.log("rating_withdrawn", tier: .personalization,
                        ["product_id": .string("ttb:1")])
        try await queue.flush()

        #expect(await sink.batches.count == 1)
        #expect(await queue.pendingCount == 0)
        // And the emptying is written down too, or the next launch would send it again.
        let next = TelemetryQueue(consent: everything(), storeURL: url)
        #expect(await next.pendingCount == 0)
    }

    @Test func withoutConsentNothingIsWrittenDown() async {
        // A withdrawal from an install that never agreed to personalization has nothing to
        // withdraw on the server either, so the drop is right -- but it must be a real drop.
        let url = freshStore()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

        let queue = TelemetryQueue(consent: everything(false), storeURL: url)
        #expect(await queue.log("rating_withdrawn", tier: .personalization,
                                ["product_id": .string("ttb:1")]) == false)
        #expect(await queue.pendingCount == 0)
    }
}
