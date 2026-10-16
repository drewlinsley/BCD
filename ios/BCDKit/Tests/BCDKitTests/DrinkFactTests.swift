import Testing
import Foundation
@testable import BCDKit

// Telling us something about a drink the catalog already has. The distinguishing rule, which
// everything here is about: a contribution is forgotten once the server has it, and a fact is
// not — the card shows the drinker their own answer until the catalog earns one.

private func freshLog() -> DrinkFactLog {
    let suite = UserDefaults(suiteName: "bcd.facts.\(UUID().uuidString)")!
    return DrinkFactLog(defaults: suite)
}

/// Accepts everything, and remembers what it was handed.
private actor AcceptingFacts: APIClientProtocol {
    private(set) var sent: [DrinkFact] = []
    func contributeFact(_ fact: DrinkFact) async throws -> ContributionAck {
        sent.append(fact)
        return ContributionAck(accepted: true, docId: "doc:\(fact.id)")
    }
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        throw APIError.http(501)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

/// Refuses everything, so a failed drain can be told from a successful one.
private actor RefusingFacts: APIClientProtocol {
    func contributeFact(_ fact: DrinkFact) async throws -> ContributionAck {
        throw APIError.http(503)
    }
    func resolveScan(_ req: ScanResolveRequest) async throws -> ScanResolveResponse {
        throw APIError.http(501)
    }
    func searchProducts(_ query: String) async throws -> [ResolvedProduct] { [] }
    func sendTelemetry(_ batch: TelemetryBatch) async throws {}
}

@Suite struct WhatThisPhoneToldUs {
    @Test func aStrengthComesBackForTheDrinkItWasAbout() {
        let log = freshLog()
        log.record(DrinkFact(productId: "ttb:1", value: 5.2))
        #expect(log.value(for: "ttb:1") == 5.2)
        #expect(log.value(for: "ttb:2") == nil, "a fact is about one row, not about the catalog")
    }

    @Test func changingYourMindReplacesRatherThanQueues() {
        // Two numbers from one account are not two accounts agreeing, and the older one is
        // something they have changed their mind about.
        let log = freshLog()
        log.record(DrinkFact(productId: "ttb:1", value: 5.2))
        log.record(DrinkFact(productId: "ttb:1", value: 6.4))
        #expect(log.value(for: "ttb:1") == 6.4)
        #expect(log.all().filter { $0.productId == "ttb:1" }.count == 1)
    }

    @Test func twoDrinksAreTwoFacts() {
        let log = freshLog()
        log.record(DrinkFact(productId: "ttb:1", value: 5.2))
        log.record(DrinkFact(productId: "ttb:2", value: 8.0))
        #expect(log.all().count == 2)
        #expect(log.value(for: "ttb:1") == 5.2)
        #expect(log.value(for: "ttb:2") == 8.0)
    }

    @Test func theCardCanRedrawBecauseTheLogSaysItMoved() {
        let log = freshLog()
        let before = log.revision
        log.record(DrinkFact(productId: "ttb:1", value: 5.2))
        #expect(log.revision > before)
    }
}

@Suite struct AFactSurvivesBeingSent {
    @Test func uploadingMarksItAndKeepsIt() async {
        let log = freshLog()
        log.record(DrinkFact(productId: "ttb:1", value: 5.2))
        let api = AcceptingFacts()

        let report = await DrinkFactUploader(log: log, api: api).drain()

        #expect(report.uploaded == 1)
        #expect(report.pending == 0)
        #expect(await api.sent.count == 1)
        // The whole point: a contribution would be gone from the log by now.
        #expect(log.value(for: "ttb:1") == 5.2, "the card still has the drinker's own answer")
        #expect(log.all().first?.uploaded == true)
    }

    @Test func aRefusedFactIsStillOwed() async {
        let log = freshLog()
        log.record(DrinkFact(productId: "ttb:1", value: 5.2))

        let report = await DrinkFactUploader(log: log, api: RefusingFacts()).drain()

        #expect(report.uploaded == 0)
        #expect(report.refused == 1)
        #expect(report.pending == 1)
        #expect(log.pending().count == 1, "and the next drain will try it again")
    }

    @Test func onlyWhatHasNotBeenSentIsSent() async {
        let log = freshLog()
        log.record(DrinkFact(productId: "ttb:1", value: 5.2))
        let api = AcceptingFacts()
        await DrinkFactUploader(log: log, api: api).drain()
        await DrinkFactUploader(log: log, api: api).drain()
        #expect(await api.sent.count == 1, "a sent fact is not sent twice")
    }

    @Test func theQueueDrainsOldestFirst() {
        let log = freshLog()
        log.record(DrinkFact(productId: "ttb:1", value: 5.2))
        log.record(DrinkFact(productId: "ttb:2", value: 6.0))
        log.record(DrinkFact(productId: "ttb:3", value: 7.0))
        // The log reads newest-first for display; the drain goes in the order they were given.
        #expect(log.pending().map(\.productId) == ["ttb:1", "ttb:2", "ttb:3"])
        #expect(log.all().map(\.productId) == ["ttb:3", "ttb:2", "ttb:1"])
    }
}

@Suite struct TheFactWire {
    @Test func theFieldGoesOutInTheServersSpelling() throws {
        // `abvPct` would come back 422, and a 422 on a field name is the kind of thing only a
        // device ever finds.
        let wire = DrinkFactWire(DrinkFact(id: "f1", productId: "ttb:1", value: 5.2))
        let json = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(wire)) as! [String: Any]
        #expect(json["field"] as? String == "abv_pct")
        #expect(json["product_id"] as? String == "ttb:1")
        #expect(json["value"] as? Double == 5.2)
        #expect(json["created_at"] is String, "a timestamp, not seconds since 2001")
    }

    @Test func theDiskEncodingIsNotTheWireEncoding() throws {
        // `DrinkFact` is what sits in UserDefaults. If it ever learned the wire's snake_case,
        // an update would stop decoding what a drinker had already told us.
        let json = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(DrinkFact(productId: "ttb:1", value: 5.2)))
            as! [String: Any]
        #expect(json["productId"] != nil)
        #expect(json["product_id"] == nil)
    }
}
