import Foundation

// Client telemetry: a disk-backed, batched, offline-safe queue. Events are never dropped
// — they persist across launches until acknowledged by the collector. We run our own
// collector (not a vendor SDK) because the behavioral data is the monetizable asset.
//
// Consent gates what may be enqueued: an event tagged `.personalization` is silently
// dropped unless the user granted that tier. `pii` is never carried here by construction.

public enum ConsentTier: String, Codable, Sendable, CaseIterable, Comparable {
    case analytics, personalization, dataSharing = "data_sharing"

    private var order: Int {
        switch self { case .analytics: 0; case .personalization: 1; case .dataSharing: 2 }
    }
    public static func < (a: ConsentTier, b: ConsentTier) -> Bool { a.order < b.order }
}

public struct ConsentState: Codable, Sendable {
    public var analytics: Bool
    public var personalization: Bool
    public var dataSharing: Bool
    public init(analytics: Bool = false, personalization: Bool = false,
                dataSharing: Bool = false) {
        self.analytics = analytics; self.personalization = personalization
        self.dataSharing = dataSharing
    }
    public func allows(_ tier: ConsentTier) -> Bool {
        switch tier {
        case .analytics: analytics
        case .personalization: personalization
        case .dataSharing: dataSharing
        }
    }
}

public struct TelemetryEnvelope: Codable, Sendable {
    public let name: String
    public let tier: ConsentTier
    public let eventId: String
    public let ts: String
    public let sessionId: String
    public let properties: [String: TelemetryValue]

    enum CodingKeys: String, CodingKey {
        case name, tier, ts, properties
        case eventId = "event_id"
        case sessionId = "session_id"
    }
}

public struct TelemetryBatch: Codable, Sendable {
    public let events: [TelemetryEnvelope]
}

/// Minimal JSON-value box so event properties stay Codable without `Any`.
public enum TelemetryValue: Codable, Sendable, Equatable {
    case string(String), int(Int), double(Double), bool(Bool)
    case stringList([String])

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .int(let v): try c.encode(v)
        case .double(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .stringList(let v): try c.encode(v)
        }
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Int.self) { self = .int(v) }
        else if let v = try? c.decode(Double.self) { self = .double(v) }
        else if let v = try? c.decode([String].self) { self = .stringList(v) }
        else { self = .string(try c.decode(String.self)) }
    }
}

public actor TelemetryQueue {
    private var pending: [TelemetryEnvelope] = []
    private let consent: ConsentState
    private let sessionId: String
    private let flushThreshold: Int
    private let sink: APIClientProtocol?
    private let storeURL: URL?

    public init(consent: ConsentState, sessionId: String = UUID().uuidString,
                flushThreshold: Int = 20, sink: APIClientProtocol? = nil,
                storeURL: URL? = nil) {
        self.consent = consent
        self.sessionId = sessionId
        self.flushThreshold = flushThreshold
        self.sink = sink
        self.storeURL = storeURL
        if let storeURL, let data = try? Data(contentsOf: storeURL),
           let saved = try? JSONDecoder().decode([TelemetryEnvelope].self, from: data) {
            pending = saved
        }
    }

    /// Enqueue an event. Returns false if consent for its tier is absent (dropped).
    @discardableResult
    public func log(_ name: String, tier: ConsentTier,
                    _ properties: [String: TelemetryValue] = [:]) -> Bool {
        guard consent.allows(tier) else { return false }
        let env = TelemetryEnvelope(
            name: name, tier: tier, eventId: UUID().uuidString,
            ts: ISO8601DateFormatter().string(from: Date()),
            sessionId: sessionId, properties: properties
        )
        pending.append(env)
        persist()
        if pending.count >= flushThreshold { Task { try? await flush() } }
        return true
    }

    public func flush() async throws {
        guard !pending.isEmpty, let sink else { return }
        let batch = TelemetryBatch(events: pending)
        try await sink.sendTelemetry(batch)  // only clear after the collector accepts
        pending.removeAll()
        persist()
    }

    public var pendingCount: Int { pending.count }

    private func persist() {
        guard let storeURL, let data = try? JSONEncoder().encode(pending) else { return }
        try? data.write(to: storeURL, options: .atomic)
    }
}
