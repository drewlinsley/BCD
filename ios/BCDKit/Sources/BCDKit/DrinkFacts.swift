import Foundation

// Something a drinker knows about a drink the catalog already has.
//
// `DrinkContribution` covers the drink nothing could place. This covers the far commoner case:
// the drink is on the screen and a field on it is blank. 517,171 of 534,104 products carry no
// strength, the label prints a question mark over every one of them, and until now that mark
// could not be answered by anybody — the only way to tell us anything was to describe a drink we
// already held (user, 2026-10-16).
//
// Two things make it different from a contribution, and both follow from the same fact — that
// the row already exists:
//
//   * it names a `productId`, so it can be attached rather than promoted into existence;
//   * it is KEPT after it has been uploaded. A contribution is a queue and is forgotten once the
//     server has it; this log is also the record of what you told us, because the card shows
//     your own number back to you until the catalog has earned one of its own.

public struct DrinkFact: Codable, Sendable, Identifiable, Equatable {
    /// Which blank is being filled. Keyed on rather than implied, because agreement is counted
    /// per product and per field: two accounts saying 5.2% is a fact earned, two accounts
    /// saying unrelated things is nothing. One member today; a second adds a case here and on
    /// the server's `FactField`, and old documents keep meaning what they meant.
    public enum Field: String, Codable, Sendable, CaseIterable {
        case abvPct
    }

    public let id: String
    public let productId: String
    public let field: Field
    /// A number because the one field this carries is a strength. A fact whose value is text
    /// will carry its own, rather than widening this into a bag that holds anything.
    public let value: Double
    public let createdAt: Date
    /// Whether the server has it. Not a reason to delete it -- see the note above.
    public var uploaded: Bool

    public init(id: String = UUID().uuidString, productId: String, field: Field = .abvPct,
                value: Double, createdAt: Date = Date(), uploaded: Bool = false) {
        self.id = id
        self.productId = productId
        self.field = field
        self.value = value
        self.createdAt = createdAt
        self.uploaded = uploaded
    }
}

/// What this install has told us about rows the catalog already holds, newest first.
///
/// `ObservableObject` with a revision counter, the same shape `ReactionLog` uses and for the
/// same reason: the label card reads it on every draw, and it has to redraw the moment a
/// strength is given. A counter rather than the facts themselves, so there is only ever one
/// copy of the answer.
public final class DrinkFactLog: ObservableObject, @unchecked Sendable {
    private let key = "bcd.drinkFacts"
    private let defaults: UserDefaults
    private let limit: Int
    private let lock = NSLock()

    @Published public private(set) var revision = 0

    public init(defaults: UserDefaults = .standard, limit: Int = 500) {
        self.defaults = defaults
        self.limit = limit
    }

    public func all() -> [DrinkFact] {
        lock.lock(); defer { lock.unlock() }
        return load()
    }

    /// What this install said about one field of one drink, if anything. The newest wins: a
    /// drinker who corrects themselves meant the correction.
    public func value(for productId: String, field: DrinkFact.Field = .abvPct) -> Double? {
        lock.lock(); defer { lock.unlock() }
        return load().first { $0.productId == productId && $0.field == field }?.value
    }

    /// Say it. A second answer about the same field of the same drink replaces the first rather
    /// than queueing beside it -- two numbers from one account are not two accounts agreeing,
    /// and the older one is a thing they have changed their mind about.
    public func record(_ fact: DrinkFact) {
        write { items in
            items.removeAll { $0.productId == fact.productId && $0.field == fact.field }
            items.insert(fact, at: 0)
        }
    }

    public func pending() -> [DrinkFact] {
        lock.lock(); defer { lock.unlock() }
        return load().filter { !$0.uploaded }.reversed()   // oldest first, as the queue drains
    }

    public func markUploaded(_ id: String) {
        write { items in
            guard let i = items.firstIndex(where: { $0.id == id }) else { return }
            items[i].uploaded = true
        }
    }

    /// Take one back. Nothing in the app calls this yet; it is here because a log the drinker
    /// cannot empty is the thing `remove(for:)` had to be added to `ReactionLog` for.
    public func forget(_ id: String) {
        write { items in items.removeAll { $0.id == id } }
    }

    private func load() -> [DrinkFact] {
        guard let data = defaults.data(forKey: key),
              let items = try? JSONDecoder().decode([DrinkFact].self, from: data) else { return [] }
        return items
    }

    private func write(_ change: (inout [DrinkFact]) -> Void) {
        lock.lock()
        var items = load()
        change(&items)
        if items.count > limit { items = Array(items.prefix(limit)) }
        if let data = try? JSONEncoder().encode(items) { defaults.set(data, forKey: key) }
        lock.unlock()
        revision += 1
    }
}

// MARK: - the wire

/// What `POST /v1/contribute/fact` takes.
///
/// Its own shape, for the reason `ContributionWire` has one: `DrinkFact` is what sits in
/// `UserDefaults`, so snake_case `CodingKeys` on it would change the stored encoding and an
/// update would fail to decode what a drinker had already told us. The wire may change; the
/// disk may not.
struct DrinkFactWire: Encodable {
    let id: String
    let productId: String
    let field: String
    let value: Double
    let createdAt: String

    init(_ f: DrinkFact) {
        id = f.id
        productId = f.productId
        // The server's spelling, which is the schema's `FactField`. Not the Swift case name:
        // `abvPct` would be refused, and a 422 on a field name is the kind of thing that is
        // only ever found on a device.
        field = {
            switch f.field {
            case .abvPct: "abv_pct"
            }
        }()
        value = f.value
        // Written as a string for the same reason the contribution's is: the client sets no
        // date strategy, so a `Date` goes out as seconds since 2001 into a timestamp field.
        createdAt = ISO8601DateFormatter().string(from: f.createdAt)
    }

    enum CodingKeys: String, CodingKey {
        case id, field, value
        case productId = "product_id"
        case createdAt = "created_at"
    }
}

// MARK: - the drain

/// Sends what the drinker knows, and marks it sent -- it does not remove it.
///
/// The one difference from `ContributionUploader`, and the whole reason this is its own type:
/// a contribution is forgotten once the server has it, and a fact is not, because the card
/// goes on showing the drinker their own answer until the catalog earns one.
public actor DrinkFactUploader {
    public struct Report: Sendable, Equatable {
        public var uploaded = 0
        public var refused = 0
        /// Still waiting, including anything refused: a refusal is not permission to forget
        /// what somebody told us.
        public var pending = 0
    }

    private let log: DrinkFactLog
    private let api: APIClientProtocol

    public init(log: DrinkFactLog, api: APIClientProtocol) {
        self.log = log
        self.api = api
    }

    @discardableResult
    public func drain() async -> Report {
        var report = Report()
        for fact in log.pending() {
            do {
                let ack = try await api.contributeFact(fact)
                // `duplicate` is not a failure: the phone may have uploaded this already and
                // lost the answer, and both answers mean the server has it.
                if ack.accepted { log.markUploaded(fact.id); report.uploaded += 1 }
                else { report.refused += 1 }
            } catch {
                report.refused += 1
            }
        }
        report.pending = log.pending().count
        return report
    }
}
