import Foundation

/// A drink a user told us about — the one place the app *authors* catalog data rather than
/// reading it. It is reached from the scan HUD's empty-state, when the camera read a label
/// plainly but nothing could place it (`ScanCoordinator.isUnknownLabel`), so this turns a dead
/// end into a contribution.
///
/// Deliberately small. A name and a category are all that is required; a maker, a strength and a
/// note are offered because the camera has often already read them and they are the fields a
/// human curating the row would most want. Nothing here is a `Product`: a contribution is a claim
/// *towards* one, not a catalog row, and it carries no id the registry would recognise.
///
/// `sightings` is what the camera read off the label at the moment the user tapped add — not
/// shown back, not required, kept so a later catalog match (or a person reviewing the
/// contribution) can be checked against what the phone actually saw. It is the same honest
/// record the vision path logs, carried here for the one label the catalog missed.
public struct DrinkContribution: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let name: String
    public let category: Category
    public let maker: String?
    public let abvPct: Double?
    public let note: String?
    public let sightings: [String]
    public let createdAt: Date

    public init(id: String = UUID().uuidString, name: String, category: Category,
                maker: String? = nil, abvPct: Double? = nil, note: String? = nil,
                sightings: [String] = [], createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.category = category
        self.maker = maker
        self.abvPct = abvPct
        self.note = note
        self.sightings = sightings
        self.createdAt = createdAt
    }
}

/// Disk-backed list of the drinks this install has contributed, newest first.
///
/// Unlike `SeenLog`, this is *not* a convenience queue rebuilt from the catalog — it is the
/// user's own authored data, so until a contribution has been uploaded this log is the record of
/// truth, and losing it loses something they typed. `ContributionUploader` drains it into
/// `POST /v1/contribute` and removes an entry only once the server has confirmed it — which is
/// why entries carry a stable `id`: it is what makes a retry land as one contribution, not two.
///
/// Consent does not gate it. The user explicitly asked to add this drink, so it is kept whatever
/// their telemetry tiers say — and the analytics breadcrumb that records a contribution happened
/// carries none of its free text, only its shape.
public final class ContributionLog: @unchecked Sendable {
    private let key = "bcd.contributions"
    private let defaults: UserDefaults
    private let limit: Int
    private let lock = NSLock()

    public init(defaults: UserDefaults = .standard, limit: Int = 500) {
        self.defaults = defaults
        self.limit = limit
    }

    public func all() -> [DrinkContribution] {
        lock.lock(); defer { lock.unlock() }
        return load()
    }

    public var count: Int {
        lock.lock(); defer { lock.unlock() }
        return load().count
    }

    /// Newest first. A contribution is never merged with an earlier one: two cans the catalog
    /// was missing are two contributions even if the user happened to type the same name twice,
    /// because the dedup belongs to whoever ingests these, with the whole catalog to check
    /// against — not to the phone, which has only what was typed.
    public func add(_ contribution: DrinkContribution) {
        lock.lock(); defer { lock.unlock() }
        var items = load()
        items.insert(contribution, at: 0)
        save(Array(items.prefix(limit)))
    }

    public func remove(_ id: String) {
        lock.lock(); defer { lock.unlock() }
        save(load().filter { $0.id != id })
    }

    private func load() -> [DrinkContribution] {
        guard let data = defaults.data(forKey: key),
              let items = try? JSONDecoder().decode([DrinkContribution].self, from: data)
        else { return [] }
        return items
    }

    private func save(_ items: [DrinkContribution]) {
        guard let data = try? JSONEncoder().encode(items) else { return }
        defaults.set(data, forKey: key)
    }
}

// MARK: - the wire

/// What `POST /v1/contribute` takes.
///
/// A separate shape from `DrinkContribution` on purpose. The model is also what sits in
/// `UserDefaults`, so giving it snake_case `CodingKeys` would change the *stored* encoding, and
/// an update would then fail to decode contributions a drinker had already typed — losing
/// exactly the thing this log exists to protect. The wire is allowed to change; the disk is not.
struct ContributionWire: Encodable {
    let id: String
    let name: String
    let category: String
    let maker: String?
    let abvPct: Double?
    let note: String?
    let sightings: [String]
    let createdAt: String

    init(_ c: DrinkContribution) {
        id = c.id
        name = c.name
        category = c.category.rawValue
        maker = c.maker
        abvPct = c.abvPct
        note = c.note
        sightings = c.sightings
        // Written out as a string, not left to the encoder: the client sets no date strategy, so
        // a `Date` would go out as a float of seconds since 2001 into a field the server reads
        // as a timestamp.
        createdAt = ISO8601DateFormatter().string(from: c.createdAt)
    }

    enum CodingKeys: String, CodingKey {
        case id, name, category, maker, note, sightings
        case abvPct = "abv_pct"
        case createdAt = "created_at"
    }
}

/// The server's answer to a contribution.
///
/// `duplicate` is not an error. The phone keeps a contribution until an upload is confirmed, so a
/// response that never arrived means the next drain sends it again — and both answers mean the
/// same thing to the phone: recorded, stop keeping it.
public struct ContributionAck: Decodable, Sendable, Equatable {
    public let accepted: Bool
    public let docId: String
    public let duplicate: Bool

    public init(accepted: Bool, docId: String, duplicate: Bool = false) {
        self.accepted = accepted
        self.docId = docId
        self.duplicate = duplicate
    }

    enum CodingKeys: String, CodingKey {
        case accepted, duplicate
        case docId = "doc_id"
    }
}

// MARK: - the drain

/// Sends what the drinker typed to the server, and forgets it only once the server says it has it.
///
/// The ordering is the whole design: upload, *then* remove from the log. Removing first would
/// lose a contribution to a dropped connection, and the log is the record of truth precisely
/// because the server might not have it yet.
///
/// Oldest first, so the queue drains in the order things were seen rather than newest-first the
/// way the log reads for display.
public actor ContributionUploader {
    /// What one drain did. `remaining` is what is still in the log afterwards — including
    /// anything `refused`, because a refusal is not permission to delete what someone typed.
    public struct Report: Sendable, Equatable {
        public var uploaded = 0
        public var refused = 0
        public var remaining = 0

        public init(uploaded: Int = 0, refused: Int = 0, remaining: Int = 0) {
            self.uploaded = uploaded
            self.refused = refused
            self.remaining = remaining
        }
    }

    private let log: ContributionLog
    private let api: APIClientProtocol
    private var draining = false

    public init(log: ContributionLog, api: APIClientProtocol) {
        self.log = log
        self.api = api
    }

    /// Try to upload everything waiting. Safe to call often — on submit, and again whenever the
    /// app comes back — because a drain already running returns an empty report rather than
    /// sending the same contribution twice. The report is there for tests and for a screen that
    /// wants to show the queue; the call sites that just want it tried can ignore it.
    @discardableResult
    public func drain() async -> Report {
        guard !draining else { return Report() }
        draining = true
        defer { draining = false }

        var report = Report()
        for contribution in log.all().reversed() {
            do {
                _ = try await api.contribute(contribution)
                log.remove(contribution.id)
                report.uploaded += 1
            } catch let APIError.http(code) where Self.isRefusal(code) {
                // The server will never take this body, so retrying it would block everything
                // behind it forever. It stays on disk and is counted, not deleted.
                report.refused += 1
            } catch {
                // Unreachable, unauthorized, rate-limited, or a route that isn't there yet: all
                // of them mean "not now", and none of them is a reason to try the rest.
                break
            }
        }
        report.remaining = log.count
        return report
    }

    /// Only the codes that say *this content* is wrong. Not 401/403 (the token can be re-minted),
    /// not 429 (ask later), not 501 (a stub client), and nothing 5xx.
    static func isRefusal(_ code: Int) -> Bool { code == 400 || code == 422 }
}
