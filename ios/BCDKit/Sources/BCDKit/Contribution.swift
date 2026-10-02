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
/// truth, and losing it loses something they typed. The server has no contribute route yet; when
/// it grows one, draining this log is the whole of the sync, which is why entries carry a stable
/// `id` and a timestamp.
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
