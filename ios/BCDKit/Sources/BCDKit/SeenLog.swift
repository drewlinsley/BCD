import Foundation

/// Drinks this install has actually looked at, newest first — the queue the rating surface
/// works through.
///
/// Recorded when a product's detail is *opened*, not when it merely crosses the scan HUD. A
/// live viewfinder sweeping a shelf resolves dozens of products a minute; opening one is a
/// deliberate act, and only deliberate ones are worth asking about later. If that proves too
/// narrow, widening it is a one-line change at the call site — nothing here assumes the
/// source.
public struct SeenProduct: Codable, Sendable, Identifiable, Equatable {
    public let id: String              // product id
    public let name: String
    public let producer: String
    public let abvPct: Double?
    public let seenAt: Date

    public init(id: String, name: String, producer: String, abvPct: Double? = nil,
                seenAt: Date = Date()) {
        self.id = id
        self.name = name
        self.producer = producer
        self.abvPct = abvPct
        self.seenAt = seenAt
    }
}

/// Disk-backed and capped. This is a convenience queue, not a record of truth — the server's
/// telemetry log remains the only durable history — so losing it costs the user nothing but a
/// list to work through.
public final class SeenLog: @unchecked Sendable {
    private let key = "bcd.seen"
    private let defaults: UserDefaults
    private let limit: Int
    private let lock = NSLock()

    public init(defaults: UserDefaults = .standard, limit: Int = 50) {
        self.defaults = defaults
        self.limit = limit
    }

    public func all() -> [SeenProduct] {
        lock.lock(); defer { lock.unlock() }
        return load()
    }

    /// Re-opening a product moves it back to the top rather than duplicating it.
    public func record(_ product: SeenProduct) {
        lock.lock(); defer { lock.unlock() }
        var items = load().filter { $0.id != product.id }
        items.insert(product, at: 0)
        save(Array(items.prefix(limit)))
    }

    public func remove(_ id: String) {
        lock.lock(); defer { lock.unlock() }
        save(load().filter { $0.id != id })
    }

    private func load() -> [SeenProduct] {
        guard let data = defaults.data(forKey: key),
              let items = try? JSONDecoder().decode([SeenProduct].self, from: data)
        else { return [] }
        return items
    }

    private func save(_ items: [SeenProduct]) {
        guard let data = try? JSONEncoder().encode(items) else { return }
        defaults.set(data, forKey: key)
    }
}
