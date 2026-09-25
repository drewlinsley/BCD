import Foundation

/// The searches this install has run, most recent first.
///
/// A catalog of 534k rows is not browsable, and the registry files a lot of near-names, so a
/// search that worked is worth more than the words that produced it: getting back to a drink
/// you found last week should not mean remembering which of "other half citra", "ddh citra"
/// and "citra nelson" was the one that landed it.
///
/// Kept on the phone. These are what the person typed, which is a sharper thing than what
/// they scanned — it says what they were looking for rather than what was in front of them —
/// and nothing here needs a server to be useful.
public final class RecentSearches: @unchecked Sendable {
    private let key = "bcd.recent_searches"
    private let defaults: UserDefaults
    private let lock = NSLock()

    /// Long enough to hold a few weeks of ordinary use, short enough that the list stays a
    /// shortcut rather than a second screen to read.
    public static let limit = 12

    public init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    public func all() -> [String] {
        lock.lock(); defer { lock.unlock() }
        return defaults.stringArray(forKey: key) ?? []
    }

    /// Remember a query that was actually run. Matching is case- and space-insensitive so
    /// "Heady Topper" does not sit above "heady topper", and the spelling kept is the most
    /// recent one — the way the person last chose to write it.
    public func record(_ query: String) {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        var all = defaults.stringArray(forKey: key) ?? []
        all.removeAll { $0.caseInsensitiveCompare(trimmed) == .orderedSame }
        all.insert(trimmed, at: 0)
        defaults.set(Array(all.prefix(Self.limit)), forKey: key)
    }

    public func remove(_ query: String) {
        lock.lock(); defer { lock.unlock() }
        var all = defaults.stringArray(forKey: key) ?? []
        all.removeAll { $0.caseInsensitiveCompare(query) == .orderedSame }
        defaults.set(all, forKey: key)
    }

    public func clear() {
        lock.lock(); defer { lock.unlock() }
        defaults.removeObject(forKey: key)
    }
}
