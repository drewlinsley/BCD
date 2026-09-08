import Foundation

// Catalog names are built for matching, not for reading.
//
// Producers arrive from TTB permits and Open Food Facts brand strings, so they carry
// their legal wrapper ("The Alchemist LLC"). Products are often stored brand-first so a
// trigram search can find them ("The Alchemist Heady Topper"), which means a detail
// screen that prints both says the same words twice.
//
// This is presentation only. Nothing here is written back to the store — the long forms
// are what the resolver matches against, and shortening them there would cost recall.
public enum DisplayName {

    /// Legal-entity wrappers. Trade words ("Brewing", "Distillery") are deliberately NOT
    /// here: "Sierra Nevada Brewing Co" should read "Sierra Nevada Brewing", not
    /// "Sierra Nevada".
    static let legalSuffixes: Set<String> = [
        "llc", "l.l.c", "inc", "ltd", "limited", "co", "company", "corp", "corporation",
        "plc", "gmbh", "bv", "nv", "sa", "srl", "spa", "ag", "kg", "ab", "aps", "oy",
        "pty", "llp", "holdings", "group", "international",
    ]

    /// Words that name a category rather than a drink. Stripping the brand off "Titos
    /// Vodka" leaves "Vodka", which identifies nothing — so these block the strip.
    static let categoryWords: Set<String> = [
        "vodka", "gin", "rum", "whisky", "whiskey", "bourbon", "scotch", "rye", "tequila",
        "mezcal", "ouzo", "brandy", "cognac", "armagnac", "liqueur", "schnapps", "absinthe",
        "aquavit", "grappa", "sake", "mead", "cider", "perry", "beer", "ale", "lager",
        "pilsner", "stout", "porter", "ipa", "wine", "seltzer", "hard", "spirit", "spirits",
        "original", "classic", "reserve", "select",
    ]

    /// "The Alchemist LLC" -> "The Alchemist". Strips repeatedly, so "Foo Brewing Co Ltd"
    /// loses both, but never strips a name down to nothing.
    public static func producer(_ raw: String) -> String {
        var parts = tokens(raw)
        while parts.count > 1, legalSuffixes.contains(normalize(parts[parts.count - 1])) {
            parts.removeLast()
        }
        let joined = parts.joined(separator: " ").trimmingCharacters(in: punctuationAndSpace)
        return joined.isEmpty ? raw : joined
    }

    /// "The Alchemist Heady Topper" shown under The Alchemist -> "Heady Topper".
    ///
    /// Refuses when what's left doesn't identify anything on its own: "Plomari Ouzo" keeps
    /// its brand, because "Ouzo" on a label under the word Plomari tells you less than the
    /// duplication costs.
    public static func product(_ raw: String, producer producerName: String) -> String {
        let name = tokens(raw)
        let brand = tokens(producer(producerName))
        guard !brand.isEmpty, name.count > brand.count else { return raw }
        guard zip(name, brand).allSatisfy({ normalize($0.0) == normalize($0.1) }) else { return raw }

        let rest = Array(name.dropFirst(brand.count))
        guard rest.contains(where: { !categoryWords.contains(normalize($0)) }) else { return raw }
        let joined = rest.joined(separator: " ").trimmingCharacters(in: punctuationAndSpace)
        return joined.isEmpty ? raw : joined
    }

    // MARK: -

    static let punctuationAndSpace = CharacterSet.punctuationCharacters
        .union(.whitespacesAndNewlines)

    static func tokens(_ raw: String) -> [String] {
        raw.split(whereSeparator: \.isWhitespace).map(String.init)
    }

    /// Case- and punctuation-insensitive form used only for comparison, never for display.
    static func normalize(_ token: String) -> String {
        token.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: nil)
            .trimmingCharacters(in: punctuationAndSpace)
    }
}
