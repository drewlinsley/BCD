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
    ///
    /// Styles too: "Bombay Sapphire London Dry Gin" under Bombay Sapphire was shown as
    /// "London Dry Gin", which is what it is, not what it is called — reported from the
    /// camera as the detail screen showing "the wrong name" (2026-09-16).
    static let categoryWords: Set<String> = [
        "vodka", "gin", "rum", "whisky", "whiskey", "bourbon", "scotch", "rye", "tequila",
        "mezcal", "ouzo", "brandy", "cognac", "armagnac", "liqueur", "schnapps", "absinthe",
        "aquavit", "grappa", "sake", "mead", "cider", "perry", "beer", "ale", "lager",
        "pilsner", "stout", "porter", "ipa", "wine", "seltzer", "hard", "spirit", "spirits",
        "original", "classic", "reserve", "select",
        "london", "dry", "extra", "special", "premium", "pale", "india", "indian", "double",
        "imperial", "hazy", "session", "draught", "draft", "light", "lite", "strong", "blonde",
        "amber", "dark", "white", "black", "gold", "red", "aperitivo", "bitter", "bitters",
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

    /// The name a label prints: the product's, with its brand in front when the name does
    /// not carry it. Open Food Facts files the brand apart from the name, so a bottle of
    /// Tito's came up as "Handmade Vodka" -- reported from the camera as "too generic"
    /// (2026-09-17) -- and the same rows would show "London Dry Gin" or "Vodka" alone.
    /// A name that already holds the brand's first word anywhere ("Bitter Campari" under
    /// Campari, "The Alchemist Heady Topper") is left as it is; a placeholder brand (the
    /// resolver names it after the product when the catalog has none) adds nothing.
    public static func label(_ raw: String, brand: String) -> String {
        let brandTokens = tokens(brand).map(comparable).filter { !$0.isEmpty }
        guard let first = brandTokens.first, brand.lowercased() != "unknown" else { return raw }
        let nameTokens = tokens(raw).map(comparable)
        if nameTokens.contains(first) { return raw }
        let joined = (brand.trimmingCharacters(in: .whitespacesAndNewlines) + " " + raw)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.isEmpty ? raw : joined
    }

    /// `normalize`, with the apostrophes gone too: "Tito's", "Tito’s" and "Titos" are one
    /// word to a label.
    static func comparable(_ token: String) -> String {
        normalize(token).filter { $0 != "'" && $0 != "’" && $0 != "`" }
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
