import Foundation

/// Routing seam for language-model calls. The app never imports a specific model — it
/// depends on this protocol, and the composition root picks an implementation:
///   - `FoundationModelsProvider` (iOS 26+, on-device, free/private/fast) for intent
///     parsing and reranking — NOT facts (the 3B model confidently hallucinates world
///     knowledge, so product facts always come from the backend).
///   - `CloudLLMProvider` (Claude / Gemini Flash) server-side for the cold path.
///   - `MockLLMProvider` for tests and previews.
///
/// On this Intel Mac the on-device model can't run in the Simulator, which is exactly why
/// this is a protocol with a cloud default — swapping providers is a one-line change.
public protocol LLMProvider: Sendable {
    /// Parse a free-text HUD query ("cheapest hazy here", "nothing over 6%") into a
    /// structured filter the client applies to in-frame candidates.
    func parseQuery(_ text: String) async throws -> QueryIntent

    /// Rerank candidates given a natural-language ask. Returns product ids best-first.
    func rerank(_ candidates: [ScoredCandidate], for ask: String) async throws -> [String]

    /// Interpret a frame's raw OCR lines — often garbled off a stylized label — into a small
    /// set of clean product/brand guesses to match against the catalog. Returns [] when nothing
    /// reads like a drink. This is the fallback for labels that defeat plain OCR matching:
    /// on-device Apple Intelligence when available, a no-op elsewhere.
    func interpretLabels(_ ocrLines: [String]) async throws -> [String]

    /// Adjudicate an `ambiguous` object: given what the camera read and the server's
    /// shortlist, pick the entry the label is — or nil when none clearly is. Constrained on
    /// purpose: the model chooses among catalog rows the evidence already surfaced, so it
    /// cannot invent a beer the way a free-form guess can.
    func pickProduct(ocr texts: [String], candidates: [ScoredCandidate]) async throws -> ProductPick?
}

public extension LLMProvider {
    // Default: no interpretation. Providers without a real model (mock, cloud-less) inherit
    // this, so the shutter's fallback simply finds nothing rather than failing to compile.
    func interpretLabels(_ ocrLines: [String]) async throws -> [String] { [] }

    // Default: no opinion. A requirement rather than only an extension method so a provider
    // reached through the protocol (the coordinator holds `LLMProvider?`) dispatches to its
    // own implementation instead of this one.
    func pickProduct(ocr texts: [String], candidates: [ScoredCandidate]) async throws -> ProductPick? {
        nil
    }
}

public struct QueryIntent: Codable, Sendable, Equatable {
    public var maxAbv: Double?
    public var minAbv: Double?
    public var styleContains: String?
    public var maxPrice: Double?
    public var sortBy: SortKey
    public var freeText: String?

    public enum SortKey: String, Codable, Sendable { case personal, price, abv, relevance }

    public init(maxAbv: Double? = nil, minAbv: Double? = nil, styleContains: String? = nil,
                maxPrice: Double? = nil, sortBy: SortKey = .personal, freeText: String? = nil) {
        self.maxAbv = maxAbv; self.minAbv = minAbv; self.styleContains = styleContains
        self.maxPrice = maxPrice; self.sortBy = sortBy; self.freeText = freeText
    }
}

/// Deterministic provider: a tiny rule-based parser good enough for tests, previews, and
/// an offline fallback when no model is reachable. Real providers override with an LLM.
public struct MockLLMProvider: LLMProvider {
    public init() {}

    public func parseQuery(_ text: String) async throws -> QueryIntent {
        var intent = QueryIntent(freeText: text)
        let lower = text.lowercased()
        // "nothing over 6%", "under 5 abv"
        if let pct = Self.firstPercent(in: lower),
           lower.contains("over") || lower.contains("under") || lower.contains("no more") {
            intent.maxAbv = pct
        }
        if lower.contains("cheap") || lower.contains("cheapest") { intent.sortBy = .price }
        for style in ["hazy", "ipa", "stout", "lager", "pilsner", "sour", "bourbon", "scotch"]
        where lower.contains(style) {
            intent.styleContains = style
            break
        }
        return intent
    }

    public func rerank(_ candidates: [ScoredCandidate], for ask: String) async throws -> [String] {
        // Fallback ranking: honor the parsed sort, else personal score.
        let intent = try await parseQuery(ask)
        let filtered = candidates.filter { c in
            if let maxAbv = intent.maxAbv,
               let abv = c.resolved.product.spec.abvPct?.value, abv > maxAbv { return false }
            if let style = intent.styleContains,
               !(c.resolved.product.style?.value.lowercased().contains(style) ?? false),
               !c.resolved.product.name.lowercased().contains(style) { return false }
            return true
        }
        return filtered
            .sorted { ($0.personalScore ?? 0) > ($1.personalScore ?? 0) }
            .map { $0.resolved.product.id }
    }

    private static func firstPercent(in s: String) -> Double? {
        // grabs the number in "6%", "6 %", "6 abv"
        let pattern = #/(\d+(?:\.\d+)?)\s*%?/#
        if let m = s.firstMatch(of: pattern) { return Double(m.1) }
        return nil
    }
}

// MARK: - constrained pick

/// The model's answer when asked to choose among the server's shortlist for one object.
public struct ProductPick: Sendable, Equatable {
    public let productId: String
    public let confidence: Double
    public init(productId: String, confidence: Double) {
        self.productId = productId; self.confidence = confidence
    }
}

public extension MockLLMProvider {
    /// Deterministic stand-in: the entry whose name the fragments cover best, if it covers
    /// most of it and clearly beats the runner-up. One misread letter per word is tolerated.
    func pickProduct(ocr texts: [String], candidates: [ScoredCandidate]) async throws -> ProductPick? {
        let ocrTokens = Set(texts.flatMap(LabelText.tokens))
        guard !ocrTokens.isEmpty, !candidates.isEmpty else { return nil }
        let scored: [(String, Double)] = candidates.map { c in
            let name = LabelText.tokens(c.resolved.product.name)
            // A name has to be *read*, not merely brushed: some word of substance in it must
            // match, or a row called `Top's` is picked off "FADY TOP" on the strength of a
            // three-letter coincidence — the very row the server refused to resolve.
            guard LabelText.readsASubstantialWord(of: name, in: ocrTokens) else {
                return (c.resolved.product.id, 0)
            }
            return (c.resolved.product.id, LabelText.coverage(of: name, by: ocrTokens))
        }.sorted { $0.1 > $1.1 }
        let best = scored[0]
        let runner = scored.count > 1 ? scored[1].1 : 0
        guard best.1 >= 0.6, best.1 - runner >= 0.2 else { return nil }
        return ProductPick(productId: best.0, confidence: best.1)
    }
}

/// Word-level helpers shared by the mock adjudicator and tests.
public enum LabelText {
    static let generic: Set<String> = [
        "ipa", "ale", "beer", "lager", "stout", "porter", "pilsner", "sour", "hazy", "double",
        "imperial", "india", "pale", "brewing", "brewery", "brewed", "company", "co", "the",
        "and", "of", "can", "bottle", "oz", "ml", "abv", "vol", "alc",
    ]

    public static func tokens(_ s: String) -> [String] {
        s.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
            .filter { $0.count >= 2 && !generic.contains($0) }
    }

    /// Fraction of `name`'s tokens that some OCR token matches (exactly, or within one
    /// edit for words of four or more letters).
    public static func coverage(of name: [String], by ocr: Set<String>) -> Double {
        guard !name.isEmpty else { return 0 }
        let hit = name.filter { n in
            ocr.contains(n) || (n.count >= 4 && ocr.contains { editDistance($0, n) <= 1 })
        }
        return Double(hit.count) / Double(name.count)
    }

    /// Whether some word of four or more letters in `name` was read (within one edit).
    public static func readsASubstantialWord(of name: [String], in ocr: Set<String>) -> Bool {
        name.contains { n in n.count >= 4 && ocr.contains { editDistance($0, n) <= 1 } }
    }

    static func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        guard abs(a.count - b.count) <= 1 else { return 2 }
        var prev = Array(0...b.count)
        for i in 1...max(a.count, 1) where i <= a.count {
            var cur = [i] + Array(repeating: 0, count: b.count)
            for j in 1...max(b.count, 1) where j <= b.count {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            }
            prev = cur
        }
        return prev[b.count]
    }
}
