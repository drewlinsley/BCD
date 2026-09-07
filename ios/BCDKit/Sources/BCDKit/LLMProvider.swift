import Foundation

/// Routing seam for language-model calls. The app never imports a specific model — it
/// depends on this protocol, and the composition root picks an implementation:
///   - `FoundationModelsProvider` (iOS 26+, on-device, free/private/fast) for intent
///     parsing, reranking and *constrained* label adjudication — NOT facts (the 3B model
///     confidently hallucinates world knowledge, so product facts always come from the
///     backend, and the model is only ever asked to choose among catalog candidates).
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

    /// The fine stage's last resort. Given the OCR fragments read off one object and the
    /// server's shortlist for it, pick the candidate the label is — or nil if none is
    /// convincing. The model can only answer with one of the given ids; it can't invent
    /// a beer, which is the whole point.
    func pickProduct(ocr texts: [String], candidates: [ScoredCandidate]) async throws -> ProductPick?
}

public struct ProductPick: Sendable, Equatable {
    public let productId: String
    public let confidence: Double
    public init(productId: String, confidence: Double) {
        self.productId = productId; self.confidence = confidence
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

    /// Rule-based adjudication: how much of each candidate's *name* the OCR accounts for,
    /// tolerating one-character misreads on longer words and ignoring generic label words.
    /// Picks only with a clear winner. This is also the offline fallback on devices
    /// without Apple Intelligence.
    public func pickProduct(ocr texts: [String], candidates: [ScoredCandidate]) async throws -> ProductPick? {
        let ocrTokens = Set(texts.flatMap(LabelText.tokens))
        guard !ocrTokens.isEmpty, !candidates.isEmpty else { return nil }
        let scored: [(String, Double)] = candidates.map { c in
            (c.resolved.product.id,
             LabelText.coverage(of: LabelText.tokens(c.resolved.product.name), by: ocrTokens))
        }.sorted { $0.1 > $1.1 }
        let best = scored[0]
        let runner = scored.count > 1 ? scored[1].1 : 0
        guard best.1 >= 0.6, best.1 - runner >= 0.2 else { return nil }
        return ProductPick(productId: best.0, confidence: best.1)
    }

    private static func firstPercent(in s: String) -> Double? {
        // grabs the number in "6%", "6 %", "6 abv"
        let pattern = #/(\d+(?:\.\d+)?)\s*%?/#
        if let m = s.firstMatch(of: pattern) { return Double(m.1) }
        return nil
    }
}

/// Small label-text helpers shared by the rule-based adjudicator and tests. Mirrors the
/// spirit of `bcd_api.matching` on the server (generic words weigh little, short OCR
/// misreads are tolerated) without trying to be the same scorer.
public enum LabelText {
    static let generic: Set<String> = [
        "ipa", "ale", "beer", "lager", "stout", "porter", "pilsner", "sour", "hazy", "double",
        "imperial", "india", "pale", "brewing", "brewery", "brewed", "company", "co", "the",
        "and", "of", "by", "fl", "oz", "ml", "abv", "alc", "vol", "alcohol", "volume",
    ]

    public static func tokens(_ s: String) -> [String] {
        s.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count >= 2 }
    }

    /// Weighted fraction of `target` tokens present in `query` (exact, or within one edit
    /// for words of 4+ characters).
    public static func coverage(of target: [String], by query: Set<String>) -> Double {
        guard !target.isEmpty else { return 0 }
        var total = 0.0, hit = 0.0
        for t in target {
            let w = generic.contains(t) || t.allSatisfy(\.isNumber) ? 0.25 : 1.0
            total += w
            if query.contains(t) { hit += w; continue }
            if t.count >= 4, query.contains(where: { $0.count >= 4 && editDistance($0, t) <= 1 }) {
                hit += w * 0.9
            }
        }
        return total > 0 ? hit / total : 0
    }

    static func editDistance(_ a: String, _ b: String) -> Int {
        let a = Array(a), b = Array(b)
        if abs(a.count - b.count) > 1 { return 2 }
        guard !a.isEmpty else { return b.count }
        guard !b.isEmpty else { return a.count }
        var prev = Array(0...b.count)
        for i in 1...a.count {
            var cur = [i] + Array(repeating: 0, count: b.count)
            for j in 1...b.count {
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                             prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1))
            }
            prev = cur
        }
        return prev[b.count]
    }
}
