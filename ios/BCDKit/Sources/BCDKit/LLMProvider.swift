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
