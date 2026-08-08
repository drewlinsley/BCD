import Foundation

// On-device LLM provider, compiled only where Apple's Foundation Models framework exists
// (iOS 26+ / macOS 26+ on Apple Intelligence hardware). Everywhere else — including this
// Intel Mac and the Simulator — the file is inert and the app falls back to
// MockLLMProvider or CloudLLMProvider. This is the "optimization, never a dependency"
// principle made literal.

#if canImport(FoundationModels)
import FoundationModels

@available(iOS 26.0, macOS 26.0, *)
public struct FoundationModelsProvider: LLMProvider {
    public init() {}

    public var isAvailable: Bool {
        SystemLanguageModel.default.availability == .available
    }

    public func parseQuery(_ text: String) async throws -> QueryIntent {
        // Guided generation into our Codable-adjacent shape. If the model is unavailable
        // (no Apple Intelligence, downloading, etc.) fall back to the rule-based parser.
        guard isAvailable else { return try await MockLLMProvider().parseQuery(text) }
        let session = LanguageModelSession()
        let prompt = """
        Extract a drink filter from this request. Reply with fields only.
        Request: \(text)
        """
        let response = try await session.respond(to: prompt)
        return Self.parseLoose(response.content, original: text)
    }

    public func rerank(_ candidates: [ScoredCandidate], for ask: String) async throws -> [String] {
        guard isAvailable else { return try await MockLLMProvider().rerank(candidates, for: ask) }
        let menu = candidates.map { "\($0.resolved.product.id): \($0.resolved.product.name)" }
            .joined(separator: "\n")
        let session = LanguageModelSession()
        let prompt = """
        The user asked: "\(ask)"
        Rank these drinks best-first for them. Reply with ids, one per line.
        \(menu)
        """
        let response = try await session.respond(to: prompt)
        let ids = Set(candidates.map { $0.resolved.product.id })
        let ranked = response.content.split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { ids.contains($0) }
        // Append anything the model dropped, preserving determinism.
        return ranked + candidates.map { $0.resolved.product.id }.filter { !ranked.contains($0) }
    }

    private static func parseLoose(_ content: String, original: String) -> QueryIntent {
        var intent = QueryIntent(freeText: original)
        let lower = content.lowercased()
        if let m = lower.firstMatch(of: #/max_?abv[:=]\s*(\d+(?:\.\d+)?)/#),
           let v = Double(m.1) {
            intent.maxAbv = v
        }
        return intent
    }
}
#endif
