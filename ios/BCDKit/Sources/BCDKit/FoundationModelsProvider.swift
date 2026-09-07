import Foundation

// On-device LLM provider, compiled only where Apple's Foundation Models framework exists
// (iOS 26+ / macOS 26+ on Apple Intelligence hardware). Everywhere else — including this
// Intel Mac and the Simulator — the file is inert and the app falls back to
// MockLLMProvider or CloudLLMProvider. This is the "optimization, never a dependency"
// principle made literal.
//
// Note on scope: the iOS 26 SDK's model is text-only. Image input to Foundation Models
// (WWDC26) needs the iOS 27 SDK, which the reference machine can't build against. So
// the on-device model's role in the scan path is *adjudication over text*: given the OCR
// fragments and the server's shortlist, which catalog entry is this — or none. That is
// exactly the task a small model does well and, because it can only answer with a given
// id, exactly the task on which it cannot hallucinate a beer.

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

    public func pickProduct(ocr texts: [String], candidates: [ScoredCandidate]) async throws -> ProductPick? {
        guard isAvailable else {
            return try await MockLLMProvider().pickProduct(ocr: texts, candidates: candidates)
        }
        guard !candidates.isEmpty, !texts.isEmpty else { return nil }
        let menu = candidates.enumerated().map { i, c in
            let p = c.resolved.product
            let style = p.style.map { " (\($0.value))" } ?? ""
            return "\(i + 1). \(p.name) — \(c.resolved.producer.name)\(style)"
        }.joined(separator: "\n")
        let fragments = texts.map { "\"\($0)\"" }.joined(separator: ", ")
        let session = LanguageModelSession(instructions: """
            You match text that OCR read off a beer, cider or spirits label to a short list of
            catalog entries. Label typefaces are stylized, so letters may be missing, split or
            substituted (for example "Chemist" for "Alchemist", "Ready" for "Heady"). Consider
            producer names and beer names. Answer with only the number of the matching entry,
            or NONE if no entry is clearly the label. Never answer with a name.
            """)
        let prompt = """
        OCR fragments: \(fragments)
        Entries:
        \(menu)
        """
        let response = try await session.respond(to: prompt)
        let answer = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let m = answer.firstMatch(of: #/(\d+)/#), let n = Int(m.1),
              (1...candidates.count).contains(n) else { return nil }
        // The model gives no calibrated confidence; treat a constrained pick as moderately
        // confident and let the HUD mark it as model-adjudicated.
        return ProductPick(productId: candidates[n - 1].resolved.product.id, confidence: 0.7)
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
