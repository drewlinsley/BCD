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

    public func interpretLabels(_ ocrLines: [String]) async throws -> [String] {
        // Stylized cans (Heady Topper's logo, etc.) defeat plain OCR — VisionKit returns
        // fragments like "FADY TOPP", "DEALCHEMIST". The on-device model reads them in context
        // and names the product; that clean name is what the catalog match then hits.
        let fragments = ocrLines.map { $0.replacingOccurrences(of: "\n", with: " ") }
            .filter { !$0.isEmpty }
        guard isAvailable, !fragments.isEmpty else { return [] }
        let session = LanguageModelSession()
        // The example is a *shape*, not a product. This used to read `e.g. "Heady Topper The
        // Alchemist"`, and the model returned that string verbatim while the camera was on a
        // Focal Banger can -- the same frame's raw OCR reads "FOCAL BAN". A one-shot example
        // naming a real beer is the answer the model reaches for whenever the fragments are
        // too garbled to read, which is precisely when it gets asked.
        //
        // Nothing else here changes, and that is deliberate. Two attempts at rewording the
        // rest of it -- "use only words you can actually see", then "match by shape and sound
        // ... but do not answer with a famous drink the fragments do not resemble" -- each
        // silenced the model completely: two sessions, a hundred and twenty frames, not one
        // answer. Reading letters that are *not* there is the entire job, since a Heady Topper
        // can OCRs as "FADY TOPPE" and "ПУТОРРЕ", and every added caution reads as a reason to
        // say NONE. This is the wording that demonstrably worked, with only the example
        // replaced by its own shape.
        let prompt = """
        These are OCR fragments from ONE alcoholic-drink label (beer or spirits), possibly \
        garbled or partial. If you recognize the product, reply with just its brand and name \
        on a single line, like "<brand> <product name>". If you can't, reply exactly NONE.
        Fragments: \(fragments.joined(separator: " | "))
        """
        let response = try await session.respond(to: prompt)
        let text = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
        let firstLine = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        let guess = firstLine.trimmingCharacters(in: .whitespaces)
        if guess.isEmpty || guess.uppercased().contains("NONE") { return [] }
        return [guess]
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
