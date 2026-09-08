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
        // Concrete examples, from producers nobody here is scanning.
        //
        // The example does more than anchor an answer: it teaches the answer's *shape*.
        // Replacing it with a placeholder ("<brand> <product name>") made the model stop naming
        // products and start echoing the input -- it replied "**HEMIST-VERME** **ALE**", then
        // "**CAN! DRINK FROM THEO**", and once handed the entire fragment list back. Across 124
        // frames of Heady Topper and Focal Banger it named nothing at all.
        //
        // Two examples rather than one, so neither becomes the default answer, and from an
        // unrelated brewery and distillery so that a parroted example cannot survive the
        // server: "Sierra Nevada Pale Ale" off an Alchemist can shares no word with the frame,
        // so nothing corroborates it and it is dropped. That safety net is exactly what the old
        // `Heady Topper The Alchemist` example defeated -- it names the same brewery as Focal
        // Banger, so the ALCHEMIST printed on the chrome corroborated the wrong beer.
        let prompt = """
        These are OCR fragments from ONE alcoholic-drink label (beer or spirits), possibly \
        garbled or partial. If you recognize the product, reply with just its brand and name \
        on a single line, e.g. "Sierra Nevada Pale Ale" or "Bombay Sapphire London Dry Gin". \
        If you can't, reply exactly NONE.
        Fragments: \(fragments.joined(separator: " | "))
        """
        let response = try await session.respond(to: prompt)
        let text = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
        let firstLine = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        // It answers in markdown -- every reply in the logs comes back as "**Heady Topper**".
        // Trigram flattening happened to ignore the asterisks, but the catalog should not be
        // asked to resolve them.
        let guess = firstLine.trimmingCharacters(in: CharacterSet(charactersIn: " *\"'`"))
        if guess.isEmpty || guess.uppercased().contains("NONE") { return [] }
        return [guess]
    }

    public func pickProduct(ocr texts: [String], candidates: [ScoredCandidate]) async throws -> ProductPick? {
        guard isAvailable else {
            return try await MockLLMProvider().pickProduct(ocr: texts, candidates: candidates)
        }
        guard !candidates.isEmpty, !texts.isEmpty else { return nil }
        // A numbered menu and an instruction to answer with a number. The model never gets
        // to name a product, so the parroted-example failure `interpretLabels` had to be
        // defended against cannot happen here: an answer is a row the server already
        // surfaced, or nothing.
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
        // The model gives no calibrated confidence; a constrained pick is moderately
        // confident and the HUD marks it as model-adjudicated.
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
