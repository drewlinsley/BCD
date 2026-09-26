import Foundation

// What else tastes like this one.
//
// A different question from `Recommendation`, and deliberately a different type. That one is
// about a *person* — it carries a predicted score and a reason written in the second person.
// This is about a *bottle*: two people asking of the same row get the same answer, so there is
// no score to show and nothing to explain about you.
//
// The app's own thesis, applied to one row: a drink with no reviews is still placed by what it
// is made of.

public struct SimilarProduct: Codable, Sendable, Identifiable, Equatable {
    public var id: String { productId }
    public let productId: String
    public let name: String
    /// Absent where the catalog has no producer for the row — the registry files plenty, and
    /// since 2026-09-26 an imported row whose brand field held a product name has none on
    /// purpose rather than carrying its importer's.
    public let producer: String?
    /// What stands behind the neighbour's own vector, the same ladder `Recommendation` uses.
    public let evidence: Recommendation.Evidence
    /// How many further rows share this exact vector and so are the same suggestion. Shown
    /// because "and 8 more taste identical" is a fact about the catalog worth admitting.
    public let also: Int

    /// Spelled out rather than left to a decoder strategy, the way `Recommendation` does it:
    /// the wire format is the server's, and a model that carries its own mapping decodes the
    /// same whoever built the `JSONDecoder`.
    enum CodingKeys: String, CodingKey {
        case name, producer, evidence, also
        case productId = "product_id"
    }

    public init(productId: String, name: String, producer: String?,
                evidence: Recommendation.Evidence, also: Int) {
        self.productId = productId
        self.name = name
        self.producer = producer
        self.evidence = evidence
        self.also = also
    }
}

/// The answer, with why it is the shape it is.
public struct SimilarResponse: Codable, Sendable, Equatable {
    /// `profile` when the row has a vector of its own; `styleOnly` when it carries its style's
    /// average, which 95% of the catalog does. For those there is nothing to be similar *to*,
    /// and the honest move is to show nothing rather than list the style back.
    public enum Basis: String, Codable, Sendable {
        case profile
        case styleOnly = "style_only"
    }

    public let basis: Basis
    public let results: [SimilarProduct]

    public init(basis: Basis, results: [SimilarProduct]) {
        self.basis = basis
        self.results = results
    }
}
