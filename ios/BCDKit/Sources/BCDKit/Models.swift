import Foundation

// Codable mirrors of the Python `bcd_schema` API contract. Field names match the JSON
// the FastAPI service emits (snake_case via CodingKeys). A future codegen step can emit
// this file from the pydantic models, same as telemetry events.

public enum Category: String, Codable, Sendable {
    case beer, cider, wine, spirit, rtd, mead, sake, other
}

public enum ExtractionMethod: String, Codable, Sendable {
    case statedByProducer = "stated_by_producer"
    case regulatoryFiling = "regulatory_filing"
    case labelOCR = "label_ocr"
    case retailerListing = "retailer_listing"
    case communityClone = "community_clone"
    case reviewConsensus = "review_consensus"
    case llmInferredFromStylePrior = "llm_inferred_from_style_prior"
    case userContributed = "user_contributed"

    /// How much visual weight the provenance chip earns. Drives the receipt UI.
    public var trustRank: Int {
        switch self {
        case .statedByProducer, .regulatoryFiling: return 3
        case .labelOCR, .retailerListing: return 2
        case .communityClone, .reviewConsensus, .userContributed: return 1
        case .llmInferredFromStylePrior: return 0
        }
    }
}

public struct Provenance: Codable, Sendable, Hashable {
    public let sourceId: String
    public let url: String?
    public let quote: String?
    public let method: ExtractionMethod
    public let confidence: Double

    enum CodingKeys: String, CodingKey {
        case sourceId = "source_id"
        case url, quote, method, confidence
    }
}

public struct Sourced<Value: Codable & Sendable>: Codable, Sendable {
    public let value: Value
    public let provenance: Provenance
}

public struct Producer: Codable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let kind: String?
    public let country: String?
    public let region: String?       // state / province
    public let city: String?
    public let lat: Double?
    public let lon: Double?
    public let website: String?
}

public struct Brand: Codable, Sendable, Identifiable {
    public let id: String
    public let producerId: String
    public let name: String

    enum CodingKeys: String, CodingKey {
        case id, name
        case producerId = "producer_id"
    }
}

public struct ProductSpec: Codable, Sendable {
    public let abvPct: Sourced<Double>?
    public let ibu: Sourced<Double>?
    public let proof: Sourced<Double>?
    public let ageStatementYears: Sourced<Double>?

    enum CodingKeys: String, CodingKey {
        case abvPct = "abv_pct"
        case ibu
        case proof
        case ageStatementYears = "age_statement_years"
    }
}

public enum IngredientRole: String, Codable, Sendable {
    case baseMalt = "base_malt", specialtyMalt = "specialty_malt"
    case bitteringHop = "bittering_hop", flavorHop = "flavor_hop"
    case aromaHop = "aroma_hop", dryHop = "dry_hop"
    case yeast, waterSalt = "water_salt", adjunct, fruit, spice, barrel
    case mashGrain = "mash_grain", fining, other
}

public struct RecipeIngredient: Codable, Sendable, Identifiable {
    public var id: String { "\(role.rawValue):\(rawName)" }
    public let role: IngredientRole
    public let entityKind: String
    public let entityRef: String?
    public let rawName: String
    public let provenance: Provenance

    enum CodingKeys: String, CodingKey {
        case role
        case entityKind = "entity_kind"
        case entityRef = "entity_ref"
        case rawName = "raw_name"
        case provenance
    }
}

public struct RecipeGraph: Codable, Sendable {
    public let ingredients: [RecipeIngredient]
    public init(ingredients: [RecipeIngredient] = []) { self.ingredients = ingredients }
}

public struct Product: Codable, Sendable, Identifiable {
    public let id: String
    public let brandId: String
    public let producerId: String
    public let category: Category
    public let name: String
    public let style: Sourced<String>?
    public let spec: ProductSpec
    public let recipe: RecipeGraph
    public let sensory: SensoryVector?

    enum CodingKeys: String, CodingKey {
        case id, category, name, style, spec, recipe, sensory
        case brandId = "brand_id"
        case producerId = "producer_id"
    }

    // Spelled out rather than left to the memberwise initializer so `sensory` can default:
    // the field arrived long after the call sites did, and none of them should have to
    // name it to keep building.
    public init(id: String, brandId: String, producerId: String, category: Category,
                name: String, style: Sourced<String>?, spec: ProductSpec,
                recipe: RecipeGraph, sensory: SensoryVector? = nil) {
        self.id = id
        self.brandId = brandId
        self.producerId = producerId
        self.category = category
        self.name = name
        self.style = style
        self.spec = spec
        self.recipe = recipe
        self.sensory = sensory
    }
}

public struct ResolvedProduct: Codable, Sendable, Identifiable {
    public var id: String { product.id }
    public let product: Product
    public let producer: Producer
    public let brand: Brand
}
