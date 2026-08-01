import SwiftUI
import BCDKit

// "The receipt" — the full decomposed product with a provenance chip on every recovered
// fact. Tapping a chip is the answer to "your data isn't verifiable": it shows the source,
// the method, the confidence, and the exact supporting quote.

struct ProductDetailView: View {
    let candidate: ScoredCandidate
    @EnvironmentObject var env: AppEnvironment

    private var product: Product { candidate.resolved.product }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    header
                }
                if let abv = product.spec.abvPct {
                    Section("Specs") {
                        SourcedRow(label: "ABV", value: "\(abv.value)%", provenance: abv.provenance)
                        if let style = product.style {
                            SourcedRow(label: "Style", value: style.value,
                                       provenance: style.provenance)
                        }
                    }
                }
                if !product.recipe.ingredients.isEmpty {
                    Section("Ingredients & process") {
                        ForEach(product.recipe.ingredients) { ing in
                            IngredientRow(ingredient: ing)
                        }
                    }
                }
                Section {
                    if candidate.coldStart {
                        Label("Scored from its recipe chemistry — no reviews needed.",
                              systemImage: "flask.fill")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle(product.name)
            .navigationBarTitleDisplayMode(.inline)
            .task {
                try? await env.telemetry.log("product_view", tier: .analytics,
                    ["product_id": .string(product.id), "referrer": .string("scan")])
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(candidate.resolved.producer.name).foregroundStyle(.secondary)
            if let s = candidate.personalScore {
                HStack {
                    ProgressView(value: s).tint(s > 0.75 ? .green : .yellow)
                    Text("\(Int(s * 100))% match").font(.subheadline.bold())
                }
                if let reason = candidate.reason {
                    Text(reason).font(.footnote).foregroundStyle(.secondary)
                }
            }
        }
    }
}

struct SourcedRow: View {
    let label: String
    let value: String
    let provenance: Provenance
    @State private var showProvenance = false

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
            ProvenanceChip(provenance: provenance)
                .onTapGesture { showProvenance = true }
        }
        .popover(isPresented: $showProvenance) {
            ProvenanceCard(provenance: provenance).presentationCompactAdaptation(.popover)
        }
    }
}

struct IngredientRow: View {
    let ingredient: RecipeIngredient
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(ingredient.rawName)
                Text(ingredient.role.rawValue.replacingOccurrences(of: "_", with: " "))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            ProvenanceChip(provenance: ingredient.provenance)
        }
    }
}

struct ProvenanceChip: View {
    let provenance: Provenance
    private var color: Color {
        switch provenance.method.trustRank {
        case 3: .green
        case 2: .blue
        case 1: .orange
        default: .gray
        }
    }
    var body: some View {
        Image(systemName: "checkmark.seal.fill")
            .font(.caption)
            .foregroundStyle(color)
            .accessibilityLabel("Provenance: \(provenance.method.rawValue)")
    }
}

struct ProvenanceCard: View {
    let provenance: Provenance
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(provenance.method.rawValue.replacingOccurrences(of: "_", with: " "),
                  systemImage: "checkmark.seal.fill").font(.headline)
            Text("Source: \(provenance.sourceId)").font(.subheadline)
            Text("Confidence: \(Int(provenance.confidence * 100))%").font(.subheadline)
            if let quote = provenance.quote {
                Text("\u{201C}\(quote)\u{201D}").font(.footnote).italic()
                    .foregroundStyle(.secondary)
            }
            if let url = provenance.url {
                Text(url).font(.caption).foregroundStyle(.blue).lineLimit(1)
            }
        }
        .padding()
        .frame(maxWidth: 280)
    }
}
