import SwiftUI
import BCDKit

// The screen you land on from a scan overlay. Its job is a *choice* — is this the one I take
// off the shelf? — so it leads with the four things that decide that: how likely you are to
// like it, ABV, style, and where it was made. Rating belongs to a drink you've already had,
// so it is deliberately not here.
//
// Every recovered fact still carries a provenance chip; tapping one is the answer to "your
// data isn't verifiable" — source, method, confidence, and the exact supporting quote.

struct ProductDetailView: View {
    let candidate: ScoredCandidate
    @EnvironmentObject var env: AppEnvironment

    private var product: Product { candidate.resolved.product }
    private var producer: Producer { candidate.resolved.producer }

    var body: some View {
        NavigationStack {
            List {
                Section { verdict }
                Section("Details") { details }
                if !product.recipe.ingredients.isEmpty {
                    // What the score is actually derived from, for a product no one has
                    // reviewed yet.
                    Section("Ingredients & process") {
                        ForEach(product.recipe.ingredients) { ing in
                            IngredientRow(ingredient: ing)
                        }
                    }
                }
            }
            .navigationTitle(product.name)
            .navigationBarTitleDisplayMode(.inline)
            .task {
                // Opening a product is the deliberate act that earns it a place in the Rate
                // queue; merely crossing the viewfinder does not.
                env.seen.record(SeenProduct(id: product.id, name: product.name,
                                            producer: producer.name,
                                            abvPct: product.spec.abvPct?.value))
                try? await env.telemetry.log("product_view", tier: .analytics,
                    ["product_id": .string(product.id), "referrer": .string("scan")])
            }
        }
    }

    // MARK: - the decision

    /// How likely you are to like it, stated plainly and up top — the one number this screen
    /// exists to deliver.
    @ViewBuilder private var verdict: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(producer.name)
                .font(.subheadline)
                .foregroundStyle(Brand.textMuted)

            if let score = candidate.personalScore {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(Int(score * 100))%")
                        .font(.system(size: 34, weight: .semibold, design: .rounded))
                        .foregroundStyle(tint(for: score))
                    Text("likely you'll like it")
                        .font(.subheadline)
                        .foregroundStyle(Brand.textMuted)
                }
                ProgressView(value: score)
                    .tint(tint(for: score))
                if let reason = candidate.reason {
                    Text(reason).font(.footnote).foregroundStyle(Brand.textMuted)
                }
                if candidate.coldStart {
                    Label("Scored from its recipe chemistry — no reviews needed.",
                          systemImage: "flask.fill")
                        .font(.caption).foregroundStyle(Brand.textMuted)
                }
            } else {
                Text("Not scored for you yet")
                    .font(.subheadline).foregroundStyle(Brand.textMuted)
            }
        }
        .padding(.vertical, 4)
    }

    /// Reuses the reaction ramp, so good-to-bad reads the same everywhere in the app.
    private func tint(for score: Double) -> Color {
        score > 0.75 ? Reaction.chuggedIt.tint
            : (score > 0.5 ? Reaction.fine.tint : Reaction.pouredItOut.tint)
    }

    // MARK: - the facts

    @ViewBuilder private var details: some View {
        // Each row stands on its own: a product with no ABV should still show its style.
        if let abv = product.spec.abvPct {
            SourcedRow(label: "ABV", value: String(format: "%.1f%%", abv.value),
                       provenance: abv.provenance)
        }
        if let style = product.style {
            SourcedRow(label: "Style", value: style.value, provenance: style.provenance)
        }
        if let place = location {
            LabeledContent("From", value: place)
        }
    }

    /// Where it was made, finest-grained first. Omitted entirely rather than shown empty —
    /// the catalog's producer records don't all carry a location.
    private var location: String? {
        let parts = [producer.city, producer.region, producer.country]
            .compactMap { $0?.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        return parts.isEmpty ? nil : parts.joined(separator: ", ")
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
