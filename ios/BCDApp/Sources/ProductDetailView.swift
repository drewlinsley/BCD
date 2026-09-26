import SwiftUI
import BCDKit

// The screen you land on from a scan overlay. Its job is a *choice* — is this the one I
// take off the shelf? — so it carries only what decides that: how likely you are to like
// it, how strong it is, what it is, where it's from, and what it tastes like. Rating
// belongs to a drink you've already had, and lives in the Rate tab.
//
// Rating opens from here as a sheet rather than living on the page. The screen's job is
// still the choice, and the five-glyph picker with its readout would be the loudest thing
// on it; but the Rate tab was the *only* door, so a drink you had just identified could not
// be rated without leaving the thing you were looking at, and in practice no verdict was
// ever given (2026-09-24). One tap, same picker, same profile.
//
// Drawn as a label rather than a settings list. The catalog can't fill a table: `style`
// exists on 9 of 915 products and a producer city on about one in eight, so a grid of
// rows is mostly empty rows. A label has a shape that survives missing fields — the name
// and the strength carry it, and everything else is an line that either appears or
// doesn't.

struct ProductDetailView: View {
    let candidate: ScoredCandidate
    @EnvironmentObject var env: AppEnvironment

    /// This install's own verdict, if it has one. Re-read when the rating sheet closes,
    /// because the Seal and the verdict line both change the moment one is given.
    @State private var myReaction: Reaction?
    @State private var rating = false
    /// Neighbours in the sensory space. Nil until asked, so the section can stay off the screen
    /// rather than flashing an empty card on every open.
    @State private var similar: [SimilarProduct]?
    @State private var openingSimilar: String?
    /// A neighbour opened from this screen. Presented rather than pushed: this view owns a
    /// `NavigationStack`, and nesting one inside another costs the back button.
    @State private var drilldown: ScoredCandidate?

    private var product: Product { candidate.resolved.product }
    private var producer: Producer { candidate.resolved.producer }

    private var producerName: String { DisplayName.producer(producer.name) }
    private var productName: String {
        DisplayName.product(DisplayName.label(product.name, brand: candidate.resolved.brand.name),
                            producer: producer.name)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    label
                    verdict
                    rateButton
                    taste
                    similarProfile
                    if !product.recipe.ingredients.isEmpty { ingredients }
                }
                .padding(16)
            }
            .background(Brand.surface)
            .navigationTitle(productName)
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $rating) {
                RatingSheet(productId: product.id, productName: productName)
            }
            .task(id: product.id) {
                similar = (try? await env.api.similar(to: product.id, limit: 6))?.results ?? []
            }
            .sheet(item: $drilldown) { ProductDetailView(candidate: $0) }
            // The sheet writes through `ReactionLog`, so the verdict is on disk by the time
            // it closes; this is what puts it on the label.
            .onChange(of: rating) { _, open in
                if !open { myReaction = env.reactions.reaction(for: product.id) }
            }
            .task {
                // Opening a product is the deliberate act that earns it a place in the Rate
                // queue; merely crossing the viewfinder does not.
                myReaction = env.reactions.reaction(for: product.id)
                env.seen.record(SeenProduct(id: product.id, name: productName,
                                            producer: producerName,
                                            abvPct: product.spec.abvPct?.value))
                try? await env.telemetry.log("product_view", tier: .analytics,
                    ["product_id": .string(product.id), "referrer": .string("scan")])
            }
        }
    }

    // MARK: - the label

    private var label: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(eyebrow)
                .font(.system(size: 10, weight: .bold))
                .tracking(2.2)
                .foregroundStyle(Brand.amber)

            Text(productName)
                .font(.system(size: 30, weight: .semibold, design: .serif))
                .foregroundStyle(Brand.cream)
                .lineLimit(3)
                .minimumScaleFactor(0.62)
                .padding(.top, 10)

            Text(origin)
                .font(.system(size: 10.5, weight: .semibold))
                .tracking(1.5)
                .foregroundStyle(Brand.cream.opacity(0.62))
                .padding(.top, 8)

            HStack(alignment: .bottom) {
                strength
                Spacer(minLength: 12)
                // Your own verdict replaces the prediction. Once you have actually had
                // the thing, what a model guessed you would think is not the number worth
                // the most prominent spot on the label.
                //
                // And once it is your verdict it is also the way to change it: the mark
                // showing what you said is the obvious thing to press when you want to say
                // something else. The row below stays as the labelled affordance -- a
                // circle in a dark card does not announce that it is a button.
                if let mine = myReaction {
                    Button { rating = true } label: { Seal(rated: mine) }
                        .buttonStyle(.plain)
                        .accessibilityHint("Change your rating")
                } else if let score = candidate.personalScore {
                    Seal(score: score)
                } else {
                    Seal()
                }
            }
            .padding(.top, 22)
        }
        .padding(20)
        .background(Brand.ink)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            // The inset keyline every printed label has. Inside the fill, not on its edge.
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .strokeBorder(Brand.cream.opacity(0.22), lineWidth: 1)
                .padding(6)
        )
        // The card is ink in both appearances, so the colours drawn on it have to resolve
        // as they would on a dark surface — otherwise the light-mode ramp puts a dark
        // green seal on a near-black ground.
        .environment(\.colorScheme, .dark)
    }

    /// "BEER · VERMONT" — category is on every product, region on most of the ones that
    /// have any location at all.
    private var eyebrow: String {
        [product.category.rawValue, producer.region]
            .compactMap { $0?.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " · ")
            .uppercased()
    }

    /// "THE ALCHEMIST · STOWE".
    private var origin: String {
        [producerName, producer.city]
            .compactMap { $0?.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " · ")
            .uppercased()
    }

    /// ABV set as a label sets it, with the provenance receipt one tap away.
    @ViewBuilder private var strength: some View {
        if let abv = product.spec.abvPct {
            SourcedStrength(abv: abv)
        } else if let style = product.style {
            // No strength on record: the style, on the rare product that has one, is the
            // next most label-like fact and keeps the foot of the card from collapsing.
            VStack(alignment: .leading, spacing: 5) {
                Text(style.value)
                    .font(.system(size: 17, weight: .semibold, design: .serif))
                    .foregroundStyle(Brand.cream)
                Text("STYLE")
                    .font(.system(size: 9.5, weight: .semibold))
                    .tracking(1.3)
                    .foregroundStyle(Brand.cream.opacity(0.6))
            }
        }
    }

    // MARK: - the verdict

    /// One line saying why the seal reads what it reads. The number is on the label; this
    /// is the sentence that makes it arguable rather than oracular.
    /// Nothing at all once you have rated it. A rating is knowledge and a score is a guess, so
    /// the guess does not get to argue with it -- and the model's reason goes with it, because
    /// it explains a prediction this screen is no longer making.
    ///
    /// It used to say "You rated this chugged it." here, which made the screen state one fact
    /// three times: the Seal with your face on it, this sentence, and the row underneath
    /// carrying the same glyph. The Seal is the one that survives, because it says it in the
    /// spot the prediction used to hold and says it with the mark rather than about it.
    @ViewBuilder private var verdict: some View {
        if myReaction != nil {
            EmptyView()
        } else if candidate.personalScore != nil {
            VStack(alignment: .leading, spacing: 6) {
                if let reason = candidate.reason {
                    Text(reason.prefix(1).uppercased() + reason.dropFirst())
                        .font(.subheadline)
                        .foregroundStyle(Brand.text)
                }
                if candidate.coldStart {
                    Text("Scored from its recipe and style — no personal reviews.")
                        .font(.caption)
                        .foregroundStyle(Brand.textMuted)
                }
            }
            .padding(.horizontal, 4)
        }
        // Nothing when there is no score either. It used to say "Not scored for you yet" in a
        // bare line under the card, which is the one shape on this screen that is neither a
        // label nor a tile -- and the Seal now says the same thing, in the spot that means it.
    }

    /// The one write this screen offers. Worded as the question it is — you can only rate
    /// what you have drunk — so it reads as an invitation rather than a demand, and says
    /// what it is for once a verdict exists.
    ///
    /// The label owns the left, the rating owns the right, in both states: un-rated that is the
    /// whole ramp, rated it is your own face. Same slot, same kind of thing, so nothing jumps
    /// across the row when you come back from the sheet.
    ///
    /// Un-rated it shows all five rather than one mark, because a single glyph would have to be
    /// one of the five verdicts and every one of them answers the question the button is asking:
    /// the top rung reads as "loved it", the pivot as "it was OK". The ramp answers nothing and
    /// is the row the sheet is about to show, so the button previews itself. (It was
    /// `hand.thumbsup`, a borrowed mark promising two directions for a five-rung scale.)
    private var rateButton: some View {
        Button { rating = true } label: {
            HStack(spacing: 8) {
                Text(myReaction == nil ? "Had it? Rate it" : "Change your rating")
                Spacer(minLength: 8)
                if let mine = myReaction {
                    ReactionGlyph(reaction: mine, size: 22)
                } else {
                    // Tight, and below the `recall` floor, so five read as one object -- a
                    // scale -- rather than five buttons competing with the sheet's real ones.
                    HStack(spacing: 1) {
                        ForEach(Reaction.allCases) { ReactionGlyph(reaction: $0, size: 20) }
                    }
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("Five point taste scale")
                }
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Brand.textMuted)
            }
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(myReaction == nil ? Brand.amber : Brand.text)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .cardChrome()
    }

    // MARK: - what it tastes like

    /// No hedge under the sentence. The confidence gate in `TasteSummary` is where that
    /// judgement is made, and it is made once: anything that survives it has been decided
    /// worth saying, so it is said plainly. A percentage printed under a claim the app has
    /// already chosen to stand behind only asks the reader to re-litigate it.
    @ViewBuilder private var taste: some View {
        if let sensory = product.sensory, let sentence = TasteSummary.sentence(for: sensory) {
            Tile(title: "Tastes like") {
                Text(sentence)
                    .font(.callout)
                    .foregroundStyle(Brand.text)
            }
        }
    }

    /// Shut by default. It is the longest block on the screen and the least load-bearing —
    /// the choice this page exists to serve is already made by the seal and the label, and
    /// a list of twelve malts underneath buries both.
    /// What else tastes like this. The app's whole argument in one card: a drink nobody has
    /// reviewed is still placed by what it is made of, so this is answerable for a bottle the
    /// catalog has never seen a rating for.
    ///
    /// About the bottle, not about you — so no score and no "matches your…" line. Two people
    /// reading this screen see the same list, which is what separates it from Discover.
    ///
    /// Silent when the server says `style_only`. 507,341 of 534,103 rows carry their style's
    /// average rather than a vector of their own, and for those the nearest rows are simply
    /// every other row of that style. A section listing them would be a style filter claiming
    /// to be a similarity, so there is no section.
    @ViewBuilder private var similarProfile: some View {
        if let similar, !similar.isEmpty {
            // Shut until asked for, like Ingredients: the screen's job is still the choice in
            // front of you, and six other drinks open on the page is an argument for leaving.
            // The count in the header is the invitation.
            DisclosureTile(title: "Similar profile", count: similar.count) {
                ForEach(similar) { row in
                    Button { Task { await openSimilar(row) } } label: {
                        SimilarRow(row: row, busy: openingSimilar == row.productId)
                    }
                    .buttonStyle(.plain)
                    if row.id != similar.last?.id {
                        Divider().overlay(Brand.hairline)
                    }
                }
                Text("Nearest flavour profiles in the catalog — not what other people drank.")
                    .font(.caption2)
                    .foregroundStyle(Brand.textMuted)
                    .padding(.top, 8)
            }
        }
    }

    /// A neighbour is a name and an id; the screen this opens wants the whole product. Same
    /// round trip `DiscoverView` makes, and the id picks the right row out of the answer
    /// because names repeat in the registry.
    private func openSimilar(_ row: SimilarProduct) async {
        openingSimilar = row.productId
        defer { openingSimilar = nil }
        let hits = (try? await env.api.searchProducts(row.name)) ?? []
        guard let match = hits.first(where: { $0.product.id == row.productId })
                ?? hits.first else { return }
        drilldown = ScoredCandidate(resolved: match, matchScore: 1.0)
    }

    private var ingredients: some View {
        DisclosureTile(title: "Ingredients & process",
                       count: product.recipe.ingredients.count) {
            ForEach(product.recipe.ingredients) { ing in
                IngredientRow(ingredient: ing)
                    .padding(.vertical, 3)
            }
        }
    }
}

// MARK: - pieces

/// The score, stamped. A circle rather than a bar because the label is the metaphor and a
/// progress bar belongs to a different one.
private struct Seal: View {
    /// Two things can occupy this spot, and they are not the same kind of claim: one is
    /// what we predict, the other is what you told us. Making that a type keeps the
    /// difference from being flattened into "a number in a circle".
    enum Verdict {
        case predicted(Double)
        case rated(Reaction)
        /// Nothing to say yet. A product reached by name rather than by the recommender has
        /// no score for this reader, and the spot has to hold that rather than go empty --
        /// a label with a hole where its stamp goes looks broken, not silent.
        case unscored
    }

    let verdict: Verdict

    init(score: Double) { self.verdict = .predicted(score) }
    init(rated reaction: Reaction) { self.verdict = .rated(reaction) }
    init() { self.verdict = .unscored }

    private var tint: Color {
        switch verdict {
        case .predicted(let score):
            score > 0.75 ? Reaction.chuggedIt.tint
                : (score > 0.5 ? Reaction.fine.tint : Reaction.pouredItOut.tint)
        case .rated(let reaction):
            reaction.tint
        // Muted, and the same grey the card's own secondary lines use: an absence should not
        // be as loud as an answer.
        case .unscored:
            Brand.cream.opacity(0.45)
        }
    }

    var body: some View {
        Group {
            switch verdict {
            case .predicted(let score):
                VStack(spacing: 1) {
                    Text("\(Int(score * 100))%")
                        .font(.system(size: 19, weight: .bold, design: .rounded))
                        .monospacedDigit()
                    Text("LIKELY")
                        .font(.system(size: 7.5, weight: .bold))
                        .tracking(1.1)
                }
                .foregroundStyle(tint)
            case .rated(let reaction):
                VStack(spacing: 3) {
                    ReactionGlyph(reaction: reaction, size: 28)
                    Text("RATED")
                        .font(.system(size: 7.5, weight: .bold))
                        .tracking(1.1)
                        .foregroundStyle(tint)
                }
            case .unscored:
                VStack(spacing: 1) {
                    Text("?")
                        .font(.system(size: 19, weight: .bold, design: .rounded))
                    Text("NOT YET")
                        .font(.system(size: 7.5, weight: .bold))
                        .tracking(1.1)
                }
                .foregroundStyle(tint)
            }
        }
        .frame(width: 68, height: 68)
        .overlay(Circle().strokeBorder(tint, lineWidth: 1.5))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(spoken)
    }

    private var spoken: String {
        switch verdict {
        case .predicted(let score): "\(Int(score * 100)) percent likely you'll like it"
        case .rated(let reaction): "You rated this \(reaction.label.lowercased())"
        case .unscored: "Not scored for you yet"
        }
    }
}

/// ABV in label type, tappable for its receipt.
private struct SourcedStrength: View {
    let abv: Sourced<Double>
    @State private var showProvenance = false

    var body: some View {
        Button { showProvenance = true } label: {
            VStack(alignment: .leading, spacing: 4) {
                Text(String(format: "%.1f", abv.value))
                    .font(.system(size: 34, weight: .semibold, design: .serif))
                    .monospacedDigit()
                    .foregroundStyle(Brand.cream)
                HStack(spacing: 5) {
                    Text("ALC / VOL")
                        .font(.system(size: 9.5, weight: .semibold))
                        .tracking(1.3)
                    ProvenanceChip(provenance: abv.provenance)
                }
                .foregroundStyle(Brand.cream.opacity(0.6))
            }
        }
        .buttonStyle(.plain)
        .popover(isPresented: $showProvenance) {
            ProvenanceCard(provenance: abv.provenance).presentationCompactAdaptation(.popover)
        }
        .accessibilityLabel(String(format: "%.1f percent alcohol by volume", abv.value))
    }
}

// Everything below the label is a quiet card, so the label stays the only loud thing on
// the page. The chrome and the heading live in one place each — an open tile and a
// closed one differ in behaviour, not in looks.

private struct CardChrome: ViewModifier {
    func body(content: Content) -> some View {
        content
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Brand.tile)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Brand.hairline, lineWidth: 0.5)
            )
    }
}

extension View {
    fileprivate func cardChrome() -> some View { modifier(CardChrome()) }
}

/// One neighbour. No score: this list is about the bottle, so there is no number that belongs
/// to the reader. The evidence chip stays, because a profile of this exact drink and a guess
/// from its style are not the same claim -- the same ladder Discover shows.
private struct SimilarRow: View {
    let row: SimilarProduct
    let busy: Bool

    var body: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(DisplayName.product(row.name, producer: row.producer ?? ""))
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(Brand.text)
                    .lineLimit(1)
                HStack(spacing: 6) {
                    if let producer = row.producer, !producer.isEmpty {
                        Text(DisplayName.producer(producer))
                            .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
                    // Admitted rather than hidden: a vector shared by nine rows is nine rows
                    // the catalog cannot tell apart, and the reader is entitled to know the
                    // one on screen was picked to stand for them.
                    if row.also > 0 {
                        Text("+\(row.also) alike")
                            .font(.caption2).foregroundStyle(Brand.textMuted)
                    }
                }
            }
            Spacer(minLength: 8)
            if busy {
                ProgressView().controlSize(.small)
            } else {
                EvidenceChip(evidence: row.evidence)
                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Brand.textMuted)
            }
        }
        .padding(.vertical, 9)
        .contentShape(Rectangle())
    }
}

private struct TileTitle: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(.system(size: 10, weight: .bold))
            .tracking(1.4)
            .foregroundStyle(Brand.textMuted)
    }
}

private struct Tile<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TileTitle(text: title)
            content
        }
        .cardChrome()
    }
}

/// A tile that stays shut until asked for. For detail that is worth having but not worth
/// the room it takes on a screen whose job is a yes-or-no decision.
///
/// The count sits in the header so the tap is an informed one — "12" is worth opening,
/// "1" usually isn't.
private struct DisclosureTile<Content: View>: View {
    let title: String
    let count: Int
    @ViewBuilder let content: Content

    @State private var isExpanded = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(reduceMotion ? nil : .snappy(duration: 0.22)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 7) {
                    TileTitle(text: title)
                    Text("\(count)")
                        .font(.system(size: 10, weight: .bold))
                        .monospacedDigit()
                        .foregroundStyle(Brand.textMuted.opacity(0.7))
                    Spacer(minLength: 8)
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Brand.textMuted)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
                // The whole header row is the target, not just the words in it.
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(title), \(count) items")
            .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")

            if isExpanded {
                VStack(alignment: .leading, spacing: 0) { content }
                    .padding(.top, 12)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .cardChrome()
        // Clip so the rows slide out from behind the card's own edge rather than over it.
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

struct IngredientRow: View {
    let ingredient: RecipeIngredient
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(ingredient.rawName)
                    .font(.subheadline)
                Text(ingredient.role.rawValue.replacingOccurrences(of: "_", with: " "))
                    .font(.caption).foregroundStyle(Brand.textMuted)
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
