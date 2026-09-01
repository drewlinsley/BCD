import SwiftUI
import BCDKit

// The taster set, in the three places the design canvas puts it: the ask on product
// detail, the recall badge in a search row, and the verdict carried on a scan HUD chip.
// Geometry and radii are lifted from the surrounding views rather than reinvented.

/// One glyph, always in its own ramp colour.
///
/// The build spec says to leave the rest state neutral grey so no level is pre-suggested.
/// On device that failed: at picker size the five differ only by a few pixels of expression,
/// and in one flat grey they are not tellable apart. Colouring *all five* red-to-green labels
/// the scale rather than nudging toward any one of them — the thing the rule was protecting
/// against — and the ramp is by far the most legible signal at this size. Selection is carried
/// by the ring instead.
struct ReactionGlyph: View {
    let reaction: Reaction
    var size: CGFloat = Brand.GlyphSize.picker

    var body: some View {
        reaction.glyph(at: size)
            .renderingMode(.template)
            .resizable()
            .scaledToFit()
            .frame(width: size, height: size)
            .foregroundStyle(reaction.tint)
            .accessibilityLabel(reaction.label)
    }
}

/// "How was it?" — the ask. A tap is a rating, which is the only thing that moves the
/// taste centroid, so this is the app's one real write.
struct ReactionPicker: View {
    let productId: String
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var consent: ConsentStore
    @State private var picked: Reaction?
    @State private var sending = false
    @State private var failed = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("How was it?")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Brand.text)

            HStack(spacing: 2) {
                ForEach(Reaction.allCases) { reaction in
                    Button { choose(reaction) } label: {
                        VStack(spacing: 4) {
                            ReactionGlyph(reaction: reaction, size: Brand.GlyphSize.picker)
                                .frame(width: Brand.GlyphSize.hitTarget,
                                       height: Brand.GlyphSize.hitTarget)
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(picked == reaction ? reaction.tint.opacity(0.12)
                                                                 : Color.clear)
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(picked == reaction ? reaction.tint : .clear,
                                                lineWidth: 1.5)
                                )
                            // The five faces are close cousins; the name is what makes each
                            // one unambiguous at a glance.
                            Text(reaction.label)
                                .font(.caption2)
                                .foregroundStyle(picked == reaction ? reaction.tint
                                                                    : Brand.textMuted)
                                .lineLimit(2)
                                .multilineTextAlignment(.center)
                                .minimumScaleFactor(0.8)
                        }
                    }
                    .buttonStyle(.plain)
                    .disabled(sending)
                    .frame(maxWidth: .infinity)
                }
            }
            .animation(.snappy(duration: 0.18), value: picked)

            readout
        }
        .padding(.vertical, 4)
        .task { picked = env.reactions.reaction(for: productId) }
    }

    @ViewBuilder private var readout: some View {
        if let picked {
            VStack(alignment: .leading, spacing: 3) {
                Text(String(format: "weight %+.1f", picked.weight))
                    .font(.caption2.monospaced()).foregroundStyle(Brand.textMuted)
                Text(picked.note).font(.caption).foregroundStyle(Brand.textMuted)
                if !consent.personalization {
                    // Honest about where it went: with personalization off this is a local
                    // note, not a signal, and nothing reaches the profile.
                    HStack(spacing: 6) {
                        Text("Kept on this phone — personalization is off.")
                            .font(.caption2).foregroundStyle(Brand.textMuted)
                        Button("Turn on") { turnOnAndSend(picked) }
                            .font(.caption2.weight(.semibold))
                    }
                } else if failed {
                    Text("Couldn't reach the server — saved locally.")
                        .font(.caption2).foregroundStyle(.orange)
                }
            }
        }
    }

    private func choose(_ reaction: Reaction) {
        picked = reaction
        env.reactions.record(reaction, for: productId)
        guard consent.personalization else { return }
        send(reaction)
    }

    private func turnOnAndSend(_ reaction: Reaction) {
        consent.personalization = true
        send(reaction)
    }

    private func send(_ reaction: Reaction) {
        sending = true
        failed = false
        Task {
            defer { sending = false }
            do {
                _ = try await env.api.submitFeedback(
                    FeedbackRequest(productId: productId, reaction: reaction),
                    userId: env.installId)
                try? await env.telemetry.log(
                    "rating_submitted", tier: .personalization,
                    ["product_id": .string(productId), "rating": .int(reaction.rawValue)])
            } catch {
                failed = true
            }
        }
    }
}

/// Recall: what this install already said about a product, at the 24pt list size.
struct ReactionBadge: View {
    let reaction: Reaction?
    var size: CGFloat = Brand.GlyphSize.recall

    var body: some View {
        if let reaction {
            ReactionGlyph(reaction: reaction, size: size)
        } else {
            Text("not rated")
                .font(.caption2)
                .foregroundStyle(Brand.textMuted)
        }
    }
}
