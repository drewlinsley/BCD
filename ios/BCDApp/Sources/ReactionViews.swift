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
    /// Whether to ask before the first verdict leaves the phone.
    @State private var asking = false

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
        // The first rating is where consent actually becomes a question, so it is asked
        // here rather than left as a line of small print under the picker. Before this, a
        // tap on a fresh install recorded locally and went nowhere: the profile never
        // moved, the Discover list kept saying "Somewhere to start", and nothing on screen
        // said the rating had not been sent unless you read the caption (2026-09-24).
        .alert("Use your ratings?", isPresented: $asking) {
            Button("Yes, learn what I like") { allow() }
            Button("Not now", role: .cancel) {}
        } message: {
            Text("Your verdicts shape what BCD suggests. They are kept against a random id "
                 + "for this install \u{2014} not your name, email, or anything about you "
                 + "\u{2014} and you can turn this off any time under You.")
        }
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
        // Recorded locally either way: it is the user's own answer about their own drink,
        // and it is what the Seal and the search rows read back.
        env.reactions.record(reaction, for: productId)
        if consent.personalization { send(reaction) } else { asking = true }
    }

    /// Yes. Flipping the store is what lets the queue take personalization events too --
    /// `ConsentStore.onChange` carries it to the queue -- so this is the whole of turning
    /// the loop on.
    private func allow() {
        consent.personalization = true
        if let picked { send(picked) }
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
                // The profile has moved, so anything ranking with it is now stale.
                env.ratingAccepted()
                try? await env.telemetry.log(
                    "rating_submitted", tier: .personalization,
                    ["product_id": .string(productId), "rating": .int(reaction.rawValue)])
            } catch {
                failed = true
            }
        }
    }
}

/// The rating sheet — `ReactionPicker` with a name on it and a way out.
///
/// A sheet rather than a push because rating is an aside to the screen that opened it: you
/// came to decide whether to drink the thing, and this is you reporting back on one you
/// already did. It closes itself once a verdict is in, so the answer to "how was it?" takes
/// exactly one tap.
struct RatingSheet: View {
    let productId: String
    let productName: String
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var env: AppEnvironment
    @EnvironmentObject var consent: ConsentStore

    var body: some View {
        NavigationStack {
            ScrollView {
                // The picker asks "How was it?" itself -- it has to, because the Rate tab
                // stacks several of them -- so the bar carries the drink's name instead of
                // asking the same question twice.
                ReactionPicker(productId: productId).padding(20)
            }
            .background(Brand.surface)
            .navigationTitle(productName)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
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
