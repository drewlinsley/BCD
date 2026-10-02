import SwiftUI
import BCDKit

// The one screen where the app writes a drink instead of reading one. It opens from the scan
// HUD's empty-state glyph: the camera read a label plainly but the catalog couldn't place it, so
// rather than leave a silent blank, this turns the gap into a contribution.
//
// Small on purpose. A name and a category are all it asks; a maker and a strength are *offered*
// because the camera has often already read them, and a note is there for the one thing the
// fields don't cover. Everything is pre-filled from what the phone saw where we can, so the
// common case is a glance and a Save, not a form filled standing in front of a shelf.
//
// The ink card at the top is the product label from `ProductDetailView`, drawn from the fields
// as they're typed — so the abstract act of "adding a drink" shows up as a label taking shape,
// which is the thing BCD makes. It doubles as the preview and as the screen's one loud element.

struct AddDrinkView: View {
    /// What the camera read off the label, best-first (`ScanCoordinator.lastSightings`). Seeds
    /// the name and is offered as one-tap fills. Empty is normal — the vision path only runs
    /// with label-photo consent, and the form stands on its own without it.
    let sightings: [String]
    /// Hand the finished contribution back to the caller to persist and log. The view owns the
    /// fields and the dismissal; the caller owns where it goes.
    let onSubmit: (DrinkContribution) -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var name = ""
    // `BCDKit.` qualified throughout: SwiftUI pulls in the ObjC runtime, whose `Category`
    // typedef makes the bare name ambiguous in this target.
    @State private var category: BCDKit.Category = .beer
    @State private var maker = ""
    @State private var abv = ""
    @State private var note = ""
    @State private var didSubmit = false
    @FocusState private var focus: Field?

    private enum Field: Hashable { case name, maker, abv, note }

    private var trimmedName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var trimmedMaker: String { maker.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var trimmedNote: String { note.trimmingCharacters(in: .whitespacesAndNewlines) }
    /// A name is the one thing a contribution can't be without — everything else is a row a
    /// curator can fill, but a nameless drink is nothing to add.
    private var canSubmit: Bool { !trimmedName.isEmpty }

    var body: some View {
        NavigationStack {
            Group {
                if didSubmit { thanks } else { form }
            }
            .background(Brand.surface)
            .navigationTitle("Add a drink")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(didSubmit ? "Done" : "Cancel") { dismiss() }
                }
            }
        }
        .onAppear(perform: seed)
    }

    // MARK: - the form

    private var form: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                labelCard
                if !sightings.isEmpty { sightingsStrip }
                field(title: "Name", text: $name, focus: .name,
                      placeholder: "What's it called?")
                categoryField
                field(title: "Maker", text: $maker, focus: .maker,
                      placeholder: "Brewery or distillery (optional)")
                abvField
                field(title: "Anything else", text: $note, focus: .note,
                      placeholder: "Style, where you found it… (optional)")
                submitButton
            }
            .padding(16)
        }
        .scrollDismissesKeyboard(.interactively)
    }

    /// The product label, authored live. Same chrome as `ProductDetailView.label` so a
    /// contribution looks like the thing it will become; the fields fill it as they're typed.
    private var labelCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(eyebrow)
                .font(.system(size: 10, weight: .bold))
                .tracking(2.2)
                .foregroundStyle(Brand.amber)

            Text(trimmedName.isEmpty ? "Name this drink" : trimmedName)
                .font(.system(size: 30, weight: .semibold, design: .serif))
                // Dim the placeholder so the card reads as waiting for a name, not naming a
                // drink "Name this drink".
                .foregroundStyle(Brand.cream.opacity(trimmedName.isEmpty ? 0.4 : 1))
                .lineLimit(3)
                .minimumScaleFactor(0.62)
                .padding(.top, 10)

            HStack(alignment: .bottom) {
                strength
                Spacer(minLength: 12)
                Text("NOT IN\nTHE CATALOG")
                    .font(.system(size: 8.5, weight: .bold))
                    .tracking(1.1)
                    .multilineTextAlignment(.trailing)
                    .foregroundStyle(Brand.cream.opacity(0.45))
            }
            .padding(.top, 22)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Brand.ink)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .strokeBorder(Brand.cream.opacity(0.22), lineWidth: 1)
                .padding(6)
        )
        // Ink in both appearances, so the cream and amber drawn on it resolve as they would on a
        // dark surface rather than flipping in light mode.
        .environment(\.colorScheme, .dark)
        .animation(reduceMotion ? nil : .easeOut(duration: 0.2), value: trimmedName.isEmpty)
    }

    /// "BEER" or "BEER · THE ALCHEMIST" — category always, maker once it's been typed. Mirrors
    /// the detail label's category · region line, with the maker standing in for a place the
    /// contribution doesn't have.
    private var eyebrow: String {
        [category.contributionLabel, trimmedMaker]
            .filter { !$0.isEmpty }
            .joined(separator: " · ")
            .uppercased()
    }

    /// ABV in label type once there's a number, like the detail label's strength. Blank keeps
    /// the foot of the card quiet rather than showing a 0.
    @ViewBuilder private var strength: some View {
        if let value = parsedAbv {
            VStack(alignment: .leading, spacing: 4) {
                Text(String(format: "%.1f", value))
                    .font(.system(size: 34, weight: .semibold, design: .serif))
                    .monospacedDigit()
                    .foregroundStyle(Brand.cream)
                Text("ALC / VOL")
                    .font(.system(size: 9.5, weight: .semibold))
                    .tracking(1.3)
                    .foregroundStyle(Brand.cream.opacity(0.6))
            }
        }
    }

    /// What the camera read, offered as one-tap fills. The brand's own reading handed back: the
    /// top line usually *is* the name, and tapping it is faster than retyping a wordmark the OCR
    /// already got. Fills the name while it's empty, then the maker, so two taps fill both.
    private var sightingsStrip: some View {
        VStack(alignment: .leading, spacing: 8) {
            fieldTitle("The camera read")
            FlowRow(spacing: 8) {
                ForEach(Array(sightings.prefix(6).enumerated()), id: \.offset) { _, text in
                    Button { useSighting(text) } label: {
                        Text(text)
                            .font(.subheadline)
                            .foregroundStyle(Brand.text)
                            .lineLimit(1)
                            .padding(.horizontal, 12).padding(.vertical, 7)
                            .background(Brand.tile, in: Capsule())
                            .overlay(Capsule().strokeBorder(Brand.hairline, lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                }
            }
            Text("Tap to use — fills the name, then the maker.")
                .font(.caption2)
                .foregroundStyle(Brand.textMuted)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var categoryField: some View {
        VStack(alignment: .leading, spacing: 8) {
            fieldTitle("What is it")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(BCDKit.Category.allCases, id: \.rawValue) { cat in
                        let on = cat == category
                        Button { category = cat } label: {
                            Text(cat.contributionLabel)
                                .font(.subheadline.weight(on ? .semibold : .regular))
                                .foregroundStyle(on ? Brand.ink : Brand.text)
                                .padding(.horizontal, 14).padding(.vertical, 8)
                                .background(on ? Brand.amber : Brand.tile, in: Capsule())
                                .overlay(Capsule().strokeBorder(Brand.hairline,
                                                                lineWidth: on ? 0 : 0.5))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 1) // so a selected end chip's capsule isn't clipped
            }
            .animation(reduceMotion ? nil : .easeOut(duration: 0.15), value: category)
        }
    }

    private var abvField: some View {
        VStack(alignment: .leading, spacing: 8) {
            fieldTitle("Strength")
            HStack(spacing: 6) {
                TextField("ABV", text: $abv)
                    .keyboardType(.decimalPad)
                    .focused($focus, equals: .abv)
                    .frame(maxWidth: 90)
                Text("% ALC / VOL")
                    .font(.caption.weight(.semibold))
                    .tracking(0.8)
                    .foregroundStyle(Brand.textMuted)
                Spacer()
            }
            .padding(14)
            .background(Brand.tile)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Brand.hairline, lineWidth: 0.5))
        }
    }

    /// The commit. One amber button rather than a toolbar Save: this flow is one-handed in front
    /// of a shelf, so the action wants a real tap target, and amber is the app's "this is the
    /// thing to press" everywhere else.
    private var submitButton: some View {
        Button(action: submit) {
            Text("Add to the catalog")
                .font(.headline)
                .foregroundStyle(canSubmit ? Brand.ink : Brand.textMuted)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
                .background(canSubmit ? Brand.amber : Brand.tile,
                            in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(!canSubmit)
        .padding(.top, 4)
    }

    // MARK: - the thanks

    /// A short, honest confirmation. We don't yet upload contributions, so this doesn't promise
    /// the drink will appear — only that we have it. Dismisses itself, or the toolbar's Done.
    private var thanks: some View {
        VStack(spacing: 14) {
            ContributionCheck(size: 72)
            Text("Thanks for the catch")
                .font(.title3.weight(.semibold))
                .foregroundStyle(Brand.text)
            Text("We saved “\(trimmedName)” — it helps fill a gap the scanner found.")
                .font(.subheadline)
                .foregroundStyle(Brand.textMuted)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task {
            // Long enough to read, short enough not to trap someone who wants to keep scanning.
            try? await Task.sleep(nanoseconds: 1_600_000_000)
            dismiss()
        }
    }

    // MARK: - pieces & actions

    private func field(title: String, text: Binding<String>, focus kind: Field,
                       placeholder: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            fieldTitle(title)
            TextField(placeholder, text: text)
                .focused($focus, equals: kind)
                .submitLabel(.done)
                .onSubmit { focus = nil }
                .padding(14)
                .background(Brand.tile)
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(Brand.hairline, lineWidth: 0.5))
        }
    }

    private func fieldTitle(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.system(size: 10, weight: .bold))
            .tracking(1.4)
            .foregroundStyle(Brand.textMuted)
    }

    /// Parse the ABV field leniently: a bare number, or one with a "%" the user typed. Out-of-range
    /// values are dropped rather than shown, so a fat-fingered "555" doesn't print on the label.
    private var parsedAbv: Double? {
        let cleaned = abv.replacingOccurrences(of: "%", with: "")
            .trimmingCharacters(in: .whitespaces)
        guard let v = Double(cleaned), v > 0, v <= 100 else { return nil }
        return v
    }

    /// Seed from what the camera saw: the first sighting is almost always the name, and the
    /// second is often the maker. Guessed, not locked — both are editable, and the sightings
    /// strip lets the user re-point them.
    private func seed() {
        guard name.isEmpty else { return }
        if let first = sightings.first { name = first }
        if sightings.count > 1 { maker = sightings[1] }
    }

    private func useSighting(_ text: String) {
        if trimmedName.isEmpty { name = text }
        else if trimmedMaker.isEmpty { maker = text }
        else { name = text }
    }

    private func submit() {
        guard canSubmit else { return }
        focus = nil
        let contribution = DrinkContribution(
            name: trimmedName,
            category: category,
            maker: trimmedMaker.isEmpty ? nil : trimmedMaker,
            abvPct: parsedAbv,
            note: trimmedNote.isEmpty ? nil : trimmedNote,
            sightings: sightings)
        onSubmit(contribution)
        withAnimation(reduceMotion ? nil : .easeInOut(duration: 0.25)) { didSubmit = true }
    }
}

/// Category names for the contribution picker. Kept here, not on the model: `rawValue`
/// ("rtd") is the wire contract and the detail label already shows it uppercased, whereas this
/// is reader-facing chrome for one screen ("Seltzer / RTD").
private extension BCDKit.Category {
    var contributionLabel: String {
        switch self {
        case .beer: "Beer"
        case .cider: "Cider"
        case .wine: "Wine"
        case .spirit: "Spirit"
        case .rtd: "Seltzer / RTD"
        case .mead: "Mead"
        case .sake: "Sake"
        case .other: "Other"
        }
    }
}

/// The confirmation mark — a check struck in the scan's own amber, in a soft ring. Drawn rather
/// than an SF Symbol to answer `ScanNotFoundGlyph` in the same hand: the glyph asked, this is
/// the yes. Laid out in the glyph's 200×200 space and scaled, so the stroke tracks the size.
private struct ContributionCheck: View {
    var size: CGFloat = 72
    private var k: CGFloat { size / 200 }

    var body: some View {
        ZStack {
            Circle()
                .strokeBorder(Brand.amber.opacity(0.3), lineWidth: 5 * k)
            CheckShape()
                .stroke(Brand.amber,
                        style: StrokeStyle(lineWidth: 10 * k, lineCap: .round, lineJoin: .round))
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

private struct CheckShape: Shape {
    func path(in rect: CGRect) -> Path {
        let s = min(rect.width, rect.height) / 200
        func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: x * s, y: y * s) }
        var path = Path()
        path.move(to: p(62, 104))
        path.addLine(to: p(88, 132))
        path.addLine(to: p(140, 72))
        return path
    }
}

/// A wrapping HStack — chips flow onto the next line rather than scrolling or clipping. The
/// camera can hand back six readings and they shouldn't run off the edge.
private struct FlowRow: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var rows: [[CGSize]] = [[]]
        var x: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width > maxWidth, !(rows.last?.isEmpty ?? true) {
                rows.append([]); x = 0
            }
            rows[rows.count - 1].append(s)
            x += s.width + spacing
        }
        let height = rows.reduce(CGFloat(0)) { acc, row in
            acc + (row.map(\.height).max() ?? 0) + spacing
        } - (rows.isEmpty ? 0 : spacing)
        return CGSize(width: maxWidth == .infinity ? x : maxWidth, height: max(0, height))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize,
                       subviews: Subviews, cache: inout Void) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x + s.width > bounds.maxX, x > bounds.minX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            v.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(s))
            x += s.width + spacing
            rowHeight = max(rowHeight, s.height)
        }
    }
}

#if DEBUG
#Preview("Add a drink — from a scan") {
    AddDrinkView(sightings: ["Crusher", "The Alchemist"]) { _ in }
}

#Preview("Add a drink — cold") {
    AddDrinkView(sightings: []) { _ in }
}
#endif
