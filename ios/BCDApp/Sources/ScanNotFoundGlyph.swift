import SwiftUI

// The scan's empty hand. It shows in the viewfinder when a label was read clearly but matched
// nothing in the catalog — the one moment the HUD has to admit it doesn't know a drink. Rather
// than an error, it turns the dead end into an invitation: an empty glass draws itself in, a
// `?` settles, and after a few seconds it offers a `+`, blinking slowly between the two — *we
// don't have this one; add it.* Tapping fires `onAdd`.
//
// Drawn, not an SF Symbol, for the same reason the reaction faces are (ReactionViews): the scan
// speaks in its own marks. The two colours are the scan-frame mark's own — `Brand.cream` for the
// glass, `Brand.amber` for the prompt — so it reads as BCD over any camera background.

/// The animated "nothing found — add it?" glyph for the scan HUD.
struct ScanNotFoundGlyph: View {
    /// The rendered square size of the glyph itself (the scrim bleeds past it).
    var size: CGFloat = 120
    /// A soft ink vignette behind the mark, so cream and amber stay legible over a bright shelf.
    /// Off when the caller already sits the glyph on a dark surface.
    var showsScrim: Bool = true
    /// What a tap means: add the unknown drink.
    var onAdd: () -> Void = {}

    // Honour the system switch. With it on there is no draw and no blink — we land straight on
    // the state that carries the meaning (an empty glass offering a `+`) and hold it still.
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var glassTrim: CGFloat = 0
    @State private var baseOpacity: Double = 0
    @State private var qOpacity: Double = 0
    @State private var qScale: CGFloat = 0.7
    @State private var plusOpacity: Double = 0
    @State private var plusScale: CGFloat = 0.7

    // Everything is laid out in a 200×200 design space (the same one the mark was drawn in) and
    // scaled to `size`, so stroke weights and the glyph track the frame together.
    private var k: CGFloat { size / 200 }

    var body: some View {
        Button(action: onAdd) {
            ZStack {
                if showsScrim {
                    RadialGradient(colors: [Brand.ink.opacity(0.55), Brand.ink.opacity(0)],
                                   center: .center, startRadius: 0, endRadius: size * 0.85)
                        .frame(width: size * 1.7, height: size * 1.7)
                        .allowsHitTesting(false)
                }

                // One open stroke, so `.trim` can pour the glass in from one rim to the other.
                GlassShape()
                    .trim(from: 0, to: glassTrim)
                    .stroke(Brand.cream,
                            style: StrokeStyle(lineWidth: 5 * k, lineCap: .round, lineJoin: .round))
                    .frame(width: size, height: size)

                GlassBaseShape()
                    .stroke(Brand.cream, style: StrokeStyle(lineWidth: 4 * k, lineCap: .round))
                    .frame(width: size, height: size)
                    .opacity(baseOpacity)

                // The prompt. `?` and `+` share a centre and cross-fade; only one is ever really
                // present, so the blink reads as the same mark changing its mind.
                ZStack {
                    Text("?")
                        .font(.system(size: 44 * k, weight: .medium))
                        .foregroundStyle(Brand.amber)
                        .opacity(qOpacity)
                        .scaleEffect(qScale)

                    PlusShape()
                        .stroke(Brand.amber, style: StrokeStyle(lineWidth: 5.5 * k, lineCap: .round))
                        .frame(width: size, height: size)
                        .opacity(plusOpacity)
                        .scaleEffect(plusScale)
                }
                // The glass bowl sits a hair below the frame's centre (it spans 66–140 of 200);
                // nudge the prompt down to float inside it rather than above the base.
                .offset(y: 3 * k)
            }
            .frame(width: size, height: size)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Add this drink")
        .accessibilityHint("Not in the catalog yet")
        // `.task` is bound to the view's lifetime: it starts on appear and is cancelled on
        // disappear, which is what ends the blink loop below.
        .task { await animate() }
    }

    @MainActor
    private func animate() async {
        guard !reduceMotion else {
            glassTrim = 1; baseOpacity = 0.45; plusOpacity = 1; plusScale = 1
            return
        }

        // 1 · The glass traces itself in, unhurried; the base settles under it near the end.
        withAnimation(.easeInOut(duration: 1.5)) { glassTrim = 1 }
        withAnimation(.easeInOut(duration: 0.5).delay(1.0)) { baseOpacity = 0.45 }
        try? await Task.sleep(for: .seconds(1.6))

        // 2 · Name the problem: the `?` pops gently into the empty glass and holds a few seconds.
        withAnimation(.spring(response: 0.5, dampingFraction: 0.62)) { qOpacity = 1; qScale = 1 }
        try? await Task.sleep(for: .seconds(3.3))

        // 3 · Hand off to the offer — `?` out, `+` in — then breathe slowly between the two.
        withAnimation(.easeInOut(duration: 0.5)) { qOpacity = 0 }
        withAnimation(.spring(response: 0.5, dampingFraction: 0.62)) { plusOpacity = 1; plusScale = 1 }
        try? await Task.sleep(for: .seconds(1.8))

        var showPlus = false
        while !Task.isCancelled {
            withAnimation(.easeInOut(duration: 0.5)) {
                plusOpacity = showPlus ? 1 : 0
                qOpacity = showPlus ? 0 : 1
            }
            showPlus.toggle()
            try? await Task.sleep(for: .seconds(1.8))
        }
    }
}

// MARK: - Marks

/// An open-topped tumbler, as a single stroke so it can be drawn on with `.trim`. Points live in
/// a 200×200 space and scale to the view's frame.
private struct GlassShape: Shape {
    func path(in rect: CGRect) -> Path {
        let s = min(rect.width, rect.height) / 200
        func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: x * s, y: y * s) }
        var path = Path()
        path.move(to: p(74, 66))
        path.addLine(to: p(82, 140))
        path.addLine(to: p(118, 140))
        path.addLine(to: p(126, 66))
        return path
    }
}

/// The thick base: a short line inset from the walls.
private struct GlassBaseShape: Shape {
    func path(in rect: CGRect) -> Path {
        let s = min(rect.width, rect.height) / 200
        var path = Path()
        path.move(to: CGPoint(x: 84 * s, y: 132 * s))
        path.addLine(to: CGPoint(x: 116 * s, y: 132 * s))
        return path
    }
}

/// The add mark — two strokes, deliberately small so it floats clear of the glass walls.
private struct PlusShape: Shape {
    func path(in rect: CGRect) -> Path {
        let s = min(rect.width, rect.height) / 200
        var path = Path()
        path.move(to: CGPoint(x: 90 * s, y: 100 * s))
        path.addLine(to: CGPoint(x: 110 * s, y: 100 * s))
        path.move(to: CGPoint(x: 100 * s, y: 90 * s))
        path.addLine(to: CGPoint(x: 100 * s, y: 110 * s))
        return path
    }
}

#if DEBUG
#Preview {
    ZStack {
        LinearGradient(colors: [.black, .gray.opacity(0.6)], startPoint: .top, endPoint: .bottom)
            .ignoresSafeArea()
        ScanNotFoundGlyph(size: 140) { }
    }
}
#endif
