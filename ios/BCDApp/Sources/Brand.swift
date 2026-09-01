import SwiftUI
import BCDKit

// The identity, in one place. Colours are asset-catalog colour sets rather than literals
// so light and dark are decided once, in the catalog, and every surface picks up the same
// pair. Names mirror the design canvas ("BCD Icon System") exactly.

enum Brand {
    // Identity — the scan-frame mark's own three colours.
    static let ink = Color("BrandInk")
    static let amber = Color("BrandAmber")
    static let cream = Color("BrandCream")

    // Surfaces.
    static let surface = Color("BrandSurface")
    static let tile = Color("BrandTile")
    static let hairline = Color("BrandHairline")
    static let text = Color("BrandText")
    static let textMuted = Color("BrandTextMuted")

    /// Rest state for an unpicked reaction — deliberately neutral so no level is
    /// pre-suggested before the user chooses one.
    static let reactionRest = Color("ReactionRest")

    // The reaction glyph size ramp. 22 is the floor: below it the blush and the spray
    // fill in and the five stop being separable.
    enum GlyphSize {
        static let recall: CGFloat = 24     // a rated row in search
        static let picker: CGFloat = 46     // five across a phone, still under `display`
        static let hitTarget: CGFloat = 52
        static let display: CGFloat = 62
    }
}

extension Reaction {
    /// The ramp colour for this level — oklch(L 0.13 H) with only the hue moving, resolved
    /// per appearance in the catalog (L 0.70 dark, L 0.52 light).
    var tint: Color { Color("Reaction\(rawValue)") }

    /// The glyph. The heavier 2.0 stroke is cut as its own asset for use at or below
    /// 24pt, where the display weight thins out.
    func glyph(at size: CGFloat) -> Image {
        Image(size <= Brand.GlyphSize.recall ? "Reaction\(rawValue)Small" : "Reaction\(rawValue)")
    }
}
