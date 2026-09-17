import Foundation

/// Where the chips go, and how they stay there.
///
/// Reported from the camera, twice: "the HUD text box jumps all over the place when it pops
/// up" (2026-09-15) and, after the first fix, "the boxes are still jumpy and kind of swimming
/// across the screen" (2026-09-17) -- with "spacing and differentiation when there are
/// multiple items on screen" the other half of the ask. Three rules, all here so they can be
/// tested without a camera:
///
///   * A chip is **pinned** where it first appeared. It ignores an anchor that wanders within
///     `deadZone` of the pin, and follows one that has stayed outside it for `settle` --
///     in one move, not a little each frame. The first fix moved the chip a third of the way
///     toward its anchor on every frame once the anchor left the dead zone, and the tracker
///     re-blends a box on every frame the recognizer emits (thirty a second on a live shelf):
///     that is a chip forever chasing a target it never reaches, which is the swim.
///   * Chips are **spread** so none covers another: a chip that would overlap one already
///     placed drops below it, deterministically, so the same pins give the same layout.
///   * An object's chip sits **above its box**, off the label, tied to the box by an anchor
///     point the HUD can draw a leader to when the chip had to move; the box itself is
///     handed along for a thin outline, which is what tells two cans' chips apart.
public enum HUDLayout {
    /// How far (normalised, straight-line) an anchor may wander before the chip is asked to
    /// move. A label's lines sit within about a tenth of the screen of each other on a
    /// bottle filling the frame, and the tracker's box wanders about that much as lines
    /// come and go.
    public static let deadZone = 0.10
    /// How long an anchor has to stay outside the dead zone before the chip follows it. Hand
    /// jitter is a few frames; a pan is longer than this.
    public static let settle: TimeInterval = 0.5

    /// The chip's footprint, normalised, estimated from what it says. The HUD's chip is at
    /// most 180 pt wide on a 393 pt screen; its title wraps to two lines past about
    /// `titleLineChars` characters, and a reason adds a line. The estimate errs wide so a
    /// near miss still spreads.
    public static let chipMaxWidth = 0.46
    public static let chipHeight = 0.065
    public static let chipHeightWithReason = 0.08
    public static let titleLineChars = 22
    public static let titleExtraLine = 0.022
    /// Space left between chips, and between a chip and the box it sits above.
    public static let gap = 0.012
    /// A chip never sits in the top or bottom band, where the status pill and chat bar are.
    public static let topMargin = 0.08
    public static let bottomMargin = 0.22

    public struct Pin: Sendable, Equatable {
        public var x: Double
        public var y: Double
        /// When the anchor first left the dead zone, if it is out now.
        public var driftingSince: Date?
    }

    /// The pin for this anchor: the old one while the anchor stays near it, or is only just
    /// out; a new one at the anchor once it has been out for `settle`.
    public static func steadied(_ pin: Pin?, anchorX: Double, anchorY: Double, now: Date) -> Pin {
        guard var pin else { return Pin(x: anchorX, y: anchorY, driftingSince: nil) }
        let (dx, dy) = (anchorX - pin.x, anchorY - pin.y)
        if (dx * dx + dy * dy).squareRoot() <= deadZone {
            pin.driftingSince = nil
            return pin
        }
        guard let since = pin.driftingSince else {
            pin.driftingSince = now
            return pin
        }
        if now.timeIntervalSince(since) >= settle {
            return Pin(x: anchorX, y: anchorY, driftingSince: nil)
        }
        return pin
    }

    /// What the chip prints as its title -- the label's name, brand and all.
    public static func title(of candidate: ScoredCandidate) -> String {
        DisplayName.label(candidate.resolved.product.name, brand: candidate.resolved.brand.name)
    }

    static func size(name: String, hasReason: Bool) -> (w: Double, h: Double) {
        let w = min(chipMaxWidth, 0.06 + 0.02 * Double(name.count))
        var h = hasReason ? chipHeightWithReason : chipHeight
        if name.count > titleLineChars { h += titleExtraLine }
        return (w, h)
    }

    /// The footprint of a chip centred at (x, y).
    public static func footprint(x: Double, y: Double, name: String, hasReason: Bool) -> BoundingBox {
        let (w, h) = size(name: name, hasReason: hasReason)
        return BoundingBox(x: x - w / 2, y: y - h / 2, w: w, h: h)
    }

    /// Where an object's chip wants to be: above the box, centred, clear of the screen's
    /// edges -- or below it when the box is up against the top. The anchor is the point on
    /// the box the chip belongs to.
    public static func perch(for box: BoundingBox, name: String, hasReason: Bool)
        -> (x: Double, y: Double, anchorX: Double, anchorY: Double) {
        let (w, h) = size(name: name, hasReason: hasReason)
        let x = min(max(box.midX, w / 2 + gap), 1 - w / 2 - gap)
        let above = box.minY - gap - h / 2
        if above - h / 2 >= topMargin {
            return (x, above, box.midX, box.minY)
        }
        let below = box.maxY + gap + h / 2
        return (x, min(below, 1 - bottomMargin - h / 2), box.midX, box.maxY)
    }

    /// Chips laid out so none overlaps another. Taken in the order given (best first), each
    /// chip keeps its place unless it overlaps one already placed, in which case it moves
    /// just below the lowest chip it collides with -- and, if that would put it under the
    /// chat bar, just above the highest instead. The same input gives the same layout.
    public static func spread(_ overlays: [ResolvedOverlay]) -> [ResolvedOverlay] {
        var placed: [BoundingBox] = []
        var out: [ResolvedOverlay] = []
        for o in overlays {
            let name = title(of: o.candidate)
            let hasReason = o.candidate.reason != nil
            var y = o.y
            var rect = footprint(x: o.x, y: y, name: name, hasReason: hasReason)
            var tries = 0
            while let hit = placed.filter({ $0.intersection(rect) != nil }).max(by: { $0.maxY < $1.maxY }),
                  tries < 8 {
                y = hit.maxY + gap + rect.h / 2
                if y + rect.h / 2 > 1 - bottomMargin {
                    let top = placed.filter { $0.intersection(rect) != nil }.min(by: { $0.minY < $1.minY })!
                    y = top.minY - gap - rect.h / 2
                }
                rect = footprint(x: o.x, y: y, name: name, hasReason: hasReason)
                tries += 1
            }
            placed.append(rect)
            out.append(o.moved(toX: o.x, y: y))
        }
        return out
    }
}
