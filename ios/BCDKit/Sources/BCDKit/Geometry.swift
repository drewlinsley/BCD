import Foundation

/// A normalized (0-1) rectangle with origin top-left — the HUD's coordinate space, and the
/// space every `DetectedText`, `ObjectRegion` and `DetectedObject` box lives in.
public struct BoundingBox: Sendable, Equatable, Hashable, Codable {
    public var x: Double
    public var y: Double
    public var w: Double
    public var h: Double

    public init(x: Double, y: Double, w: Double, h: Double) {
        self.x = x; self.y = y; self.w = w; self.h = h
    }

    public static let unit = BoundingBox(x: 0, y: 0, w: 1, h: 1)

    public var minX: Double { x }
    public var minY: Double { y }
    public var maxX: Double { x + w }
    public var maxY: Double { y + h }
    public var midX: Double { x + w / 2 }
    public var midY: Double { y + h / 2 }
    public var area: Double { max(0, w) * max(0, h) }

    public func contains(x px: Double, y py: Double) -> Bool {
        px >= minX && px <= maxX && py >= minY && py <= maxY
    }

    public func intersection(_ o: BoundingBox) -> BoundingBox? {
        let ix = max(minX, o.minX), iy = max(minY, o.minY)
        let ax = min(maxX, o.maxX), ay = min(maxY, o.maxY)
        guard ax > ix, ay > iy else { return nil }
        return BoundingBox(x: ix, y: iy, w: ax - ix, h: ay - iy)
    }

    public func iou(_ o: BoundingBox) -> Double {
        guard let i = intersection(o) else { return 0 }
        let u = area + o.area - i.area
        return u > 0 ? i.area / u : 0
    }

    public func union(_ o: BoundingBox) -> BoundingBox {
        let nx = min(minX, o.minX), ny = min(minY, o.minY)
        return BoundingBox(x: nx, y: ny, w: max(maxX, o.maxX) - nx, h: max(maxY, o.maxY) - ny)
    }

    /// Grow by `pad` (a fraction of the frame) on every side, clamped to the unit square.
    public func expanded(by pad: Double) -> BoundingBox {
        let nx = max(0, minX - pad), ny = max(0, minY - pad)
        return BoundingBox(x: nx, y: ny, w: min(1, maxX + pad) - nx, h: min(1, maxY + pad) - ny)
    }

    /// Linear blend toward `o` (0 = keep self, 1 = take o). Used to smooth tracked boxes.
    public func blended(toward o: BoundingBox, _ t: Double) -> BoundingBox {
        BoundingBox(x: x + (o.x - x) * t, y: y + (o.y - y) * t,
                    w: w + (o.w - w) * t, h: h + (o.h - h) * t)
    }

    public static func enclosing(_ boxes: [BoundingBox]) -> BoundingBox? {
        guard var acc = boxes.first else { return nil }
        for b in boxes.dropFirst() { acc = acc.union(b) }
        return acc
    }

    public func distance(to o: BoundingBox) -> Double {
        let dx = midX - o.midX, dy = midY - o.midY
        return (dx * dx + dy * dy).squareRoot()
    }
}

/// Maps boxes normalized to a *content* (a camera buffer with a fixed aspect) onto a view
/// that displays that content aspect-fill — the way `AVCaptureVideoPreviewLayer` with
/// `.resizeAspectFill` does. Without this, overlays drift toward the edges on any phone
/// whose screen isn't the sensor's aspect. `contentAspect` is width / height.
public struct AspectFillMapper: Sendable, Equatable {
    public let contentAspect: Double

    public init(contentAspect: Double) { self.contentAspect = contentAspect }

    /// The rectangle (in view-normalized coordinates) the content occupies; it overflows
    /// the unit square on the axis that gets cropped.
    public func contentFrame(inViewAspect viewAspect: Double) -> BoundingBox {
        guard contentAspect > 0, viewAspect > 0 else { return .unit }
        if contentAspect > viewAspect {
            // content is wider than the view: full height, cropped sides
            let w = contentAspect / viewAspect
            return BoundingBox(x: (1 - w) / 2, y: 0, w: w, h: 1)
        } else {
            let h = viewAspect / contentAspect
            return BoundingBox(x: 0, y: (1 - h) / 2, w: 1, h: h)
        }
    }

    /// Content-normalized box -> view-normalized box.
    public func toView(_ b: BoundingBox, viewAspect: Double) -> BoundingBox {
        let f = contentFrame(inViewAspect: viewAspect)
        return BoundingBox(x: f.x + b.x * f.w, y: f.y + b.y * f.h, w: b.w * f.w, h: b.h * f.h)
    }

    /// View-normalized box -> content-normalized box (clamped to the content).
    public func toContent(_ b: BoundingBox, viewAspect: Double) -> BoundingBox {
        let f = contentFrame(inViewAspect: viewAspect)
        let raw = BoundingBox(x: (b.x - f.x) / f.w, y: (b.y - f.y) / f.h, w: b.w / f.w, h: b.h / f.h)
        let nx = max(0, raw.minX), ny = max(0, raw.minY)
        return BoundingBox(x: nx, y: ny, w: max(0, min(1, raw.maxX) - nx),
                           h: max(0, min(1, raw.maxY) - ny))
    }
}
