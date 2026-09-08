import Foundation

#if canImport(Vision) && os(iOS)
import Vision
import CoreGraphics
import ImageIO

/// The fine stage's OCR: Vision's `RecognizeTextRequest` at `.accurate` on a crop of one
/// object, run two ways and unioned —
///   1. language correction **off**: raw glyph reads. The coarse scanner's correction is
///      what turns "ALCHEMIST" into "Chemist" and "HEADY" into "Ready"; the server's
///      matcher would rather have the uncorrected string and tolerate a wrong letter.
///   2. language correction **on** with the catalog lexicon as `customWords`: the
///      recognizer is steered toward "Alchemist" rather than any dictionary word.
/// Also reads barcodes in the crop. Compiled only into the iOS app.
@available(iOS 18.0, *)
public struct VisionTextReader: Sendable {
    public var lexicon: [String]

    public init(lexicon: [String] = []) { self.lexicon = lexicon }

    /// Read `image` (upright), optionally only inside `box` (normalized, top-left origin,
    /// padded a little to survive imprecise object boxes). Boxes in the result are
    /// normalized to the *whole* image.
    public func read(_ image: CGImage, in box: BoundingBox? = nil) async throws -> [DetectedText] {
        let crop = box.map { $0.expanded(by: 0.03) } ?? .unit
        let W = Double(image.width), H = Double(image.height)
        let rect = CGRect(x: crop.x * W, y: crop.y * H, width: crop.w * W, height: crop.h * H)
            .integral
        guard rect.width >= 8, rect.height >= 8, let cg = image.cropping(to: rect) else { return [] }

        var raw = RecognizeTextRequest()
        raw.recognitionLevel = .accurate
        raw.usesLanguageCorrection = false

        var steered = RecognizeTextRequest()
        steered.recognitionLevel = .accurate
        steered.usesLanguageCorrection = true
        steered.customWords = lexicon

        var out: [DetectedText] = []
        var seen = Set<String>()
        func add(_ t: DetectedText) {
            let key = t.kind + ":" + t.text.uppercased()
            guard !seen.contains(key) else { return }
            seen.insert(key)
            out.append(t)
        }
        for req in [raw, steered] {
            let observations = try await req.perform(on: cg)
            for o in observations {
                guard let top = o.topCandidates(1).first, !top.string.isEmpty else { continue }
                let r = o.boundingBox.toImageCoordinates(CGSize(width: 1, height: 1), origin: .upperLeft)
                add(DetectedText(text: top.string, kind: "text",
                                 x: crop.x + r.minX * crop.w, y: crop.y + r.minY * crop.h,
                                 w: r.width * crop.w, h: r.height * crop.h,
                                 confidence: Double(top.confidence)))
            }
        }
        if let codes = try? await DetectBarcodesRequest().perform(on: cg) {
            for c in codes {
                guard let payload = c.payloadString, !payload.isEmpty else { continue }
                let r = c.boundingBox.toImageCoordinates(CGSize(width: 1, height: 1), origin: .upperLeft)
                add(DetectedText(text: payload, kind: "barcode",
                                 symbology: String(describing: c.symbology),
                                 x: crop.x + r.minX * crop.w, y: crop.y + r.minY * crop.h,
                                 w: r.width * crop.w, h: r.height * crop.h))
            }
        }
        return out
    }
}
#endif
