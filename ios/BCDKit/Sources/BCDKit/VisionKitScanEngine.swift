import Foundation

#if canImport(VisionKit) && canImport(UIKit) && os(iOS)
import VisionKit
import UIKit

/// The VisionKit scanner. Wraps `DataScannerViewController` with
/// `recognizesMultipleItems: true` and `qualityLevel: .fast` per the latency budget
/// (barcode < 100ms, text line < 400ms p50). Emits normalized bounding boxes (view space)
/// so the tracker can group lines into objects and the HUD can anchor overlays.
///
/// What it can't do: segment cans/bottles, or turn off the language correction that
/// mangles stylized brand names. It gets by with text clustering for the coarse stage,
/// and for the fine stage it captures a still (`capturePhoto`) and runs the careful
/// `VisionTextReader` on the object's crop. `VisionFrameScanEngine` is the fuller
/// pipeline; this one is the known-good fallback.
///
/// Compiled only into the iOS app. On the macOS host this file is empty, so BCDKit still
/// builds for `swift test`.
@available(iOS 18.0, *)
public final class VisionKitScanEngine: NSObject, ScanEngine, @unchecked Sendable {
    public let frames: AsyncStream<ScanFrame>
    private var continuation: AsyncStream<ScanFrame>.Continuation?
    private var scanner: DataScannerViewController?
    private let lock = NSLock()
    private var _lexicon: [String] = []
    private var viewSize: CGSize = .zero

    public override init() {
        var cont: AsyncStream<ScanFrame>.Continuation!
        self.frames = AsyncStream { cont = $0 }
        self.continuation = cont
        super.init()
    }

    public var viewController: UIViewController? { scanner }

    @MainActor public func makeScanner() -> DataScannerViewController {
        let scanner = DataScannerViewController(
            recognizedDataTypes: [.text(), .barcode()],
            // `.balanced`, not `.fast`. Reported from the camera as "the barcode is a bit hard
            // to read", and the log agreed: one clean decode in roughly two minutes, while
            // VisionKit repeatedly read the digits *printed under* the code as text -- "11726",
            // "3573 11726" -- instead of decoding the symbology. `.fast` trades exactly that
            // accuracy away. The budget is there: a barcode frame now resolves server-side in
            // ~50ms against a 350ms tick.
            qualityLevel: .balanced,
            recognizesMultipleItems: true,
            isHighFrameRateTrackingEnabled: true,
            isHighlightingEnabled: false // we draw our own HUD overlays
        )
        scanner.delegate = self
        self.scanner = scanner
        return scanner
    }

    public func start() async {
        // Hop to the main actor: DataScannerViewController is @MainActor-isolated, but
        // ScanEngine.start() is a non-isolated protocol requirement. Capture self (which is
        // @unchecked Sendable), not the non-Sendable scanner, to stay race-clean.
        await MainActor.run { try? self.scanner?.startScanning() }
    }

    public func stop() {
        continuation?.finish()
        Task { @MainActor in self.scanner?.stopScanning() }
    }

    /// The frame itself, for the labels OCR cannot read.
    ///
    /// A craft can's wordmark is a drawing, not type, and by the time it reaches this file it
    /// is already "FADY TOPPE" — the information is gone before any matching starts. Sending
    /// the picture is the only way to get it back. `capturePhoto()` reuses the scanner's own
    /// session, so this costs no second camera and no interruption to the live viewfinder.
    ///
    /// Downscaled hard on the way out. A 48MP still is nothing but upload latency: a label
    /// legible at 1024px is legible to the model, and this runs on someone's cellular
    /// connection in a shop.
    public func captureFrame() async -> Data? {
        guard let scanner else { return nil }
        guard let photo = try? await scanner.capturePhoto() else { return nil }
        return await MainActor.run { Self.jpeg(photo) }
    }

    static let maxCaptureEdge: CGFloat = 1024
    static let captureQuality: CGFloat = 0.6

    @MainActor static func jpeg(_ image: UIImage) -> Data? {
        let longest = max(image.size.width, image.size.height)
        guard longest > 0 else { return nil }
        let scale = min(1, maxCaptureEdge / longest)
        guard scale < 1 else { return image.jpegData(compressionQuality: captureQuality) }
        let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1                      // points, not device pixels — this is already 3x
        let shrunk = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
        return shrunk.jpegData(compressionQuality: captureQuality)
    }

    private func emit(_ items: [RecognizedItem], in bounds: CGSize) {
        lock.withLock { viewSize = bounds }
        let detections: [DetectedText] = items.compactMap { item in
            switch item {
            case .text(let text):
                return Self.detected(text.transcript, kind: "text", bounds: item.bounds, in: bounds)
            case .barcode(let code):
                return Self.detected(code.payloadStringValue ?? "", kind: "barcode",
                                     symbology: code.observation.symbology.rawValue,
                                     bounds: item.bounds, in: bounds)
            @unknown default:
                return nil
            }
        }
        continuation?.yield(ScanFrame(texts: detections))
    }

    private static func detected(_ text: String, kind: String, symbology: String? = nil,
                                 bounds: RecognizedItem.Bounds, in size: CGSize) -> DetectedText? {
        guard !text.isEmpty, size.width > 0, size.height > 0 else { return nil }
        let minX = min(bounds.topLeft.x, bounds.bottomLeft.x)
        let minY = min(bounds.topLeft.y, bounds.topRight.y)
        let maxX = max(bounds.topRight.x, bounds.bottomRight.x)
        let maxY = max(bounds.bottomLeft.y, bounds.bottomRight.y)
        return DetectedText(
            text: text, kind: kind, symbology: symbology,
            x: minX / size.width, y: minY / size.height,
            w: (maxX - minX) / size.width, h: (maxY - minY) / size.height
        )
    }
}

@available(iOS 18.0, *)
extension VisionKitScanEngine: LexiconConsumer {
    public var lexicon: [String] {
        get { lock.withLock { _lexicon } }
        set { lock.withLock { _lexicon = newValue } }
    }
}

@available(iOS 18.0, *)
extension VisionKitScanEngine: FineTextReader {
    /// Snap a still, map the (view-space) object box onto it, and read it carefully.
    /// The scanner's preview is aspect-fill, so the photo overflows the view on one axis;
    /// `AspectFillMapper` undoes that. The crop is padded, so a few points of error in
    /// the mapping don't cut the label.
    public func readText(in box: BoundingBox) async -> [String] {
        let words = lexicon
        let size = lock.withLock { viewSize }
        let image: UIImage? = await Task { @MainActor in
            try? await self.scanner?.capturePhoto()
        }.value
        guard let image, let cg = Self.upright(image) else { return [] }
        var target = box
        if size.width > 0, size.height > 0, cg.width > 0, cg.height > 0 {
            let mapper = AspectFillMapper(contentAspect: Double(cg.width) / Double(cg.height))
            target = mapper.toContent(box, viewAspect: Double(size.width / size.height))
        }
        let reader = VisionTextReader(lexicon: words)
        let texts = (try? await reader.read(cg, in: target.expanded(by: 0.05))) ?? []
        return texts.map(\.text)
    }

    /// `UIImage.cgImage` is in sensor orientation; redraw so Vision sees it upright.
    private static func upright(_ image: UIImage) -> CGImage? {
        if image.imageOrientation == .up { return image.cgImage }
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: image.size)) }.cgImage
    }
}

@available(iOS 18.0, *)
extension VisionKitScanEngine: DataScannerViewControllerDelegate {
    public func dataScanner(_ scanner: DataScannerViewController,
                            didAdd addedItems: [RecognizedItem],
                            allItems: [RecognizedItem]) {
        emit(allItems, in: scanner.view.bounds.size)
    }

    public func dataScanner(_ scanner: DataScannerViewController,
                            didUpdate updatedItems: [RecognizedItem],
                            allItems: [RecognizedItem]) {
        emit(allItems, in: scanner.view.bounds.size)
    }

    public func dataScanner(_ scanner: DataScannerViewController,
                            didRemove removedItems: [RecognizedItem],
                            allItems: [RecognizedItem]) {
        emit(allItems, in: scanner.view.bounds.size)
    }
}
#endif
