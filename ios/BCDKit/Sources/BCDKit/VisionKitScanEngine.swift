import Foundation

#if canImport(VisionKit) && canImport(UIKit) && os(iOS)
import VisionKit
import UIKit

/// Real on-device scanner. Wraps `DataScannerViewController` with
/// `recognizesMultipleItems: true` and `qualityLevel: .fast` per the latency budget
/// (barcode < 100ms, text line < 400ms p50). Emits normalized bounding boxes so the HUD
/// can anchor overlays onto the exact text/barcode in the camera frame.
///
/// Compiled only into the iOS app. On the macOS host this file is empty, so BCDKit still
/// builds for `swift test`.
@available(iOS 18.0, *)
public final class VisionKitScanEngine: NSObject, ScanEngine, @unchecked Sendable {
    public let frames: AsyncStream<[DetectedText]>
    private var continuation: AsyncStream<[DetectedText]>.Continuation?
    private var scanner: DataScannerViewController?

    public override init() {
        var cont: AsyncStream<[DetectedText]>.Continuation!
        self.frames = AsyncStream { cont = $0 }
        self.continuation = cont
        super.init()
    }

    public var viewController: UIViewController? { scanner }

    @MainActor public func makeScanner() -> DataScannerViewController {
        let scanner = DataScannerViewController(
            recognizedDataTypes: [.text(), .barcode()],
            qualityLevel: .fast,
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

    private func emit(_ items: [RecognizedItem], in bounds: CGSize) {
        let detections: [DetectedText] = items.compactMap { item in
            switch item {
            case .text(let text):
                return Self.detected(text.transcript, kind: "text", bounds: item.bounds, in: bounds)
            case .barcode(let code):
                return Self.detected(code.payloadStringValue ?? "", kind: "barcode",
                                     bounds: item.bounds, in: bounds)
            @unknown default:
                return nil
            }
        }
        continuation?.yield(detections)
    }

    private static func detected(_ text: String, kind: String,
                                 bounds: RecognizedItem.Bounds, in size: CGSize) -> DetectedText? {
        guard !text.isEmpty, size.width > 0, size.height > 0 else { return nil }
        let minX = min(bounds.topLeft.x, bounds.bottomLeft.x)
        let minY = min(bounds.topLeft.y, bounds.topRight.y)
        let maxX = max(bounds.topRight.x, bounds.bottomRight.x)
        let maxY = max(bounds.bottomLeft.y, bounds.bottomRight.y)
        return DetectedText(
            text: text, kind: kind,
            x: minX / size.width, y: minY / size.height,
            w: (maxX - minX) / size.width, h: (maxY - minY) / size.height
        )
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
