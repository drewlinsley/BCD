import SwiftUI
import Combine
import BCDKit
#if canImport(VisionKit) && os(iOS)
import VisionKit
import AVFoundation
#endif

// The camera HUD — the whole product thesis in one screen. Live detections become
// overlays anchored to their bounding boxes, color-coded by predicted enjoyment. A
// persistent chat bar routes natural-language asks against the items currently in frame.

struct ScanView: View {
    @EnvironmentObject var env: AppEnvironment
    @StateObject private var model = ScanViewModel()
    @State private var ask: String = ""
    @State private var selected: ScoredCandidate?

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraLayer(engine: model.engine)  // live DataScanner on device; gradient in Sim/host
                .ignoresSafeArea()

            // Once captured, dim the live feed so the frozen result reads as "analyzed".
            if model.captured {
                Color.black.opacity(0.35).ignoresSafeArea().allowsHitTesting(false)
            }

            GeometryReader { geo in
                ForEach(model.overlays) { overlay in
                    OverlayChip(candidate: overlay.candidate)
                        .position(x: overlay.anchor.x * geo.size.width,
                                  y: overlay.anchor.y * geo.size.height)
                        .onTapGesture { selected = overlay.candidate }
                }
            }

            VStack(spacing: 12) {
                statusPill
                if model.captured && !model.overlays.isEmpty { chatBar }
                shutterRow
            }
            .padding()
        }
        .task { model.configure(env: env); model.start() }
        .onDisappear { model.stop() }
        .sheet(item: $selected) { cand in
            ProductDetailView(candidate: cand)
        }
    }

    // A one-line status: what to do, that we're working, or what we found.
    @ViewBuilder private var statusPill: some View {
        if model.isResolving {
            pill("Analyzing…", system: "hourglass")
        } else if model.captured {
            let n = model.overlays.count
            pill(n == 0 ? "No products found" : "\(n) found"
                    + (model.lastLatencyMs.map { " · \(Int($0))ms" } ?? ""),
                 system: n == 0 ? "questionmark.circle" : "checkmark.circle.fill")
        } else {
            pill("Point at a shelf, then tap to scan", system: "viewfinder")
        }
    }

    private func pill(_ text: String, system: String) -> some View {
        Label(text, systemImage: system)
            .font(.caption).foregroundStyle(.white)
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
    }

    // Shutter when live; Rescan when a result is frozen.
    @ViewBuilder private var shutterRow: some View {
        if model.captured {
            Button { model.rescan() } label: {
                Label("Rescan", systemImage: "arrow.counterclockwise")
                    .font(.headline).foregroundStyle(.white)
                    .padding(.horizontal, 24).padding(.vertical, 14)
                    .background(.ultraThinMaterial, in: Capsule())
            }
        } else {
            Button { model.capture() } label: {
                ZStack {
                    Circle().strokeBorder(.white, lineWidth: 4).frame(width: 74, height: 74)
                    Circle().fill(.white).frame(width: 60, height: 60)
                }
            }
            .disabled(model.isResolving)
            .opacity(model.isResolving ? 0.5 : 1)
        }
    }

    private var chatBar: some View {
        HStack {
            Image(systemName: "sparkles")
            TextField("cheapest hazy here · nothing over 6%", text: $ask)
                .textFieldStyle(.plain)
                .submitLabel(.search)
                .onSubmit { Task { await model.applyAsk(ask) } }
            if !ask.isEmpty {
                Button { ask = "" } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

/// One anchored overlay: a candidate + where to draw it (normalized 0-1).
struct HUDOverlay: Identifiable {
    let id: String
    let candidate: ScoredCandidate
    let anchor: CGPoint
}

struct OverlayChip: View {
    let candidate: ScoredCandidate

    private var tint: Color {
        guard let s = candidate.personalScore else { return .gray }
        return s > 0.75 ? .green : (s > 0.5 ? .yellow : .orange)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(candidate.resolved.product.name).font(.subheadline.bold()).lineLimit(1)
            HStack(spacing: 6) {
                if let s = candidate.personalScore {
                    Label("\(Int(s * 100))", systemImage: "hand.thumbsup.fill").font(.caption2)
                }
                if candidate.coldStart {
                    Image(systemName: "flask.fill").font(.caption2)  // scored from chemistry
                }
            }
            if let reason = candidate.reason {
                Text(reason).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(tint, lineWidth: 2))
        .frame(maxWidth: 180)
    }
}

@MainActor
final class ScanViewModel: ObservableObject {
    @Published var overlays: [HUDOverlay] = []
    @Published var lastLatencyMs: Double?
    @Published var captured = false
    @Published var isResolving = false
    /// The engine the coordinator consumes. Exposed so the camera layer can present *this*
    /// engine's scanner view — it must be the same instance, or detections wouldn't reach the HUD.
    @Published private(set) var engine: ScanEngine?

    private var coordinator: ScanCoordinator?
    private var env: AppEnvironment?

    func configure(env: AppEnvironment) {
        guard coordinator == nil else { return }
        self.env = env
        let engine = env.makeScanEngine()
        self.engine = engine
        let coord = ScanCoordinator(engine: engine, api: env.api, telemetry: env.telemetry)
        self.coordinator = coord
        // Mirror the coordinator's frozen, box-anchored overlays straight into the view.
        coord.$overlays
            .map { ovs in
                ovs.map { HUDOverlay(id: $0.id, candidate: $0.candidate,
                                     anchor: CGPoint(x: $0.x, y: $0.y)) }
            }
            .assign(to: &$overlays)
        coord.$lastLatencyMs.assign(to: &$lastLatencyMs)
        coord.$captured.assign(to: &$captured)
        coord.$isResolving.assign(to: &$isResolving)
    }

    func start() { coordinator?.start() }
    func stop() { coordinator?.stop() }
    func capture() { Task { await coordinator?.capture() } }
    func rescan() { coordinator?.rescan() }

    /// Chat-bar rerank: ask the LLM to order the captured products, then re-pin in that order.
    func applyAsk(_ ask: String) async {
        guard let env, let coord = coordinator, !ask.isEmpty else { return }
        let order = (try? await env.llm.rerank(coord.capturedCandidates, for: ask)) ?? []
        coord.rerank(order: order)
    }
}

/// Camera layer. On a real device it presents VisionKit's `DataScannerViewController` (live
/// text + barcode) driven by the shared engine; in the Simulator or on the host — no camera —
/// it falls back to a neutral gradient so the HUD stays previewable.
struct CameraLayer: View {
    var engine: ScanEngine?

    var body: some View {
        #if canImport(VisionKit) && os(iOS)
        if #available(iOS 18.0, *), DataScannerViewController.isSupported,
           let vk = engine as? VisionKitScanEngine {
            DataScannerView(engine: vk)
        } else {
            placeholder
        }
        #else
        placeholder
        #endif
    }

    private var placeholder: some View {
        LinearGradient(colors: [.black, .gray.opacity(0.6)],
                       startPoint: .top, endPoint: .bottom)
    }
}

#if canImport(VisionKit) && os(iOS)
/// Presents the engine's `DataScannerViewController` and drives its lifecycle: request camera
/// access (this is what makes iOS show the permission prompt), then start scanning. Detections
/// flow out through the engine's delegate → `frames` → the coordinator, which the HUD renders.
@available(iOS 18.0, *)
struct DataScannerView: UIViewControllerRepresentable {
    let engine: VisionKitScanEngine

    func makeUIViewController(context: Context) -> DataScannerViewController {
        engine.makeScanner()
    }

    func updateUIViewController(_ scanner: DataScannerViewController, context: Context) {
        guard !context.coordinator.started else { return }
        context.coordinator.started = true
        let engine = self.engine  // @unchecked Sendable — safe to hand to the async closure
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return }
            Task { await engine.start() }  // hops to @MainActor, starts the created scanner
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var started = false }
}
#endif
