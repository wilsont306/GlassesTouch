//
// PhoneCamera.swift
//
// Streams this phone's back camera (0.5x ultra-wide, else 1x wide) through FrameServer, so the
// Mac can read it from a stand instead of the glasses. Runs only while the view is on screen.
//

import AVFoundation
import CoreImage
import SwiftUI
import UIKit

// @unchecked: the session objects are only touched on sessionQueue, state on the main actor.
final class PhoneCamera: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
  @Published private(set) var isRunning = false
  @Published private(set) var lens = ""
  @Published private(set) var errorMessage: String?
  @Published private(set) var device: AVCaptureDevice?

  let session = AVCaptureSession()
  private let sessionQueue = DispatchQueue(label: "PhoneCamera.session")
  private let videoQueue = DispatchQueue(label: "PhoneCamera.video")
  private let ciContext = CIContext()
  // Touched only on sessionQueue.
  private var rotation: AVCaptureDevice.RotationCoordinator?
  private var rotationObservation: NSKeyValueObservation?
  // Main actor only: false once stop() runs, so a start still waiting on the camera permission won't run after it.
  private var wanted = false

  @MainActor
  func start() {
    wanted = true
    Task {
      guard await AVCaptureDevice.requestAccess(for: .video) else {
        errorMessage = "Allow camera access for GlassesTouch in Settings."
        return
      }
      guard wanted else { return }  // stopped while the permission prompt was up
      errorMessage = nil
      sessionQueue.async {
        let device = self.configure()
        if device != nil {
          FrameServer.shared.start()
          self.session.startRunning()
        }
        Task { @MainActor in
          guard let device else {
            self.errorMessage = "This phone's back camera is not available."
            return
          }
          self.device = device
          self.lens = device.deviceType == .builtInUltraWideCamera ? "0.5x ultra wide" : "1x wide"
          UIApplication.shared.isIdleTimerDisabled = true
          self.isRunning = true
        }
      }
    }
  }

  /// Stops the camera only. FrameServer keeps running because the glasses path shares it.
  @MainActor
  func stop() {
    wanted = false
    sessionQueue.async {
      self.session.stopRunning()
      self.session.inputs.forEach { self.session.removeInput($0) }
      self.session.outputs.forEach { self.session.removeOutput($0) }
      self.rotationObservation = nil
      self.rotation = nil
    }
    UIApplication.shared.isIdleTimerDisabled = false
    isRunning = false
  }

  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection
  ) {
    guard FrameServer.shared.hasClients, let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    // Sample timestamps are on the host clock, so map them to Unix time via the current host time.
    let age = CMClockGetTime(CMClockGetHostTimeClock()).seconds
      - CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
    let captureTime = Date().timeIntervalSince1970 - age
    guard
      let jpeg = ciContext.jpegRepresentation(
        of: CIImage(cvPixelBuffer: pixels),
        colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
        options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.7])
    else { return }
    FrameServer.shared.publish(jpeg: jpeg, captureTime: captureTime)
  }

  // MARK: - Private (sessionQueue only)

  /// Builds the session and returns the chosen camera, or nil if the session could not be set up.
  private func configure() -> AVCaptureDevice? {
    guard
      let device = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)
        ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
      let input = try? AVCaptureDeviceInput(device: device)
    else { return nil }
    let output = AVCaptureVideoDataOutput()
    output.alwaysDiscardsLateVideoFrames = true
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    output.setSampleBufferDelegate(self, queue: videoQueue)

    session.beginConfiguration()
    defer { session.commitConfiguration() }
    session.sessionPreset = .inputPriority
    guard (try? device.lockForConfiguration()) != nil else { return nil }
    if let format = bestFormat(of: device) { device.activeFormat = format }
    // a frame duration outside the format's ranges raises an exception, so only set 24 fps where it is allowed
    if device.activeFormat.videoSupportedFrameRateRanges.contains(where: { $0.minFrameRate <= 24 && 24 <= $0.maxFrameRate }) {
      device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 24)
      device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 24)
    }
    if device.isGeometricDistortionCorrectionSupported { device.isGeometricDistortionCorrectionEnabled = true }
    device.unlockForConfiguration()

    guard session.canAddInput(input), session.canAddOutput(output) else { return nil }
    session.addInput(input)
    session.addOutput(output)

    // Keep frames upright as the phone turns; the observer also fires once with the current angle.
    let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: nil)
    let connection = output.connection(with: .video)
    rotation = coordinator
    rotationObservation = coordinator.observe(
      \.videoRotationAngleForHorizonLevelCapture, options: [.initial, .new]
    ) { c, _ in
      let angle = c.videoRotationAngleForHorizonLevelCapture
      guard let connection, connection.isVideoRotationAngleSupported(angle) else { return }
      connection.videoRotationAngle = angle
    }
    return device
  }

  /// Largest format up to 1920x1440 that can run at 24 fps; 4:3 wins ties.
  private func bestFormat(of device: AVCaptureDevice) -> AVCaptureDevice.Format? {
    func size(_ format: AVCaptureDevice.Format) -> (w: Int, h: Int) {
      let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
      return (Int(d.width), Int(d.height))
    }
    func is43(_ s: (w: Int, h: Int)) -> Bool { s.w * 3 == s.h * 4 || s.h * 3 == s.w * 4 }
    return device.formats
      .filter { format in
        let s = size(format)
        return s.w * s.h <= 1920 * 1440 && format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 24 }
      }
      .max { a, b in
        let sa = size(a), sb = size(b)
        if sa.w * sa.h != sb.w * sb.h { return sa.w * sa.h < sb.w * sb.h }
        return !is43(sa) && is43(sb)
      }
  }
}

/// Full-screen view: live preview, stream address, Mac link status and a Stop button.
struct PhoneCameraView: View {
  @Environment(\.dismiss) private var dismiss
  @StateObject private var camera = PhoneCamera()
  @ObservedObject private var pairing = PairingModel.shared

  var body: some View {
    ZStack {
      Color.gtBg.ignoresSafeArea()

      VStack(alignment: .leading, spacing: 20) {
        VStack(alignment: .leading, spacing: 6) {
          Text("This phone is the camera")
            .font(.system(size: 28, weight: .semibold))
            .foregroundStyle(Color.gtText)
          Text(camera.lens)
            .font(.system(size: 16))
            .foregroundStyle(Color.gtMuted)
        }

        CameraPreview(session: camera.session, device: camera.device)
          .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
          .frame(maxWidth: .infinity, maxHeight: .infinity)

        VStack(alignment: .leading, spacing: 16) {
          // FrameServer's client count isn't published, so re-read it every second
          TimelineView(.periodic(from: .now, by: 1)) { _ in
            let status = macStatus(receiving: FrameServer.shared.hasClients)
            StatusRow(icon: "desktopcomputer", title: "Mac", value: status.value, tint: status.tint)
          }
          StatusRow(
            icon: "wifi",
            title: "Wi-Fi",
            tint: streamURL == nil ? Color.gtMuted : Color.gtLive)
          {
            if let streamURL = streamURL {
              AddressChip(address: streamURL)
            } else {
              Text("No Wi-Fi")
                .font(.system(.subheadline, design: .monospaced))
                .foregroundStyle(Color.gtMuted)
            }
          }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .gtCard()

        Text("Put the phone on a stand, back camera facing your Mac, with the whole screen and your hand in view. Keep this app open while you use it. If the Mac isn't reading, press Start in its panel, or Pair phone there again.")
          .font(.system(size: 14))
          .foregroundStyle(Color.gtMuted)

        if let errorMessage = camera.errorMessage {
          Text(errorMessage)
            .font(.system(size: 14))
            .foregroundStyle(Color.gtError)
        }

        CustomButton(title: "Stop", style: .destructive, isDisabled: false) {
          camera.stop()
          dismiss()
        }
      }
      .padding(24)
    }
    .onAppear { camera.start() }
    .onDisappear { camera.stop() }
  }

  private var streamURL: String? {
    FrameServer.wifiIPAddress().map { "http://\($0):8080/" }
  }

  /// "Linked" alone only means the phone paired once; the Mac may since have switched to another camera.
  private func macStatus(receiving: Bool) -> (value: String, tint: Color) {
    if receiving { return ("Receiving frames", Color.gtLive) }
    switch pairing.state {
    case .linked:
      return ("Not reading yet", Color.gtMuted)
    case .linking:
      return ("Linking…", Color.gtAccent)
    case .failed:
      return ("Failed", Color.gtError)
    case .notLinked:
      return ("Not linked", Color.gtMuted)
    }
  }
}

private struct CameraPreview: UIViewRepresentable {
  let session: AVCaptureSession
  let device: AVCaptureDevice?

  final class Coordinator {
    var device: AVCaptureDevice?
    var rotation: AVCaptureDevice.RotationCoordinator?
    var observation: NSKeyValueObservation?
  }

  func makeCoordinator() -> Coordinator { Coordinator() }

  func makeUIView(context: Context) -> PreviewLayerView {
    let view = PreviewLayerView()
    view.previewLayer.session = session
    view.previewLayer.videoGravity = .resizeAspect
    return view
  }

  func updateUIView(_ view: PreviewLayerView, context: Context) {
    // The device is only known once start() has picked it, so the rotation is attached then.
    guard let device, context.coordinator.device !== device else { return }
    let coordinator = context.coordinator
    coordinator.device = device
    let rotation = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: view.previewLayer)
    coordinator.rotation = rotation
    coordinator.observation = rotation.observe(
      \.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]
    ) { [weak layer = view.previewLayer] r, _ in
      let angle = r.videoRotationAngleForHorizonLevelPreview
      guard let connection = layer?.connection, connection.isVideoRotationAngleSupported(angle) else { return }
      connection.videoRotationAngle = angle
    }
  }
}

private final class PreviewLayerView: UIView {
  override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
  var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}
