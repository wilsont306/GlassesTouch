/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// StreamSessionViewModel.swift
//
// Core view model demonstrating video streaming from Meta wearable devices using the DAT SDK.
// This class showcases the key streaming patterns: device selection, session management,
// video frame handling, photo capture, and error handling.
//

import MWDATCamera
import MWDATCore
import SwiftUI
import os

enum StreamingStatus {
  case streaming
  case waiting
  case stopped
}

/// Holds only the newest decoded frame until the main actor takes it, so at most one frame is pending.
private final class LatestFrameSlot: @unchecked Sendable {
  private let lock = OSAllocatedUnfairLock<UIImage?>(uncheckedState: nil)

  /// Stores `image`, replacing any pending one. Returns true if the slot was empty, meaning the caller must schedule a drain.
  func put(_ image: UIImage) -> Bool {
    lock.withLockUnchecked { slot in
      let wasEmpty = slot == nil
      slot = image
      return wasEmpty
    }
  }

  func take() -> UIImage? {
    lock.withLockUnchecked { slot in
      let image = slot
      slot = nil
      return image
    }
  }

  func clear() {
    lock.withLockUnchecked { $0 = nil }
  }
}

@MainActor
class StreamSessionViewModel: ObservableObject {
  @Published var currentVideoFrame: UIImage?
  @Published var hasReceivedFirstFrame: Bool = false
  @Published var streamingStatus: StreamingStatus = .stopped
  @Published var showError: Bool = false
  @Published var errorMessage: String = ""
  @Published var hasActiveDevice: Bool = false
  // http://<phone-ip>:8080/ for OpenCV clients on the same Wi-Fi; nil if no Wi-Fi IP
  @Published var streamURL: String?

  var isStreaming: Bool {
    streamingStatus != .stopped
  }

  // Photo capture properties
  @Published var capturedPhoto: UIImage?
  @Published var showPhotoPreview: Bool = false
  // The core DAT SDK StreamSession - handles all streaming operations
  private var streamSession: StreamSession
  // Listener tokens are used to manage DAT SDK event subscriptions
  private var stateListenerToken: AnyListenerToken?
  private var videoFrameListenerToken: AnyListenerToken?
  private var errorListenerToken: AnyListenerToken?
  private var photoDataListenerToken: AnyListenerToken?
  private let wearables: WearablesInterface
  private let deviceSelector: AutoDeviceSelector
  private var deviceMonitorTask: Task<Void, Never>?
  private let frameServer = FrameServer.shared
  // Decodes hvc1 (HEVC) frames to UIImage. Thread-safe; used from the SDK callback thread.
  private let videoFrameDecoder = VideoFrameDecoder()
  // The newest decoded frame waiting for the main actor.
  private let latestFrame = LatestFrameSlot()

  init(wearables: WearablesInterface) {
    self.wearables = wearables
    // Let the SDK auto-select from available devices
    self.deviceSelector = AutoDeviceSelector(wearables: wearables)
    // hvc1 (compressed HEVC) keeps streaming when the app is backgrounded; raw pauses.
    let config = StreamSessionConfig(
      videoCodec: VideoCodec.hvc1,
      resolution: StreamingResolution.high,
      frameRate: 15)
    streamSession = StreamSession(streamSessionConfig: config, deviceSelector: deviceSelector)

    // Serve frames as MJPEG on port 8080 for OpenCV clients
    frameServer.start()
    streamURL = FrameServer.wifiIPAddress().map { "http://\($0):8080/" }

    // Monitor device availability
    deviceMonitorTask = Task { @MainActor in
      for await device in deviceSelector.activeDeviceStream() {
        self.hasActiveDevice = device != nil
      }
    }

    // Subscribe to session state changes using the DAT SDK listener pattern
    // State changes tell us when streaming starts, stops, or encounters issues
    stateListenerToken = streamSession.statePublisher.listen { [weak self] state in
      Task { @MainActor [weak self] in
        self?.updateStatusFromState(state)
      }
    }

    // Subscribe to video frames from the device camera.
    // Each VideoFrame carries a compressed HEVC sampleBuffer, which we decode to UIImage.
    let decoder = videoFrameDecoder
    let slot = latestFrame
    let server = frameServer
    let clock = CaptureClock()
    videoFrameListenerToken = streamSession.videoFramePublisher.listen { [weak self] videoFrame in
      // Decodes on this callback thread; the decoder serialises its own VideoToolbox work.
      guard let image = decoder.decode(videoFrame.sampleBuffer) else { return }
      // Every frame goes to the Mac with its capture time; the preview below only needs the newest one.
      server.publish(image, captureTime: clock.wallTime(of: videoFrame.sampleBuffer))
      // Keep only the newest frame. Schedule one main-thread delivery when the slot goes from empty to full.
      if slot.put(image) {
        DispatchQueue.main.async {
          MainActor.assumeIsolated { self?.deliverLatestFrame() }
        }
      }
    }

    // Subscribe to streaming errors
    // Errors include device disconnection, streaming failures, etc.
    errorListenerToken = streamSession.errorPublisher.listen { [weak self] error in
      Task { @MainActor [weak self] in
        guard let self else { return }
        let newErrorMessage = formatStreamingError(error)
        if newErrorMessage != self.errorMessage {
          showError(newErrorMessage)
        }
      }
    }

    updateStatusFromState(streamSession.state)

    // Subscribe to photo capture events
    // PhotoData contains the captured image in the requested format (JPEG/HEIC)
    photoDataListenerToken = streamSession.photoDataPublisher.listen { [weak self] photoData in
      Task { @MainActor [weak self] in
        guard let self else { return }
        if let uiImage = UIImage(data: photoData.data) {
          self.capturedPhoto = uiImage
          self.showPhotoPreview = true
        }
      }
    }
  }

  func handleStartStreaming() async {
    let permission = Permission.camera
    do {
      let status = try await wearables.checkPermissionStatus(permission)
      if status == .granted {
        await startSession()
        return
      }
      let requestStatus = try await wearables.requestPermission(permission)
      if requestStatus == .granted {
        await startSession()
        return
      }
      showError("Permission denied")
    } catch {
      showError("Permission error: \(error.description)")
    }
  }

  func startSession() async {
    await streamSession.start()
  }

  /// Main-actor end of the preview: shows the newest decoded frame (the MJPEG server gets every frame directly).
  private func deliverLatestFrame() {
    guard let image = latestFrame.take() else { return }
    currentVideoFrame = image
    hasReceivedFirstFrame = true
  }

  private func showError(_ message: String) {
    errorMessage = message
    showError = true
  }

  func stopSession() async {
    await streamSession.stop()
  }

  func dismissError() {
    showError = false
    errorMessage = ""
  }

  func capturePhoto() {
    streamSession.capturePhoto(format: .jpeg)
  }

  func dismissPhotoPreview() {
    showPhotoPreview = false
    capturedPhoto = nil
  }

  private func updateStatusFromState(_ state: StreamSessionState) {
    switch state {
    case .stopped:
      currentVideoFrame = nil
      streamingStatus = .stopped
      latestFrame.clear()
      videoFrameDecoder.reset()
    case .waitingForDevice, .starting, .stopping, .paused:
      streamingStatus = .waiting
    case .streaming:
      streamingStatus = .streaming
      // Refresh in case the Wi-Fi address changed since init
      streamURL = FrameServer.wifiIPAddress().map { "http://\($0):8080/" }
    }
  }

  private func formatStreamingError(_ error: StreamSessionError) -> String {
    switch error {
    case .internalError:
      return "An internal error occurred. Please try again."
    case .deviceNotFound:
      return "Device not found. Please ensure your device is connected."
    case .deviceNotConnected:
      return "Device not connected. Please check your connection and try again."
    case .timeout:
      return "The operation timed out. Please try again."
    case .videoStreamingError:
      return "Video streaming failed. Please try again."
    case .permissionDenied:
      return "Camera permission denied. Please grant permission in Settings."
    case .hingesClosed:
      return "The hinges on the glasses were closed. Please open the hinges and try again."
    case .thermalCritical:
      return "Device is overheating. Streaming has been paused to protect the device."
    @unknown default:
      return "An unknown streaming error occurred."
    }
  }
}
