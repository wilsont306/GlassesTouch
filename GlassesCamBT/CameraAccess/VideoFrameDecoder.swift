/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import CoreImage
import CoreMedia
import UIKit
import VideoToolbox

// MARK: - VideoFrameDecoder

/// Decompresses HEVC (`VideoCodec.hvc1`) `CMSampleBuffer`s into `UIImage`s for the on-screen
/// preview and the MJPEG server. Adapted from Meta's GlassesCam sample.
///
/// - Software decoding always (VideoToolbox with hardware acceleration disabled), as in Meta's sample:
///   hardware sessions stutter.
/// - All VideoToolbox work (create, decode, invalidate) and all mutable state run on one serial queue,
///   so no session is used after another thread has invalidated it.
/// - Output is rendered with a CPU `CIContext`, because GPU (Metal) work fails in the background.
/// - After `failureThreshold` consecutive failures the session is invalidated. A new one is created
///   after a backoff that starts at 0.25 s, doubles, and is capped at 4 s.
/// - The last good frame is returned for at most `staleLimit` seconds. After that `decode` returns nil,
///   so the feed stalls instead of repeating one image.
///
/// Thread-safe: `decode(_:)` may be called from any thread and blocks until its work is done.
final class VideoFrameDecoder: @unchecked Sendable {
  private let failureThreshold = 3
  private let initialRetryDelay: TimeInterval = 0.25
  private let maxRetryDelay: TimeInterval = 4
  private let staleLimit: TimeInterval = 2

  // Everything below is touched only on `queue`.
  private let queue = DispatchQueue(label: "VideoFrameDecoder")
  // CPU rendering: Metal/GPU work fails when submitted from the background.
  private let ciContext = CIContext(options: [.useSoftwareRenderer: true])
  private var session: VTDecompressionSession?
  private var formatDescription: CMFormatDescription?
  private var awaitingKeyframe = false
  private var consecutiveFailures = 0
  private var retryDelay: TimeInterval = 0
  private var retryNotBefore: TimeInterval?
  private var lastGoodImage: UIImage?
  private var lastSuccess: TimeInterval?
  private var stalled = false

  deinit {
    if let session {
      VTDecompressionSessionInvalidate(session)
    }
  }

  /// Decodes one HEVC sample buffer. Returns nil when there is no fresh frame to show.
  func decode(_ sampleBuffer: CMSampleBuffer) -> UIImage? {
    queue.sync { decodeOnQueue(sampleBuffer) }
  }

  /// Tears down the session and clears all state. Call when the stream stops.
  func reset() {
    queue.async {
      self.invalidateSession(reason: "stream stopped")
      self.lastGoodImage = nil
      self.lastSuccess = nil
      self.awaitingKeyframe = false
      self.consecutiveFailures = 0
      self.retryDelay = 0
      self.retryNotBefore = nil
      self.stalled = false
      print("VideoFrameDecoder: reset")
    }
  }

  // MARK: - Queue-only

  private func decodeOnQueue(_ sampleBuffer: CMSampleBuffer) -> UIImage? {
    guard CMSampleBufferGetDataBuffer(sampleBuffer) != nil,
      let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer)
    else {
      return fallbackImage()
    }

    let isKeyframe = sampleBuffer.isHEVCKeyframe()

    // Recreate the session if the format changed (e.g. a resolution switch mid-stream).
    if let currentFormat = self.formatDescription,
      !CMFormatDescriptionEqual(currentFormat, otherFormatDescription: formatDescription)
    {
      invalidateSession(reason: "format changed")
    }
    // Catch an invalidated session before a decode attempt fails.
    if let existing = session,
      !VTDecompressionSessionCanAcceptFormatDescription(existing, formatDescription: formatDescription)
    {
      invalidateSession(reason: "session cannot accept format")
    }

    if session == nil {
      // Backing off after failures: don't call VTDecompressionSessionCreate on every frame.
      if let notBefore = retryNotBefore, ProcessInfo.processInfo.systemUptime < notBefore {
        return fallbackImage()
      }
      guard createSession(for: formatDescription), let session else {
        return fallbackImage()
      }
      return decodeFrame(sampleBuffer, with: session, isKeyframe: isKeyframe)
    }

    guard let session else { return fallbackImage() }
    return decodeFrame(sampleBuffer, with: session, isKeyframe: isKeyframe)
  }

  private func decodeFrame(
    _ sampleBuffer: CMSampleBuffer, with session: VTDecompressionSession, isKeyframe: Bool
  ) -> UIImage? {
    // Hold the last good frame until a keyframe arrives after a (re)creation.
    if awaitingKeyframe && !isKeyframe {
      return fallbackImage()
    }

    // flags is empty, so the output handler runs synchronously on this thread, and the captures
    // below are safe despite the @Sendable warning. Do not add ._EnableAsynchronousDecompression.
    nonisolated(unsafe) var outputPixelBuffer: CVPixelBuffer?
    nonisolated(unsafe) var outputStatus: OSStatus = noErr
    let decodeStatus = VTDecompressionSessionDecodeFrame(
      session,
      sampleBuffer: sampleBuffer,
      flags: [],
      infoFlagsOut: nil
    ) { status, _, imageBuffer, _, _ in
      outputStatus = status
      if status == noErr, let imageBuffer {
        outputPixelBuffer = imageBuffer
      }
    }

    // A noErr status with no pixel buffer counts as a failure too.
    guard decodeStatus == noErr, outputStatus == noErr, let pixelBuffer = outputPixelBuffer else {
      let status = decodeStatus != noErr ? decodeStatus : outputStatus
      recordFailure("decode status \(status), no image")
      return fallbackImage()
    }

    let cgImage = ciContext.createCGImage(
      CIImage(cvPixelBuffer: pixelBuffer),
      from: CGRect(
        x: 0, y: 0,
        width: CVPixelBufferGetWidth(pixelBuffer),
        height: CVPixelBufferGetHeight(pixelBuffer)))
    guard let cgImage else {
      recordFailure("CIContext produced no CGImage")
      return fallbackImage()
    }

    consecutiveFailures = 0
    retryDelay = 0
    retryNotBefore = nil
    if isKeyframe {
      awaitingKeyframe = false
    }

    let image = UIImage(cgImage: cgImage)
    lastGoodImage = image
    lastSuccess = ProcessInfo.processInfo.systemUptime
    if stalled {
      stalled = false
      print("VideoFrameDecoder: frames resumed")
    }
    return image
  }

  /// Creates a software session. Returns false (and schedules a retry) if VideoToolbox refuses it.
  private func createSession(for formatDescription: CMFormatDescription) -> Bool {
    let decoderSpec = [
      kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder as String: false
    ] as CFDictionary
    let outputAttrs: [CFString: Any] = [
      kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
    ]

    var newSession: VTDecompressionSession?
    let status = VTDecompressionSessionCreate(
      allocator: kCFAllocatorDefault,
      formatDescription: formatDescription,
      decoderSpecification: decoderSpec,
      imageBufferAttributes: outputAttrs as CFDictionary,
      outputCallback: nil,
      decompressionSessionOut: &newSession)

    guard status == noErr, let newSession else {
      print("VideoFrameDecoder: session creation failed (status \(status))")
      scheduleRetry()
      return false
    }

    session = newSession
    self.formatDescription = formatDescription
    awaitingKeyframe = true
    print("VideoFrameDecoder: software session created")
    return true
  }

  private func recordFailure(_ reason: String) {
    consecutiveFailures += 1
    print("VideoFrameDecoder: decode failure \(consecutiveFailures) (\(reason))")
    guard consecutiveFailures >= failureThreshold else { return }
    consecutiveFailures = 0
    invalidateSession(reason: "\(failureThreshold) consecutive failures")
    scheduleRetry()
  }

  /// Doubles the backoff (0.25 s, 0.5 s, 1 s, 2 s, then 4 s) and blocks session creation until it passes.
  private func scheduleRetry() {
    retryDelay = retryDelay == 0 ? initialRetryDelay : min(retryDelay * 2, maxRetryDelay)
    retryNotBefore = ProcessInfo.processInfo.systemUptime + retryDelay
    print("VideoFrameDecoder: next session attempt in \(retryDelay) s")
  }

  private func invalidateSession(reason: String) {
    if let session {
      VTDecompressionSessionInvalidate(session)
      print("VideoFrameDecoder: session invalidated (\(reason))")
    }
    session = nil
    formatDescription = nil
    awaitingKeyframe = true
  }

  /// The last good frame while it is at most `staleLimit` seconds old. Otherwise nil, so the stream stalls.
  private func fallbackImage() -> UIImage? {
    guard let lastGoodImage, let lastSuccess else { return nil }
    if ProcessInfo.processInfo.systemUptime - lastSuccess <= staleLimit {
      return lastGoodImage
    }
    if !stalled {
      stalled = true
      print("VideoFrameDecoder: no fresh frame for \(staleLimit) s, stalling output")
    }
    return nil
  }
}

// MARK: - HEVC Keyframe Detection

fileprivate extension CMSampleBuffer {
  /// Returns true if any NAL unit in the buffer is an HEVC keyframe type.
  /// Keyframe NAL types: BLA (16-18), IDR (19-20), CRA (21).
  func isHEVCKeyframe() -> Bool {
    guard let dataBuffer = CMSampleBufferGetDataBuffer(self) else { return false }
    let totalLength = CMBlockBufferGetDataLength(dataBuffer)
    var offset = 0

    // NAL units are in HVCC format: a 4-byte big-endian length prefix, then NAL data. Assumes 4-byte prefixes, which is what the live stream uses.
    while offset + 4 < totalLength {
      var nalLengthBE: UInt32 = 0
      guard CMBlockBufferCopyDataBytes(dataBuffer, atOffset: offset, dataLength: 4, destination: &nalLengthBE) == kCMBlockBufferNoErr
      else { break }
      let nalLength = Int(UInt32(bigEndian: nalLengthBE))
      offset += 4

      guard nalLength > 0, offset + nalLength <= totalLength else { break }

      // HEVC NAL header byte 0: forbidden_zero_bit(1) | nal_unit_type(6) | nuh_layer_id MSB(1)
      var nalHeader: UInt8 = 0
      guard CMBlockBufferCopyDataBytes(dataBuffer, atOffset: offset, dataLength: 1, destination: &nalHeader) == kCMBlockBufferNoErr
      else { break }
      let nalUnitType = (nalHeader >> 1) & 0x3F

      if nalUnitType >= 16 && nalUnitType <= 21 {
        return true
      }

      offset += nalLength
    }

    return false
  }
}
