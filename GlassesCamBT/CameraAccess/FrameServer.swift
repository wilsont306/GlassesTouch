//
// FrameServer.swift
//
// Serves published video frames as MJPEG over HTTP so a Mac on the same Wi-Fi
// can read them with OpenCV: cv2.VideoCapture("http://<phone-ip>:8080/").
//

import CoreMedia
import Foundation
import Network
import UIKit
import os

final class FrameServer {
  private static let port = NWEndpoint.Port(rawValue: 8080)!
  private static let header = Data(
    "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
      .utf8)

  private final class Client {
    let connection: NWConnection
    // True while a send is in flight (or the header is pending); frames are skipped meanwhile.
    var busy = false
    init(_ connection: NWConnection) { self.connection = connection }
  }

  // All state below is only touched on `queue`.
  private let queue = DispatchQueue(label: "FrameServer")
  private var listener: NWListener?
  private var clients: [ObjectIdentifier: Client] = [:] {
    didSet { clientCount.withLock { $0 = clients.count } }
  }
  // Mirrors clients.count so other threads can check it without hopping to `queue`.
  private let clientCount = OSAllocatedUnfairLock<Int>(initialState: 0)
  var hasClients: Bool { clientCount.withLock { $0 > 0 } }

  // One listener per process: a second NWListener cannot bind port 8080.
  static let shared = FrameServer()
  private init() {}

  func start() {
    queue.async { [self] in
      guard self.listener == nil else { return }
      do {
        let listener = try NWListener(using: .tcp, on: FrameServer.port)
        listener.newConnectionHandler = { [self] connection in
          self.accept(connection)
        }
        listener.stateUpdateHandler = { [weak self] state in
          guard case .failed(let error) = state else { return }
          print("FrameServer listener failed: \(error), retrying in 2s")
          self?.queue.async {
            self?.stop()
            self?.queue.asyncAfter(deadline: .now() + 2) { self?.start() }
          }
        }
        listener.start(queue: self.queue)
        self.listener = listener
      } catch {
        print("FrameServer could not create listener: \(error)")
      }
    }
  }

  func stop() {
    queue.async {
      self.listener?.cancel()
      self.listener = nil
      for client in self.clients.values { client.connection.cancel() }
      self.clients.removeAll()
    }
  }

  /// JPEG-encodes the frame once and sends it to every client that is ready for it.
  /// captureTime (Unix seconds) goes out as X-Capture-Time so the Mac can line frames up with taps.
  func publish(_ image: UIImage, captureTime: Double) {
    queue.async {
      guard !self.clients.isEmpty, let jpeg = image.jpegData(compressionQuality: 0.85) else { return }
      self.broadcast(jpeg, captureTime: captureTime, extraHeader: "")
    }
  }

  /// Same as publish(_:captureTime:) for frames that are already JPEG (the phone camera).
  func publish(jpeg: Data, captureTime: Double) {
    queue.async {
      guard !self.clients.isEmpty else { return }
      self.broadcast(jpeg, captureTime: captureTime, extraHeader: "X-Capture-Source: phone-camera\r\n")
    }
  }

  // MARK: - Private (queue only)

  private func broadcast(_ jpeg: Data, captureTime: Double, extraHeader: String) {
    let head = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: \(jpeg.count)\r\n"
      + String(format: "X-Capture-Time: %.3f\r\n", captureTime) + extraHeader + "\r\n"
    let frame = Data(head.utf8) + jpeg + Data("\r\n".utf8)
    for client in clients.values where !client.busy {
      send(frame, to: client)
    }
  }

  private func accept(_ connection: NWConnection) {
    let client = Client(connection)
    clients[ObjectIdentifier(connection)] = client
    client.busy = true  // keep frames away until the header is out
    connection.stateUpdateHandler = { [weak self, weak client] state in
      switch state {
      case .failed, .cancelled:
        self?.queue.async {
          guard let self, let client else { return }
          self.remove(client)
        }
      default:
        break
      }
    }
    connection.start(queue: queue)
    // Read and ignore the HTTP request, then answer with the MJPEG header.
    connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] _, _, _, _ in
      self?.queue.async {
        guard let self, self.clients[ObjectIdentifier(connection)] === client else { return }
        self.send(FrameServer.header, to: client)
      }
    }
  }

  private func send(_ data: Data, to client: Client) {
    client.busy = true
    client.connection.send(content: data, completion: .contentProcessed { [weak self] error in
      self?.queue.async {
        guard let self, self.clients[ObjectIdentifier(client.connection)] === client else { return }
        client.busy = false
        if error != nil { self.remove(client) }
      }
    })
  }

  private func remove(_ client: Client) {
    guard clients.removeValue(forKey: ObjectIdentifier(client.connection)) != nil else { return }
    // Break the handler -> client reference cycle before cancelling.
    client.connection.stateUpdateHandler = nil
    client.connection.cancel()
  }

  // MARK: - Helpers

  /// The phone's Wi-Fi (en0) IPv4 address, or nil if unavailable.
  static func wifiIPAddress() -> String? {
    var ifaddr: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return nil }
    defer { freeifaddrs(ifaddr) }

    for ifa in sequence(first: first, next: { $0.pointee.ifa_next }) {
      guard let addr = ifa.pointee.ifa_addr,
        Int32(addr.pointee.sa_family) == AF_INET,
        String(cString: ifa.pointee.ifa_name) == "en0"
      else { continue }

      var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
      let result = getnameinfo(
        addr, socklen_t(addr.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST)
      return result == 0 ? String(cString: host) : nil
    }
    return nil
  }
}

/// Wall-clock capture times for frames from their presentation timestamps (the glasses' own clock).
/// Bluetooth delivers frames in bursts, so arrival time is a poor clock. The offset is the smallest
/// (arrival - pts) seen: pts + offset is when a frame would have arrived without being held in a burst
/// (its capture time plus the link's minimum latency, which the Mac subtracts).
final class CaptureClock: @unchecked Sendable {
  private let offset = OSAllocatedUnfairLock<Double?>(uncheckedState: nil)

  func wallTime(of buffer: CMSampleBuffer) -> Double {
    let now = Date().timeIntervalSince1970
    let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer))
    guard pts.isFinite else { return now }
    return offset.withLockUnchecked { stored in
      let d = now - pts
      var o = min((stored ?? d) + 0.0005, d)  // creeps up slowly, so clock drift cannot pin it
      if d - o > 5 { o = d }  // the pts clock restarted (new session): start over
      stored = o
      return pts + o
    }
  }
}
