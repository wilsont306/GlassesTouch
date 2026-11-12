//
// FrameServer.swift
//
// Serves published video frames as MJPEG over HTTP so a Mac on the same Wi-Fi
// can read them with OpenCV: cv2.VideoCapture("http://<phone-ip>:8080/").
//

import Foundation
import Network
import UIKit

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
  private var clients: [ObjectIdentifier: Client] = [:]

  // One listener per process: a second NWListener cannot bind port 8080.
  static let shared = FrameServer()
  private init() {}

  func start() {
    queue.async {
      guard self.listener == nil else { return }
      do {
        let listener = try NWListener(using: .tcp, on: FrameServer.port)
        listener.newConnectionHandler = { [weak self] connection in
          self?.accept(connection)
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
  func publish(_ image: UIImage) {
    queue.async {
      guard !self.clients.isEmpty, let jpeg = image.jpegData(compressionQuality: 0.85) else { return }
      let frame = Data("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: \(jpeg.count)\r\n\r\n".utf8)
        + jpeg + Data("\r\n".utf8)
      for client in self.clients.values where !client.busy {
        self.send(frame, to: client)
      }
    }
  }

  // MARK: - Private (queue only)

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
