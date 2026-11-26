/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// PairingModel.swift
//
// Links this phone to the GlassesTouch Mac app. The Mac shows a QR code holding
// cameraaccess://glasstouch-pair?h=<ip>&p=<port>&c=<code>; scanning it opens the app,
// which posts the code to the Mac at http://<ip>:<port>/pair.
//

import Foundation
import Network

@MainActor
final class PairingModel: ObservableObject {
  static let shared = PairingModel()

  enum State {
    case notLinked, linking, linked(mac: String), failed(String)
  }

  @Published var state: State

  private static let pairedMacKey = "pairedMac"
  private static let codeCharacters = Set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")

  private init() {
    if let mac = UserDefaults.standard.string(forKey: Self.pairedMacKey) {
      state = .linked(mac: mac)
    } else {
      state = .notLinked
    }
  }

  /// Handles a cameraaccess://glasstouch-pair link. Returns false for any other URL.
  func handle(_ url: URL) -> Bool {
    guard url.host == "glasstouch-pair" else { return false }
    let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
    func value(_ name: String) -> String? { items.first { $0.name == name }?.value }

    // Validate everything before sending anything to the Mac.
    guard let host = value("h"), IPv4Address(host) != nil,
      let portText = value("p"), let port = Int(portText), (1...65535).contains(port),
      let code = value("c"), (1...64).contains(code.count),
      code.allSatisfy({ Self.codeCharacters.contains($0) })
    else {
      state = .failed("Not a GlassesTouch pairing code")
      return true
    }

    state = .linking
    Task { await send(code: code, host: host, port: port) }
    return true
  }

  func forget() {
    UserDefaults.standard.removeObject(forKey: Self.pairedMacKey)
    state = .notLinked
  }

  private func send(code: String, host: String, port: Int) async {
    do {
      guard let url = URL(string: "http://\(host):\(port)/pair") else { throw URLError(.badURL) }
      var request = URLRequest(url: url, timeoutInterval: 5)
      request.httpMethod = "POST"
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
      request.httpBody = try JSONEncoder().encode(["code": code])

      let (data, response) = try await URLSession.shared.data(for: request)
      let status = (response as? HTTPURLResponse)?.statusCode ?? 0
      switch status {
      case 200:
        guard let reply = try? JSONDecoder().decode(PairReply.self, from: data), reply.ok else {
          state = .failed("Unexpected reply from the Mac.")
          return
        }
        UserDefaults.standard.set(reply.mac, forKey: Self.pairedMacKey)
        state = .linked(mac: reply.mac)
      case 403:
        state = .failed("Wrong code. Check the code on the Mac.")
      case 410:
        state = .failed("Code expired. Make a new one on the Mac.")
      default:
        state = .failed("Pairing failed (HTTP \(status)).")
      }
    } catch {
      state = .failed("Can't reach the Mac. Check that both are on the same Wi-Fi.")
    }
  }
}

private struct PairReply: Decodable {
  let ok: Bool
  let mac: String
}
