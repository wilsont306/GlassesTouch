/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// Theme.swift
//
// GlassesTouch dark palette, card surface and wordmark shared across screens.
//

import SwiftUI

extension Color {
  init(hex: UInt32) {
    self.init(
      red: Double((hex >> 16) & 0xFF) / 255,
      green: Double((hex >> 8) & 0xFF) / 255,
      blue: Double(hex & 0xFF) / 255)
  }

  static let gtBg = Color(hex: 0x0F1113)
  static let gtSurface = Color(hex: 0x171A1D)
  static let gtLine = Color(hex: 0x2A2F34)
  static let gtText = Color(hex: 0xE8EAED)
  static let gtMuted = Color(hex: 0x8B939C)
  static let gtAccent = Color(hex: 0xF0A020)
  static let gtAccentInk = Color(hex: 0x1A1200)
  static let gtLive = Color(hex: 0x3DDC84)
  static let gtError = Color(hex: 0xFF6B5E)
}

extension View {
  /// Surface card: gtSurface fill, 1 pt gtLine border, 16 pt continuous corners.
  func gtCard() -> some View {
    let shape = RoundedRectangle(cornerRadius: 16, style: .continuous)
    return self
      .background(Color.gtSurface, in: shape)
      .overlay(shape.stroke(Color.gtLine, lineWidth: 1))
  }
}

struct Wordmark: View {
  var body: some View {
    HStack(spacing: 8) {
      Image(systemName: "viewfinder")
        .font(.system(size: 20, weight: .semibold))
        .foregroundStyle(Color.gtAccent)
      Text("GlassesTouch")
        .font(.system(size: 22, weight: .semibold))
        .foregroundStyle(Color.gtText)
    }
  }
}
