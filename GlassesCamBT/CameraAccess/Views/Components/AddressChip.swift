/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// AddressChip.swift
//
// Stream address shown blurred until tapped, so it is not read off the screen by accident.
//

import SwiftUI

struct AddressChip: View {
  let address: String
  @State private var revealed = false

  var body: some View {
    Button {
      revealed.toggle()
    } label: {
      HStack(spacing: 6) {
        Text(address)
          .font(.system(.caption, design: .monospaced))
          .foregroundStyle(Color.gtText)
          .blur(radius: revealed ? 0 : 6)
        Image(systemName: revealed ? "eye.slash" : "eye")
          .font(.system(size: 12))
          .foregroundStyle(Color.gtMuted)
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 6)
      .background(Color.gtSurface.opacity(0.85), in: Capsule())
      .overlay(Capsule().stroke(Color.gtLine, lineWidth: 1))
    }
    .buttonStyle(.plain)
    .accessibilityLabel(revealed ? address : "Stream address, hidden")
    .accessibilityHint("Double-tap to show or hide")
  }
}
