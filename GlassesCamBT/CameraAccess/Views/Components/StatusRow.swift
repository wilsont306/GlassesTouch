/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// StatusRow.swift
//
// One status line: a tinted dot, an icon and title, and a trailing value.
//

import SwiftUI

struct StatusRow<Trailing: View>: View {
  let icon: String
  let title: String
  let tint: Color
  let trailing: Trailing

  init(icon: String, title: String, tint: Color, @ViewBuilder trailing: () -> Trailing) {
    self.icon = icon
    self.title = title
    self.tint = tint
    self.trailing = trailing()
  }

  var body: some View {
    HStack(spacing: 12) {
      Circle()
        .fill(tint)
        .frame(width: 8, height: 8)
      Image(systemName: icon)
        .font(.system(size: 16))
        .foregroundStyle(Color.gtMuted)
        .frame(width: 22)
      Text(title)
        .font(.system(size: 16))
        .foregroundStyle(Color.gtText)
      Spacer()
      trailing
    }
  }
}

extension StatusRow where Trailing == Text {
  init(icon: String, title: String, value: String, tint: Color) {
    self.init(icon: icon, title: title, tint: tint) {
      Text(value)
        .font(.system(.subheadline, design: .monospaced))
        .foregroundStyle(Color.gtMuted)
    }
  }
}
