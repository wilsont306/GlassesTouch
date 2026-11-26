/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// CustomButton.swift
//
// Reusable button component used throughout the CameraAccess app for consistent styling.
//

import SwiftUI

struct CustomButton: View {
  let title: String
  let style: ButtonStyle
  let isDisabled: Bool
  let action: () -> Void

  enum ButtonStyle {
    case primary, destructive

    var backgroundColor: Color {
      switch self {
      case .primary:
        return .gtAccent
      case .destructive:
        return .gtError.opacity(0.15)
      }
    }

    var foregroundColor: Color {
      switch self {
      case .primary:
        return .gtAccentInk
      case .destructive:
        return .gtError
      }
    }
  }

  var body: some View {
    Button(action: action) {
      Text(title)
        .font(.system(size: 16, weight: .semibold))
        .foregroundStyle(style.foregroundColor)
        .frame(maxWidth: .infinity)
        .frame(height: 52)
        .background(style.backgroundColor, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
    .disabled(isDisabled)
    .opacity(isDisabled ? 0.6 : 1.0)
  }
}
