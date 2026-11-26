/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// HomeScreenView.swift
//
// Welcome screen that guides users through the DAT SDK registration process.
// This view is displayed when the app is not yet registered.
//

import MWDATCore
import SwiftUI

struct HomeScreenView: View {
  @ObservedObject var viewModel: WearablesViewModel
  @State private var showPhoneCamera = false

  var body: some View {
    let registering = viewModel.registrationState == .registering

    VStack(alignment: .leading, spacing: 24) {
      Wordmark()

      VStack(alignment: .leading, spacing: 12) {
        Text("Touch your Mac screen.")
          .font(.system(size: 28, weight: .semibold))
          .foregroundStyle(Color.gtText)
        Text("Ray-Ban Meta glasses watch your hand; your MacBook feels the tap.")
          .font(.system(size: 16))
          .foregroundStyle(Color.gtMuted)
      }

      VStack(alignment: .leading, spacing: 16) {
        HomeStep(number: 1, icon: "eyeglasses", text: "Connect your glasses")
        HomeStep(number: 2, icon: "qrcode.viewfinder", text: "Scan the code in the Mac panel with the Camera app")
        HomeStep(number: 3, icon: "hand.tap", text: "Start streaming and touch the screen")
      }
      .padding(20)
      .frame(maxWidth: .infinity, alignment: .leading)
      .gtCard()

      Spacer()

      VStack(spacing: 16) {
        Text("Opens the Meta AI app to confirm.")
          .font(.footnote)
          .foregroundStyle(Color.gtMuted)

        Button("Use this phone's camera instead") {
          showPhoneCamera = true
        }
        .font(.system(size: 15, weight: .medium))
        .foregroundStyle(Color.gtText)
        .frame(maxWidth: .infinity)

        CustomButton(
          title: registering ? "Connecting…" : "Connect glasses",
          style: .primary,
          isDisabled: registering
        ) {
          viewModel.connectGlasses()
        }
      }
    }
    .padding(24)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    .background(Color.gtBg.ignoresSafeArea())
    .fullScreenCover(isPresented: $showPhoneCamera) {
      PhoneCameraView()
    }
  }
}

private struct HomeStep: View {
  let number: Int
  let icon: String
  let text: String

  var body: some View {
    HStack(spacing: 14) {
      Text("\(number)")
        .font(.system(size: 14, weight: .semibold, design: .monospaced))
        .foregroundStyle(Color.gtAccentInk)
        .frame(width: 26, height: 26)
        .background(Color.gtAccent, in: Circle())
      Image(systemName: icon)
        .font(.system(size: 18))
        .foregroundStyle(Color.gtMuted)
        .frame(width: 24)
      Text(text)
        .font(.system(size: 16))
        .foregroundStyle(Color.gtText)
        .fixedSize(horizontal: false, vertical: true)
    }
  }
}
