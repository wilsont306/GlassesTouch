/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// NonStreamView.swift
//
// Default screen showing glasses, Mac and Wi-Fi status before streaming.
// Initiates streaming
//

import MWDATCore
import SwiftUI

struct NonStreamView: View {
  @ObservedObject var viewModel: StreamSessionViewModel
  @ObservedObject var wearablesVM: WearablesViewModel
  @ObservedObject private var pairing = PairingModel.shared
  @State private var sheetHeight: CGFloat = 300
  @State private var showPhoneCamera = false

  var body: some View {
    ZStack {
      Color.gtBg.edgesIgnoringSafeArea(.all)

      VStack(alignment: .leading, spacing: 20) {
        HStack {
          Wordmark()
          Spacer()
          Menu {
            Button("Disconnect", role: .destructive) {
              wearablesVM.disconnectGlasses()
            }
            .disabled(wearablesVM.registrationState != .registered)
            if isLinked {
              Button("Forget Mac") {
                pairing.forget()
              }
            }
          } label: {
            Image(systemName: "gearshape")
              .font(.system(size: 20))
              .foregroundStyle(Color.gtText)
              .frame(width: 44, height: 44)
          }
        }

        VStack(alignment: .leading, spacing: 16) {
          StatusRow(
            icon: "eyeglasses",
            title: "Glasses",
            value: viewModel.hasActiveDevice ? "Connected" : "Waiting",
            tint: viewModel.hasActiveDevice ? Color.gtLive : Color.gtAccent)
          StatusRow(
            icon: "desktopcomputer",
            title: "Mac",
            value: macStatus.value,
            tint: macStatus.tint)
          StatusRow(
            icon: "wifi",
            title: "Wi-Fi",
            tint: viewModel.streamURL == nil ? Color.gtMuted : Color.gtLive)
          {
            if let streamURL = viewModel.streamURL {
              AddressChip(address: streamURL)
            } else {
              Text("No Wi-Fi")
                .font(.system(.subheadline, design: .monospaced))
                .foregroundStyle(Color.gtMuted)
            }
          }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .gtCard()

        if case .notLinked = pairing.state {
          Text("On the Mac, open the GlassesTouch panel and press Pair phone, then scan the code with the Camera app.")
            .font(.system(size: 14))
            .foregroundStyle(Color.gtMuted)
        }
        if case .failed(let message) = pairing.state {
          Text(message)
            .font(.system(size: 14))
            .foregroundStyle(Color.gtError)
        }

        Spacer()

        Button("Use this phone's camera instead") {
          showPhoneCamera = true
        }
        .font(.system(size: 15, weight: .medium))
        .foregroundStyle(Color.gtText)
        .frame(maxWidth: .infinity)

        CustomButton(
          title: "Start streaming",
          style: .primary,
          isDisabled: !viewModel.hasActiveDevice
        ) {
          Task {
            await viewModel.handleStartStreaming()
          }
        }
      }
      .padding(.all, 24)
    }
    .sheet(isPresented: $wearablesVM.showGettingStartedSheet) {
      if #available(iOS 16.0, *) {
        GettingStartedSheetView(height: $sheetHeight)
          .presentationDetents([.height(sheetHeight)])
          .presentationDragIndicator(.visible)
      } else {
        GettingStartedSheetView(height: $sheetHeight)
      }
    }
    .fullScreenCover(isPresented: $showPhoneCamera) {
      PhoneCameraView()
    }
  }

  private var isLinked: Bool {
    if case .linked = pairing.state { return true }
    return false
  }

  private var macStatus: (value: String, tint: Color) {
    switch pairing.state {
    case .linked(let mac):
      return (mac, Color.gtLive)
    case .linking:
      return ("Linking…", Color.gtAccent)
    case .failed:
      return ("Failed", Color.gtError)
    case .notLinked:
      return ("Not linked", Color.gtMuted)
    }
  }
}

struct GettingStartedSheetView: View {
  @Environment(\.dismiss) var dismiss
  @Binding var height: CGFloat

  var body: some View {
    VStack(spacing: 24) {
      Text("Getting started")
        .font(.system(size: 18, weight: .semibold))
        .foregroundStyle(Color.gtText)

      VStack(spacing: 12) {
        TipItemView(
          resource: .videoIcon,
          text: "First, Camera Access needs permission to use your glasses camera."
        )
        TipItemView(
          resource: .tapIcon,
          text: "Capture photos by tapping the camera button."
        )
        TipItemView(
          resource: .smartGlassesIcon,
          text: "The capture LED lets others know when you're capturing content or going live."
        )
      }
      .padding(.bottom, 16)

      CustomButton(
        title: "Continue",
        style: .primary,
        isDisabled: false
      ) {
        dismiss()
      }
    }
    .padding(.all, 24)
    .background(Color.gtSurface)
    .background(
      GeometryReader { geo -> Color in
        DispatchQueue.main.async {
          height = geo.size.height
        }
        return Color.clear
      }
    )
  }
}

struct TipItemView: View {
  let resource: ImageResource
  let text: String

  var body: some View {
    HStack(alignment: .top, spacing: 12) {
      Image(resource)
        .resizable()
        .renderingMode(.template)
        .foregroundStyle(Color.gtMuted)
        .aspectRatio(contentMode: .fit)
        .frame(width: 24)
        .padding(.leading, 4)
        .padding(.top, 4)

      Text(text)
        .font(.system(size: 15))
        .foregroundStyle(Color.gtText)
        .fixedSize(horizontal: false, vertical: true)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }
}
