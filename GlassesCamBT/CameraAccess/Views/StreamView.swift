/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// StreamView.swift
//
// Main UI for video streaming from Meta wearable devices using the DAT SDK.
// This view demonstrates the complete streaming API: video streaming with real-time display, photo capture,
// and error handling.
//

import MWDATCore
import SwiftUI

struct StreamView: View {
  @ObservedObject var viewModel: StreamSessionViewModel
  @ObservedObject var wearablesVM: WearablesViewModel
  @ObservedObject private var pairing = PairingModel.shared

  var body: some View {
    ZStack {
      // Black background for letterboxing/pillarboxing
      Color.black
        .edgesIgnoringSafeArea(.all)

      // Video backdrop
      if let videoFrame = viewModel.currentVideoFrame, viewModel.hasReceivedFirstFrame {
        GeometryReader { geometry in
          Image(uiImage: videoFrame)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: geometry.size.width, height: geometry.size.height)
            .clipped()
        }
        .edgesIgnoringSafeArea(.all)
      } else {
        ProgressView()
          .scaleEffect(1.5)
          .tint(Color.gtAccent)
      }

      // Top overlay: live status, stream address for OpenCV clients on the same Wi-Fi, Mac link
      VStack(alignment: .leading, spacing: 8) {
        HStack(spacing: 8) {
          StatusBadge(isLive: viewModel.hasReceivedFirstFrame)
          Spacer()
          if let streamURL = viewModel.streamURL {
            AddressChip(address: streamURL)
          }
        }
        if case .linked(let mac) = pairing.state {
          Text("Linked to \(mac)")
            .font(.system(size: 12))
            .foregroundStyle(Color.gtText)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.gtSurface.opacity(0.85), in: Capsule())
            .overlay(Capsule().stroke(Color.gtLine, lineWidth: 1))
        }
        Spacer()
      }
      .padding(.horizontal, 16)
      .padding(.top, 8)

      // Bottom controls layer

      VStack {
        Spacer()
        ControlsView(viewModel: viewModel)
      }
      .padding(.all, 24)
    }
    .onDisappear {
      Task {
        if viewModel.streamingStatus != .stopped {
          await viewModel.stopSession()
        }
      }
    }
    // Show captured photos from DAT SDK in a preview sheet
    .sheet(isPresented: $viewModel.showPhotoPreview) {
      if let photo = viewModel.capturedPhoto {
        PhotoPreviewView(
          photo: photo,
          onDismiss: {
            viewModel.dismissPhotoPreview()
          }
        )
      }
    }
  }
}

// "LIVE" once the first frame arrives, "STARTING" before that
struct StatusBadge: View {
  let isLive: Bool

  var body: some View {
    let tint = isLive ? Color.gtLive : Color.gtAccent
    HStack(spacing: 6) {
      Circle()
        .fill(tint)
        .frame(width: 8, height: 8)
      Text(isLive ? "LIVE" : "STARTING")
        .font(.system(size: 12, weight: .bold, design: .monospaced))
        .foregroundStyle(isLive ? Color.gtText : Color.gtAccent)
    }
    .padding(.horizontal, 10)
    .padding(.vertical, 6)
    .background(Color.black.opacity(0.55), in: Capsule())
  }
}

// Extracted controls for clarity
struct ControlsView: View {
  @ObservedObject var viewModel: StreamSessionViewModel
  var body: some View {
    // Controls row
    HStack(spacing: 8) {
      CustomButton(
        title: "Stop streaming",
        style: .destructive,
        isDisabled: false
      ) {
        Task {
          await viewModel.stopSession()
        }
      }

      // Photo button
      CircleButton(icon: "camera.fill", text: nil) {
        viewModel.capturePhoto()
      }
      .accessibilityIdentifier("capture_photo_button")
    }
  }
}
