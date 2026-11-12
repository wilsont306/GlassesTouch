#!/usr/bin/env python3
"""Turn a camera view of the MacBook screen into touch input.

Index fingertip moves the cursor; thumb-index pinch holds the mouse button.
"""
import argparse
import json
import math
import os
import signal
import functools
import select
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PARENT_PID = os.getppid()  # before the slow imports below, so a parent that dies during them is still noticed

import cv2
import mediapipe as mp
import mss
import numpy as np
import pyautogui
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (HandLandmarker, HandLandmarkerOptions,
                                           RunningMode)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def make_marker_detector(thresh_min=3, thresh_max=23, thresh_step=4):
    """ArUco detector for small markers: sub-pixel corner refinement, smaller candidates allowed (the default
    minimum perimeter rate of 0.03 is too strict for a ~32 px marker in a 1280 px frame), and the adaptive
    threshold window sizes given (pixels)."""
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 4
    p.minMarkerPerimeterRate = 0.01
    p.adaptiveThreshWinSizeMin = thresh_min
    p.adaptiveThreshWinSizeMax = thresh_max
    p.adaptiveThreshWinSizeStep = thresh_step
    # read bits from cell centres only: at ~32 px the default 0.13 margin bleeds blur from neighbouring cells, and
    # ids 0 and 2 (thin cells) never decoded live; on 206 recorded glasses frames this took frames with 2+ markers
    # from 44 to 195, with all-4 fit residuals under 3.1 px
    p.perspectiveRemoveIgnoredMarginPerCell = 0.3
    p.perspectiveRemovePixelPerCell = 8
    return cv2.aruco.ArucoDetector(ARUCO_DICT, p)


ARUCO_DETECTOR = make_marker_detector()
# guided ROIs are upscaled 2x, so the threshold windows must grow with the marker to binarise it well
ROI_DETECTOR = make_marker_detector(5, 61, 4)
GUIDE_PAD = 0.75      # guided ROI: padding beyond the predicted marker window, in window widths
GUIDE_SCALE = 2       # guided ROI: upscale factor before detection
GUIDE_MATCH = 1.0     # guided re-detection counts only if its centre is within this many window widths of the prediction
DRIFT_FRAMES = 3         # 2-marker fits that disagree with the tracker and agree with each other: markers win
TRACK_ONLY_MAX = 8.0         # seconds without a marker fix before a tracker-only M is not trusted (it may have drifted)
TRUST_FRAC = 0.04     # 2-marker M vs tracker M: disagreement above this fraction of the frame width keeps the tracker's M
PAIR_SIDES = ({0, 1}, {2, 3}, {0, 3}, {1, 2})  # marker pairs on one side of the screen (top, bottom, left, right)
CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]  # ids 0..3
HOLD_SECONDS = 0.5   # keep last homography this long after markers are lost
TRACK_FEATURES = 300    # ScreenTracker: features to follow inside the screen quad
TRACK_REDETECT = 100    # re-detect features when fewer than this survive
TRACK_MIN_INLIERS = 25  # fewer tracked inliers than this counts as a tracking failure
SMOOTH_ALPHA = 0.5
TOUCH_PORT = 9787       # imu_taps.py sends tap datagrams here
TOUCH_SETTLE = 0.25     # decide a tap this long after it (in capture time): its fingertip samples have arrived
TOUCH_WINDOW = (-0.12, 0.12)  # fingertip samples that decide a tap's point, seconds around the tap
TOUCH_SPAN_MAX = 250.0        # samples in the window spanning more than this: a sweeping hand, not a tap
TOUCH_TAP_AGE, TOUCH_SAMPLE_AGE = 2.0, 3.0  # drop undecided taps / old fingertip samples after this
TOUCH_MIN_GAP = 0.4           # accepted clicks at least this far apart (seconds): one knock, one click
TOUCH_DRAIN_MAX = 64          # tap datagrams read per camera frame
GESTURE_HOLD = 0.35          # a touch-down still for this long (capture time) is a click; a drag needs this hold
GESTURE_MOVE = 30.0          # points the fingertip must move from the touch point to start a drag or scroll
GESTURE_LIFT_WINDOW = 0.35   # lift-off: average fingertip speed over this much capture time ...
GESTURE_LIFT_SPEED = 12.0    # ... below this (points/s) means the finger has come to rest (slow drags stay above it)
GESTURE_OFF = 0.6            # lift-off: off screen, hand lost or no frame for longer than this (bursty camera frames)
GESTURE_MAX = 10.0           # hard cap on one drag or scroll, so a button is never held for long
GESTURE_STALL = 0.5          # wall-clock seconds without a camera frame while a button is held: release it
GESTURE_SMOOTH = 0.6         # drag position: weight of the newest fingertip sample
GESTURE_HIST = 3.0           # fingertip frames kept so a late-accepted touch-down can be replayed
SCROLL_GAIN = 0.1            # finger points -> scroll units (about 10 pt per unit)
SCROLL_DEADBAND = 3.0        # scrolling ignores finger motion below this many points (jitter)
VIEW_MAX_W = 540             # --view-port: frames are at most this wide ...
VIEW_FPS = 15                # ... and at most this many a second (shared by all clients)
TAP_FEATURES = {"gyro_peak": "g", "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz", "acc_peak": "a",
                "acc_x": "ax", "acc_y": "ay", "acc_z": "az", "lid_delta": "lid"}  # imu_taps.py fields -> debug names
TAP_MIN_ACC, TAP_MAX_ACC = 0.003, 0.08  # accel change (g) of a screen tap: below min is weak, above max a knock
TAP_MAX_GYRO = 3.0    # rad/s: a bigger gyro peak is a knock on the case, not a tap on the screen
TYPING_BELOW = 150.0  # points: a second hand whose index tip is this far under the screen is over the keyboard


def detect_marker_corners(frame):
    """Return {marker_id: (4, 2) corners in TL, TR, BR, BL order} for ArUco markers in a BGR or gray frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c[0] for i, c in zip(ids.flatten(), corners)}


def detect_markers(frame):
    """Return {marker_id: (cx, cy)} for ArUco markers found in a BGR or gray frame."""
    return {i: c.mean(axis=0) for i, c in detect_marker_corners(frame).items()}


def homography_from_markers(centres_by_id, W, H, targets=None):
    """Camera pixels -> screen homography from markers 0..3, or None if any is missing.
    targets: optional {id: (x, y)} screen point for each marker centre; default is the screen corners."""
    if not all(i in centres_by_id for i in range(4)):
        return None
    src = np.float32([centres_by_id[i] for i in range(4)])
    if targets is None:
        dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    else:
        dst = np.float32([targets[i] for i in range(4)])
    M, _ = cv2.findHomography(src, dst)
    return M


def homography_from_corners(corners_by_id, targets):
    """Camera pixels -> screen homography from every corner of the overlay markers seen (>= 2).
    targets: {id: (4, 2) screen corners}. RANSAC rejects a bad corner."""
    ids = [i for i in targets if i in corners_by_id]
    if len(ids) < 2:
        return None
    src = np.float32(np.concatenate([corners_by_id[i] for i in ids]))
    dst = np.float32(np.concatenate([targets[i] for i in ids]))
    # fit screen -> camera so that the RANSAC threshold (5 px) is in camera pixels, not screen points
    G, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    if G is None or mask is None or mask.sum() < 6 or not np.isfinite(G).all() or abs(np.linalg.det(G)) <= 1e-9:
        return None  # too few inliers, or a degenerate (near-collinear) fit
    M = np.linalg.inv(G)
    return M if np.isfinite(M).all() else None


def _poly(pts, shape):
    """int32 polygon for cv2 fill/draw; coordinates are clipped so far-off values cannot overflow."""
    lim = 4 * max(shape)
    return np.int32(np.clip(np.round(pts), -lim, lim))


def screen_quad(M, W, H):
    """Screen corners (TL, TR, BR, BL) in camera pixels; M maps camera pixels -> screen points.
    None if M is missing, singular or non-finite."""
    if M is None:
        return None
    try:
        quad = cv2.perspectiveTransform(np.float32([[[0, 0], [W, 0], [W, H], [0, H]]]), np.linalg.inv(M))[0]
    except np.linalg.LinAlgError:
        return None
    return quad if np.isfinite(quad).all() else None


def _on_mask(mask, pts):
    """Which (N, 1, 2) points land on a nonzero mask pixel; points off the image never do."""
    p = np.nan_to_num(pts.reshape(-1, 2), nan=-1e9, posinf=1e9, neginf=-1e9)
    h, w = mask.shape
    x, y = np.round(p[:, 0]), np.round(p[:, 1])
    on = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return on & (mask[np.clip(y, 0, h - 1).astype(int), np.clip(x, 0, w - 1).astype(int)] > 0)


def window_quads(targets):
    """{id: (4, 2) screen corners of the whole overlay window}, from each black-square quad in targets."""
    lo, hi = marker_extent()
    out = {}
    for i, q in targets.items():
        c = q.mean(axis=0)
        out[i] = c + (q - c) / (hi - lo)
    return out


def guided_redetect(gray, M, targets, found):
    """Re-detect overlay markers missing from `found` ({id: corners}). For each, the window is predicted in the
    camera image through inv(M), a ROI padded by GUIDE_PAD window widths is upscaled GUIDE_SCALE x and searched once.
    A hit counts only if its id matches and its centre is within GUIDE_MATCH window widths of the prediction.
    Returns {id: (4, 2) full-frame corners} for the markers found this way."""
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return {}
    h, w = gray.shape[:2]
    windows = window_quads(targets)
    out = {}
    for i in sorted(targets):
        if i in found:
            continue
        pred = cv2.perspectiveTransform(windows[i][None].astype(np.float32), Minv)[0]
        if not np.isfinite(pred).all():
            continue
        side = np.linalg.norm(pred - np.roll(pred, -1, axis=0), axis=1).mean()  # window width in camera px
        centre = pred.mean(axis=0)
        pad = GUIDE_PAD * side
        x0, x1 = int(max(0, np.floor(pred[:, 0].min() - pad))), int(min(w, np.ceil(pred[:, 0].max() + pad)))
        y0, y1 = int(max(0, np.floor(pred[:, 1].min() - pad))), int(min(h, np.ceil(pred[:, 1].max() + pad)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        roi = cv2.resize(gray[y0:y1, x0:x1], None, fx=GUIDE_SCALE, fy=GUIDE_SCALE, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = ROI_DETECTOR.detectMarkers(roi)
        if ids is None:
            continue
        best = None
        for cid, c in zip(ids.flatten(), corners):
            if int(cid) != i:
                continue
            full = (c[0] + 0.5) / GUIDE_SCALE - 0.5 + np.float32([x0, y0])  # ROI pixel centres -> frame pixels
            d = np.linalg.norm(full.mean(axis=0) - centre)
            if d <= GUIDE_MATCH * side and (best is None or d < best[0]):
                best = (d, full)
        if best is not None:
            out[i] = best[1].astype(np.float32)
    return out


def overlay_guess(corners, targets, M_track, M_held):
    """Estimate of M for predicting where missed overlay markers are: the markers' own fit when 3 or more are seen,
    else the ScreenTracker's M, else the 2-marker fit, else the held M. None if there is nothing."""
    seen = sum(i in corners for i in targets)
    M_pair = homography_from_corners(corners, targets) if seen >= 2 else None
    if seen >= 3 and M_pair is not None:
        return M_pair
    if M_track is not None:
        return M_track
    if M_pair is not None:
        return M_pair
    return M_held


def refine_overlay_corners(gray, corners, targets, M_guess):
    """Add guided re-detections of the overlay markers that `corners` misses. Returns (corners, redetected ids)."""
    if M_guess is None or all(i in corners for i in targets):
        return dict(corners), []
    extra = guided_redetect(gray, M_guess, targets, corners)
    return {**corners, **extra}, sorted(extra)


def choose_overlay_M(corners, targets, M_track, W, H, frame_w, drift=None):
    """Trust rules for the overlay homography from the markers seen (corners: {id: (4, 2)}, direct or re-detected).
    M_track: the ScreenTracker's current M, or None. Returns (M, source, reset): source is 'markers', 'tracking'
    or None; reset says whether the tracker should restart from M. drift: a dict the caller keeps across frames,
    for the drift guard (see _drift_vote); None disables it."""
    seen = [i for i in targets if i in corners]
    M_pair = homography_from_corners(corners, targets) if len(seen) >= 2 else None
    if M_pair is not None and screen_quad(M_pair, W, H) is None:
        M_pair = None  # singular fit counts as a failure
    if M_pair is None:
        if drift is not None:
            drift.clear()
        return (M_track, "tracking", False) if M_track is not None else (None, None, False)
    if len(seen) >= 3:
        if drift is not None:
            drift.clear()
        return M_pair, "markers", True
    if M_track is not None:  # two markers: a diagonal pair is poorly conditioned, so check it against the tracker
        qa, qb = screen_quad(M_pair, W, H), screen_quad(M_track, W, H)
        if qa is not None and qb is not None and np.linalg.norm(qa - qb, axis=1).mean() > TRUST_FRAC * frame_w:
            if drift is not None and _drift_vote(drift, qa, set(seen) in PAIR_SIDES, frame_w):
                drift.clear()
                return M_pair, "markers", True  # the tracker has drifted; the markers agree with each other
            return M_track, "tracking", False
        if drift is not None:
            drift.clear()
        return M_pair, "markers", True
    if drift is not None:
        drift.clear()
    return M_pair, "markers", set(seen) in PAIR_SIDES


def _drift_vote(drift, quad, same_side, frame_w):
    """One more 2-marker fit that disagrees with the tracker (quad: its screen corners in camera px). drift counts
    consecutive such fits that agree with each other. True when the markers should win: DRIFT_FRAMES in a row, or at
    once for a same-side pair that agrees with the previous fit."""
    prev = drift.get("quad")
    agree = prev is not None and np.linalg.norm(quad - prev, axis=1).mean() < TRUST_FRAC * frame_w
    drift["n"] = drift.get("n", 0) + 1 if agree else 1
    drift["quad"] = quad
    return drift["n"] >= DRIFT_FRAMES or (same_side and agree)


def tracking_age_guard(M, source, now, last_marker):
    """A tracker-only M is not trusted once markers have not reset the tracker for TRACK_ONLY_MAX s (it may have
    drifted): no source, so no touches, until markers return."""
    if source == "tracking" and now - last_marker > TRACK_ONLY_MAX:
        return None, None
    return M, source


class ScreenTracker:
    """Follow the screen between marker fixes: LK-track features and chain frame-to-frame homographies
    into M (camera -> screen). Features come from the whole screen quad (printed markers), or only from
    static_regions plus a bezel ring (overlay mode, where screen content may scroll). After a failure it
    stays lost until reset()."""
    LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def __init__(self, W, H, static_regions=None):
        self.W, self.H, self.static_regions = W, H, static_regions  # screen-point polygons, or None
        self.prev, self.M, self.pts, self.area, self.lost = None, None, None, 0.0, True

    def _mask(self, shape, M, exclude=None):
        """uint8 mask of where features may be, minus the hand polygon `exclude` (camera pixels).
        None if M gives no screen quad."""
        quad = screen_quad(M, self.W, self.H)
        if quad is None:
            return None
        mask = np.zeros(shape, np.uint8)
        if self.static_regions is None:
            cv2.fillConvexPoly(mask, _poly(quad, shape), 255)
        else:
            c = quad.mean(axis=0)
            cv2.fillConvexPoly(mask, _poly(c + 1.04 * (quad - c), shape), 255)  # bezel ring: outer quad...
            cv2.fillConvexPoly(mask, _poly(quad, shape), 0)                     # ...minus the screen
            Minv = np.linalg.inv(M)
            for region in self.static_regions:
                cv2.fillConvexPoly(mask, _poly(cv2.perspectiveTransform(np.float32([region]), Minv)[0], shape), 255)
        if exclude is not None:
            cv2.fillConvexPoly(mask, _poly(exclude, shape), 0)
        return mask

    def _detect(self, gray, mask):
        pts = cv2.goodFeaturesToTrack(gray, TRACK_FEATURES, 0.01, 7, mask=mask)
        return pts if pts is not None else np.empty((0, 1, 2), np.float32)

    def reset(self, gray, M):
        """Start from a fresh marker homography M for the frame `gray`. Features are found lazily in update()."""
        quad = screen_quad(M, self.W, self.H)
        self.prev, self.M, self.pts = gray, M, None
        self.area = 0.0 if quad is None else cv2.contourArea(quad)
        self.lost = self.area <= 0

    def _fail(self):
        self.lost = True
        return None

    def _sane(self, quad, shape):
        """Reject folds, jumps and wild zooms: convex, area within 30% of the last quad, corners near the frame."""
        h, w = shape
        if not cv2.isContourConvex(quad) or abs(cv2.contourArea(quad) / self.area - 1) > 0.3:
            return False
        return bool(((quad >= [-w, -h]) & (quad <= [2 * w, 2 * h])).all())

    def update(self, gray, exclude=None):
        """Track into `gray`. exclude: optional camera-pixel polygon (the hand) whose features are ignored.
        Return the new camera -> screen M, or None (then lost until reset)."""
        if self.lost:
            return None
        if self.pts is None:  # first update after reset: find features in the reset frame, under its M
            mask = self._mask(self.prev.shape, self.M, exclude)
            if mask is None:
                return self._fail()
            self.pts = self._detect(self.prev, mask)
        if len(self.pts) < TRACK_MIN_INLIERS:
            return self._fail()
        p0 = self.pts
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, **self.LK)
        p0b, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **self.LK)  # forward-backward check
        fb = np.linalg.norm(p0b - p0, axis=2).ravel()
        ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb < 1.0)
        if ok.sum() < TRACK_MIN_INLIERS:
            return self._fail()
        H_ff, mask = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 3.0)  # prev frame -> this frame
        if H_ff is None or mask.sum() < TRACK_MIN_INLIERS or not np.isfinite(H_ff).all() \
                or abs(np.linalg.det(H_ff)) <= 1e-9:
            return self._fail()
        M_new = self.M @ np.linalg.inv(H_ff)
        quad = screen_quad(M_new, self.W, self.H)
        if quad is None or not self._sane(quad, gray.shape):
            return self._fail()
        mask_img = self._mask(gray.shape, M_new, exclude)
        inl = p1[ok][mask.ravel().astype(bool)]
        self.prev, self.M, self.area = gray, M_new, cv2.contourArea(quad)
        self.pts = inl[_on_mask(mask_img, inl)]
        if len(self.pts) < TRACK_REDETECT:
            self.pts = self._detect(gray, mask_img)
        return M_new


def map_point(M, pt, screen):
    """Apply homography M to pt; clamped to [1, W-2] x [1, H-2] so it never sits on
    pyautogui's four failsafe corners (screen = (W, H))."""
    x, y = cv2.perspectiveTransform(np.float32([[pt]]), M)[0, 0]
    W, H = screen
    return min(max(float(x), 1.0), W - 2.0), min(max(float(y), 1.0), H - 2.0)


def pinch_ratio(pts):
    """Thumb-tip to index-tip distance over wrist to middle-MCP distance. pts: 21 (x, y) pixels."""
    def dist(a, b):
        return np.linalg.norm(np.subtract(pts[a], pts[b]))
    return dist(4, 8) / max(dist(0, 9), 1e-6)


def two_finger(pts):
    """True if index and middle fingers point together: both extended, ring and pinky folded, index and middle
    tips within 0.6 hand sizes of each other. pts: 21 (x, y) landmark pixels, as for pinch_ratio."""
    def dist(a, b):
        return np.linalg.norm(np.subtract(pts[a], pts[b]))
    hand = dist(0, 9)
    return bool(dist(0, 8) > dist(0, 6) and dist(0, 12) > dist(0, 10)
                and dist(0, 16) < dist(0, 14) and dist(0, 20) < dist(0, 18)
                and dist(8, 12) < 0.6 * hand)


class PinchDetector:
    """Hysteresis on the pinch ratio: goes down below `enter`, up above `exit`."""

    def __init__(self, enter=0.30, exit=0.45):
        self.enter, self.exit = enter, exit
        self.down = False

    def update(self, ratio):
        """Return 'down', 'up', or None for this sample."""
        if not self.down and ratio < self.enter:
            self.down = True
            return "down"
        if self.down and ratio > self.exit:
            self.down = False
            return "up"
        return None

    def release(self):
        """Force 'up' if currently held (e.g. the hand disappeared mid-pinch)."""
        if self.down:
            self.down = False
            return "up"
        return None


def make_marker_sheet(path, px=300, margin=80):
    """Write a printable 2x2 sheet of the four corner markers, each labelled with its corner."""
    cell = px + 2 * margin
    sheet = np.full((2 * cell, 2 * cell), 255, np.uint8)
    # grid position (row, col) for ids 0..3: TL, TR, BR, BL
    grid = [(0, 0), (0, 1), (1, 1), (1, 0)]
    for i, (row, col) in enumerate(grid):
        x0, y0 = col * cell, row * cell
        marker = cv2.aruco.generateImageMarker(ARUCO_DICT, i, px)
        sheet[y0 + margin:y0 + margin + px, x0 + margin:x0 + margin + px] = marker
        label = f"id {i}: {CORNER_NAMES[i]}"
        cv2.putText(sheet, label, (x0 + margin, y0 + margin - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    cv2.imwrite(path, sheet)


class RegionGrabber:
    """Grab a rectangle of the Mac's own screen with mss, shaped like cv2.VideoCapture.read()."""

    def __init__(self, x, y, w, h):
        self.sct = mss.mss()
        self.mon = {"left": x, "top": y, "width": w, "height": h}

    def read(self):
        img = np.asarray(self.sct.grab(self.mon))  # BGRA
        return True, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def release(self):
        self.sct.close()


def open_source(args):
    if args.region:
        x, y, w, h = map(int, args.region.split(","))
        return RegionGrabber(x, y, w, h)
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:  # network stream: fail a hung read after 3 s (default 30 s), so a held drag can't stay down that long
        cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")
    return cap


def make_hand_landmarker():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
        os.replace(MODEL_PATH + ".part", MODEL_PATH)
    opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                 running_mode=RunningMode.VIDEO, num_hands=2)  # 2 so the other hand is logged
    return HandLandmarker.create_from_options(opts)


def draw_debug(frame, centres, lm, pinched, status, redetected=(), m_source=None):
    """Markers in green (decoded directly) or yellow (guided re-detection); line 2 says which ids were used
    and whether M came from the markers or from tracking."""
    h, w = frame.shape[:2]
    for i, (cx, cy) in centres.items():
        col = (0, 255, 255) if i in redetected else (0, 255, 0)
        cv2.circle(frame, (int(cx), int(cy)), 8, col, -1)
        cv2.putText(frame, str(i), (int(cx) + 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if lm:
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (255, 0, 0), -1)
    cv2.putText(frame, f"{status} pinch={'DOWN' if pinched else 'up'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    direct = sorted(i for i in centres if i not in redetected)
    cv2.putText(frame, f"ids direct {direct} re-detected {sorted(redetected)} M from {m_source or 'none'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


class FrameHub:
    """Latest annotated frame as JPEG for the view clients. put() replaces the frame (no queue) and encodes at most
    VIEW_FPS times a second; get() blocks until a frame newer than `seen` arrives."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpg, self.seq, self.last = None, 0, -1e9

    def put(self, frame):
        now = time.monotonic()
        if now - self.last < 1 / VIEW_FPS:
            return
        self.last = now
        h, w = frame.shape[:2]
        if w > VIEW_MAX_W:
            frame = cv2.resize(frame, (VIEW_MAX_W, round(h * VIEW_MAX_W / w)), interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        with self.cond:
            self.jpg, self.seq = jpg, self.seq + 1
            self.cond.notify_all()

    def get(self, seen):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seen)
            return self.seq, self.jpg


class ViewHandler(BaseHTTPRequestHandler):
    """Any GET streams the view as multipart JPEG. Only the local Host names are served (DNS rebinding)."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        port = self.server.server_port
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self.send_error(403)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = 0
        try:
            while True:
                seen, jpg = self.server.hub.get(seen)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                                 + jpg + b"\r\n")
                self.wfile.flush()
        except OSError:
            pass  # the client went away


class ViewServer(ThreadingHTTPServer):
    """--view-port: the debug view on 127.0.0.1 from a daemon thread. The thread polls handle_request, so close()
    can stop it without shutdown() (which would hang if the thread had not started its loop yet)."""
    timeout = 0.5

    def __init__(self, port):
        super().__init__(("127.0.0.1", port), ViewHandler)
        self.hub, self.stopping = FrameHub(), threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while not self.stopping.is_set():
            self.handle_request()

    def close(self):
        self.stopping.set()
        self.thread.join()
        self.server_close()


def marker_png(i):
    """PNG bytes of marker i with a white quiet zone of one module on every side (detection needs it)."""
    module = 100  # pixels per module; generateImageMarker's 4x4 bits plus 1-module black border = 6 modules
    marker = cv2.aruco.generateImageMarker(ARUCO_DICT, i, 6 * module)
    img = np.full((8 * module, 8 * module), 255, np.uint8)
    img[module:7 * module, module:7 * module] = marker
    return cv2.imencode(".png", img)[1].tobytes()


@functools.lru_cache(maxsize=None)
def marker_extent():
    """(lo, hi): where the black marker sits in its window, as fractions of the side (measured, not assumed)."""
    img = cv2.imdecode(np.frombuffer(marker_png(0), np.uint8), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.nonzero(img < 128)
    n = img.shape[0]  # square image; the black marker is square too
    return xs.min() / n, (xs.max() + 1) / n


def overlay_child(size):
    """Child process: one borderless always-on-top window per marker in the main screen's corners.
    Prints each marker's four black-square corners (pyautogui points, top-left origin), then runs the AppKit loop."""
    import AppKit
    lo, hi = marker_extent()
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # no Dock icon
    sf = AppKit.NSScreen.screens()[0].frame()  # menu-bar display: matches pyautogui's origin and size
    sx, sy, sw, sh = sf.origin.x, sf.origin.y, sf.size.width, sf.size.height
    # Cocoa origin is bottom-left; corners by marker id: TL, TR, BR, BL
    corners = {0: (sx, sy + sh - size), 1: (sx + sw - size, sy + sh - size),
               2: (sx + sw - size, sy), 3: (sx, sy)}
    windows, quads = [], {}
    for i, (x, y) in corners.items():
        png = marker_png(i)
        image = AppKit.NSImage.alloc().initWithData_(AppKit.NSData.dataWithBytes_length_(png, len(png)))
        win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(x, y, size, size), AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered, False)
        win.setLevel_(AppKit.NSStatusWindowLevel + 1)
        win.setIgnoresMouseEvents_(True)
        win.setOpaque_(True)
        win.setBackgroundColor_(AppKit.NSColor.whiteColor())
        win.setHasShadow_(False)
        win.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                   | AppKit.NSWindowCollectionBehaviorStationary
                                   | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary)
        view = AppKit.NSImageView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, size, size))
        view.setImage_(image)
        view.setImageScaling_(AppKit.NSImageScaleAxesIndependently)
        win.setContentView_(view)
        win.orderFrontRegardless()
        f = win.frame()  # macOS may have moved the window (menu bar, notch), so read the real frame
        fx, fy, fw, fh = f.origin.x, sh - (f.origin.y + f.size.height), f.size.width, f.size.height  # top-left origin
        # black-square corners in TL, TR, BR, BL order, matching cv2.aruco corner order
        quads[str(i)] = [[fx + u * fw, fy + v * fh] for u, v in
                         ((lo, lo), (hi, lo), (hi, hi), (lo, hi))]
        windows.append((win, view))  # keep references alive for the run loop
    print(json.dumps(quads), flush=True)

    def exit_with_parent():  # parent holds our stdin pipe; its EOF means the parent is gone
        sys.stdin.read()
        os._exit(0)
    threading.Thread(target=exit_with_parent, daemon=True).start()
    app.run()


def start_overlay(size=140):
    """Spawn the overlay child; return (proc, {id: (4, 2) corners}) with the markers' on-screen corners."""
    proc = subprocess.Popen([sys.executable, __file__, "--overlay-child", str(size)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while True:
            left = deadline - time.monotonic()
            ready = select.select([proc.stdout], [], [], left)[0] if left > 0 else []
            if not ready:
                raise SystemExit("marker overlay gave no positions within 15 s (see its stderr above)")
            line = proc.stdout.readline()
            if not line:  # EOF: the child exited before reporting
                proc.wait()
                raise SystemExit(f"marker overlay exited with code {proc.returncode}")
            if line.startswith("{"):
                try:
                    quads = {int(k): np.float32(v) for k, v in json.loads(line).items()}
                except (ValueError, TypeError) as e:
                    raise SystemExit(f"marker overlay sent bad positions: {e}")
                if any(q.shape != (4, 2) for q in quads.values()):
                    raise SystemExit("marker overlay sent bad positions: expected 4 corners per marker")
                return proc, quads
    except BaseException:
        stop_overlay(proc)
        raise


def stop_overlay(proc):
    proc.stdin.close()  # child sees EOF and exits on its own
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    proc.stdout.close()


def accessibility_trusted():
    """True if macOS lets this process post mouse events (pyautogui fails silently otherwise)."""
    import ctypes
    ax = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
    ax.AXIsProcessTrusted.restype = ctypes.c_bool
    return ax.AXIsProcessTrusted()


def overlay_regions(targets):
    """Screen-point polygons of the overlay windows: each black-square quad scaled up to its whole window."""
    lo, hi = marker_extent()
    out = []
    for q in targets.values():
        c = q.mean(axis=0)
        out.append(c + (q - c) / (hi - lo))
    return out


def hand_exclusion(lm, w, h, r=20):
    """Camera-pixel polygon around the hand: its landmarks grown by r px (convex hull of disc samples)."""
    pts = np.float32([(p.x * w, p.y * h) for p in lm])
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    grown = (pts[:, None, :] + r * np.stack([np.cos(ang), np.sin(ang)], 1)[None]).reshape(-1, 2)
    return cv2.convexHull(grown.astype(np.float32)).reshape(-1, 2)


def pointing_hand(hands, w, h, ref, on_screen=None):
    """Index in MediaPipe's hand list (non-empty) of the pointing hand: among the hands whose index tip is on the
    screen (on_screen, a list of bools; all hands if none is), the one whose tip is nearest ref (camera pixels).
    Preferring on-screen tips stops it sticking to a hand that moved to the keyboard."""
    pick = [i for i in range(len(hands)) if on_screen and on_screen[i]] or range(len(hands))
    return min(pick, key=lambda i: math.hypot(hands[i][8].x * w - ref[0], hands[i][8].y * h - ref[1]))


def typing_flag(hands, pointing, w, h, M, fresh, H, below=TYPING_BELOW):
    """True when a hand other than the pointing one (index in `hands`) is over the keyboard: its index tip, mapped
    through M, is more than `below` points under the screen (height H), or there is no fresh M to map it with."""
    others = [hd for i, hd in enumerate(hands) if i != pointing]
    if not others:
        return False
    if not fresh or M is None:
        return True
    return bool(raw_map(M, (others[0][8].x * w, others[0][8].y * h))[1] > H + below)


class TouchFuser:
    """Pair sensor taps with the fingertip. A tap is accepted when an on-screen fingertip sample lies within
    TOUCH_WINDOW of it, and its point is the fingertip at the tap time (contact_point). Fingertip times are capture
    times (arrival minus camera latency); poll() decides taps at least TOUCH_SETTLE old. Every tap poll() decides or
    drops is recorded in self.decisions (see format_decision)."""

    def __init__(self, min_acc=TAP_MIN_ACC, max_acc=TAP_MAX_ACC):
        self.min_acc, self.max_acc = min_acc, max_acc  # tap strength limits (accel change, g)
        self.samples = deque()  # (t, x, y, on_screen)
        self.hands = deque()    # (t, hands MediaPipe saw in that frame)
        self.typing = deque()   # (t, True when a second hand was over the keyboard in that frame)
        self.taps = deque()     # (tap time, peak or None, feature dict) from imu_taps.py
        self.last_click = None  # tap time of the last accepted click
        self.decisions = []     # debug records from the last poll()
        self.downs = []        # (tap time, x, y) of the touch-downs accepted by the last poll()

    def add_fingertip(self, t, x, y, on_screen):
        self.samples.append((t, x, y, on_screen))

    def add_hands(self, t, n):
        self.hands.append((t, n))

    def add_typing(self, t, flag):
        self.typing.append((t, bool(flag)))

    def _typing(self, tap):
        return any(f and tap + TOUCH_WINDOW[0] <= t <= tap + TOUCH_WINDOW[1] for t, f in self.typing)

    def add_tap(self, t, peak=None, feats=None):
        self.taps.append((t, peak, feats or {}))

    def _window(self, tap):
        return [(t, x, y, on) for t, x, y, on in self.samples if tap + TOUCH_WINDOW[0] <= t <= tap + TOUCH_WINDOW[1]]

    def _record(self, tap, peak, outcome, point=None, feats=None):
        offs = [t - tap for t, *_ in self.samples]
        ctx = [o for o in offs if abs(o) <= 0.5]  # the 1 s around the tap
        seen = [n for t, n in self.hands if tap + TOUCH_WINDOW[0] <= t <= tap + TOUCH_WINDOW[1]]
        self.decisions.append({
            "tap": tap, "outcome": outcome, "peak": peak, "n": len(self._window(tap)),
            "hands": max(seen) if seen else None, "typing": self._typing(tap), "feats": feats or {},
            "nearest_ms": min(offs, key=abs) * 1000 if offs else None,
            "ctx_n": len(ctx), "ctx_min_ms": min(ctx) * 1000 if ctx else None,
            "ctx_max_ms": max(ctx) * 1000 if ctx else None,
            "point": None if point is None else (round(point[0], 1), round(point[1], 1)),
            "xy": [(round(x), round(y)) for _, x, y, _ in self._window(tap)]})  # where the fingertip mapped

    def poll(self, now):
        """Decide each tap at least TOUCH_SETTLE old. Return the click points (x, y) of the accepted ones.
        Taps older than TOUCH_TAP_AGE are dropped first (a stall must not click late), and a tap within
        TOUCH_MIN_GAP of the last accepted click is dropped (one knock can ring twice)."""
        clicks, self.decisions, self.downs = [], [], []
        while self.taps and now - self.taps[0][0] > TOUCH_TAP_AGE:
            tap, peak, feats = self.taps.popleft()
            self._record(tap, peak, "stale", feats=feats)
        while self.taps and now - self.taps[0][0] >= TOUCH_SETTLE:
            tap, peak, feats = self.taps.popleft()
            near = self._window(tap)
            point = None
            acc = feats.get("acc_peak")  # taps from an old helper have no strength fields: no strength checks
            if self.last_click is not None and tap - self.last_click < TOUCH_MIN_GAP:
                outcome = "gap"
            elif acc is not None and acc < self.min_acc:
                outcome = f"weak acc={acc:.4f}"
            elif acc is not None and (acc > self.max_acc or feats.get("gyro_peak", 0.0) > TAP_MAX_GYRO):
                outcome = f"knock acc={acc:.3f} gyro={feats.get('gyro_peak', 0.0):.1f}"
            elif self._typing(tap):
                outcome = "typing"
            elif not near:
                outcome = "no-samples"
            elif not min(near, key=lambda s: abs(s[0] - tap))[3]:  # the nearest sample is off screen (keyboard)
                outcome = f"off-screen n_off={sum(not s[3] for s in near)}/{len(near)}"
            else:
                xy = np.float32([(x, y) for _, x, y, _ in near])
                span = float(np.ptp(xy, axis=0).max())
                if span > TOUCH_SPAN_MAX:
                    outcome = f"sweeping span={span:.0f}"
                else:
                    point = contact_point(near, tap)
                    outcome = "CLICK"
                    clicks.append(point)
                    self.downs.append((tap, *point))
                    self.last_click = tap
            self._record(tap, peak, outcome, point, feats)
        while self.samples and now - self.samples[0][0] > TOUCH_SAMPLE_AGE:
            self.samples.popleft()
        while self.hands and now - self.hands[0][0] > TOUCH_SAMPLE_AGE:
            self.hands.popleft()
        while self.typing and now - self.typing[0][0] > TOUCH_SAMPLE_AGE:
            self.typing.popleft()
        return clicks


def contact_point(samples, tap):
    """Fingertip (x, y) at the tap time, from (t, x, y, on_screen) samples near it: linear interpolation between the
    on-screen samples just before and just after the tap, or the nearest on-screen sample when only one side has one."""
    on = [s for s in samples if s[3]]
    before = [s for s in on if s[0] <= tap]
    after = [s for s in on if s[0] >= tap]
    if before and after:
        b, a = max(before, key=lambda s: s[0]), min(after, key=lambda s: s[0])
        if a[0] > b[0]:
            w = (tap - b[0]) / (a[0] - b[0])
            return (float(b[1] + w * (a[1] - b[1])), float(b[2] + w * (a[2] - b[2])))
        return (float(b[1]), float(b[2]))
    s = min(on, key=lambda s: abs(s[0] - tap))
    return (float(s[1]), float(s[2]))


def debug_log(line):
    """Print a --touch-debug line and append it to build/touch_debug.log so it can be read afterwards."""
    print(line)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "touch_debug.log"), "a") as f:
        f.write(line + "\n")


def format_decision(d):
    """One debug line for a TouchFuser decision record."""
    nearest = "-" if d["nearest_ms"] is None else f"{d['nearest_ms']:+.0f}ms"
    peak = "-" if d["peak"] is None else f"{d['peak']:.1f}"
    ctx = "" if not d["ctx_n"] else f" [{d['ctx_min_ms']:+.0f}..{d['ctx_max_ms']:+.0f}ms]"
    pt = "" if d.get("point") is None else f" point=({d['point'][0]:.0f},{d['point'][1]:.0f})"
    hands = "-" if d.get("hands") is None else d["hands"]
    feats = "".join(f" {short}={d['feats'][k]:.1f}" for k, short in TAP_FEATURES.items() if k in d.get("feats", {}))
    return (f"touch tap={d['tap']:.3f} {d['outcome']} n={d['n']} hands={hands} typing={int(d.get('typing', False))} "
            f"nearest={nearest} peak={peak}{feats} "
            f"ctx1s={d['ctx_n']}{ctx}{pt} xy={d.get('xy', [])}")


class GestureEngine:
    """Touch-downs and fingertip frames to mouse actions. Pure: no pyautogui.
    A touch-down (tap time from TouchFuser) opens a decision. Two fingers (majority vote of two_finger) that move
    more than GESTURE_MOVE from the touch point scroll at once. One finger that moves before GESTURE_HOLD is a tap
    withdrawing: a click at the touch point, nothing held. One finger still for GESTURE_HOLD is a click at the touch
    point; if it then moves more than GESTURE_MOVE it is a drag (down, moves, up). A drag or scroll ends on lift-off: the fingertip
    has come to rest (average speed below GESTURE_LIFT_SPEED over GESTURE_LIFT_WINDOW), it has been off screen or
    absent for GESTURE_OFF, or GESTURE_MAX has passed. Frames are kept for GESTURE_HIST seconds so that a touch-down
    accepted late is replayed from its tap time.
    Actions: ("click"|"down"|"move"|"up"|"point", x, y) and ("scroll", dx, dy) in pyautogui units."""

    def __init__(self):
        self.hist = deque()     # (t, x, y, on_screen, two_finger), for replaying a late touch-down
        self.state = None       # None (idle), "decide", "press" (clicked, still touching), "drag" or "scroll"
        self.start = None       # (t, x, y) of the touch-down
        self.pos = None         # drag: smoothed fingertip; scroll: last fingertip that counted
        self.track = deque()    # (t, x, y) on-screen fingertip samples of the gesture, for the speed rule
        self.votes = []         # two_finger flags of the on-screen frames in the decide window
        self.off_n = 0          # consecutive frames off screen or without a hand
        self.off_t = None       # capture time of the first of them
        self.last_t = None      # capture time of the last frame
        self.acc = [0.0, 0.0]   # scroll units not yet sent
        self.t_gesture = None   # capture time the drag or scroll started (GESTURE_MAX counts from here)

    @property
    def held(self):
        """True while a drag holds the button down."""
        return self.state == "drag"

    def frame(self, t, x, y, on, two):
        self.hist.append((t, x, y, on, two))
        while self.hist and t - self.hist[0][0] > GESTURE_HIST:
            self.hist.popleft()
        self.last_t = t
        return self._step(t, x, y, on, two) if self.state else []

    def touch_down(self, t, x, y):
        acts = [("click", *self.start[1:])] if self.state == "decide" else []  # a pending touch that never moved
        acts += self.release()  # a new touch-down ends the current gesture first
        self.state, self.start = "decide", (t, x, y)
        self.pos, self.votes, self.off_n, self.off_t, self.acc = (x, y), [], 0, None, [0.0, 0.0]
        self.track = deque()
        for f in list(self.hist):  # frames that arrived after the tap
            if f[0] > t:
                acts += self._step(*f)
        return acts

    def release(self):
        """End the gesture without a new frame. A held drag sends its up; a pending decision is dropped, not clicked."""
        acts = [("up", *self.pos)] if self.state == "drag" else []
        self.state = None
        return acts

    def tick(self, t):
        """Lift-off checks with no new frame (a camera stall): GESTURE_OFF since the last frame, or the cap."""
        if self.state in ("drag", "scroll") and self.last_t is not None and t - self.last_t > GESTURE_OFF:
            return self.release()
        return self._cap(t)

    def _cap(self, t):
        return self.release() if self.state in ("drag", "scroll") and t - self.t_gesture >= GESTURE_MAX else []

    def _step(self, t, x, y, on, two):
        if self.state is None:  # a replayed frame after the gesture has ended (a withdrawal click) changes nothing
            return []
        t0, px, py = self.start
        if self.state in ("decide", "press"):  # press: the click was emitted at the hold; a move makes it a drag
            if on:
                self.votes.append(two)
            if on and math.hypot(x - px, y - py) > GESTURE_MOVE:
                if 2 * sum(self.votes) > len(self.votes):  # two fingers: scroll at once
                    self.state, self.t_gesture = "scroll", t
                    return [("point", px, py)] + self._track_frame(t, x, y, on)
                if self.state == "decide" and t - t0 < GESTURE_HOLD:  # moved before the hold: a tap withdrawing
                    self.state = None
                    return [("click", px, py)]
                self.state, self.t_gesture = "drag", t
                self.pos = (px, py)
                return [("down", px, py)] + self._track_frame(t, x, y, on)
            if self.state == "decide" and t - t0 >= GESTURE_HOLD:  # still for the whole hold: a click
                self.state = "press"
                return [("click", px, py)]
            return []
        return self._track_frame(t, x, y, on)

    def _track_frame(self, t, x, y, on):
        if not on:
            self.off_n += 1
            if self.off_n == 1:
                self.off_t = t
            if self.off_n >= 2 and t - self.off_t > GESTURE_OFF:  # one dropped frame is a burst, not a lift
                return self.release()
            return self._cap(t)
        if self.off_n:  # back on screen: continue from here, no leap across the gap
            self.off_n = 0
            self.pos = (x, y)
        acts = []
        if self.state == "drag":
            a = GESTURE_SMOOTH
            self.pos = (a * x + (1 - a) * self.pos[0], a * y + (1 - a) * self.pos[1])
            acts.append(("move", *self.pos))
        else:  # content follows the finger (natural scrolling); motion under the deadband is jitter
            dx, dy = x - self.pos[0], y - self.pos[1]
            if math.hypot(dx, dy) >= SCROLL_DEADBAND:
                self.acc[0] -= dx * SCROLL_GAIN
                self.acc[1] += dy * SCROLL_GAIN
                self.pos = (x, y)
            ix, iy = int(self.acc[0]), int(self.acc[1])
            if ix or iy:
                self.acc[0] -= ix
                self.acc[1] -= iy
                acts.append(("scroll", ix, iy))
        self.track.append((t, x, y))
        while t - self.track[0][0] > GESTURE_LIFT_WINDOW:
            self.track.popleft()
        if len(self.track) >= 2:
            (t1, x1, y1), (t2, x2, y2) = self.track[0], self.track[-1]
            if t2 > t1 and math.hypot(x2 - x1, y2 - y1) / (t2 - t1) < GESTURE_LIFT_SPEED:
                return acts + self.release()  # the finger has come to rest
        return acts + self._cap(t)


def perform_gestures(acts, args):
    """Carry out GestureEngine actions with pyautogui, or print them in a dry run. Returns the last point to flash."""
    flash = None
    for act in acts:
        msg = f"gesture {act}"
        if args.touch_debug:
            debug_log(msg)
        elif args.dry_run:
            print(msg)
        if act[0] in ("click", "down"):
            flash = act[1:]
        if args.dry_run:
            continue
        kind, *v = act
        if kind == "click":
            pyautogui.click(*v)
        elif kind == "down":
            pyautogui.mouseDown(*v, button="left")
        elif kind == "move":
            pyautogui.dragTo(*v, duration=0, button="left", mouseDownUp=False)
        elif kind == "up":
            pyautogui.mouseUp(*v, button="left")
        elif kind == "point":
            pyautogui.moveTo(*v)
        elif kind == "scroll":
            if v[1]:
                pyautogui.scroll(v[1])
            if v[0]:
                pyautogui.hscroll(v[0])
    return flash


def touch_step(engine, fuser, cap, tip, on, two):
    """One camera frame of touch handling, as run() does it. tip: fingertip (x, y) in screen points, or None when
    no hand was found; on: fingertip on screen under a trusted M; cap: the frame's capture time. Returns the actions
    in order: the frame's, then those of the touch-downs TouchFuser accepted, then any lift-off without a frame."""
    if tip is None:
        acts = engine.frame(cap, 0.0, 0.0, False, False)
    else:
        fuser.add_fingertip(cap, float(tip[0]), float(tip[1]), bool(on))
        acts = engine.frame(cap, float(tip[0]), float(tip[1]), bool(on), two)
    fuser.poll(cap)  # taps at least TOUCH_SETTLE old, decided on capture time
    for t_tap, x, y in fuser.downs:
        acts += engine.touch_down(t_tap, x, y)
    return acts + engine.tick(cap)


def release_touch(engine, args):
    """End any gesture and never leave the button down: the release actions run first, then mouseUp always follows,
    even if an action raises. A FailSafe raised by an action is swallowed here because mouseUp releases the button."""
    try:
        try:
            perform_gestures(engine.release(), args)
        except pyautogui.FailSafeException:
            pass  # cursor parked on a failsafe corner: the mouseUp below still releases the button
    finally:
        if not args.dry_run:
            try:
                pyautogui.mouseUp()  # never leave the button stuck
            except pyautogui.FailSafeException:
                # cursor parked on a failsafe corner makes mouseUp raise; retry without the check
                pyautogui.FAILSAFE = False
                pyautogui.mouseUp()


def open_touch_socket(port):
    """Non-blocking UDP socket on 127.0.0.1 for the tap datagrams from imu_taps.py."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as e:
        sock.close()
        raise SystemExit(f"cannot bind UDP port {port} on 127.0.0.1 for taps ({e}); another process may hold it. "
                         f"imu_taps.py must send to the same port (--touch-port, default {TOUCH_PORT}).")
    sock.setblocking(False)
    return sock


def _number(v):
    """float for a finite int or float (not bool), else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def drain_taps(sock, fuser, limit=TOUCH_DRAIN_MAX):
    """Feed up to `limit` queued tap datagrams to the fuser. Malformed datagrams are ignored; t must be a finite
    int or float (not bool, not a string). The optional peak and motion features (TAP_FEATURES) are kept only
    when they pass _number; the rest are dropped."""
    for _ in range(limit):
        try:
            data = sock.recv(512)
        except BlockingIOError:
            return
        try:
            obj = json.loads(data)
            t = obj["t"]
            if isinstance(t, bool) or not isinstance(t, (int, float)):
                continue
            t = float(t)
        except (ValueError, KeyError, TypeError, OverflowError):
            continue
        peak = _number(obj.get("peak"))  # optional: the sensor's peak, shown by --touch-debug
        feats = {k: v for k in TAP_FEATURES if (v := _number(obj.get(k))) is not None}
        if math.isfinite(t):
            fuser.add_tap(t, peak, feats)


def capture_time(arrival, camera_latency):
    """Monotonic capture time of a camera frame that arrived at `arrival`: frames lag real time by camera_latency."""
    return arrival - camera_latency


def raw_map(M, pt):
    """Unclamped point mapping through M (no clamping to the screen, unlike map_point)."""
    return cv2.perspectiveTransform(np.float32([[pt]]), M)[0, 0]


EDGE_TOL = 60  # screen points: from the glasses the fingertip shows a little past the contact point at edges


def screen_point(tip, W, H, tol=EDGE_TOL):
    """Fingertip mapped to the screen: (point clamped onto it, on_screen). Up to tol outside still counts (a tap
    on the Dock or an edge); the clamp keeps 1 pt off every edge, because pyautogui's FailSafe fires on corners."""
    x, y = float(tip[0]), float(tip[1])
    on = -tol <= x <= W + tol and -tol <= y <= H + tol
    return (min(max(x, 1.0), W - 2.0), min(max(y, 1.0), H - 2.0)), on


def hover_moves_cursor(args):
    """A hovering fingertip moves the cursor only without --touch; with --touch the cursor moves only on a click."""
    return not args.touch


def start_orphan_watchdog(dry_run):
    """Daemon thread: if the panel or shell that started us dies, release the mouse and exit. The frame loop
    cannot check this while it is blocked on the camera. A dry run never moves the mouse, so it doesn't click either."""
    def watch():
        while True:
            time.sleep(0.5)
            if os.getppid() != PARENT_PID:
                if not dry_run:
                    try:
                        pyautogui.mouseUp()
                    except Exception:
                        pass  # best effort: exiting below is what matters
                os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def run(args):
    start_orphan_watchdog(args.dry_run)  # first, so a hang in model loading or the camera is covered too
    W, H = pyautogui.size()
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True
    if not args.dry_run and not accessibility_trusted():
        raise SystemExit("macOS hasn't granted Accessibility permission to the app running this "
                         "(Terminal or Claude), so the cursor can't move. Enable it in System Settings > "
                         "Privacy & Security > Accessibility, restart that app, and run again.")
    landmarker = make_hand_landmarker()  # before the camera opens, so a download failure cannot leak it
    src = open_source(args)
    pinch = PinchDetector()
    M, last_ok, smooth, t0, last_ts = None, 0.0, None, time.monotonic(), -1
    overlay, targets = None, None
    touch_sock, touch_flash = None, None  # touch_flash: ((x, y) screen point, monotonic time to stop showing)
    engine = GestureEngine()  # touch-downs and fingertip frames -> click / drag / scroll actions
    view, prev_tip = None, None  # prev_tip: last frame's pointing fingertip (camera px), None without a hand
    try:
        if args.view_port:
            try:
                view = ViewServer(args.view_port)
            except OSError as e:
                raise SystemExit(f"cannot serve the view on 127.0.0.1:{args.view_port} ({e}); "
                                 "another process may hold it.")
        if args.show_markers:
            overlay, targets = start_overlay(args.marker_size)
        tracker = ScreenTracker(W, H, overlay_regions(targets) if targets is not None else None)
        hand_poly = None  # previous frame's hand, kept out of the tracker's features
        fuser = TouchFuser(args.tap_min_acc, args.tap_max_acc)
        if args.touch:
            touch_sock = open_touch_socket(args.touch_port)
        last_frame_wall = time.monotonic()  # wall clock of the last camera frame (stall detection)
        last_marker, stale_logged, drift = time.monotonic(), False, {}  # last marker reset; drift guard state
        stats = {"t": time.monotonic(), "frames": 0, "hand": 0, "markers": 0, "tracking": 0, "none": 0,
                 "wait": 0.0, "proc": 0.0}
        prev_now = None
        while True:
            if overlay is not None and overlay.poll() is not None:
                raise SystemExit("marker overlay died")
            t_read = time.monotonic()
            if prev_now is not None:
                stats["proc"] += t_read - prev_now  # time spent processing the previous frame
            ok, frame = src.read()
            if not ok:
                print("video source ended")
                break
            now = time.monotonic()
            stats["wait"] += now - t_read  # time blocked waiting for the next frame
            prev_now = now
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_marker_corners(gray)
            redetected = []
            if targets is None:
                centres = {i: c.mean(axis=0) for i, c in corners.items()}
                M_new = homography_from_markers(centres, W, H)
                if M_new is not None and screen_quad(M_new, W, H) is None:
                    M_new = None  # singular fit counts as a failure
                if M_new is not None:
                    M, last_ok, source = M_new, now, "markers"
                    tracker.reset(gray, M)
                else:  # markers missing: follow the screen; HOLD_SECONDS applies only if this fails too
                    M_t = tracker.update(gray, hand_poly)
                    if M_t is not None:
                        M, last_ok, source = M_t, now, "tracking"
                    else:
                        source = None
            else:  # overlay: guided re-detection of missed markers, then the trust rules choose M
                M_track = tracker.update(gray, hand_poly)
                held = M if M is not None and now - last_ok <= HOLD_SECONDS else None
                corners, redetected = refine_overlay_corners(
                    gray, corners, targets, overlay_guess(corners, targets, M_track, held))
                centres = {i: c.mean(axis=0) for i, c in corners.items()}
                M_new, source, reset = choose_overlay_M(corners, targets, M_track, W, H, gray.shape[1], drift)
                if reset:
                    last_marker, stale_logged = now, False
                M_g, source_g = tracking_age_guard(M_new, source, now, last_marker)
                if source is not None and source_g is None and not stale_logged:
                    debug_log(f"no marker fix for {now - last_marker:.1f} s: tracked M not trusted, "
                              "no touches until markers return")
                    stale_logged = True
                M_new, source = M_g, source_g
                if M_new is not None:
                    M, last_ok = M_new, now
                    if reset:
                        tracker.reset(gray, M)
                else:
                    source = None
            m_valid = M is not None and now - last_ok <= HOLD_SECONDS
            if not m_valid:
                smooth = None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_ts = max(int((now - t0) * 1000), last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ts)
            h, w = frame.shape[:2]
            lm, k = None, None
            if res.hand_landmarks:  # the pointing hand: nearest the last fingertip, else nearest the screen centre
                ref = prev_tip
                if ref is None:
                    ref = raw_map(np.linalg.inv(M), (W / 2, H / 2)) if M is not None else (w / 2, h / 2)
                on = None
                if M is not None and source is not None:  # fresh M: which hands point at the screen
                    on = [screen_point(raw_map(M, (hd[8].x * w, hd[8].y * h)), W, H)[1] for hd in res.hand_landmarks]
                k = pointing_hand(res.hand_landmarks, w, h, ref, on)
                lm = res.hand_landmarks[k]
            prev_tip = (lm[8].x * w, lm[8].y * h) if lm is not None else None
            if args.touch_debug:  # once a second: is the hand found, where M comes from, is the loop keeping up
                stats["frames"] += 1
                stats["hand"] += lm is not None
                stats[source or "none"] += 1
                if now - stats["t"] >= 1.0:
                    n = max(stats["frames"], 1)
                    debug_log(f"stats fps={stats['frames'] / (now - stats['t']):.1f} hand={stats['hand']}/{n} "
                              f"M markers={stats['markers']} tracking={stats['tracking']} none={stats['none']} "
                              f"proc={1000 * stats['proc'] / n:.0f}ms wait={1000 * stats['wait'] / n:.0f}ms")
                    stats.update(t=now, frames=0, hand=0, markers=0, tracking=0, none=0, wait=0.0, proc=0.0)
            hand_poly = hand_exclusion(lm, w, h) if lm is not None else None
            status, ev = ("no hand" if lm is None else "tracking"), None
            if args.touch:
                drain_taps(touch_sock, fuser)
                cap = capture_time(now, args.camera_latency)
                fuser.add_hands(cap, len(res.hand_landmarks))
                fuser.add_typing(cap, typing_flag(res.hand_landmarks, k, w, h, M, source is not None, H))
                # no camera frame for GESTURE_STALL while a button is held: release it before this frame
                acts = engine.release() if engine.held and now - last_frame_wall > GESTURE_STALL else []
                last_frame_wall = now
                if lm is not None:  # fingertip sample, unclamped; off-screen or unmapped counts as not on screen
                    # only an M from markers or the tracker this frame counts; the HOLD_SECONDS hold is stale
                    fresh = source is not None
                    tip = raw_map(M, (lm[8].x * w, lm[8].y * h)) if fresh else (0.0, 0.0)
                    tip, on = screen_point(tip, W, H)
                    on = on and fresh
                    acts += touch_step(engine, fuser, cap, tip, on, two_finger([(p.x * w, p.y * h) for p in lm]))
                else:
                    acts += touch_step(engine, fuser, cap, None, False, False)  # hand lost
                flash = perform_gestures(acts, args)
                if flash is not None:
                    touch_flash = (flash, time.monotonic() + 0.3)
                if args.touch_debug:
                    for d in fuser.decisions:
                        debug_log(format_decision(d))
            if lm is not None and m_valid:
                pts = [(p.x * w, p.y * h) for p in lm]
                if hover_moves_cursor(args):
                    target = np.array(map_point(M, pts[8], (W, H)))
                    smooth = target if smooth is None else SMOOTH_ALPHA * target + (1 - SMOOTH_ALPHA) * smooth
                    sx, sy = smooth
                    if args.dry_run:
                        print(f"cursor {sx:.0f},{sy:.0f}")
                    elif pinch.down:  # button held: send a drag event, do not press or release here
                        pyautogui.dragTo(sx, sy, duration=0, button="left", mouseDownUp=False)
                    else:
                        pyautogui.moveTo(sx, sy)
                ev = None if args.touch else pinch.update(pinch_ratio(pts))  # move happens before the press/release below
            else:
                ev = pinch.release()  # never press without a valid mapping; drop a held button

            if ev:
                if args.dry_run:
                    print(f"pinch {ev}")
                elif ev == "down":
                    pyautogui.mouseDown()
                else:
                    pyautogui.mouseUp()

            quad = screen_quad(M, W, H) if source is not None else None
            if quad is not None:  # screen quad: green from markers, yellow from tracking
                cv2.polylines(frame, [_poly(quad, frame.shape[:2])], True,
                              (0, 255, 0) if source == "markers" else (0, 255, 255), 2)
            if touch_flash is not None and time.monotonic() < touch_flash[1] and m_valid:
                cx, cy = raw_map(np.linalg.inv(M), touch_flash[0])  # screen point back into the camera image
                cv2.circle(frame, (int(cx), int(cy)), 14, (255, 0, 255), 3)
            draw_debug(frame, centres, lm, pinch.down, status, redetected, source)
            if view is not None:
                view.hub.put(frame)
            if not args.no_window:
                cv2.imshow("glasstouch", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except (KeyboardInterrupt, pyautogui.FailSafeException) as e:
        print(f"stopped: {type(e).__name__}")
    finally:
        try:
            release_touch(engine, args)  # a drag still held is released here
        finally:
            if touch_sock is not None:
                touch_sock.close()
            if overlay is not None:
                stop_overlay(overlay)
            if view is not None:
                view.close()
            src.release()
            landmarker.close()
            cv2.destroyAllWindows()


def marker_size(s):
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {s!r}")
    if n < 40:
        raise argparse.ArgumentTypeError(f"must be at least 40 points, got {n}")
    return n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Camera-based touch input for macOS.")
    p.add_argument("--source", help="webcam index (e.g. 0) or RTSP/HTTP URL")
    p.add_argument("--region", help="x,y,w,h of the Mac screen to grab with mss")
    p.add_argument("--dry-run", action="store_true", help="print events, do not move the mouse")
    p.add_argument("--make-markers", metavar="OUT.png", help="write printable marker sheet and exit")
    p.add_argument("--show-markers", action="store_true",
                   help="draw the markers in the screen corners instead of using printed ones")
    p.add_argument("--marker-size", type=marker_size, default=140, metavar="N",
                   help="overlay marker window size in points (default 140)")
    p.add_argument("--touch", action="store_true",
                   help="click where a sensor tap (imu_taps.py) meets a still fingertip on the screen; "
                        "hovering does not move the cursor and pinch clicking is off")
    p.add_argument("--touch-port", type=int, default=TOUCH_PORT, help="UDP port for imu_taps.py (default 9787)")
    p.add_argument("--touch-debug", action="store_true",
                   help="with --touch: print one line per tap (CLICK or the reason it was rejected)")
    p.add_argument("--camera-latency", type=float, default=0.3, metavar="S",
                   help="seconds camera frames lag real time (default 0.3)")
    p.add_argument("--tap-min-acc", type=float, default=TAP_MIN_ACC, metavar="G",
                   help="with --touch: reject taps whose accel change is below this, in g (default 0.003)")
    p.add_argument("--tap-max-acc", type=float, default=TAP_MAX_ACC, metavar="G",
                   help="with --touch: reject taps whose accel change is above this, a knock (default 0.08)")
    p.add_argument("--view-port", type=int, metavar="N",
                   help="serve the annotated debug frame as MJPEG on 127.0.0.1:N (the control panel shows it)")
    p.add_argument("--no-window", action="store_true",
                   help="no OpenCV window (the q key is then unavailable; stop with Ctrl-C)")
    p.add_argument("--overlay-child", type=int, help=argparse.SUPPRESS)  # internal: marker overlay process
    a = p.parse_args(argv)
    if not (a.make_markers or a.overlay_child is not None) and bool(a.source) == bool(a.region):
        p.error("give exactly one of --source or --region")
    if not 0 <= a.tap_min_acc < a.tap_max_acc:
        p.error("need 0 <= --tap-min-acc < --tap-max-acc")
    return a


def main(argv=None):
    a = parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGHUP):  # terminate or closed Terminal -> KeyboardInterrupt, so finally releases the mouse
        signal.signal(sig, signal.default_int_handler)
    if a.overlay_child is not None:
        overlay_child(a.overlay_child)
        return
    if a.make_markers:
        make_marker_sheet(a.make_markers)
        print(f"wrote {a.make_markers}")
        return
    run(a)


if __name__ == "__main__":
    main()
