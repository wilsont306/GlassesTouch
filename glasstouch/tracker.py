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
import http.client
import select
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
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

import maccam

HERE = os.path.dirname(os.path.abspath(__file__))
HOST_APP = os.environ.get("GLASSTOUCH_APP", "Terminal")  # the app macOS gives permissions to (app.py sets it)
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
REPROJ_PX = 6.0      # 2-3 marker fit: a marker whose corners sit this far (mean, camera px) from the fitted M is an outlier
JUMP_FRAC = 0.08     # marker M moving the screen quad this far (fraction of frame width) in a frame with <4 markers is suspect
PAIR_SIDES = ({0, 1}, {2, 3}, {0, 3}, {1, 2})  # marker pairs on one side of the screen (top, bottom, left, right)
CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]  # ids 0..3
HOLD_SECONDS = 0.5   # keep last homography this long after markers are lost
TRACK_FEATURES = 300    # ScreenTracker: features to follow inside the screen quad
TRACK_REDETECT = 100    # re-detect features when fewer than this survive
TRACK_MIN_INLIERS = 25  # fewer tracked inliers than this counts as a tracking failure
SMOOTH_ALPHA = 0.5
TOUCH_PORT = 9787       # imu_taps.py sends tap datagrams here
TOUCH_SETTLE = 0.13     # decide a tap this long after it (in capture time): its fingertip samples (up to
                        # TOUCH_WINDOW[1] after it) have arrived; frames carry real capture times, so no extra margin
TOUCH_WINDOW = (-0.12, 0.12)  # fingertip samples that decide a tap's point, seconds around the tap
TOUCH_SPAN_MAX = 250.0        # samples in the window spanning more than this: a sweeping hand, not a tap
TOUCH_TAP_AGE, TOUCH_SAMPLE_AGE = 2.0, 3.0  # drop undecided taps / old fingertip samples after this
TOUCH_MIN_GAP = 0.4           # accepted clicks at least this far apart (seconds): one knock, one click
TOUCH_DRAIN_MAX = 64          # tap datagrams read per camera frame
GESTURE_DECIDE = 0.25        # a press or two-finger touch-down is decided this long (capture time) after it
GESTURE_DOUBLE = 0.9         # (taps < 0.4 s apart are dropped as ringing) a touch-down this soon after the last click, and near it, is a press (double-click or drag)
GESTURE_DOUBLE_DIST = 150.0  # points: how near the last click a press must land (live 2nd taps landed 50-110 pt off)
GESTURE_MOVE = 30.0          # points the fingertip must move from the touch point to start a drag or scroll
GESTURE_LIFTOFF = 0.2        # a press that moves this soon after the 2nd tap is the finger lifting off (live: it leaves
                             # at 20-50 pt per frame at once; at 15 fps that crosses GESTURE_MOVE within ~0.2 s), so a
                             # double-click; a drag rests on the glass first, then slides
GESTURE_PRESS_WAIT = 0.6     # a still press waits this long for the slide before it is a double-click
GESTURE_HOLD = 0.15          # a tap whose finger stays put this long (within GESTURE_MOVE) is held: sliding starts a drag
                             # (user's choice, from 0.3 then 0.2; a slow lift-off can cross GESTURE_MOVE after ~0.2 s, see
                             # above, so some may now read as drags: watch for "drag" right after plain taps)
GESTURE_HOLD_WAIT = 1.5      # a held tap that hasn't slid by then is just its click
GESTURE_DRAG_GO = 15.0       # ... and still moving away by this much the next frame: a slide, not a one-off jump of the
                             # fingertip estimate (between a double-tap's taps it jumps 60-110 pt, then stays)
GESTURE_DRAG_MOVE = 60.0     # points a press must slide to start a drag (live: a hovering hand wobbled 25-40 pt after
                             # the 2nd tap and crossed GESTURE_MOVE)
GESTURE_LIFT_GROW = 0.12     # lift-off: the hand looks this much bigger (wrist-to-knuckle, on the screen plane) than at the
                             # start of the drag or scroll, for two frames: it came toward the eye, off the glass
GESTURE_PAUSE_SPEED = 225.0  # a drag frame slower than this (points/s, ~15 pt per frame) is a pause ...
GESTURE_RESUME = 25.0        # ... after which the finger must move this far (points) to say where it goes next:
GESTURE_TURN = 45.0          # within this many degrees of the drag's direction it carries on; turned further it was the
                             # pull-away after a drop (live: carrying on turned 1-18 deg, pull-aways 53-179, mostly >90)
GESTURE_LIFT_WINDOW = 0.45   # backup for a lift the size can't show: a real stop, average speed over this much time ...
GESTURE_LIFT_SPEED = 100.0   # ... below this (points/s). Was 0.15 s / 300 pt/s, which also dropped drags that only slowed
                             # down (live, "releases too early when my hand doesn't even release")
GESTURE_OFF = 0.6            # lift-off: off screen, hand lost or no frame for longer than this (bursty camera frames)
GESTURE_MAX = 10.0           # hard cap on one drag or scroll, so a button is never held for long
GESTURE_STALL = 0.5          # wall-clock seconds without a camera frame while a button is held: release it
GESTURE_SMOOTH = 0.6         # drag position: weight of the newest fingertip sample
GESTURE_HIST = 3.0           # fingertip frames kept so a late-accepted touch-down can be replayed
SCROLL_GAIN = 0.1            # finger points -> scroll units (about 10 pt per unit)
SCROLL_DEADBAND = 3.0        # scrolling ignores finger motion below this many points (jitter)
SCROLL_SIDEWAYS = 2.0        # a scroll is sideways only if its first stroke is this many times wider than tall (live: the
                             # fingertip jumps 30-150 pt between frames, and "wider than tall" locked 16 of 21 scrolls sideways)
SCROLL_LIFT_GROW = 0.5       # a scroll ends on hand size only past this growth (live: real lifts grow < 8 %; scrolls were
                             # cut after 1-8 steps at 13-48 % growth, size noise as the hand sweeps the screen)
MARKER_SHOW_AFTER = 0.5      # on-demand markers: seconds without a marker or content fix before they fade in
MARKER_PROBE_EVERY = 0.5     # ... while shown, seconds between checks that content alone finds the screen
MARKER_HIDE_AFTER = 6        # ... checks in a row (3 s) that do, before they fade out again
VIEW_MAX_W = 540             # --view-port: frames are at most this wide ...
VIEW_FPS = 15                # ... and at most this many a second (shared by all clients)
TAP_FEATURES = {"gyro_peak": "g", "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz", "acc_peak": "a",
                "acc_x": "ax", "acc_y": "ay", "acc_z": "az", "lid_delta": "lid"}  # imu_taps.py fields -> debug names
TAP_MIN_ACC, TAP_MAX_ACC = 0.003, 0.08  # accel change (g) of a screen tap: below min is weak, above max a knock
TAP_MIN_ACC_SECOND = 0.002  # a tap 0.4-0.9 s after a click (the 2nd of a double-tap) lands lighter: live 0.002 g
TAP_MAX_GYRO = 3.0    # rad/s: a bigger gyro peak is a knock on the case, not a tap on the screen
TYPING_BELOW = 150.0  # points: a second hand whose index tip is this far under the screen is over the keyboard
TYPING_MOVE = 8.0     # camera px a fingertip of that hand must move between frames for it to count as typing (live:
                      # once the hand crop made hands larger, a hand resting on the palm rest was found in every
                      # frame and every tap was rejected as typing)
FINGERTIPS = (8, 12, 16, 20)  # index, middle, ring, little (MediaPipe landmark ids)
MIN_HAND_PT = 200.0  # wrist to middle knuckle, in screen points: a real hand in front of the screen spans ~450+;
                     # a hand shown ON the screen (the tutorial's 3D hand, live) spans under 100, so it is ignored


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


def reproj_errors(corners_by_id, targets, M):
    """{id: mean corner distance (camera px)} between each marker's seen corners and where M puts its screen quad."""
    G = np.linalg.inv(M)
    return {i: float(np.linalg.norm(cv2.perspectiveTransform(np.float32([targets[i]]), G)[0] - c, axis=1).mean())
            for i, c in corners_by_id.items()}


def fit_markers(corners, targets, W, H):
    """Marker homography from the seen overlay markers, with a consistency guard for 2-3 of them: the worst marker
    more than REPROJ_PX off the fit is dropped and the fit redone; a 2-marker fit that is off is rejected.
    Returns (M or None, ids used)."""
    ids = [i for i in targets if i in corners]
    while True:
        sub = {i: corners[i] for i in ids}
        M = homography_from_corners(sub, targets) if len(ids) >= 2 else None
        if M is None or screen_quad(M, W, H) is None:
            return None, ids
        if len(ids) > 3:  # 4 markers: RANSAC over all corners is the check
            return M, ids
        err = reproj_errors(sub, targets, M)
        worst = max(err, key=err.get)
        if err[worst] <= REPROJ_PX:
            return M, ids
        if len(ids) == 2:
            return None, ids
        ids.remove(worst)


def _grow(pts, r):
    """Convex hull of pts (N, 2) grown by r px: the hull of disc samples around every point."""
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    grown = (pts[:, None, :] + r * np.stack([np.cos(ang), np.sin(ang)], 1)[None]).reshape(-1, 2)
    return cv2.convexHull(grown.astype(np.float32)).reshape(-1, 2)


def drop_occluded(corners_by_id, hand_poly, pad):
    """Split overlay markers by whether the hand covers them: (kept, dropped), each {id: (4, 2) corners}. A marker is
    covered when its quad meets the previous hand polygon grown by pad camera px (about one marker side)."""
    if hand_poly is None or not corners_by_id:
        return dict(corners_by_id), {}
    hand = _grow(np.float32(hand_poly), pad)
    kept, dropped = {}, {}
    for i, c in corners_by_id.items():
        area, _ = cv2.intersectConvexConvex(cv2.convexHull(np.float32(c)), hand)
        (dropped if area > 0 else kept)[i] = c
    return kept, dropped


def _side(corners):
    """Mean marker side (camera px) over the corners given; 0 for none."""
    return float(np.mean([np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean() for c in corners.values()])) \
        if corners else 0.0


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


def overlay_fix(gray, corners, targets, M_track, held, hand_poly, W, H, drift, prev_M):
    """One frame of the overlay path: drop markers the hand covers, guided re-detection of the missing ones, drop
    the covered ones again (a re-detection can be damaged too), then the trust rules. Returns
    (M, source, reset, corners, redetected, dropped): corners are the markers used, dropped is {id: corners} ignored."""
    corners, dropped = drop_occluded(corners, hand_poly, _side(corners))
    corners, redetected = refine_overlay_corners(gray, corners, targets, overlay_guess(corners, targets, M_track, held))
    corners, late = drop_occluded(corners, hand_poly, _side(corners))
    dropped = {**dropped, **late}
    redetected = [i for i in redetected if i in corners]
    M, source, reset = choose_overlay_M(corners, targets, M_track, W, H, gray.shape[1], drift, prev_M)
    return M, source, reset, corners, redetected, dropped


def choose_overlay_M(corners, targets, M_track, W, H, frame_w, drift=None, prev_M=None):
    """Trust rules for the overlay homography from the markers seen (corners: {id: (4, 2)}, direct or re-detected).
    M_track: the ScreenTracker's current M, or None. Returns (M, source, reset): source is 'markers', 'tracking'
    or None; reset says whether the tracker should restart from M. drift: a dict the caller keeps across frames,
    for the drift guard (see _drift_vote); None disables it. prev_M: last frame's M, for the one-frame jump rule."""
    M_pair, seen = fit_markers(corners, targets, W, H)
    if M_pair is None:
        if drift is not None:
            drift.clear()
        return (M_track, "tracking", False) if M_track is not None else (None, None, False)
    if prev_M is not None and M_track is not None and len(seen) < 4:
        # a marker fit that jumps the screen while the tracker holds still is suspect: keep the tracker this frame,
        # and let _drift_vote count it so a persistent change still wins after DRIFT_FRAMES
        qa, qp, qt = screen_quad(M_pair, W, H), screen_quad(prev_M, W, H), screen_quad(M_track, W, H)
        if qp is not None and qt is not None and np.linalg.norm(qa - qp, axis=1).mean() > JUMP_FRAC * frame_w \
                and np.linalg.norm(qt - qp, axis=1).mean() <= TRUST_FRAC * frame_w:
            if drift is not None and _drift_vote(drift, qa, False, frame_w):
                drift.clear()
                return M_pair, "markers", True
            return M_track, "tracking", False
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
    tips within 0.6 hand sizes of each other. pts: 21 landmarks, (x, y) or (x, y, z). run() passes MediaPipe's 3D
    world landmarks: 2D pixel distances misread a one-finger point as two fingers when the glasses look down."""
    return pose_is_two(*finger_extension(pts))


def pose_is_two(ext, gap):
    """The two-finger decision from finger_extension()'s (ext, gap)."""
    # live: in 3D, "tip closer to the wrist than the PIP" demanded fully curled ring/pinky and never fired for a
    # relaxed two-finger pose; extension ratios allow half-bent ring and pinky
    fold = max(ext[2], ext[3])  # ring and pinky: below the middle finger, never as straight as it
    return bool(ext[0] > TWO_INDEX and ext[1] > TWO_MIDDLE and gap < TWO_GAP
                and fold < TWO_FOLD and fold < ext[1] - TWO_MARGIN)


# fitted to a live session (wrist-to-tip / wrist-to-knuckle; gap in hand sizes): pointing = middle 1.0-1.4,
# gap 0.5-0.9; two fingers = index 1.45-1.75, middle 1.45-1.95, ring/pinky 1.15-1.55, gap 0.14-0.32.
# This rule: 22/22 two-finger and 0/41 pointing seconds
TWO_INDEX, TWO_MIDDLE, TWO_GAP, TWO_FOLD, TWO_MARGIN = 1.35, 1.45, 0.4, 1.6, 0.08


def finger_extension(pts):
    """Per finger (index, middle, ring, pinky): wrist-to-tip over wrist-to-knuckle (MCP) distance, plus the
    index-middle tip gap in hand sizes (wrist to middle knuckle). pts: 21 landmarks, 2D or 3D."""
    def dist(a, b):
        return float(np.linalg.norm(np.subtract(pts[a], pts[b])))
    ext = [dist(0, tip) / max(dist(0, mcp), 1e-9) for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20))]
    return ext, dist(8, 12) / max(dist(0, 9), 1e-9)


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


class MjpegSource:
    """The phone's MJPEG stream read directly (not via FFmpeg) so each frame's X-Capture-Time header is kept.
    stamp: the last frame's capture time in Unix seconds, or None when the stream sends none."""

    def __init__(self, url, timeout=3.0):
        u = urllib.parse.urlsplit(url)
        conn = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        self.conn = conn(u.hostname, u.port, timeout=timeout)  # connect and every read: a hung stream fails in 3 s
        try:
            self.conn.request("GET", u.path or "/")
            self.resp = self.conn.getresponse()
            ctype = self.resp.getheader("Content-Type", "")
            if self.resp.status != 200 or "multipart" not in ctype:
                raise OSError(f"not an MJPEG stream ({self.resp.status} {ctype})")
        except BaseException:
            self.conn.close()
            raise
        self.stamp = None
        self.exact = False  # stamp is the true capture time (X-Capture-Source: phone-camera), no link delay in it

    def isOpened(self):
        return True

    def read(self):
        try:
            headers = {}
            while True:  # boundary line, then "Name: value" lines up to the blank line before the JPEG
                line = self.resp.readline(1024)
                if not line:
                    return False, None
                line = line.strip()
                if not line:
                    if "content-length" in headers:
                        break
                    continue
                if b":" in line:
                    k, v = line.split(b":", 1)
                    headers[k.strip().lower().decode("latin-1")] = v.strip().decode("latin-1")
            n = int(headers["content-length"])
            if not 0 < n <= 20_000_000:
                return False, None
            data = self.resp.read(n)
            if len(data) < n:
                return False, None
        except (OSError, ValueError, http.client.HTTPException):
            return False, None
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        try:
            self.stamp = float(headers["x-capture-time"])
        except (KeyError, ValueError):
            self.stamp = None
        # the phone's own camera stamps the real capture time; the glasses' stamps still hold the link's delay
        self.exact = headers.get("x-capture-source") == "phone-camera"
        return True, frame

    def release(self):
        self.conn.close()


def open_source(args):
    if args.region:
        x, y, w, h = map(int, args.region.split(","))
        return RegionGrabber(x, y, w, h)
    if args.source.isdigit():  # a camera on this Mac (the panel's "camera pointed at the screen")
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():  # macOS asks for Camera permission the first time and fails this try
            raise SystemExit(f"cannot open camera {args.source}: allow {HOST_APP} under System Settings > Privacy & "
                             "Security > Camera (macOS asks the first time), check the camera is on (an iPhone: "
                             "nearby, same Apple ID), then start again.")
        cams = maccam.devices()  # OpenCV opened it at ~480x360 (preset Medium): switch it to its best format
        if int(args.source) < len(cams):
            size = maccam.use_best_format(cams[int(args.source)])
            print(f"camera {args.source}: {cams[int(args.source)].localizedName()}, format "
                  + (f"{size[0]}x{size[1]}" if size else "unchanged (no format at 24 fps or more)"), flush=True)
    else:
        try:  # the phone's stream: read it ourselves to keep each frame's capture time
            return MjpegSource(args.source)
        except (OSError, http.client.HTTPException):
            pass  # not multipart MJPEG (or not reachable): let FFmpeg try
        # network stream: fail a hung read after 3 s (default 30 s), so a held drag can't stay down that long
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
                                 running_mode=RunningMode.VIDEO, num_hands=3)  # 3: live, a hand shown on the screen (the
    # tutorial's) plus a hand resting on the keyboard held both slots of 2, and the pointing hand wasn't found again
    return HandLandmarker.create_from_options(opts)


def draw_debug(frame, centres, lm, pinched, status, redetected=(), m_source=None, dropped=None):
    """Markers in green (decoded directly) or yellow (guided re-detection), grey when the hand covers them and
    they are ignored (dropped: {id: centre}); line 2 says which ids were used and whether M came from the markers
    or from tracking."""
    h, w = frame.shape[:2]
    for i, (cx, cy) in centres.items():
        col = (0, 255, 255) if i in redetected else (0, 255, 0)
        cv2.circle(frame, (int(cx), int(cy)), 8, col, -1)
        cv2.putText(frame, str(i), (int(cx) + 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for i, (cx, cy) in (dropped or {}).items():
        cv2.circle(frame, (int(cx), int(cy)), 8, (160, 160, 160), -1)
        cv2.putText(frame, f"{i} hand", (int(cx) + 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)
    if lm:
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (255, 0, 0), -1)
    cv2.putText(frame, f"{status} pinch={'DOWN' if pinched else 'up'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    direct = sorted(i for i in centres if i not in redetected)
    cv2.putText(frame, f"ids direct {direct} re-detected {sorted(redetected)} M from {m_source or 'none'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if dropped:
        cv2.putText(frame, f"ignored, covered by hand: {sorted(dropped)}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)


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


def overlay_child(size, hidden=False):
    """Child process: one borderless always-on-top window per marker: the main screen's corners and side edges.
    Prints each marker's four black-square corners (pyautogui points, top-left origin), then runs the AppKit loop.
    hidden: the windows start invisible; "show" / "hide" lines on stdin fade them in and out (on-demand markers)."""
    import AppKit
    lo, hi = marker_extent()
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # no Dock icon
    sf = AppKit.NSScreen.screens()[0].frame()  # menu-bar display: matches pyautogui's origin and size
    sx, sy, sw, sh = sf.origin.x, sf.origin.y, sf.size.width, sf.size.height
    # Cocoa origin is bottom-left; corners by marker id: TL, TR, BR, BL, then the middle of the left and right edges,
    # so most views of part of the screen still hold the 2 markers a fit needs (none mid-top/bottom: notch, Dock)
    corners = {0: (sx, sy + sh - size), 1: (sx + sw - size, sy + sh - size),
               2: (sx + sw - size, sy), 3: (sx, sy),
               4: (sx, sy + (sh - size) / 2), 5: (sx + sw - size, sy + (sh - size) / 2)}
    windows, quads = [], {}
    for i, (x, y) in corners.items():
        png = marker_png(i)
        image = AppKit.NSImage.alloc().initWithData_(AppKit.NSData.dataWithBytes_length_(png, len(png)))
        # a non-activating panel at screen-saver level: live, plain NSWindows vanished behind other apps'
        # full-screen spaces; panels may join them (FullScreenAuxiliary) and don't hide when we're inactive
        win = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(x, y, size, size),
            AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered, False)
        win.setFloatingPanel_(True)
        win.setHidesOnDeactivate_(False)
        win.setLevel_(AppKit.NSScreenSaverWindowLevel)
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
        if hidden:
            win.setAlphaValue_(0.0)  # in place but invisible (it already lets clicks through)
        win.orderFrontRegardless()
        f = win.frame()  # macOS may have moved the window (menu bar, notch), so read the real frame
        fx, fy, fw, fh = f.origin.x, sh - (f.origin.y + f.size.height), f.size.width, f.size.height  # top-left origin
        # black-square corners in TL, TR, BR, BL order, matching cv2.aruco corner order
        quads[str(i)] = [[fx + u * fw, fy + v * fh] for u, v in
                         ((lo, lo), (hi, lo), (hi, hi), (lo, hi))]
        windows.append((win, view))  # keep references alive for the run loop
    print(json.dumps(quads), flush=True)

    def fade(alpha):  # AppKit work belongs on the main thread
        def run():
            for win, _ in windows:
                win.animator().setAlphaValue_(alpha)  # a short fade, not a flash
        AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(run)

    def follow_parent():  # "show" / "hide" from the parent; EOF on our stdin pipe means the parent is gone
        for line in sys.stdin:
            if line.strip() in ("show", "hide"):
                fade(1.0 if line.strip() == "show" else 0.0)
        os._exit(0)
    threading.Thread(target=follow_parent, daemon=True).start()
    app.run()


def start_overlay(size=140, hidden=False):
    """Spawn the overlay child; return (proc, {id: (4, 2) corners}) with the markers' on-screen corners.
    hidden: the markers start invisible (on-demand markers; overlay_set shows them)."""
    proc = subprocess.Popen([sys.executable, __file__, "--overlay-child", str(size)] + (["--overlay-hidden"] if hidden else []),
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


def overlay_set(proc, shown):
    """Fade the overlay's markers in (shown) or out. A dead child is caught by the run loop's poll."""
    try:
        proc.stdin.write("show\n" if shown else "hide\n")
        proc.stdin.flush()
    except (BrokenPipeError, ValueError):
        pass


class MarkerToggle:
    """On-demand markers: hidden while the screen is found from what it shows; shown once it has gone unfound (no
    marker or content fix) for MARKER_SHOW_AFTER s; hidden again after MARKER_HIDE_AFTER checks in a row,
    MARKER_PROBE_EVERY s apart, find it from content alone. The matcher skips the marker windows, so the markers
    themselves can't pass that check (which would hide them, lose the screen and show them again, in a loop)."""

    def __init__(self, now):
        self.shown, self.last_fix, self.t_probe, self.probes = False, now, -1e9, 0

    def step(self, now, source, probe):
        """source: this frame's fix ("markers", "content", "tracking" or None). probe(): True if content alone finds
        the screen in this frame (called at most every MARKER_PROBE_EVERY s, while shown). Returns "show", "hide"
        or None."""
        if not self.shown:
            if source in ("markers", "content"):
                self.last_fix = now
            elif now - self.last_fix > MARKER_SHOW_AFTER:
                self.shown, self.probes = True, 0
                return "show"
            return None
        if now - self.t_probe < MARKER_PROBE_EVERY:
            return None
        self.t_probe = now
        found = source == "content" or (source == "markers" and probe())
        self.probes = self.probes + 1 if found else 0
        if self.probes >= MARKER_HIDE_AFTER:
            self.shown, self.last_fix, self.probes = False, now, 0
            return "hide"
        return None


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


def accessibility_trusted(prompt=False):
    """True if macOS lets this process post mouse events (pyautogui fails silently otherwise). prompt: when it doesn't,
    macOS adds the app this runs in to the Accessibility list and shows its dialog (the only sure way to list
    GlassesTouch.app, whose program is really Homebrew's Python)."""
    import Foundation
    import objc
    fns = {}
    objc.loadBundleFunctions(Foundation.NSBundle.bundleWithPath_("/System/Library/Frameworks/ApplicationServices.framework"),
                             fns, [("AXIsProcessTrustedWithOptions", b"Z@")])
    return bool(fns["AXIsProcessTrustedWithOptions"]({"AXTrustedCheckOptionPrompt": prompt}))


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
    return _grow(np.float32([(p.x * w, p.y * h) for p in lm]), r)


def pointing_hand(hands, w, h, ref, on_screen=None):
    """Index in MediaPipe's hand list (non-empty) of the pointing hand: among the hands whose index tip is on the
    screen (on_screen, a list of bools; all hands if none is), the one whose tip is nearest ref (camera pixels).
    Preferring on-screen tips stops it sticking to a hand that moved to the keyboard."""
    pick = [i for i in range(len(hands)) if on_screen and on_screen[i]] or range(len(hands))
    return min(pick, key=lambda i: math.hypot(hands[i][8].x * w - ref[0], hands[i][8].y * h - ref[1]))


def real_hands(hands, M, w, h, min_pt=MIN_HAND_PT):
    """Indexes of the hands big enough to be real, measured on the screen plane through M (camera px -> points):
    a picture of a hand on the screen (the tutorial's 3D hand) also looks like a hand to MediaPipe. Without M all count."""
    if M is None:
        return list(range(len(hands)))
    return [i for i, hd in enumerate(hands)
            if math.dist(raw_map(M, (hd[0].x * w, hd[0].y * h)), raw_map(M, (hd[9].x * w, hd[9].y * h))) >= min_pt]


def other_hand_tips(hands, pointing, w, h):
    """Camera-pixel fingertips (FINGERTIPS) of the hand that is not the pointing one, or None with one hand."""
    others = [hd for i, hd in enumerate(hands) if i != pointing]
    return np.float32([(others[0][j].x * w, others[0][j].y * h) for j in FINGERTIPS]) if others else None


def typing_flag(hands, pointing, w, h, M, fresh, H, prev_tips=None, below=TYPING_BELOW, move=TYPING_MOVE):
    """True when a hand other than the pointing one (index in `hands`) is typing: its index tip, mapped through M, is
    more than `below` points under the screen (height H), and one of its fingertips moved more than `move` camera px
    since the previous frame (prev_tips from other_hand_tips; None, the hand just appeared, counts as moving). A
    hand resting on the keyboard or palm rest is not typing. No fresh M to map it with also counts as typing."""
    tips = other_hand_tips(hands, pointing, w, h)
    if tips is None:
        return False
    if not fresh or M is None:
        return True
    if raw_map(M, tuple(tips[0]))[1] <= H + below:
        return False
    return prev_tips is None or float(np.linalg.norm(tips - prev_tips, axis=1).max()) > move


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
            elif acc is not None and acc < self.min_acc and not (  # a double-tap's 2nd tap is lighter: allow it
                    self.last_click is not None and tap - self.last_click <= GESTURE_DOUBLE and acc >= TAP_MIN_ACC_SECOND):
                outcome = f"weak acc={acc:.4f}"
            elif acc is not None and (acc > self.max_acc or feats.get("gyro_peak", 0.0) > TAP_MAX_GYRO):
                outcome = f"knock acc={acc:.3f} gyro={feats.get('gyro_peak', 0.0):.1f}"
            # (no typing veto: live, the keyboard hand moved as much resting as typing, and the veto rejected 41 real screen
            # taps in one session; real typing puts the pointing finger on the keyboard, so "off-screen" rejects it.
            # The typing flag stays in the log record.)
            elif not near:
                outcome = "no-samples"
            elif not min(near, key=lambda s: abs(s[0] - tap))[3]:  # the nearest sample is off screen (keyboard)
                outcome = f"off-screen n_off={sum(not s[3] for s in near)}/{len(near)}"
            else:
                xy = np.float32([(x, y) for _, x, y, on in near if on])  # off-screen samples are (0, 0) placeholders
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
    A one-finger touch-down (tap time from TouchFuser) is a click at once, and nothing after it can be a drag. Two
    fingers (majority vote of two_finger over the frames after the touch-down) that move more than GESTURE_MOVE scroll
    at once; two fingers still for GESTURE_DECIDE are a click, and the touch ends there. A double-tap press
    that moves is always a drag, whatever the two-finger check says. A touch-down within
    GESTURE_DOUBLE s and GESTURE_DOUBLE_DIST of the last click is a press, decided within GESTURE_DECIDE: moved more
    than GESTURE_MOVE it is a drag (down at the touch point, moves, up on lift-off); still, it is a second click.
    A drag or a scroll ends at its first pause (average speed below GESTURE_LIFT_SPEED over GESTURE_LIFT_WINDOW: the
    camera can't see the finger leave the glass, so the pull-away would keep dragging or scroll back), when it has
    been off screen or absent for GESTURE_OFF, or at GESTURE_MAX. A scroll or a two-finger press also ends when the
    two-finger pose is lost, or the fingertip is off screen, for two frames (about 0.2 s). Frames are kept for GESTURE_HIST seconds so that a touch-down accepted
    late is replayed from its tap time.
    Actions: ("click"|"down"|"move"|"up"|"point", x, y) and ("scroll", dx, dy) in pyautogui units."""

    def __init__(self):
        self.hist = deque()     # (t, x, y, on_screen, two_finger, hand span), for replaying a late touch-down
        self.span0, self.grow_n = None, 0  # a drag's or scroll's hand size at its start; frames it looked bigger
        self.notes = []          # debug lines: why each drag or scroll ended, the hand-size ratio per drag frame
        self.state = None       # None (idle), "decide" (pending press or two fingers), "press" (two fingers clicked,
                                # still touching), "drag" or "scroll"
        self.start = None       # (t, x, y) of the touch-down
        self.double = False     # the touch-down is a press: near the last click in time and place
        self.last_click = None  # (t, x, y) of the last click's touch-down
        self.pos = None         # drag: smoothed fingertip; scroll: last fingertip that counted
        self.track = deque()    # (t, x, y) on-screen fingertip samples of the drag, for the speed rule
        self.votes = []         # two_finger flags of the on-screen frames since the touch-down
        self.offset = (0.0, 0.0)  # a drag: where it grabbed minus where the 2nd touch landed
        self.scroll_dir = (0.0, 1.0)  # a scroll's axis and direction, from its first stroke
        self.paused, self.dhist = None, []  # a drag's pause (spot, direction) and its recent points
        self.off_n = 0          # consecutive frames off screen or without a hand
        self.off_t = None       # capture time of the first of them
        self.last_t = None      # capture time of the last frame
        self.lost_n = 0         # consecutive frames without the two-finger pose on screen (press and scroll)
        self.acc = [0.0, 0.0]   # scroll units not yet sent
        self.t_gesture = None   # capture time the drag or scroll started (GESTURE_MAX counts from here)

    @property
    def held(self):
        """True while a drag holds the button down."""
        return self.state == "drag"

    def frame(self, t, x, y, on, two, span=None):
        self.hist.append((t, x, y, on, two, span))
        while self.hist and t - self.hist[0][0] > GESTURE_HIST:
            self.hist.popleft()
        self.last_t = t
        return self._step(t, x, y, on, two, span) if self.state else []

    def touch_down(self, t, x, y):
        if self.state in ("drag", "scroll"):  # live: sliding or pressing on the lid fires the tap sensor 0.1-0.35 s
            self.notes.append(f"tap ignored: {self.state} held")  # into a drag, which ended it (and could double-click)
            return []
        acts = [self._click(*self.start)] if self.state == "decide" else []  # a pending touch that never moved
        acts += self.release()  # a new touch-down ends the current gesture first
        replay = [f for f in self.hist if f[0] > t]  # frames that arrived after the tap
        votes = [f[4] for f in replay if f[3]]
        two = 2 * sum(votes) > len(votes)
        near = (self.last_click is not None and t - self.last_click[0] <= GESTURE_DOUBLE
                and math.hypot(x - self.last_click[1], y - self.last_click[2]) <= GESTURE_DOUBLE_DIST)
        if near and not two:  # a second tap soon and near: a double-click, at once, at the first tap's spot (macOS
            # needs both clicks at nearly the same point; the 2nd touch lands up to ~110 pt away)
            _, lx, ly = self.last_click
            self.last_click = None  # a third tap starts afresh
            return acts + [("dclick", lx, ly)]
        # one finger: the click is now; if the finger then holds still and slides, it becomes a drag ("hold")
        # two fingers: decide whether they scroll
        if not two:
            acts.append(self._click(t, x, y))
        self.state, self.start, self.double = ("decide" if two else "hold"), (t, x, y), False
        self.pos, self.votes, self.off_n, self.off_t, self.acc = (x, y), [], 0, None, [0.0, 0.0]
        self.offset, self.track, self.lost_n, self.hold_prev = (0.0, 0.0), deque(), 0, None
        for f in replay:
            acts += self._step(*f)
        return acts

    def release(self, why="ended"):
        """End the gesture without a new frame. A held drag sends its up; a pending or pressed touch is dropped."""
        if self.state == "drag" and self.paused is not None:  # released during a pause: drop where the finger paused
            self.pos = self.paused[0]
        acts = [("up", *self.pos)] if self.state == "drag" else []
        if self.state in ("drag", "scroll"):
            self.notes.append(f"{self.state} released: {why}")
        self.state, self.span0, self.grow_n = None, None, 0
        return acts

    def tick(self, t):
        """Lift-off checks with no new frame (a camera stall): GESTURE_OFF since the last frame, or the cap."""
        if self.state in ("drag", "scroll") and self.last_t is not None and t - self.last_t > GESTURE_OFF:
            return self.release("no frames")
        return self._cap(t)

    def _cap(self, t):
        return self.release("time cap") if self.state in ("drag", "scroll") and t - self.t_gesture >= GESTURE_MAX else []

    def _click(self, t, x, y):
        self.last_click = (t, x, y)
        return ("click", x, y)

    def _step(self, t, x, y, on, two, span=None):
        if self.state is None:  # a replayed frame after the gesture has ended changes nothing
            return []
        if self.state in ("drag", "scroll"):
            return self._track_frame(t, x, y, on, two, span)
        t0, px, py = self.start
        if self.state == "hold":  # a clicked tap: still for GESTURE_HOLD, then a slide, is a drag from the tap point
            if not on:
                self.lost_n += 1
                if self.lost_n >= 2:  # the finger left the screen: just the click
                    self.state = None
                return []
            self.lost_n = 0
            moved = math.hypot(x - px, y - py)
            if t - t0 < GESTURE_HOLD:
                if moved > GESTURE_MOVE:  # moved before it was held: the finger lifting off, the click stands
                    self.state = None
                self.hold_prev = moved
                return []
            prev, self.hold_prev = self.hold_prev, moved
            if prev is not None and prev > GESTURE_DRAG_MOVE and moved > prev + GESTURE_DRAG_GO:  # past 60 pt and
                # still sliding away a frame later
                self.state, self.t_gesture, self.pos = "drag", t, (px, py)
                self.last_click = None  # the tap that became a drag isn't half of a double-click
                self.paused, self.dhist = None, [(t0, px, py)]  # pause-then-turn state; the drag's recent points
                return [("down", px, py)] + self._track_frame(t, x, y, on, two, span)
            if t - t0 > GESTURE_HOLD_WAIT:
                self.state = None
            return []
        if on:
            self.votes.append(two)
        two_f = 2 * sum(self.votes) > len(self.votes)  # two fingers by majority of the on-screen frames
        if self.state == "press":
            self.lost_n = 0 if on and two else self.lost_n + 1
            if self.lost_n >= 2:
                self.state = None
                return []
        elif t - t0 >= (GESTURE_PRESS_WAIT if self.double else GESTURE_DECIDE):  # still for the whole window: a click, and the touch is over (keeping a
            # two-finger touch open made the hand's withdrawal scroll, live, when the pose check misread one finger)
            self.state, self.lost_n = None, 0
            if self.double:  # a still double-tap: a real double-click at the first tap's spot (macOS needs both
                # clicks at nearly the same point, quickly; the 2nd touch lands up to 100 pt away and 0.25 s late)
                _, lx, ly = self.last_click
                self.last_click = None  # a third tap starts afresh
                return [("dclick", lx, ly)]
            return [self._click(t0, px, py)]
        moved = math.hypot(x - px, y - py) if on else 0.0
        if self.double and t - t0 < GESTURE_LIFTOFF and moved > GESTURE_MOVE:  # moved away at once: lift-off after a
            self.state, self.lost_n = None, 0                                    # double-tap, not a drag
            _, lx, ly = self.last_click
            self.last_click = None
            return [("dclick", lx, ly)]
        if moved <= (GESTURE_DRAG_MOVE if self.double else GESTURE_MOVE):
            return []
        if two_f and not self.double:  # two fingers: scroll at once (a double-tap press is always a drag: live, the
            # two-finger check said yes during one-finger drags and turned every drag into a scroll)
            self.state, self.t_gesture, self.lost_n = "scroll", t, 0
            ddx, ddy = x - px, y - py  # the first stroke sets the scroll's axis and direction (scroll_dir)
            sideways = abs(ddx) > SCROLL_SIDEWAYS * abs(ddy)  # vertical unless clearly sideways
            self.scroll_dir = (math.copysign(1.0, ddx), 0.0) if sideways else (0.0, math.copysign(1.0, ddy))
            self.notes.append(f"scroll start stroke=({ddx:.0f},{ddy:.0f}) {'sideways' if sideways else 'vertical'}")
            return [("point", px, py)] + self._track_frame(t, x, y, on, two, span)
        if self.state == "press":  # a tap that was clicked never becomes a drag
            return []
        if self.double:  # a press that rests, then slides: grab at the FIRST tap's spot (the 2nd touch lands 60-110 pt
            # off, live, and missed what was double-tapped) and follow the finger's movement from there
            _, lx, ly = self.last_click
            self.offset = (lx - px, ly - py)
            self.state, self.t_gesture, self.pos, self.last_click = "drag", t, (lx, ly), None
            return [("down", lx, ly)] + self._track_frame(t, x, y, on, two, span)
        self.state = None  # one finger that moved inside the window: its click, and nothing else
        return [self._click(t0, px, py)]

    def _track_frame(self, t, x, y, on, two, span=None):
        if self.state == "scroll":  # the two-finger pose lost for two frames: the scroll is over
            self.lost_n = 0 if on and two else self.lost_n + 1
            if self.lost_n >= 2:
                return self.release("two-finger pose lost")
        if not on:
            self.off_n += 1
            if self.off_n == 1:
                self.off_t = t
            if self.off_n >= 2 and t - self.off_t > GESTURE_OFF:  # one dropped frame is a burst, not a lift
                return self.release("off screen")
            return self._cap(t)
        if span:  # lift-off: the hand came toward the eye (looks bigger) for two frames; end before this frame moves
            if self.span0 is None:
                self.span0 = span
            ratio = span / self.span0
            self.notes.append(f"{self.state} size={ratio:.2f}")
            if ratio > 1 + (SCROLL_LIFT_GROW if self.state == "scroll" else GESTURE_LIFT_GROW):
                self.grow_n += 1
                if self.grow_n >= 2:
                    return self.release(f"hand grew x{ratio:.2f}")
                return self._cap(t)  # maybe lifting: hold this frame's movement back until the next frame says
            else:
                self.grow_n = 0
                self.span0 = 0.9 * self.span0 + 0.1 * span  # follow slow changes (the hand turning as it drags)
        if self.state == "drag":  # the finger's movement, carried over to where the drag grabbed
            x, y = x + self.offset[0], y + self.offset[1]
        if self.off_n:  # back on screen: continue from here, no leap across the gap
            self.off_n = 0
            self.pos = (x, y)
        acts = []
        if self.state == "drag":  # pause then turn: the pull-away after a drop, not more dragging
            prev = self.dhist[-1]
            if self.paused is None:
                if t > prev[0] and math.dist(prev[1:], (x, y)) / (t - prev[0]) < GESTURE_PAUSE_SPEED and len(self.dhist) >= 3:
                    back = self.dhist[-3]
                    self.paused = ((x, y), (x - back[1], y - back[2]))  # where it paused, the way it was going
                self.dhist = (self.dhist + [(t, x, y)])[-6:]
            if self.paused is not None:
                (qx, qy), (bx, by) = self.paused
                ax, ay = x - qx, y - qy
                if math.hypot(ax, ay) < GESTURE_RESUME:  # still at the pause: hold the movement back
                    return self._stop_check(t, x, y, acts)
                cos = (ax * bx + ay * by) / (math.hypot(ax, ay) * math.hypot(bx, by) or 1.0)
                if math.hypot(bx, by) >= 20 and cos < math.cos(math.radians(GESTURE_TURN)):
                    self.pos = (qx, qy)  # dropped where it paused; the turn is never sent
                    return acts + self.release(f"turned {math.degrees(math.acos(max(-1.0, min(1.0, cos)))):.0f} deg after a pause")
                self.paused = None  # carries on the same way: drag on
                self.dhist = (self.dhist + [(t, x, y)])[-6:]
        if self.state == "drag":
            a = GESTURE_SMOOTH
            self.pos = (a * x + (1 - a) * self.pos[0], a * y + (1 - a) * self.pos[1])
            acts.append(("move", *self.pos))
        else:  # content follows the finger (natural scrolling); motion under the deadband is jitter
            dx, dy = x - self.pos[0], y - self.pos[1]
            if math.hypot(dx, dy) >= SCROLL_DEADBAND:
                # only along the first stroke's direction: live, repeated strokes scrolled back on every return stroke
                # (the fingers moving back just above the glass look the same as touching it)
                ux, uy = self.scroll_dir
                d = max(dx * ux + dy * uy, 0.0)
                self.acc[0] -= ux * d * SCROLL_GAIN
                self.acc[1] += uy * d * SCROLL_GAIN
                self.pos = (x, y)
            ix, iy = int(self.acc[0]), int(self.acc[1])
            if ix or iy:
                self.acc[0] -= ix
                self.acc[1] -= iy
                acts.append(("scroll", ix, iy))
        return self._stop_check(t, x, y, acts)

    def _stop_check(self, t, x, y, acts):
        self.track.append((t, x, y))
        while t - self.track[0][0] > GESTURE_LIFT_WINDOW:
            self.track.popleft()
        if len(self.track) >= 2:  # drag or scroll: a real stop also ends it (a lift the hand size didn't show)
            (t1, x1, y1), (t2, x2, y2) = self.track[0], self.track[-1]
            if self.state == "scroll":  # distance travelled: back-and-forth strokes net to ~0 but aren't a stop
                dist = sum(math.dist(p[1:], q[1:]) for p, q in zip(self.track, list(self.track)[1:]))
            else:  # net movement: a drag's stop jitters 15-22 pt per frame, which travelled distance would miss
                dist = math.hypot(x2 - x1, y2 - y1)
            if t2 - t1 >= GESTURE_LIFT_WINDOW - 0.07 and dist / (t2 - t1) < GESTURE_LIFT_SPEED:
                return acts + self.release("stopped")
        return acts + self._cap(t)


def double_click(x, y):
    """The second half of a double-click: one press with click state 2. The first tap was already clicked at this spot
    (GestureEngine only emits dclick after a click there), so a whole double-click here made apps see three presses.
    (pyautogui's doubleClick posts plain clicks, which browsers and the Finder count as single clicks.)"""
    import Quartz  # PyObjC, installed with pyautogui on macOS
    for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
        ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventClickState, 2)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def perform_gestures(acts, args):
    """Carry out GestureEngine actions with pyautogui, or print them in a dry run. Returns the last point to flash."""
    flash = None
    for act in acts:
        msg = f"gesture {act}"
        if args.touch_debug:
            debug_log(msg)
        elif args.dry_run:
            print(msg)
        if act[0] in ("click", "dclick", "down"):
            flash = act[1:]
        if args.dry_run:
            continue
        kind, *v = act
        if kind in ("click", "dclick", "down", "move", "up", "point"):  # onto the screen, off FailSafe corners
            v = screen_point(v, *pyautogui.size())[0]
        if kind == "click":
            pyautogui.click(*v)
        elif kind == "dclick":
            double_click(*v)
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


def touch_step(engine, fuser, cap, tip, on, two, span=None):
    """One camera frame of touch handling, as run() does it. tip: fingertip (x, y) in screen points, or None when
    no hand was found; on: fingertip on screen under a trusted M; cap: the frame's capture time. Returns the actions
    in order: the frame's, then those of the touch-downs TouchFuser accepted, then any lift-off without a frame."""
    if tip is None:
        acts = engine.frame(cap, 0.0, 0.0, False, False)
    else:
        fuser.add_fingertip(cap, float(tip[0]), float(tip[1]), bool(on))
        acts = engine.frame(cap, float(tip[0]), float(tip[1]), bool(on), two, span)
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


def stamped_capture_time(now_mono, now_wall, stamp, link_latency):
    """Monotonic capture time of a frame the phone stamped (X-Capture-Time, Unix seconds). The stamp still holds the
    glasses-to-phone minimum delay, link_latency. None when the stamp is from the future or over 3 s old: the
    phone's and the Mac's clocks disagree, so the caller falls back to capture_time()."""
    lag = now_wall - stamp
    if not -0.5 <= lag <= 3.0:
        return None
    return now_mono - lag - link_latency


HAND_CROP_MARGIN = (0.3, 0.2, 0.3, 0.7)  # left, top, right, bottom, as fractions of the screen's size in the frame


def hand_crop(M, W, H, w, h, prev):
    """Frame box (x0, y0, x1, y1) for hand detection: the screen's bounding box in the camera image plus margins
    (most below: the arm comes from there), so the hand is larger in the landmarker's input. The previous box is
    kept unless the new one moved by over 10 % of the screen's size: the landmarker tracks the hand between frames
    in crop coordinates. With no M the previous box (or None, the whole frame) stays."""
    if M is None:
        return prev
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2),
                                    np.linalg.inv(M)).reshape(-1, 2)
    (x0, y0), (x1, y1) = quad.min(axis=0), quad.max(axis=0)
    sw, sh = x1 - x0, y1 - y0
    l, t, r, b = HAND_CROP_MARGIN
    box = (max(0, int(x0 - l * sw)), max(0, int(y0 - t * sh)), min(w, int(x1 + r * sw)), min(h, int(y1 + b * sh)))
    if box[2] - box[0] < 64 or box[3] - box[1] < 64:
        return prev
    if prev is not None and max(abs(a - c) for a, c in zip(box, prev)) < 0.1 * max(sw, sh):
        return prev
    return box


def remap_landmarks(hands, box, w, h):
    """Landmarks found in the crop box -> normalised coordinates of the whole w x h frame (in place)."""
    x0, y0, x1, y1 = box
    for hand in hands:
        for p in hand:
            p.x = (x0 + p.x * (x1 - x0)) / w
            p.y = (y0 + p.y * (y1 - y0)) / h


class ScreenMatcher:
    """Finds the screen in a camera frame by matching it against a screenshot of the Mac's own display (ORB features,
    RANSAC homography), so any visible part of the screen gives a fix, with no markers in view. Synthetic tests:
    1.4-3.9 pt error, 5-8 ms per frame. macOS gives apps without Screen Recording permission only the wallpaper,
    so run() checks that first. grab: returns a BGR screenshot (any scale); tests pass their own."""
    SHOT_SCALE = 0.5  # features on a half-size screenshot (in points): about the camera's pixels per point
    SHOT_EVERY = 1.0  # seconds between screenshots: the screen changes, and matching tolerates a little staleness
    MIN_INLIERS = 25  # matches a fit must explain; fewer is a guess

    def __init__(self, W, H, grab=None):
        self.W, self.H = W, H
        if grab is None:
            sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
            mon = sct.monitors[1]  # the menu-bar display, as pyautogui's coordinates
            grab = lambda: np.asarray(sct.grab(mon))[:, :, :3]
            self._close = sct.close
        else:
            self._close = lambda: None
        self.grab = grab
        self.ignore = []  # screen-point polygons left out of the screenshot's features (the marker windows)
        # fastThreshold 6 (default 20): faint corners count too. Simulated glasses views at 504x896: a desktop matched
        # 34/40 either way, the tutorial's faint camouflage 40/40 (0/40 at 20); about twice the time (3-5 -> 6-12 ms)
        self.orb = cv2.ORB_create(3000, fastThreshold=6)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.t_shot, self.des, self.pts = -1e9, None, None

    def _refresh(self, now):
        g = cv2.cvtColor(np.ascontiguousarray(self.grab()), cv2.COLOR_BGR2GRAY)
        f = self.W * self.SHOT_SCALE / g.shape[1]  # the screenshot may be in points or Retina pixels
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        mask = None
        if self.ignore:  # the screen as it is without markers: matching must not lean on them
            mask = np.full(g.shape, 255, np.uint8)
            cv2.fillPoly(mask, [np.int32(np.round(p * self.SHOT_SCALE)) for p in self.ignore], 0)
        kp, self.des = self.orb.detectAndCompute(g, mask)
        self.pts = np.float32([k.pt for k in kp]) / self.SHOT_SCALE if kp else None  # in screen points
        self.t_shot = now

    def match(self, gray, now, box=None):
        """M (camera px -> screen points) or None. box: only look for features in this frame region (x0, y0, x1, y1)."""
        if now - self.t_shot >= self.SHOT_EVERY:
            self._refresh(now)
        if self.des is None or len(self.des) < self.MIN_INLIERS:
            return None
        mask = None
        if box is not None:
            mask = np.zeros(gray.shape[:2], np.uint8)
            mask[box[1]:box[3], box[0]:box[2]] = 255
        kc, dc = self.orb.detectAndCompute(gray, mask)
        if dc is None or len(kc) < self.MIN_INLIERS:
            return None
        pairs = self.matcher.knnMatch(dc, self.des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]  # Lowe's ratio test
        if len(good) < self.MIN_INLIERS:
            return None
        pc = np.float32([kc[g.queryIdx].pt for g in good])
        ps = self.pts[[g.trainIdx for g in good]]
        M, inliers = cv2.findHomography(pc, ps, cv2.RANSAC, 6.0)
        if M is None or int(inliers.sum()) < self.MIN_INLIERS:
            return None
        return M

    def close(self):
        self._close()


def screen_recording_allowed():
    """True if macOS lets this process see other apps' windows in screenshots (Screen Recording permission)."""
    import Quartz
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def capture_time(arrival, camera_latency):
    """Monotonic capture time of a camera frame that arrived at `arrival`: frames lag real time by camera_latency."""
    return arrival - camera_latency


def raw_map(M, pt):
    """Unclamped point mapping through M (no clamping to the screen, unlike map_point)."""
    return cv2.perspectiveTransform(np.float32([[pt]]), M)[0, 0]


EDGE_TOL = 90  # screen points: edge taps land up to ~70 pt outside (calibration: left edge -38..-68)
TIP_DY = 0.0  # screen points: a -32 shift from one calibration run (fingertip ~32 pt below the dot) felt "a lot worse" live; off
                # the touch; the remaining 20-50 pt is per-tap scatter, so only this shift is corrected


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
    if not args.dry_run and not accessibility_trusted(prompt=True):
        raise SystemExit(f"macOS hasn't granted Accessibility permission to {HOST_APP} (the app the tracker runs in), "
                         f"so the cursor can't move. Turn {HOST_APP} on in System Settings > Privacy & Security > "
                         f"Accessibility, then quit {HOST_APP} and open it again.")
    landmarker = make_hand_landmarker()  # before the camera opens, so a download failure cannot leak it
    src = open_source(args)
    pinch = PinchDetector()
    M, last_ok, smooth, t0, last_ts = None, 0.0, None, time.monotonic(), -1
    overlay, targets, matcher = None, None, None  # matcher: finds the screen from its content
    touch_sock, touch_flash = None, None  # touch_flash: ((x, y) screen point, monotonic time to stop showing)
    engine = GestureEngine()  # touch-downs and fingertip frames -> click / drag / scroll actions
    view, prev_tip = None, None  # prev_tip: last frame's pointing fingertip (camera px), None without a hand
    crop, stamp_warned = None, False  # crop: hand-detection box in the frame, None for the whole frame
    prev_other = None  # last frame's fingertips of the non-pointing hand (typing_flag)
    frame_size = None  # (w, h) of the first frame, logged once
    try:
        if args.view_port:
            try:
                view = ViewServer(args.view_port)
            except OSError as e:
                raise SystemExit(f"cannot serve the view on 127.0.0.1:{args.view_port} ({e}); "
                                 "another process may hold it.")
        if not args.no_content_match:  # with markers it fills their gaps; without them it is how the screen is found
            if screen_recording_allowed():
                matcher = ScreenMatcher(W, H)
            else:  # asks macOS once (it adds the app to the list)
                import Quartz
                Quartz.CGRequestScreenCaptureAccess()
                print(f"Screen Recording is off for {HOST_APP} (the app the tracker runs in, not the browser), so the "
                      "screen can't be found from what it shows" + (", only from the markers" if args.show_markers else
                      ": turn Screen markers on, or") + f". Turn {HOST_APP} on in System Settings > Privacy & Security > "
                      f"Screen Recording, then quit {HOST_APP} and open GlassesTouch again.", flush=True)
        toggle = None  # on-demand markers: hidden until the screen can't be found from its content
        if args.show_markers:
            on_demand = args.markers_on_demand and matcher is not None  # no content matching: markers stay on
            overlay, targets = start_overlay(args.marker_size, hidden=on_demand)
            if matcher is not None:
                matcher.ignore = overlay_regions(targets)
            if on_demand:
                toggle = MarkerToggle(time.monotonic())
                print("markers hidden until the screen can't be found from what it shows", flush=True)
        tracker = ScreenTracker(W, H, overlay_regions(targets) if targets is not None else None)
        hand_poly = None  # previous frame's hand, kept out of the tracker's features
        fuser = TouchFuser(args.tap_min_acc, args.tap_max_acc)
        if args.touch:
            touch_sock = open_touch_socket(args.touch_port)
        last_frame_wall = time.monotonic()  # wall clock of the last camera frame (stall detection)
        last_marker, stale_logged, drift = time.monotonic(), False, {}  # last marker reset; drift guard state
        stats = {"t": time.monotonic(), "frames": 0, "hand": 0, "markers": 0, "content": 0, "tracking": 0, "none": 0, "two": 0,
                 "wait": 0.0, "proc": 0.0, "lags": [], "kb": [], "spans": [], "pics": 0}
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
            if frame_size is None:  # once: the size that really arrives (a camera's format switch can fail quietly)
                frame_size = frame.shape[1::-1]
                print(f"frames arrive at {frame_size[0]}x{frame_size[1]}", flush=True)
            now = time.monotonic()
            stamp = getattr(src, "stamp", None)  # the phone's capture time for this frame, if it sent one
            if stamp is not None:
                cap_t = stamped_capture_time(now, time.time(), stamp, 0.0 if src.exact else args.link_latency)
                if cap_t is None and not stamp_warned:
                    print("frame capture times are over 3 s off: are the phone's and the Mac's clocks set "
                          "automatically? Guessing from arrival time instead.", flush=True)
                    stamp_warned = True
                if cap_t is not None and args.touch_debug:  # the stats line reports it once a second
                    stats["lags"].append(now - cap_t)
            else:
                cap_t = None
            if cap_t is None:
                cap_t = capture_time(now, args.camera_latency)
            stats["wait"] += now - t_read  # time blocked waiting for the next frame
            prev_now = now
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = detect_marker_corners(gray)
            redetected, dropped_centres = [], {}
            if targets is None:
                centres = {i: c.mean(axis=0) for i, c in corners.items()}
                M_new = homography_from_markers(centres, W, H)
                if M_new is not None and screen_quad(M_new, W, H) is None:
                    M_new = None  # singular fit counts as a failure
                M_c = matcher.match(gray, now) if matcher is not None and M_new is None else None
                if M_c is not None and screen_quad(M_c, W, H) is None:
                    M_c = None
                if M_new is not None:
                    M, last_ok, source = M_new, now, "markers"
                    tracker.reset(gray, M)
                elif M_c is not None:  # no markers on screen: found from what the screen shows
                    M, last_ok, source = M_c, now, "content"
                    tracker.reset(gray, M)
                else:  # markers missing: follow the screen; HOLD_SECONDS applies only if this fails too
                    M_t = tracker.update(gray, hand_poly)
                    if M_t is not None:
                        M, last_ok, source = M_t, now, "tracking"
                    else:
                        source = None
            else:  # overlay: drop hand-covered markers, guided re-detection, then the trust rules choose M
                M_track = tracker.update(gray, hand_poly)
                held = M if M is not None and now - last_ok <= HOLD_SECONDS else None
                M_new, source, reset, corners, redetected, dropped = overlay_fix(
                    gray, corners, targets, M_track, held, hand_poly, W, H, drift, M)
                if matcher is not None and source != "markers":  # too few markers: find the screen from its content
                    M_c = matcher.match(gray, now)
                    if M_c is not None and screen_quad(M_c, W, H) is not None:
                        M_new, source, reset = M_c, "content", True  # a fresh fix, like the markers'
                if toggle is not None:
                    def content_finds_it():  # this frame, from content alone (the matcher skips the markers)
                        M_p = matcher.match(gray, now)
                        return M_p is not None and screen_quad(M_p, W, H) is not None
                    act = toggle.step(now, source, content_finds_it)
                    if act:
                        overlay_set(overlay, act == "show")
                        print("markers shown: the screen was lost" if act == "show" else
                              "markers hidden: the screen is found from what it shows", flush=True)
                centres = {i: c.mean(axis=0) for i, c in corners.items()}
                dropped_centres = {i: c.mean(axis=0) for i, c in dropped.items()}
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

            h, w = frame.shape[:2]
            if not args.no_hand_crop:
                crop = hand_crop(M if m_valid else None, W, H, w, h, crop)
            box = crop or (0, 0, w, h)
            rgb = np.ascontiguousarray(cv2.cvtColor(frame[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2RGB))
            last_ts = max(int((now - t0) * 1000), last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ts)
            if crop:
                remap_landmarks(res.hand_landmarks, crop, w, h)
            keep = real_hands(res.hand_landmarks, M if m_valid else None, w, h)
            if len(keep) < len(res.hand_landmarks):  # drop pictures of hands on the screen, keeping the lists in step
                res.hand_landmarks = [res.hand_landmarks[i] for i in keep]
                res.hand_world_landmarks = [res.hand_world_landmarks[i] for i in keep] if res.hand_world_landmarks else []
                res.handedness = [res.handedness[i] for i in keep] if res.handedness else []
                if args.touch_debug:
                    stats["pics"] += 1
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
                if lm is not None and M is not None and m_valid:  # the pointing hand's size on the screen plane
                    stats["spans"].append(math.dist(raw_map(M, (lm[0].x * w, lm[0].y * h)), raw_map(M, (lm[9].x * w, lm[9].y * h))))
                stats[source or "none"] += 1
                if lm is not None and res.hand_world_landmarks:  # the pointing hand's pose, to tune two_finger
                    pw = [(p.x, p.y, p.z) for p in res.hand_world_landmarks[k]]
                    stats["pose"] = finger_extension(pw)
                    stats["two"] += two_finger(pw)
                if now - stats["t"] >= 1.0:
                    n = max(stats["frames"], 1)
                    pose = stats.get("pose")
                    pose = (" ext=" + "/".join(f"{e:.2f}" for e in pose[0]) + f" gap={pose[1]:.2f}") if pose else ""
                    lags = sorted(stats["lags"])  # frame delay from capture, when the phone stamps frames
                    lag = f"{1000 * lags[len(lags) // 2]:.0f}/{1000 * lags[-1]:.0f}ms" if lags else "-"
                    cw = f"{crop[2] - crop[0]}x{crop[3] - crop[1]}" if crop else "full"
                    kbs = sorted(stats["kb"])  # the other hand's biggest fingertip move per frame (px): resting vs typing
                    kb = f"{kbs[len(kbs) // 2]:.0f}/{kbs[-1]:.0f}px" if kbs else "-"
                    sp = sorted(stats["spans"])  # real hand size (pt): must stay well above MIN_HAND_PT
                    span = f"{sp[0]:.0f}-{sp[-1]:.0f}pt" if sp else "-"
                    debug_log(f"stats fps={stats['frames'] / (now - stats['t']):.1f} hand={stats['hand']}/{n} "
                              f"M markers={stats['markers']} content={stats['content'] if matcher is not None else 'off'} tracking={stats['tracking']} none={stats['none']} "
                              f"proc={1000 * stats['proc'] / n:.0f}ms wait={1000 * stats['wait'] / n:.0f}ms "
                              f"two={stats['two']}/{n}{pose} lag={lag} crop={cw} kb={kb} span={span} pics={stats['pics']}")
                    stats.update(t=now, frames=0, hand=0, markers=0, content=0, tracking=0, none=0, wait=0.0, proc=0.0, two=0, lags=[], kb=[], spans=[], pics=0,
                                 pose=None)
            hand_poly = hand_exclusion(lm, w, h) if lm is not None else None
            status, ev = ("no hand" if lm is None else "tracking"), None
            if args.touch:
                drain_taps(touch_sock, fuser)
                cap = cap_t
                fuser.add_hands(cap, len(res.hand_landmarks))
                tips_other = other_hand_tips(res.hand_landmarks, k, w, h)
                fuser.add_typing(cap, typing_flag(res.hand_landmarks, k, w, h, M, source is not None, H, prev_other))
                if tips_other is not None and prev_other is not None and args.touch_debug:
                    stats["kb"].append(float(np.linalg.norm(tips_other - prev_other, axis=1).max()))
                prev_other = tips_other
                # no camera frame for GESTURE_STALL while a button is held: release it before this frame
                acts = engine.release() if engine.held and now - last_frame_wall > GESTURE_STALL else []
                last_frame_wall = now
                if lm is not None:  # fingertip sample, unclamped; off-screen or unmapped counts as not on screen
                    # only an M from markers or the tracker this frame counts; the HOLD_SECONDS hold is stale
                    fresh = source is not None
                    tip = raw_map(M, (lm[8].x * w, lm[8].y * h)) if fresh else (0.0, 0.0)
                    tip = (tip[0], tip[1] + TIP_DY) if fresh else tip
                    on = fresh and screen_point(tip, W, H)[1]  # tip stays raw for the log; actions clamp it
                    world = res.hand_world_landmarks[k] if res.hand_world_landmarks else None
                    pose = [(p.x, p.y, p.z) for p in world] if world else [(p.x * w, p.y * h) for p in lm]
                    span = (math.dist(raw_map(M, (lm[0].x * w, lm[0].y * h)), raw_map(M, (lm[9].x * w, lm[9].y * h)))
                            if fresh else None)  # hand size on the screen plane: grows when the finger lifts off
                    acts += touch_step(engine, fuser, cap, tip, on, two_finger(pose), span)
                else:
                    acts += touch_step(engine, fuser, cap, None, False, False)  # hand lost
                if args.touch_debug:
                    for note in engine.notes:
                        debug_log(note)
                engine.notes.clear()
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
            if quad is not None:  # screen outline: green from markers, light blue from content, yellow from tracking
                colour = {"markers": (0, 255, 0), "content": (255, 200, 80)}.get(source, (0, 255, 255))
                cv2.polylines(frame, [_poly(quad, frame.shape[:2])], True, colour, 2)
            if touch_flash is not None and time.monotonic() < touch_flash[1] and m_valid:
                cx, cy = raw_map(np.linalg.inv(M), touch_flash[0])  # screen point back into the camera image
                cv2.circle(frame, (int(cx), int(cy)), 14, (255, 0, 255), 3)
            draw_debug(frame, centres, lm, pinch.down, status, redetected, source, dropped_centres)
            if view is not None:  # the panel's view: the whole frame (never cropped) with the live overlay drawn on it
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
            if matcher is not None:
                matcher.close()
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
    p.add_argument("--markers-on-demand", action="store_true",
                   help="with --show-markers: keep the markers hidden while the screen is found from what it shows, "
                        "and show them only when it is lost (needs Screen Recording)")
    p.add_argument("--touch", action="store_true",
                   help="click where a sensor tap (imu_taps.py) meets a still fingertip on the screen; "
                        "hovering does not move the cursor and pinch clicking is off")
    p.add_argument("--touch-port", type=int, default=TOUCH_PORT, help="UDP port for imu_taps.py (default 9787)")
    p.add_argument("--touch-debug", action="store_true",
                   help="with --touch: print one line per tap (CLICK or the reason it was rejected)")
    p.add_argument("--camera-latency", type=float, default=0.3, metavar="S",
                   help="seconds camera frames lag real time when the stream has no capture times (default 0.3)")
    p.add_argument("--link-latency", type=float, default=0.1, metavar="S",
                   help="glasses-to-phone minimum delay left in the phone's capture times (default 0.1)")
    p.add_argument("--no-content-match", action="store_true",
                   help="find the screen from the markers only, not by matching a screenshot of it")
    p.add_argument("--no-hand-crop", action="store_true",
                   help="find the hand in the whole frame instead of a crop around the screen")
    p.add_argument("--tap-min-acc", type=float, default=TAP_MIN_ACC, metavar="G",
                   help="with --touch: reject taps whose accel change is below this, in g (default 0.003)")
    p.add_argument("--tap-max-acc", type=float, default=TAP_MAX_ACC, metavar="G",
                   help="with --touch: reject taps whose accel change is above this, a knock (default 0.08)")
    p.add_argument("--view-port", type=int, metavar="N",
                   help="serve the annotated debug frame as MJPEG on 127.0.0.1:N (the control panel shows it)")
    p.add_argument("--no-window", action="store_true",
                   help="no OpenCV window (the q key is then unavailable; stop with Ctrl-C)")
    p.add_argument("--overlay-child", type=int, help=argparse.SUPPRESS)  # internal: marker overlay process
    p.add_argument("--overlay-hidden", action="store_true", help=argparse.SUPPRESS)  # internal: it starts invisible
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
        overlay_child(a.overlay_child, a.overlay_hidden)
        return
    if a.make_markers:
        make_marker_sheet(a.make_markers)
        print(f"wrote {a.make_markers}")
        return
    run(a)


if __name__ == "__main__":
    main()
