"""Camera-free, mouse-free tests for tracker.py. Run: .venv/bin/python test_tracker.py"""
import math
import socket
import time
from types import SimpleNamespace

import cv2
import numpy as np

import tracker
from imu_taps import TapDetector, tap_datagram, tap_features
from tracker import (ARUCO_DICT, GESTURE_MAX, SCROLL_GAIN, TOUCH_SETTLE, TRUST_FRAC, GestureEngine,
                     ViewServer, pointing_hand, typing_flag, drop_occluded, fit_markers, overlay_fix, _side,
                     release_touch, touch_step,
                     PinchDetector, ScreenTracker, TouchFuser, two_finger,
                     capture_time, choose_overlay_M, format_decision, tracking_age_guard,
                     detect_marker_corners, detect_markers, drain_taps, guided_redetect, homography_from_corners,
                     homography_from_markers, hover_moves_cursor, map_point, marker_extent, marker_png, open_touch_socket,
                     parse_args, pose_is_two, screen_point, screen_quad)


def test_screen_point_edge_tolerance():
    W, H = 1512, 982
    # live Dock taps mapped just below the screen: accepted, clamped 1 pt inside the edge
    assert screen_point((1111, 993), W, H) == ((1111.0, H - 2.0), True)
    assert screen_point((1113, 1032), W, H)[1]
    # a hand on the keyboard (live y~2400) and far past the sides: rejected
    assert not screen_point((1134, 2364), W, H)[1]
    assert not screen_point((-100, 400), W, H)[1]
    # a corner tap never lands on pyautogui's FailSafe corner points
    assert screen_point((-20, -20), W, H) == ((1.0, 1.0), True)
    assert screen_point((W + 30, H + 30), W, H) == ((W - 2.0, H - 2.0), True)
    assert screen_point((700, 400), W, H) == ((700.0, 400.0), True)

W, H = 1920, 1080
CAM = (1000, 800)  # synthetic camera image size
# camera-space marker centres for ids 0..3 (TL, TR, BR, BL)
P = np.float32([[150, 120], [850, 140], [830, 640], [170, 660]])
G = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), P)  # screen -> camera
MW = cv2.getPerspectiveTransform(  # known perspective warp applied to the whole camera image
    np.float32([[0, 0], [CAM[0], 0], [CAM[0], CAM[1]], [0, CAM[1]]]),
    np.float32([[60, 40], [960, 90], [930, 760], [40, 700]]))


def apply(M, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(1, -1, 2), M).reshape(-1, 2)


def draw_scene():
    img = np.full((CAM[1], CAM[0]), 255, np.uint8)
    for i, (cx, cy) in enumerate(P):
        x, y = int(cx) - 60, int(cy) - 60
        img[y:y + 120, x:x + 120] = cv2.aruco.generateImageMarker(ARUCO_DICT, i, 120)
    return img


def test_marker_homography_roundtrip():
    warped = cv2.warpPerspective(draw_scene(), MW, CAM, borderValue=255)
    centres = detect_markers(warped)
    assert all(i in centres for i in range(4)), f"markers not all found: {sorted(centres)}"
    Hm = homography_from_markers(centres, W, H)
    assert Hm is not None
    screen_pts = np.float32([[0, 0], [W, 0], [W, H], [0, H], [W / 2, H / 2]])
    warped_pts = apply(MW, apply(G, screen_pts))  # where each screen point appears in the warped image
    for sp, wp in zip(screen_pts, warped_pts):
        x, y = map_point(Hm, wp, (W, H))
        assert abs(x - sp[0]) <= 5 and abs(y - sp[1]) <= 5, f"{sp} -> ({x:.1f}, {y:.1f})"


def test_homography_custom_targets():
    centres = {i: P[i] for i in range(4)}
    targets = {0: (100, 100), 1: (1800, 120), 2: (1750, 950), 3: (150, 1000)}
    Hm = homography_from_markers(centres, W, H, targets)
    assert Hm is not None
    for i in range(4):
        x, y = map_point(Hm, P[i], (W, H))
        assert abs(x - targets[i][0]) <= 1 and abs(y - targets[i][1]) <= 1, (i, x, y)


def test_overlay_corner_homography_two_markers():
    # only overlay markers 0 (TL) and 2 (BR) are visible: a diagonal pair spans the screen, so the
    # extrapolation to the other corners is well conditioned. Two top markers alone give ~8 px here.
    S = 400
    sq = {0: np.float32([[80, 80], [80 + S, 80], [80 + S, 80 + S], [80, 80 + S]]),
          2: np.float32([[W - 80 - S, H - 80 - S], [W - 80, H - 80 - S], [W - 80, H - 80], [W - 80 - S, H - 80]])}
    screen = np.full((H, W), 255, np.uint8)
    for i, c in sq.items():
        x0, y0 = int(c[0][0]), int(c[0][1])
        screen[y0:y0 + S, x0:x0 + S] = cv2.aruco.generateImageMarker(ARUCO_DICT, i, S)
    SC = MW @ G  # screen -> warped camera image
    warped = cv2.warpPerspective(screen, SC, CAM, borderValue=255)
    corners = detect_marker_corners(warped)
    assert sorted(corners) == [0, 2], f"markers found: {sorted(corners)}"
    Hm = homography_from_corners(corners, sq)
    assert Hm is not None
    screen_pts = np.float32([[0, 0], [W, 0], [W, H], [0, H], [W / 2, H / 2]])
    back = apply(Hm, apply(SC, screen_pts))
    err = np.abs(back - screen_pts).max()
    print(f"  overlay corner homography max error {err:.2f} px")
    assert err <= 5, f"max error {err:.2f} px: {back.round(1).tolist()}"


def test_collinear_corners_rejected():
    # all corner points of two markers squashed onto one line: the homography is not determined
    t = np.arange(4, dtype=np.float32) * 300
    corners = {0: np.float32(np.c_[t, 0.5 * t]), 2: np.float32(np.c_[t, 0.5 * t + 0.001 * np.arange(4)])}
    targets = {0: np.float32([[0, 0], [400, 0], [400, 400], [0, 400]]),
               2: np.float32([[1000, 600], [1400, 600], [1400, 1000], [1000, 1000]])}
    assert homography_from_corners(corners, targets) is None


def test_pinch_hysteresis():
    d = PinchDetector(enter=0.30, exit=0.45)
    seq = [0.50, 0.40, 0.29, 0.35, 0.44, 0.46, 0.31, 0.20, 0.50]
    expected = [None, None, "down", None, None, "up", None, "down", "up"]
    got = [d.update(r) for r in seq]
    assert got == expected, got


def test_map_point_clamped():
    I = np.eye(3)
    x, y = map_point(I, (-50, 5000), (W, H))
    assert (x, y) == (1.0, H - 2.0), (x, y)  # off the failsafe corners
    x, y = map_point(I, (2500, 500), (W, H))
    assert (x, y) == (W - 2.0, 500.0), (x, y)
    x, y = map_point(I, (-50, -50), (1440, 900))
    assert (x, y) == (1.0, 1.0), (x, y)
    x, y = map_point(I, (9999, 9999), (1440, 900))
    assert (x, y) == (1438.0, 898.0), (x, y)
    x, y = map_point(I, (100, 200), (W, H))
    assert abs(x - 100) < 1e-6 and abs(y - 200) < 1e-6, (x, y)


def _texture(rng, h, w, n_rect=60, n_text=25):
    """Random rectangles and text on a grey canvas: corners for the tracker to follow."""
    img = np.full((h, w), 200, np.uint8)
    for _ in range(n_rect):
        x, y = rng.integers(0, w - 100), rng.integers(0, h - 60)
        bw, bh = rng.integers(20, 200), rng.integers(10, 120)
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), int(rng.integers(0, 256)), -1)
    for k in range(n_text):
        cv2.putText(img, f"word {k} {rng.integers(1000)}", (int(rng.integers(0, w - 200)), int(rng.integers(20, h - 20))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, int(rng.integers(0, 256)), 2)
    return img


def _drift(P0, k):
    """Screen -> camera for frame k: rotation, scale and translation that grow gradually from P0."""
    R = cv2.getRotationMatrix2D((600, 400), 0.4 * k, 1.0 + 0.004 * k)
    R[:, 2] += (1.5 * k, -1.0 * k)
    return np.vstack([R, [0, 0, 1]]) @ P0


def _tracker_setup():
    """Textured 1512x982 screen seen at pose P0 in a 1280x720 camera frame; tracker reset on it."""
    SW, SH, S = 1512, 982, (1280, 720)
    screen = _texture(np.random.default_rng(0), SH, SW)
    corners = np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]])
    P0 = cv2.getPerspectiveTransform(corners, np.float32([[200, 120], [1000, 150], [980, 640], [220, 680]]))
    gray0 = cv2.warpPerspective(screen, P0, S, borderValue=60)
    tr = ScreenTracker(SW, SH)
    tr.reset(gray0, np.linalg.inv(P0))
    return screen, corners, P0, S, tr, gray0


def test_screen_tracker_drift():
    screen, corners, P0, S, tr, _ = _tracker_setup()
    for k in range(1, 21):  # gradual drift from P0
        T = _drift(P0, k)  # screen -> camera for this frame
        M_k = tr.update(cv2.warpPerspective(screen, T, S, borderValue=60))
        assert M_k is not None, f"frame {k}: tracker lost"
        back = apply(M_k, apply(T, corners))  # true screen corners of this frame, mapped back
        err = np.abs(back - corners).max()
        assert err <= 6, f"frame {k}: max error {err:.2f} px"
    print(f"  screen tracker after 20 frames: last error {err:.2f} px")


def test_screen_tracker_noise_lost():
    _, _, _, S, tr, gray0 = _tracker_setup()
    noise = np.random.default_rng(1).integers(0, 256, (S[1], S[0]), dtype=np.uint8)
    assert tr.update(noise) is None
    assert tr.lost
    assert tr.update(gray0) is None  # stays lost until the next reset


def test_screen_tracker_scrolling_static_regions():
    # central window scrolls 10 px/frame; with static regions only the 4 corner squares and the bezel are tracked
    rng = np.random.default_rng(2)
    SW, SH, S, OX, OY = 1512, 982, (1280, 720), 76, 50  # bezel padding: wider than the 4% ring
    screen = _texture(rng, SH, SW)
    plane = _texture(rng, SH + 2 * OY, SW + 2 * OX)  # textured bezel around the screen
    plane[OY:OY + SH, OX:OX + SW] = screen
    scroll = _texture(rng, 500, 1100)
    corners = np.float32([[0, 0], [SW, 0], [SW, SH], [0, SH]])
    P0 = cv2.getPerspectiveTransform(corners, np.float32([[200, 120], [1000, 150], [980, 640], [220, 680]]))
    Tt = np.float32([[1, 0, -OX], [0, 1, -OY], [0, 0, 1]])  # plane coords -> screen coords

    def render(k, T):
        p = plane.copy()
        p[OY + 250:OY + 750, OX + 400:OX + 1100] = scroll[:, 10 * k:10 * k + 700]  # scrolling window
        return cv2.warpPerspective(p, T @ Tt, S, borderValue=60)

    def sq(x, y):
        return [(x, y), (x + 140, y), (x + 140, y + 140), (x, y + 140)]
    tr = ScreenTracker(SW, SH, [sq(0, 0), sq(SW - 140, 0), sq(SW - 140, SH - 140), sq(0, SH - 140)])
    tr.reset(render(0, P0), np.linalg.inv(P0))
    for k in range(1, 21):
        T = _drift(P0, k)
        M_k = tr.update(render(k, T))
        assert M_k is not None, f"frame {k}: tracker lost"
        err = np.abs(apply(M_k, apply(T, corners)) - corners).max()
        assert err <= 6, f"frame {k}: max error {err:.2f} px"
    print(f"  scrolling content, static regions: last error {err:.2f} px")


def test_screen_tracker_fold_and_jump_rejected():
    screen, corners, P0, S, tr, _ = _tracker_setup()
    # bow-tie quad: the screen folds over itself, which no real pose does
    bow = cv2.getPerspectiveTransform(corners, np.float32([[200, 120], [1000, 650], [1000, 150], [200, 680]]))
    assert tr.update(cv2.warpPerspective(screen, bow, S, borderValue=60)) is None
    assert tr.lost
    # wild zoom: the quad's area grows far more than 30% in one frame
    _, _, P0, S, tr, _ = _tracker_setup()
    zoom = np.vstack([cv2.getRotationMatrix2D((600, 400), 0, 1.6), [0, 0, 1]]) @ P0
    assert tr.update(cv2.warpPerspective(screen, zoom, S, borderValue=60)) is None
    assert tr.lost


def test_screen_tracker_realistic_noise():
    screen, corners, P0, S, _, _ = _tracker_setup()
    rng = np.random.default_rng(3)

    def degrade(img):  # sensor noise (sigma 3), then a slight 3x3 blur
        noisy = img + rng.normal(0, 3, img.shape)
        return cv2.GaussianBlur(np.clip(noisy, 0, 255).astype(np.uint8), (3, 3), 0)
    tr = ScreenTracker(1512, 982)
    tr.reset(degrade(cv2.warpPerspective(screen, P0, S, borderValue=60)), np.linalg.inv(P0))
    tracked, good, worst = 0, 0, 0.0
    for k in range(1, 101):
        T = _drift(P0, k)
        M_k = tr.update(degrade(cv2.warpPerspective(screen, T, S, borderValue=60)))
        if M_k is None:
            continue
        tracked += 1
        err = np.abs(apply(M_k, apply(T, corners)) - corners).max()
        worst = max(worst, err)
        good += err <= 8
    print(f"  noisy frames: {tracked} tracked, {good} within 8 px, worst tracked error {worst:.2f} px")
    assert good >= 90, f"only {good} of 100 frames within 8 px"


def test_tap_detector_bursts():
    # 1 kHz accel magnitude: uniform noise (max |change| 0.002 stays under 6 x the median change of 0.00059)
    # plus 10 taps, each a burst of three 5-ms spikes of 0.3 within 60 ms
    rng = np.random.default_rng(4)
    t = np.arange(0, 10, 0.001)
    mag = 1.0 + rng.uniform(-0.001, 0.001, t.size)
    taps = 1.0 + 0.9 * np.arange(10)
    for t0 in taps:
        for dt in (0.0, 0.03, 0.06):
            i = int(round((t0 + dt) * 1000))
            mag[i:i + 5] += 0.3
    det = TapDetector()
    events = [e for ti, m in zip(t, mag) for e in det.feed(ti, m)]
    assert len(events) == 10, events
    assert all(abs(e - t0) < 0.002 for e, t0 in zip(events, taps)), (events, taps)


def test_touch_fuser():
    def still(fu, x, y, t0, t1, on=True):
        for t in np.arange(t0, t1, 1 / 30):
            fu.add_fingertip(float(t), x, y, on)
    fu = TouchFuser()  # still fingertip on screen at the tap time: one touch at the right spot
    still(fu, 500, 300, 0.0, 1.0)
    fu.add_tap(0.5)
    clicks = fu.poll(0.8)
    assert len(clicks) == 1 and np.hypot(clicks[0][0] - 500, clicks[0][1] - 300) < 1, clicks
    assert fu.poll(1.0) == []  # decided once
    fu = TouchFuser()  # fingertip sweeping 280 points through the tap window (over 250): no touch
    for t in np.arange(0.0, 1.0, 1 / 30):
        fu.add_fingertip(float(t), 400 + 1400 * t, 300, True)
    fu.add_tap(0.5)
    assert fu.poll(0.8) == []
    fu = TouchFuser()  # fingertip off the screen (on the keyboard): no touch
    still(fu, 500, 300, 0.0, 1.0, on=False)
    fu.add_tap(0.5)
    assert fu.poll(0.8) == []
    fu = TouchFuser()  # no fingertip samples: no touch, and the tap is gone
    fu.add_tap(5.0)
    assert fu.poll(5.3) == [] and not fu.taps
    assert fu.poll(7.5) == [] and not fu.taps


def test_tap_udp_round_trip():
    sock = open_touch_socket(0)  # bound like --touch, on an ephemeral port
    port = sock.getsockname()[1]
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(b"{not json", ("127.0.0.1", port))  # malformed: ignored
    tx.sendto(b'{"peak": 3.0}', ("127.0.0.1", port))  # no t: ignored
    tx.sendto(tap_datagram(123.5, 40.0), ("127.0.0.1", port))
    fuser = TouchFuser()
    drain_taps(sock, fuser)
    assert list(fuser.taps) == [(123.5, 40.0, {})], list(fuser.taps)
    tx.close()
    sock.close()


def _overlay_targets():
    """Overlay black-square quads (screen points) for ids 0..3, laid out as overlay_child lays them out."""
    lo, hi = marker_extent()
    out = {}
    for i, (x, y) in {0: (0, 0), 1: (OW - OWIN, 0), 2: (OW - OWIN, OH - OWIN), 3: (0, OH - OWIN)}.items():
        out[i] = np.float32([[x + u * OWIN, y + v * OWIN] for u, v in ((lo, lo), (hi, lo), (hi, hi), (lo, hi))])
    return out


# overlay scene: a 1512x982-point screen with four 140-point marker windows in its corners, seen by a
# 720x1280 portrait camera with the screen about 360 px wide, so each black square is ~31 px
OW, OH, OWIN = 1512, 982, 140
OCAM = (720, 1280)  # (w, h)
OTARGETS = _overlay_targets()
OG = cv2.getPerspectiveTransform(np.float32([[0, 0], [OW, 0], [OW, OH], [0, OH]]),  # screen -> camera
                                 np.float32([[150, 470], [510, 510], [480, 770], [130, 800]]))


def render_overlay(ids=(0, 1, 2, 3), blur=9, jpeg=60, G=OG):
    """Camera view of the overlay markers in `ids` (grey desk, white windows), blurred and JPEG round-tripped."""
    img = np.full((OCAM[1], OCAM[0]), 40, np.uint8)
    corners = np.float32([[0, 0], [OW, 0], [OW, OH], [0, OH]])
    cv2.fillConvexPoly(img, np.int32(apply(G, corners)), 200)
    sz = 160
    for i in ids:
        x, y = {0: (0, 0), 1: (OW - OWIN, 0), 2: (OW - OWIN, OH - OWIN), 3: (0, OH - OWIN)}[i]
        win = cv2.imdecode(np.frombuffer(marker_png(i), np.uint8), cv2.IMREAD_GRAYSCALE)
        win = cv2.resize(win, (sz, sz), interpolation=cv2.INTER_AREA)
        wcam = apply(G, [[x, y], [x + OWIN, y], [x + OWIN, y + OWIN], [x, y + OWIN]])
        Wm = cv2.getPerspectiveTransform(np.float32([[0, 0], [sz, 0], [sz, sz], [0, sz]]), wcam)
        out = cv2.warpPerspective(win, Wm, OCAM, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((sz, sz), 255, np.uint8), Wm, OCAM)
        img[mask > 0] = out[mask > 0]
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    if jpeg:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_GRAYSCALE)
    return img


def _true_corners(G=OG):
    """Where each overlay marker's black square really is in the camera image, (4, 2) per id."""
    return {i: apply(G, OTARGETS[i]) for i in range(4)}


def test_small_blurred_guided_redetection():
    img = render_overlay(blur=9, jpeg=60)
    plain = detect_marker_corners(img)
    # M estimate ~14 px off in camera pixels: the prediction of every window is shifted by that much
    M_est = np.linalg.inv(OG) @ np.float64([[1, 0, -9], [0, 1, 11], [0, 0, 1]])
    extra = guided_redetect(img, M_est, OTARGETS, plain)
    n_plain, n_guided = len(plain), len(plain) + len(extra)
    print(f"  32 px, 9x9 blur, JPEG 60: plain {n_plain} ids {sorted(plain)}, "
          f"guided {n_guided} ids (re-detected {sorted(extra)})")
    # the cell-centre decoding now reads this synthetic scene directly, so it no longer forces guided work;
    # test_live_frames_guided_finds_all_four covers the real case. Here: guided adds no duplicates and is accurate.
    assert not set(extra) & set(plain), "guided re-detected an id plain already had"
    tc = _true_corners()
    for i, c in extra.items():
        assert np.abs(c - tc[i]).max() <= 4, f"re-detected id {i} off by {np.abs(c - tc[i]).max():.2f} px"


def test_live_frames_guided_finds_all_four():
    # real glasses frames (screen content blurred) where plain detection found only some markers
    import glob
    import json
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
    t = {int(k): np.float32(v) for k, v in json.load(open(os.path.join(here, "live_targets.json"))).items()}
    frames = sorted(glob.glob(os.path.join(here, "live_*.jpg")))
    assert frames, "testdata/live_*.jpg missing"
    for f in frames:
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        plain = {i: c for i, c in detect_marker_corners(g).items() if i < 4}
        assert len(plain) >= 3, f"{os.path.basename(f)}: plain found only {sorted(plain)}"
        M = homography_from_corners(plain, t)
        found = len(plain) + len(guided_redetect(g, M, t, plain))
        assert found == 4, f"{os.path.basename(f)}: only {found} markers with guided re-detection"


def test_no_false_marker_ids():
    rng = np.random.default_rng(5)
    scenes = {"texture": _texture(rng, OCAM[1], OCAM[0]),
              "noise": rng.integers(0, 256, (OCAM[1], OCAM[0]), dtype=np.uint8),
              "no markers, blurred": render_overlay(ids=(), blur=9, jpeg=60)}
    for name, img in scenes.items():
        found = detect_marker_corners(img)
        assert not found, f"{name}: false ids {sorted(found)}"


def test_diagonal_pair_trust():
    # strong perspective: a 2-marker fit (8 points, no redundancy) extrapolates the noise to the far corners.
    # On the milder OG scene only ~1 in 300 noisy fits exceeds the threshold, too few to exercise the rule.
    TILT = cv2.getPerspectiveTransform(np.float32([[0, 0], [OW, 0], [OW, OH], [0, OH]]),
                                       np.float32([[120, 520], [600, 380], [560, 820], [60, 700]]))
    # tracker M ~1.4 px from the truth; 2-marker diagonal fit from corners with +-2 px noise
    M_track = np.linalg.inv(TILT) @ np.float64([[1, 0, 1], [0, 1, -1], [0, 0, 1]])
    tc, thr, exceeded, guarded = _true_corners(TILT), TRUST_FRAC * OCAM[0], 0, 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        for pair in ((0, 2), (1, 3)):
            corners = {i: (tc[i] + rng.uniform(-2, 2, (4, 2))).astype(np.float32) for i in pair}
            M_pair = homography_from_corners(corners, OTARGETS)
            d = np.linalg.norm(screen_quad(M_pair, OW, OH) - screen_quad(M_track, OW, OH), axis=1).mean()
            M, source, reset = choose_overlay_M(corners, OTARGETS, M_track, OW, OH, OCAM[0])
            # the consistency guard rejects a noisy fit whose own corners sit > REPROJ_PX off it: the tracker M
            rejected = fit_markers(corners, OTARGETS, OW, OH)[0] is None
            guarded += rejected
            if d > thr:
                exceeded += 1
            if d > thr or rejected:
                assert source == "tracking" and not reset and np.array_equal(M, M_track), (seed, pair, d)
            else:
                assert source == "markers" and np.array_equal(M, M_pair), (seed, pair, d)
    print(f"  diagonal pairs, 600 noisy fits: {exceeded} disagree by more than {thr:.1f} px and keep the tracker M; "
          f"{guarded} rejected by the consistency guard")
    assert exceeded >= 10, f"only {exceeded} fits exceeded the threshold: the rule was not exercised"
    # same-side pair with no tracker M: used, and it restarts the tracker; diagonal pair: used, no restart
    _, source, reset = choose_overlay_M({i: tc[i] for i in (0, 1)}, OTARGETS, None, OW, OH, OCAM[0])
    assert source == "markers" and reset
    _, source, reset = choose_overlay_M({i: tc[i] for i in (0, 2)}, OTARGETS, None, OW, OH, OCAM[0])
    assert source == "markers" and not reset
    _, source, reset = choose_overlay_M({i: tc[i] for i in range(4)}, OTARGETS, M_track, OW, OH, OCAM[0])
    assert source == "markers" and reset


def test_guided_timing_two_missing():
    img = render_overlay(blur=9, jpeg=60)
    found = {i: c for i, c in _true_corners().items() if i in (0, 2)}  # 1 and 3 treated as not decoded
    M_est = np.linalg.inv(OG)
    assert len(guided_redetect(img, M_est, OTARGETS, found)) == 2  # both re-detected, so both ROIs ran
    n, t0 = 50, time.perf_counter()
    for _ in range(n):
        guided_redetect(img, M_est, OTARGETS, found)
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  guided re-detection, 2 missing, 720x1280: {ms:.2f} ms extra per frame (average of {n})")
    assert ms < 10, f"{ms:.2f} ms"


def _still_finger(fu, x, y, t0, t1, on=True):
    for t in np.arange(t0, t1, 1 / 30):
        fu.add_fingertip(float(t), x, y, on)


def test_stale_tap_dropped():
    fu = TouchFuser()  # a perfect still finger, but the tap is 2.5 s old when decided: a stall, so no click
    _still_finger(fu, 500, 300, 0.0, 3.0)
    fu.add_tap(0.5)
    assert fu.poll(3.0) == [] and not fu.taps


def test_touch_point_settle_and_window():
    fu = TouchFuser()  # on-screen samples within 0.12 s of the tap count; the point is the fingertip at the tap time
    tap = 1.0
    for t in (tap - 0.1, tap - 0.05, tap, tap + 0.05, tap + 0.1):
        fu.add_fingertip(t, 500, 300, True)
    for t in (tap - 0.13, tap + 0.13):  # just outside the window: not counted
        fu.add_fingertip(t, 900, 600, True)
    fu.add_tap(tap)
    assert fu.poll(tap + TOUCH_SETTLE - 0.01) == [] and len(fu.taps) == 1  # not decided before TOUCH_SETTLE
    clicks = fu.poll(tap + TOUCH_SETTLE + 1e-6)
    assert len(clicks) == 1 and np.hypot(clicks[0][0] - 500, clicks[0][1] - 300) < 1, clicks
    assert fu.decisions[0]["n"] == 5 and fu.decisions[0]["point"] == (500.0, 300.0), fu.decisions[0]


def test_two_samples_click_and_no_samples_do_not():
    fu = TouchFuser()  # two on-screen samples are enough; the point is the later one at the tap time
    fu.add_fingertip(0.95, 500, 300, True)
    fu.add_fingertip(1.0, 502, 301, True)
    fu.add_tap(1.0)
    clicks = fu.poll(1.3)
    assert len(clicks) == 1 and np.hypot(clicks[0][0] - 502, clicks[0][1] - 301) < 1, clicks
    fu = TouchFuser()  # samples only outside the window: no point to click
    fu.add_fingertip(0.5, 500, 300, True)
    fu.add_tap(1.0)
    assert fu.poll(1.3) == [] and fu.decisions[0]["outcome"] == "no-samples", fu.decisions


def test_keyboard_samples_off_screen_rejected():
    fu = TouchFuser()  # the fingertip is over the keyboard around the tap: no click
    for t in (0.9, 0.95, 1.0, 1.05, 1.1):
        fu.add_fingertip(t, 500, 300, False)
    fu.add_tap(1.0)
    assert fu.poll(1.3) == [] and fu.decisions[0]["outcome"].startswith("off-screen"), fu.decisions


def test_live_moving_taps_click_at_contact_point():
    # the three 'moving' taps from the live log: samples about 75 ms apart (the tap at a sample, or between two)
    cases = [  # (sample times, screen xy, tap time, expected point)
        ((0.925, 1.0, 1.075), [(888, 327), (948, 378), (993, 436)], 1.0, (948, 378)),
        ((0.85, 0.925, 1.0, 1.075), [(655, 581), (614, 547), (593, 507), (590, 482)], 0.96, (604.2, 528.3)),
        ((0.9, 0.925, 0.95, 0.975, 1.0), [(496, 305), (508, 317), (523, 330), (534, 343), (528, 327)], 0.95, (523, 330)),
    ]
    for times, xy, tap, want in cases:
        fu, eng = TouchFuser(), GestureEngine()
        for t, (x, y) in zip(times, xy):
            fu.add_fingertip(t, x, y, True)
            eng.frame(t, x, y, True, False)
        fu.add_tap(tap)
        clicks = fu.poll(tap + 0.3)
        assert len(clicks) == 1 and fu.decisions[0]["outcome"] == "CLICK", fu.decisions
        assert np.hypot(clicks[0][0] - want[0], clicks[0][1] - want[1]) <= 1.0, (clicks, want)
        acts = eng.touch_down(*fu.downs[0])  # the engine: a withdrawal is a click, never a drag or a hold
        acts += eng.frame(tap + 0.4, xy[-1][0], xy[-1][1], True, False)  # still past the hold: no second click
        assert [a[0] for a in acts] == ["click"], acts


def test_ringing_gives_one_event_and_one_click():
    # a knock with a second spike 0.3 s later (ringing) is one tap event; a second knock 0.5 s later is two
    rng = np.random.default_rng(6)
    t = np.arange(0, 5, 0.001)
    mag = 1.0 + rng.uniform(-0.001, 0.001, t.size)
    for t0 in (1.0, 1.3, 3.0, 3.5):  # spikes: knock at 1.0 rings at 1.3; a separate knock at 3.0, 3.5
        for dt in (0.0, 0.03):
            i = int(round((t0 + dt) * 1000))
            mag[i:i + 5] += 0.3
    det = TapDetector()
    events = [e for ti, m in zip(t, mag) for e in det.feed(ti, m)]
    assert len(events) == 3 and abs(events[0] - 1.0) < 0.002 and abs(events[1] - 3.0) < 0.002, events
    fu = TouchFuser()  # two taps 0.26 s apart on a still finger: one click
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_tap(1.0)
    fu.add_tap(1.26)
    assert len(fu.poll(1.8)) == 1 and not fu.taps


def test_malformed_datagrams_ignored():
    sock = open_touch_socket(0)
    port = sock.getsockname()[1]
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    bad = [b'{"t": true}', b'{"t": "1.5"}', b'{"t": NaN}', b'{"t": Infinity}', b'{"t": -Infinity}',
           b'{"t": 1e400}', b'{"t": ' + b"9" * 400 + b'}', b'{"t": null}', b"[1, 2]"]
    for d in bad:
        tx.sendto(d, ("127.0.0.1", port))
    tx.sendto(b'{"t": 12.5}', ("127.0.0.1", port))
    fuser = TouchFuser()
    drain_taps(sock, fuser)
    assert list(fuser.taps) == [(12.5, None, {})], list(fuser.taps)
    tx.close()
    sock.close()


def test_drain_at_most_64_per_frame():
    sock = open_touch_socket(0)
    port = sock.getsockname()[1]
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for k in range(100):
        tx.sendto(b'{"t": %d}' % (k + 1), ("127.0.0.1", port))
    fuser = TouchFuser()
    drain_taps(sock, fuser)
    assert len(fuser.taps) == 64, len(fuser.taps)
    drain_taps(sock, fuser)
    assert len(fuser.taps) == 100, len(fuser.taps)
    tx.close()
    sock.close()


def test_bind_failure_names_port_and_imu_taps():
    holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    holder.bind(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    try:
        open_touch_socket(port)
    except SystemExit as e:
        msg = str(e)
        assert str(port) in msg and "imu_taps.py" in msg, msg
    else:
        raise AssertionError("bind on a busy port did not exit")
    finally:
        holder.close()


def test_capture_time_shift():
    assert abs(capture_time(10.0, 0.3) - 9.7) < 1e-12
    assert capture_time(10.0, 0.0) == 10.0


def test_hover_moves_cursor_only_without_touch():
    assert hover_moves_cursor(parse_args(["--source", "0", "--touch"])) is False
    assert hover_moves_cursor(parse_args(["--source", "0"])) is True


def test_touch_decisions_per_reason():
    def fuse(samples, tap=1.0, peak=None, now=None):
        fu = TouchFuser()
        for t, x, y, on in samples:
            fu.add_fingertip(t, x, y, on)
        fu.add_tap(tap, peak)
        fu.poll(tap + 0.5 if now is None else now)
        return fu.decisions
    still = [(1.0 + k * 0.03, 500, 300, True) for k in range(-3, 4)]  # 7 samples around the tap at 1.0
    cases = {
        "CLICK": still,
        "off-screen": [(0.9, 500, 300, True), (0.95, 500, 300, True)] + [(t, 500, 300, False) for t in (1.0, 1.05, 1.1)],
        "sweeping": [(1.0 + k * 0.03, 500 + 1800 * k * 0.03, 300, True) for k in range(-3, 4)],  # 324 pt across
    }
    for name, samples in cases.items():
        (d,) = fuse(samples, peak=3.5)
        assert d["outcome"].split()[0] == name, (name, d["outcome"])
        assert d["peak"] == 3.5 and d["n"] == len(samples) and d["ctx_n"] == len(samples), d
        assert d["nearest_ms"] is not None and abs(d["nearest_ms"]) <= 90, d
        line = format_decision(d)
        assert line.startswith("touch tap=1.000 " + d["outcome"]) and "peak=3.5" in line, line
    (d,) = fuse(still, now=3.01)  # just over 2 s old: stale, whatever the finger did
    assert d["outcome"] == "stale", d["outcome"]
    fu = TouchFuser()  # ringing: the second tap 0.2 s after an accepted click is a gap
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_tap(1.0)
    fu.add_tap(1.2)
    fu.poll(1.6)
    assert [d["outcome"] for d in fu.decisions] == ["CLICK", "gap"], [d["outcome"] for d in fu.decisions]
    assert fu.decisions[1]["peak"] is None and "peak=-" in format_decision(fu.decisions[1])


def _frames(t_from, t_to, pos, on=lambda t: True, two=False, span=None):
    """Camera frames at 15 fps, capture times from t_from to t_to: ("frame", t, (x, y), on, two, hand size or None)."""
    out, k = [], round(t_from * 15)
    while k / 15 <= t_to + 1e-9:
        t = round(k / 15, 6)
        out.append(("frame", t, pos(t), on(t), two, span(t) if span else None))
        k += 1
    return out


def _with_tap(events, *taps):
    """Insert each tap datagram once its frame would have arrived (tap + 50 ms)."""
    for tap in taps:
        i = next(i for i, e in enumerate(events) if e[0] == "frame" and e[1] >= tap + 0.05)
        events = events[:i] + [("tap", tap)] + events[i:]
    return events


def _accepted_at(tap):
    """The camera frame at which TouchFuser decides a tap: the first one TOUCH_SETTLE after it (15 fps grid)."""
    return round(float(np.ceil(round((tap + TOUCH_SETTLE) * 15, 9))) / 15, 6)


def _drive(events):
    """Feed frames, taps and ticks through touch_step (as run() does). Returns ([(time, action)], engine).
    ("tick", t) calls engine.tick directly: a camera stall with no frame."""
    eng, fuser, out = GestureEngine(), TouchFuser(), []
    for ev in events:
        if ev[0] == "tap":
            fuser.add_tap(ev[1])
            continue
        if ev[0] == "tick":
            acts = eng.tick(ev[1])
            t = ev[1]
        else:
            _, t, tip, on, two, *rest = ev
            acts = touch_step(eng, fuser, t, None if tip is None else tip, on, two, rest[0] if rest else None)
        out += [(t, a) for a in acts]
    return out, eng


def _swipe(dist=200.0, t0=0.1, speed=600.0):
    """Finger rests at (500, 300); 50 ms after the tap it moves `speed` pt/s (live drags: 500-1000) until `dist`
    points, then rests."""
    return lambda t: (500.0 + min(speed * max(t - t0 - 0.05, 0.0), dist), 300.0)


def _slow(slow_end, t0=0.1, hold=0.35):
    """Rests `hold` s after the touch (a drag rests before it slides; moving at once is lift-off), then a fast start
    (72 pt, past GESTURE_DRAG_MOVE), then a steady 450 pt/s (above GESTURE_LIFT_SPEED: still dragging) until capture
    time slow_end, then rests (a pause: the drop)."""
    def pos(t):
        s_ = max(min(t - t0 - hold, slow_end - t0 - hold), 0.0)
        return (500.0 + 600.0 * min(s_, 0.12) + 450.0 * max(s_ - 0.12, 0.0), 300.0)
    return pos


def test_gesture_400_still_taps_one_click_each():
    rng = np.random.default_rng(7)  # 400 taps, 1 s apart; still finger; 30-60 ms processing behind each frame
    eng, fuser = GestureEngine(), TouchFuser()
    taps = [1.0 + i for i in range(400)]
    pending, acts, k = list(taps), [], 0
    while k / 15 < taps[-1] + 1.5:
        cap = round(k / 15 + rng.uniform(0.03, 0.06), 6)
        k += 1
        while pending and pending[0] + 0.05 <= cap:
            fuser.add_tap(pending.pop(0))
        tip = (500 + rng.uniform(-1, 1), 300 + rng.uniform(-1, 1))
        acts += [(cap, a) for a in touch_step(eng, fuser, cap, tip, True, False)]
    assert not pending
    assert not [a for _, a in acts if a[0] != "click"], [a for _, a in acts if a[0] != "click"][:3]
    good = sum(1 for tap in taps if sum(1 for t, a in acts if a[0] == "click" and tap <= t < tap + 1.0) == 1)
    print(f"  400 still taps: {good} with exactly one click, no down")
    assert good >= 396, good


def test_gesture_hold_drag_down_moves_one_up():
    # a tap (a click at once) whose finger rests 0.35 s, then slides 400 pt: down at the tap point, moves, one up
    timed, eng = _drive(_with_tap(_frames(0.0, 3.0, _swipe(t0=1.75, dist=400.0)), 1.45))
    kinds = [a[0] for _, a in timed]
    assert kinds[:2] == ["click", "down"] and kinds.count("up") == 1 and kinds[-1] == "up", kinds
    assert kinds.count("move") > 6 and "scroll" not in kinds and kinds.count("click") == 1, kinds
    assert timed[0] == (_accepted_at(1.45), ("click", 500.0, 300.0)), timed[0]  # the tap clicks at once
    assert timed[1][0] >= _accepted_at(1.45) and timed[1][1][0] == "down", timed[1]  # once it starts sliding
    assert abs(timed[1][1][1] - 500) < 1 and abs(timed[1][1][2] - 300) < 1, timed[1]  # the second touch's point
    t_up = next(t for t, a in timed if a[0] == "up")
    assert 2.47 <= t_up <= 3.0, t_up  # the finger stops at 2.47 s: the backup stop rule drops it ~0.45 s later
    assert not eng.held


def test_gesture_short_hold_drags():
    # user: the hold felt long. Resting ~0.22 s (past GESTURE_HOLD 0.15) then sliding is a drag; with the old 0.3 s
    # hold the slide came inside the hold and read as lift-off (the click only)
    timed, eng = _drive(_with_tap(_frames(0.0, 3.0, _swipe(t0=1.45 + 0.17, dist=400.0)), 1.45))
    kinds = [a[0] for _, a in timed]
    assert kinds[:2] == ["click", "down"] and kinds.count("up") == 1 and kinds.count("click") == 1, kinds
    assert not eng.held


def test_gesture_double_tap_lift_off_is_a_double_click():
    # live: after the 2nd tap the finger leaves at once (20-50 pt per frame) and that read as a drag; moving within
    # GESTURE_LIFTOFF of the touch is lift-off: a double-click at the first tap's spot, nothing held
    timed, eng = _drive(_with_tap(_frames(0.0, 2.2, _swipe(t0=1.45 - 0.04)), 1.0, 1.45))
    assert [a for _, a in timed] == [("click", 500.0, 300.0), ("dclick", 500.0, 300.0)], timed
    assert not eng.held


def test_gesture_double_tap_wobble_is_not_a_drag():
    # live: after a resting 2nd tap the hovering hand wobbled 25-40 pt around the touch point: a double-click, no down
    wobble = lambda t: (500.0, 300.0) if t < 1.75 else (500.0 - 30.0 * math.sin(9 * (t - 1.75)), 300.0 + 25.0 * math.sin(7 * (t - 1.75)))
    timed, eng = _drive(_with_tap(_frames(0.0, 2.6, wobble), 1.0, 1.45))
    assert [a[0] for _, a in timed] == ["click", "dclick"], timed
    assert not eng.held


def test_gesture_double_tap_still_is_two_clicks():
    # two taps 0.45 s apart, no move: a click, then a real double-click at the first tap's spot when the second
    # tap's window closes (macOS only counts clicks at nearly the same point), and no down
    timed, eng = _drive(_with_tap(_frames(0.0, 2.2, lambda t: (500.0, 300.0)), 1.0, 1.45))
    assert [a for _, a in timed] == [("click", 500.0, 300.0), ("dclick", 500.0, 300.0)], timed
    assert timed[0][0] == _accepted_at(1.0), timed
    assert timed[1][0] == _accepted_at(1.45), timed  # the double-click goes out as soon as the 2nd tap is accepted
    assert not eng.held and eng.state is None
    # live: the 2nd tap landed 50-80 pt off the first: still a double-click, at the first spot
    pos = lambda t: (500.0, 300.0) if t < 1.3 else (560.0, 350.0)
    timed, _ = _drive(_with_tap(_frames(0.0, 2.2, pos), 1.0, 1.45))
    assert [a for _, a in timed] == [("click", 500.0, 300.0), ("dclick", 500.0, 300.0)], timed
    # a third tap right after is a fresh click, not another double
    timed, _ = _drive(_with_tap(_frames(0.0, 2.8, lambda t: (500.0, 300.0)), 1.0, 1.45, 2.0))
    assert [a[0] for _, a in timed] == ["click", "dclick", "click"], timed


def test_gesture_taps_one_second_apart_are_two_clicks():
    # a second tap 1.0 s later is not a press: an immediate click each, nothing held
    timed, eng = _drive(_with_tap(_frames(0.0, 3.0, lambda t: (500.0, 300.0)), 1.0, 2.0))
    assert [a[0] for _, a in timed] == ["click", "click"], timed
    assert [t for t, _ in timed] == [_accepted_at(1.0), _accepted_at(2.0)], timed
    assert not eng.held and eng.state in (None, "hold"), eng.state  # "hold": still watching for a slide, no button


def test_gesture_two_finger_scroll_ends_when_pose_lost():
    # two fingers swipe, then at 1.0 s the two-finger pose is lost while the hand keeps moving: scrolling stops
    frames = _frames(0.0, 2.0, lambda t: (500.0 + 600.0 * max(t - 0.15, 0.0), 300.0), two=True)
    frames = [("frame", t, tip, on, two and t < 1.0, sp) for _, t, tip, on, two, sp in frames]
    timed, eng = _drive(_with_tap(frames, 0.1))
    kinds = {a[0] for _, a in timed}
    assert kinds <= {"point", "scroll"} and "scroll" in kinds, kinds  # no click, down or up
    scrolls = [t for t, a in timed if a[0] == "scroll"]
    assert max(scrolls) >= 0.9, scrolls[-3:]  # it scrolled while the pose held
    assert max(scrolls) <= 1.0 + 1 / 15 + 1e-6, scrolls[-3:]  # nothing after the second frame without the pose
    assert eng.state is None and not eng.held


def test_gesture_tap_then_withdrawal_is_one_click():
    def pos(t):  # rests, then withdraws 200 pt in 0.2 s, then rests again
        return (500.0 + 200 * min(max(t - 1.05, 0.0) / 0.2, 1.0), 300.0)
    timed, eng = _drive(_with_tap(_frames(0.0, 2.0, pos), 1.0))
    assert timed == [(_accepted_at(1.0), ("click", 500.0, 300.0))], timed  # one click, the moment the tap is accepted
    assert not eng.held


def test_gesture_drag_stays_held_while_moving():
    timed, _ = _drive(_with_tap(_frames(0.0, 3.2, _slow(2.6, t0=0.3)), 0.3))  # tap, then a second touch: a press
    kinds_t = [(t, a[0]) for t, a in timed]
    assert [k for _, k in kinds_t].count("up") == 1, kinds_t
    assert not [t for t, k in kinds_t if k == "up" and t < 2.6], "released while still moving"
    assert sum(1 for t, k in kinds_t if k == "move" and 1.2 <= t <= 2.5) > 10, "no moves during the drag"
    t_up = next(t for t, k in kinds_t if k == "up")
    assert 2.6 <= t_up <= 3.15, t_up  # the finger stops at 2.6 s; no size data, so the backup stop rule (0.45 s)


def test_gesture_drag_drops_at_pause_before_pull_away():
    # live: drags ended with a 1-3 frame pause, then the finger's pull-away jumped 50-80 pt and kept dragging.
    # Now the pause drops it: one up, at the pause point, and nothing moves after it
    def path(t):
        if t < 0.95:
            return (500.0, 300.0)                      # 2nd touch at 0.6 s, rests on the glass
        if t < 1.6:
            return (500.0 + 800.0 * (t - 0.95), 300.0)  # drags right at 800 pt/s (~53 pt per frame)
        if t < 1.75:
            return (1100.0, 300.0)                     # a short pause at the drop point
        return (1100.0 + 900.0 * (t - 1.75), 300.0 - 600.0 * (t - 1.75))  # pull-away
    timed, eng = _drive(_with_tap(_frames(0.0, 2.4, path, span=lambda t: 500.0 * (1 + 4 * max(t - 1.75, 0.0))), 0.6))
    kinds = [a[0] for _, a in timed]
    assert kinds.count("down") == 1 and kinds.count("up") == 1, kinds
    i_up = kinds.index("up")
    assert "move" not in kinds[i_up:], kinds  # nothing follows the pull-away
    up = timed[i_up][1]
    assert abs(up[1] - 1100) < 50 and abs(up[2] - 300) < 30, up  # dropped where the finger paused (one pull-away
    # frame can slip in before the size check sees two frames)
    assert not eng.held


def test_gesture_two_finger_swipe_scrolls_no_button():
    def jitter(t):  # 1 px jitter while resting: under the scroll deadband, so it must not scroll
        return (_swipe()(t)[0] + (1.0 if 0.85 < t < 1.0 and round(t * 15) % 2 else 0.0), 300.0)
    timed, _ = _drive(_with_tap(_frames(0.0, 2.0, jitter, two=True), 0.1))
    acts = [a for _, a in timed]
    kinds = {a[0] for a in acts}
    assert kinds <= {"point", "scroll"} and "scroll" in kinds, kinds  # no down, up or click
    total_dx = sum(a[1] for a in acts if a[0] == "scroll")
    assert abs(total_dx - (-200 * SCROLL_GAIN)) <= 0.2 * 200 * SCROLL_GAIN, total_dx  # content follows the finger
    rest = [a[1] for t, a in timed if a[0] == "scroll" and 0.85 < t < 1.0]
    assert abs(sum(rest)) <= 1, rest  # at most the sub-deadband residual of the swipe; jitter itself scrolls nothing


def test_gesture_scroll_with_sideways_jitter_stays_vertical():
    # live: the fingertip jumps 30-150 pt between frames; a stroke up whose first frame also jumped 40 pt sideways
    # locked the whole scroll sideways (16 of 21 scrolls in one session)
    def path(t):
        if t < 0.15:
            return (500.0, 300.0)
        return (540.0, 300.0 - 700.0 * (t - 0.15))  # 40 pt sideways jump, then straight up
    timed, eng = _drive(_with_tap(_frames(0.0, 0.8, path, two=True), 0.1))
    sc = [a for _, a in timed if a[0] == "scroll"]
    assert sc and all(a[1] == 0 for a in sc), sc  # never sideways
    assert sum(a[2] for a in sc) < -20, sc  # up stroke: content follows it
    assert any("vertical" in n for n in eng.notes), eng.notes


def test_gesture_scroll_survives_hand_size_noise():
    # live: 15 of 21 scrolls ended after 1-8 steps on "hand grew x1.13-1.48"; real lifts grow < 8 %
    def path(t):
        return (500.0, 300.0) if t < 0.15 else (500.0, 300.0 - 700.0 * (t - 0.15))
    timed, eng = _drive(_with_tap(_frames(0.0, 0.85, path, two=True, span=lambda t: 650.0 if t > 0.4 else 500.0), 0.1))
    total = -sum(a[2] for _, a in timed if a[0] == "scroll")
    assert total >= 0.8 * 700.0 * 0.65 * SCROLL_GAIN, total  # the whole stroke scrolled, not cut at the size jump
    assert not any("hand grew" in n for n in eng.notes), eng.notes


def test_marker_toggle_shows_when_lost_hides_when_content_holds():
    probes = []
    def probe(ok):
        def f():
            probes.append(ok)
            return ok
        return f
    def run(tg, t, secs, source, ok):  # frames at 15 fps for `secs` s; returns (end time, [(t, action)])
        out, end = [], t + secs
        while t < end:
            t = round(t + 1 / 15, 6)
            out.append((t, tg.step(t, source, probe(ok))))
        return t, [(tt, a) for tt, a in out if a]
    tg = tracker.MarkerToggle(0.0)
    t, acts = run(tg, 0.0, 2.0, "content", True)  # found from content: the markers stay hidden
    assert not acts and not tg.shown and not probes, acts
    t_lost = t
    t, acts = run(tg, t, 1.0, "tracking", False)  # lost: shown once, about MARKER_SHOW_AFTER later
    assert [a for _, a in acts] == ["show"] and tg.shown, acts
    assert tracker.MARKER_SHOW_AFTER < acts[0][0] - t_lost <= tracker.MARKER_SHOW_AFTER + 0.1, acts
    t1 = t
    t, acts = run(tg, t, 10.0, "markers", False)  # a plain page: content alone can't find it, the markers stay
    assert not acts and tg.shown, acts
    assert len(probes) <= 10.0 / tracker.MARKER_PROBE_EVERY + 1, len(probes)  # checks are rationed
    t2 = t
    t, acts = run(tg, t, 2.6, "markers", True)  # content finds it, but one check fails partway: count restarts
    t, acts2 = run(tg, t, tracker.MARKER_PROBE_EVERY + 0.1, "markers", False)  # long enough for a check to land in it
    t3 = t
    t, acts3 = run(tg, t, 5.0, "markers", True)
    assert not acts and not acts2, (acts, acts2)
    assert [a for _, a in acts3] == ["hide"] and not tg.shown, acts3
    took = acts3[0][0] - t3
    assert (tracker.MARKER_HIDE_AFTER - 1) * tracker.MARKER_PROBE_EVERY <= took <= (tracker.MARKER_HIDE_AFTER + 1) * \
        tracker.MARKER_PROBE_EVERY, took


def test_screen_matcher_ignores_marker_windows():
    # with markers on screen, content matching must not lean on them (on-demand markers would hide, lose, show...)
    rng = np.random.default_rng(3)
    shot = cv2.cvtColor(cv2.GaussianBlur((rng.random((900, 1440)) * 255).astype(np.uint8), (3, 3), 0), cv2.COLOR_GRAY2BGR)
    m = tracker.ScreenMatcher(1440, 900, grab=lambda: shot)
    m.ignore = [np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])]  # screen points
    m._refresh(0.0)
    assert m.pts is not None and len(m.pts) > 100, m.pts
    inside = [(x, y) for x, y in m.pts if x < 300 and y < 300]
    assert not inside, inside[:5]


def test_gesture_stall_releases_held_drag():
    timed, eng = _drive(_with_tap(_frames(0.0, 1.0, _slow(3.0, t0=0.3)), 0.3))  # frames stop at 1.0 s
    assert eng.held
    t = 1.0
    while t < 2.0:
        t = round(t + 1 / 30, 6)
        timed += [(t, a) for a in eng.tick(t)]
    ups = [t for t, a in timed if a[0] == "up"]
    assert len(ups) == 1 and 1.6 <= ups[0] <= 1.7, ups


def test_gesture_0p4s_frame_gap_keeps_drag():
    ev = _with_tap(_frames(0.0, 1.0667, _slow(3.0, t0=0.3)), 0.3)  # rests, slides from 0.85: a drag by 1.0
    ev += [("tick", round(1.0667 + k / 30, 6)) for k in range(1, 13)]  # 0.4 s with no frame, ticks as in run()
    ev += _frames(1.4667, 2.7, _slow(3.0, t0=0.6))
    timed, eng = _drive(ev)
    assert [a[0] for t, a in timed if t < 1.07].count("down") == 1, timed  # the drag began before the gap
    assert not [t for t, a in timed if a[0] == "up" and t < 2.7], "the gap released the drag"
    assert eng.held and sum(1 for t, a in timed if a[0] == "move" and t > 1.4) > 5


def test_gesture_forced_up_after_max():
    timed, _ = _drive(_with_tap(_frames(0.0, 12.0, _slow(12.5, t0=0.3)), 0.3))  # still moving: only the cap can end it
    start = 0.8667  # the frame the held tap is confirmed sliding (two frames past GESTURE_DRAG_MOVE): the cap starts
    ups = [t for t, a in timed if a[0] == "up"]
    assert len(ups) == 1 and start + GESTURE_MAX - 1 / 15 <= ups[0] <= start + GESTURE_MAX + 0.2, (start, ups)
    assert timed[-1][1][0] == "up", timed[-3:]  # nothing after the forced release


def test_gesture_lost_hand_lifts():
    ev = _with_tap(_frames(0.0, 1.0, _slow(3.0, t0=0.3)), 0.3)  # rests, slides from 0.85: a drag by 1.0
    ev += _frames(1.0667, 2.2, _slow(3.0, t0=0.6), on=lambda t: False)  # hand gone from 1.07 s
    timed, _ = _drive(ev)
    ups = [t for t, a in timed if a[0] == "up"]
    assert len(ups) == 1 and 1.6 <= ups[0] <= 1.9, ups


def test_gesture_tap_while_dragging_is_ignored():
    # live: pressing or sliding on the lid fired the tap sensor 0.1-0.35 s into a drag; that "tap" ended the drag,
    # clicked, and near the first tap even double-clicked it. A tap while a drag is held changes nothing
    path = _slow(2.0, t0=0.3)
    ev = _with_tap(_frames(0.0, 1.0, path), 0.3)   # held, then sliding by ~0.9 s
    ev.append(("tap", 0.98))                        # a sensor "tap" mid-drag, near the first tap
    ev += _frames(1.0667, 2.6, path)
    timed, eng = _drive(ev)
    kinds = [a[0] for _, a in timed]
    assert kinds.count("down") == 1 and kinds.count("up") == 1, kinds
    i_down, i_up = kinds.index("down"), kinds.index("up")
    assert not {"click", "dclick"} & set(kinds[i_down:]), kinds  # no click or double-click from it
    assert timed[i_up][0] > 2.0, timed[i_up]  # the drag carried on to the finger's stop at 2.0 s
    assert any("tap ignored" in n for n in eng.notes), eng.notes

def test_release_touch_reaches_mouse_up_on_exception():
    import types
    import tracker as T
    real_pg, real_perform = T.pyautogui, T.perform_gestures

    class FakePG(types.SimpleNamespace):
        pass
    fake = FakePG(FAILSAFE=True, FailSafeException=real_pg.FailSafeException, ups=0, fail_up=0,
                  size=lambda: (1512, 982))

    def mouse_up(*a, **k):
        fake.ups += 1
        if fake.fail_up:
            fake.fail_up -= 1
            raise fake.FailSafeException("parked on a corner")
    fake.mouseUp = mouse_up
    args = types.SimpleNamespace(dry_run=False, touch_debug=False)
    eng = _drive(_with_tap(_frames(0.0, 1.0, _slow(3.0, t0=0.3)), 0.3))[1]  # a drag still held
    assert eng.held
    try:
        T.pyautogui = fake
        T.perform_gestures = lambda acts, args: (_ for _ in ()).throw(RuntimeError("boom"))  # a non-FailSafe error
        try:
            release_touch(eng, args)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the error was swallowed")
        assert fake.ups == 1, fake.ups
        # the FailSafe path: the release action's mouseUp is parked, the retry in the finally releases
        T.perform_gestures = real_perform
        fake.ups, fake.fail_up, fake.FAILSAFE = 0, 2, True  # parked: the release action and the finally both raise
        eng2 = _drive(_with_tap(_frames(0.0, 1.0, _slow(3.0, t0=0.3)), 0.3))[1]
        release_touch(eng2, args)
        assert fake.ups == 3 and fake.FAILSAFE is False, (fake.ups, fake.FAILSAFE)
    finally:
        T.pyautogui, T.perform_gestures = real_pg, real_perform


def test_drift_trap_markers_win_after_three_consistent_fits():
    # the tracker has drifted by 10% of the frame width (72 camera px); the marker fits are right and agree
    M_true = np.linalg.inv(OG)
    M_track = M_true @ np.float64([[1, 0, 72], [0, 1, 0], [0, 0, 1]])
    tc = _true_corners()
    diag = {0: tc[0], 2: tc[2]}
    drift = {}
    out = [choose_overlay_M(diag, OTARGETS, M_track, OW, OH, OCAM[0], drift) for _ in range(3)]
    assert [o[1] for o in out] == ["tracking", "tracking", "markers"] and out[2][2] is True, [o[1:] for o in out]
    assert np.abs(screen_quad(out[2][0], OW, OH) - screen_quad(M_true, OW, OH)).max() < 1.0
    # a same-side pair that agrees with the previous fit resets at once
    drift = {}
    side = {0: tc[0], 1: tc[1]}
    out = [choose_overlay_M(side, OTARGETS, M_track, OW, OH, OCAM[0], drift) for _ in range(2)]
    assert [o[1] for o in out] == ["tracking", "markers"] and out[1][2] is True, [o[1:] for o in out]


def test_hand_covering_marker_one_drops_only_it():
    tc = _true_corners()
    c = tc[1].mean(axis=0)
    hand = (c + np.float32([[-10, -10], [10, -10], [10, 10], [-10, 10]])).astype(np.float32)  # over marker 1
    kept, dropped = drop_occluded(tc, hand, _side(tc))
    assert sorted(dropped) == [1] and sorted(kept) == [0, 2, 3], (sorted(kept), sorted(dropped))
    far = np.float32([[300, 60], [420, 60], [420, 140], [300, 140]])  # above the screen, well clear of every marker
    kept, dropped = drop_occluded(tc, far, _side(tc))
    assert not dropped and sorted(kept) == [0, 1, 2, 3], (sorted(kept), sorted(dropped))
    assert drop_occluded(tc, None, _side(tc))[1] == {}


def _displaced(c, rng, px=15.0):
    """Corners c with each one moved px camera px in its own random direction (uneven, as an occluder leaves them)."""
    ang = rng.uniform(0, 2 * np.pi, len(c))
    return (c + px * np.stack([np.cos(ang), np.sin(ang)], 1)).astype(np.float32)


def test_marker_with_displaced_corners_excluded_by_guard():
    # Not every case is caught: the 3-marker fit can absorb an uneven 15 px displacement (reprojection < 6 px) and
    # then its screen corners are off by tens of points. A uniform shift of a whole marker is absorbed the same way.
    # So this checks the rate: most cases excluded and then exact, and the misses are counted and printed.
    tc = _true_corners()
    scr = np.float32([[0, 0], [OW, 0], [OW, OH], [0, OH]])
    cam = apply(OG, scr)  # true camera positions of the screen corners
    total = caught = 0
    for seed in range(50):
        for bad in (0, 1, 2):
            rng = np.random.default_rng(seed)
            seen = {i: tc[i] for i in (0, 1, 2)}
            seen[bad] = _displaced(tc[bad], rng)
            M, used = fit_markers(seen, OTARGETS, OW, OH)
            total += 1
            if M is not None and sorted(used) == sorted({0, 1, 2} - {bad}):
                caught += 1
                err = np.abs(apply(M, cam) - scr).max()
                assert err < 5, f"seed {seed} bad {bad}: excluded, yet screen corners off by {err:.2f} pt"
    print(f"  displaced-corner occlusion, 3 markers: guard excluded the bad marker in {caught} of {total}")
    assert caught >= 0.95 * total, f"only {caught} of {total} excluded"
    # two markers, one displaced: no refit possible, so the marker M is rejected and the tracker M is used
    M_track = np.linalg.inv(OG)
    bad2 = {0: tc[0], 2: _displaced(tc[2], np.random.default_rng(0))}
    M, source, _ = choose_overlay_M(bad2, OTARGETS, M_track, OW, OH, OCAM[0])
    assert source == "tracking" and np.array_equal(M, M_track), source


def _jump_corners(dx=80):
    """Markers as seen after the screen moved dx camera px in one frame, and the M that matches them."""
    G_jump = np.float64([[1, 0, dx], [0, 1, 0], [0, 0, 1]]) @ OG
    return {i: c for i, c in _true_corners(G_jump).items() if i in (0, 2)}, np.linalg.inv(G_jump)


def test_one_frame_jump_with_two_markers_keeps_tracker():
    corners, M_jump = _jump_corners()
    M_prev = np.linalg.inv(OG)  # tracker agrees with the previous M, the marker pair does not
    M, source, reset = choose_overlay_M(corners, OTARGETS, M_prev, OW, OH, OCAM[0], {}, M_prev)
    assert source == "tracking" and not reset and np.array_equal(M, M_prev), source
    # if the tracker moved with the markers, the change is real (not a jump the tracker disagrees with): markers win
    M, source, reset = choose_overlay_M(corners, OTARGETS, M_jump, OW, OH, OCAM[0], {}, M_prev)
    assert source == "markers" and reset and np.allclose(M, M_jump, rtol=1e-4, atol=1e-6), source


def test_persistent_jump_markers_win_after_drift_frames():
    corners, M_jump = _jump_corners()
    M_prev, drift = np.linalg.inv(OG), {}
    out = [choose_overlay_M(corners, OTARGETS, M_prev, OW, OH, OCAM[0], drift, M_prev) for _ in range(3)]
    assert [o[1] for o in out] == ["tracking", "tracking", "markers"] and out[2][2] is True, [o[1:] for o in out]
    assert np.abs(screen_quad(out[2][0], OW, OH) - screen_quad(M_jump, OW, OH)).max() < 1.0


def test_live_frames_no_hand_all_four_used():
    # the full overlay path (overlay_fix, as run() calls it) on the real frames with no hand: all 4 markers used
    import glob
    import json
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
    t = {int(k): np.float32(v) for k, v in json.load(open(os.path.join(here, "live_targets.json"))).items()}
    frames = sorted(glob.glob(os.path.join(here, "live_*.jpg")))
    assert frames, "testdata/live_*.jpg missing"
    for f in frames:
        g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        seen = {i: c for i, c in detect_marker_corners(g).items() if i < 4}
        M, source, reset, used, redetected, dropped = overlay_fix(g, seen, t, None, None, None, OW, OH, {}, None)
        assert source == "markers" and sorted(used) == [0, 1, 2, 3] and not dropped, \
            f"{os.path.basename(f)}: source {source}, used {sorted(used)}, dropped {sorted(dropped)}"


def test_tracking_only_for_9s_gives_no_source():
    M = np.eye(3)
    assert tracking_age_guard(M, "tracking", now=9.0, last_marker=0.0) == (None, None)
    assert tracking_age_guard(M, "tracking", now=7.9, last_marker=0.0)[1] == "tracking"
    assert tracking_age_guard(M, "markers", now=9.0, last_marker=0.0)[1] == "markers"


def test_two_finger_pose():
    # wrist at the origin, knuckles (MCPs) 9 up: hand size 9. Each finger: (tip height, tip x); a straight finger's
    # tip is ~18 up (extension 2.0), a curled one ~10 (1.1), a relaxed half-bent one ~12.5 (1.4)
    def pose(index, middle, ring, pinky):
        pts = [(0.0, 0.0)] * 21
        for mcp, pip, tip, (h, x) in zip((5, 9, 13, 17), (6, 10, 14, 18), (8, 12, 16, 20), (index, middle, ring, pinky)):
            pts[mcp], pts[pip], pts[tip] = (x, -9.0), (x, -12.0), (x, -h)
        return pts
    two = pose((18, -1.0), (18, 1.0), (10, 2.0), (10, 3.0))  # index and middle up, close together
    relaxed = pose((18, -1.0), (17.5, 1.0), (12.5, 2.0), (12.5, 3.0))  # ring and pinky only half bent (live case)
    point = pose((18, -1.0), (10, 1.0), (10, 2.0), (10, 3.0))  # pointing: middle curled
    open_hand = pose((18, -2.0), (18, 0.0), (17, 2.0), (16, 4.0))  # all four extended
    spread = pose((18, -4.0), (18, 4.0), (10, 2.0), (10, 3.0))  # V sign, tips 8 apart (> 0.6 * 9)
    assert two_finger(two) is True and two_finger(relaxed) is True
    assert two_finger(point) is False and two_finger(open_hand) is False and two_finger(spread) is False
    # 3D world landmarks (what run() passes): same poses with a z coordinate
    z = lambda pts: [(x, y, 0.2 * i) for i, (x, y) in enumerate(pts)]
    assert two_finger(z(two)) is True and two_finger(z(relaxed)) is True
    assert two_finger(z(point)) is False and two_finger(z(spread)) is False


def test_two_finger_rule_on_live_poses():
    # per-second pose samples logged live (ext index/middle/ring/pinky, index-middle gap)
    two = [(1.61, 1.72, 1.41, 1.40, 0.18), (1.58, 1.62, 1.15, 1.13, 0.21), (1.64, 1.76, 1.31, 1.36, 0.14),
           (1.49, 1.68, 1.51, 1.56, 0.23), (1.50, 1.56, 1.41, 1.45, 0.23), (1.73, 1.95, 1.29, 1.35, 0.18)]
    point = [(1.72, 1.03, 0.87, 0.93, 0.91), (1.63, 1.41, 1.16, 1.32, 0.50), (1.52, 1.16, 1.05, 1.18, 0.58),
             (1.61, 1.26, 1.18, 1.34, 0.64), (1.43, 1.19, 1.12, 1.28, 0.47), (1.67, 1.12, 1.10, 1.30, 0.86)]
    assert all(pose_is_two(r[:4], r[4]) for r in two)
    assert not any(pose_is_two(r[:4], r[4]) for r in point)
    assert not pose_is_two((1.8, 1.85, 1.8, 1.75), 0.3)  # an open hand: ring and pinky as straight as the middle


def test_touch_fuser_reports_accepted_downs():
    fu = TouchFuser()
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_tap(1.0)
    fu.poll(1.5)
    assert len(fu.downs) == 1 and fu.downs[0][0] == 1.0 and np.hypot(fu.downs[0][1] - 500, fu.downs[0][2] - 300) < 1


def _series(t0, t1, val=(0.0, 0.0, 0.0), dt=0.001):
    """Samples (t, x, y, z) every dt from t0 to t1, all equal to val."""
    return [(t0 + k * dt, *val) for k in range(int(round((t1 - t0) / dt)) + 1)]


def _at(series, t):
    return int(round((t - series[0][0]) / 0.001))  # index of the sample at time t (1 ms grid)


def test_tap_features_window_and_components():
    T = 1.0  # window [0.95, 1.15]
    gyro = _series(0.7, 1.4)
    gyro[_at(gyro, 1.02)] = (1.02, 3.0, -4.0, 0.0)  # |w| = 5 inside the window
    gyro[_at(gyro, 1.3)] = (1.3, 100.0, 0.0, 0.0)   # outside the window: must be ignored
    accel = _series(0.7, 1.4, (0.0, 0.0, 1.0))
    for i in range(_at(accel, 1.01), len(accel)):  # a step of +0.5 on x at 1.01
        accel[i] = (accel[i][0], 0.5, 0.0, 1.0)
    lid = [(0.5, 100.0), (0.9, 100.0), (1.1, 97.0), (1.3, 97.0)]
    f = tap_features(T, accel, gyro, lid)
    assert abs(f["gyro_peak"] - 5.0) < 1e-9 and f["gyro_x"] == 3.0 and f["gyro_y"] == 4.0 and f["gyro_z"] == 0.0, f
    assert abs(f["acc_peak"] - 0.5) < 1e-9 and f["acc_x"] == 0.5 and f["acc_y"] == 0.0 and f["acc_z"] == 0.0, f
    assert abs(f["lid_delta"] - 3.0) < 1e-9, f  # 97 against the 100 read at 0.9, the last before the window
    assert "lid_delta" not in tap_features(T, accel, gyro), "no lid samples: no lid feature"
    assert tap_features(T, [], []) == {}  # no samples: no features at all
    quiet = tap_features(T, _series(0.7, 1.4), _series(0.7, 1.4))
    assert quiet["gyro_peak"] == 0.0 and quiet["acc_peak"] == 0.0, quiet  # the spike at 1.3 is outside


def test_drain_keeps_valid_feature_fields():
    sock = open_touch_socket(0)
    port = sock.getsockname()[1]
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(b'{"t": 5.0, "peak": 2.0, "gyro_peak": 3.5, "gyro_x": true, "gyro_y": "1", "gyro_z": NaN, '
              b'"acc_peak": 9, "acc_x": -2.25, "acc_y": 1e400}', ("127.0.0.1", port))
    tx.sendto(b'{"t": 6.0, "peak": true, "gyro_x": 1}', ("127.0.0.1", port))
    tx.sendto(b'{"t": 7.0, "lid_delta": ' + b"9" * 320 + b'}', ("127.0.0.1", port))  # int too big for a float
    time.sleep(0.05)  # loopback delivery is not always immediate: a non-blocking drain right after send can miss
    fuser = TouchFuser()
    drain_taps(sock, fuser)
    assert list(fuser.taps) == [(5.0, 2.0, {"gyro_peak": 3.5, "acc_peak": 9.0, "acc_x": -2.25}),
                                (6.0, None, {"gyro_x": 1.0}), (7.0, None, {})], list(fuser.taps)
    tx.close()
    sock.close()


def test_decision_line_has_features_and_hands():
    fu = TouchFuser()
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_hands(1.0, 2)
    fu.add_tap(1.0, 3.5, {"gyro_peak": 2.0, "gyro_x": 1.5, "acc_peak": 0.06, "lid_delta": 3.0})  # inside the limits
    fu.poll(1.5)
    (d,) = fu.decisions
    assert d["hands"] == 2 and d["outcome"] == "CLICK", d
    line = format_decision(d)
    assert " hands=2 " in line and " g=2.0 gx=1.5 a=0.1 lid=3.0 " in line, line
    fu = TouchFuser()  # no hand frames near the tap: hands is unknown, not 0
    fu.add_tap(1.0)
    fu.poll(2.0)
    assert fu.decisions[0]["hands"] is None and " hands=- " in format_decision(fu.decisions[0])


def test_pointing_hand_nearest_previous_tip():
    def hand(x, y):  # 21 landmarks (normalised); only the index tip (8) matters here
        return [SimpleNamespace(x=x, y=y) if i == 8 else SimpleNamespace(x=0.0, y=0.0) for i in range(21)]
    hands = [hand(0.2, 0.2), hand(0.7, 0.6)]  # 1000 x 800 frame: index tips at (200, 160) and (700, 480)
    assert pointing_hand(hands, 1000, 800, (690, 470)) == 1  # previous tip near the second hand
    assert pointing_hand(hands, 1000, 800, (150, 150)) == 0
    assert pointing_hand(hands[:1], 1000, 800, (690, 470)) == 0
    # live bug: the previous tip was the keyboard hand; the hand on the screen must win anyway
    assert pointing_hand(hands, 1000, 800, (690, 470), on_screen=[True, False]) == 0
    assert pointing_hand(hands, 1000, 800, (690, 470), on_screen=[False, False]) == 1  # none on screen: nearest


def _hand_at(x, y):
    """21 landmarks (normalised) with the index tip (8) at (x, y); the rest sit at the origin."""
    return [SimpleNamespace(x=x, y=y) if i == 8 else SimpleNamespace(x=0.0, y=0.0) for i in range(21)]


def test_tap_strength_and_knock_gates():
    def outcome(feats, **kw):
        fu = TouchFuser(**kw)
        _still_finger(fu, 500, 300, 0.0, 2.0)
        fu.add_tap(1.0, 3.5, feats)
        fu.poll(1.5)
        return fu.decisions[0]["outcome"].split()[0]
    assert outcome({"acc_peak": 0.0025}) == "weak"
    assert outcome({"acc_peak": 0.006}) == "CLICK"
    assert outcome({"acc_peak": 0.25, "gyro_peak": 10.0}) == "knock"
    assert outcome({"acc_peak": 0.006, "gyro_peak": 10.0}) == "knock"  # gyro over 3 rad/s
    assert outcome({}) == "CLICK" and outcome({"gyro_peak": 10.0}) == "CLICK"  # no acc field: as before
    assert outcome({"acc_peak": 0.0025}, min_acc=0.002) == "CLICK"  # the limits are the fuser's arguments
    assert outcome({"acc_peak": 0.006}, min_acc=0.01) == "weak"
    # live: a double-tap's 2nd tap measured 0.002 g; accepted 0.4-0.9 s after a click, not otherwise
    fu = TouchFuser()
    _still_finger(fu, 500, 300, 0.0, 3.0)
    fu.add_tap(1.0, 3.5, {"acc_peak": 0.006})
    fu.add_tap(1.5, 3.5, {"acc_peak": 0.002})
    fu.add_tap(2.6, 3.5, {"acc_peak": 0.002})  # 1.1 s after the last click: a lone light jolt
    fu.poll(3.0)
    assert [d["outcome"].split()[0] for d in fu.decisions] == ["CLICK", "CLICK", "weak"]
    fu = TouchFuser()
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_tap(1.0, 3.5, {"acc_peak": 0.001})
    fu.poll(1.5)
    assert fu.decisions[0]["outcome"].startswith("weak")  # below the 2nd-tap floor too


def test_typing_flag_over_keyboard():
    M = np.eye(3)  # camera pixels = screen points; frame 1000 x 1000 px, screen 800 pt high
    pointing = _hand_at(0.2, 0.2)
    assert typing_flag([pointing, _hand_at(0.5, 1.2)], 0, 1000, 1000, M, True, 800) is True  # 400 pt under
    assert typing_flag([pointing, _hand_at(0.5, 0.9)], 0, 1000, 1000, M, True, 800) is False  # 100 pt: within 150
    assert typing_flag([pointing, _hand_at(0.6, 0.5)], 0, 1000, 1000, M, True, 800) is False  # on the screen
    assert typing_flag([pointing, _hand_at(0.6, 0.5)], 0, 1000, 1000, M, False, 800) is True  # no fresh M
    assert typing_flag([pointing], 0, 1000, 1000, M, True, 800) is False  # one hand: nothing to check
    assert typing_flag([_hand_at(0.5, 1.2), _hand_at(0.6, 0.5)], 1, 1000, 1000, M, True, 800) is True  # 2nd is pointing
    # a hand resting over the keyboard is not typing; one whose fingers move is
    resting = _hand_at(0.5, 1.2)
    still = tracker.other_hand_tips([pointing, resting], 0, 1000, 1000)
    assert typing_flag([pointing, resting], 0, 1000, 1000, M, True, 800, prev_tips=still) is False
    assert typing_flag([pointing, resting], 0, 1000, 1000, M, True, 800, prev_tips=still + [[3, 2]] * 4) is False  # jitter
    moved = still.copy()
    moved[0] += (0, 15)  # the index tip moved 15 px: a key press
    assert typing_flag([pointing, resting], 0, 1000, 1000, M, True, 800, prev_tips=moved) is True


def test_typing_is_logged_not_vetoed():
    fu = TouchFuser()
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_tap(1.0, 3.5, {})
    fu.add_typing(1.1, True)  # inside the tap window: logged, but a still on-screen finger still clicks (no veto)
    fu.poll(1.5)
    assert fu.decisions[0]["outcome"] == "CLICK" and fu.decisions[0]["typing"] is True
    assert "typing=1" in format_decision(fu.decisions[0])
    fu = TouchFuser()
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_tap(1.0)
    fu.add_typing(0.9, False)
    fu.add_typing(1.5, True)  # after the window: this tap is not typing
    fu.poll(1.5)
    assert fu.decisions[0]["outcome"] == "CLICK" and fu.decisions[0]["typing"] is False


def test_tap_flags_validated():
    assert parse_args(["--source", "0", "--touch"]).tap_min_acc == 0.003
    assert parse_args(["--source", "0", "--tap-min-acc", "0.006"]).tap_min_acc == 0.006
    for bad in (["--tap-min-acc", "0.08"], ["--tap-min-acc", "-1"], ["--tap-max-acc", "0.001"]):
        try:
            parse_args(["--source", "0", *bad])
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")


def test_view_server_mjpeg_and_host_check():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    view = ViewServer(port)
    try:
        view.hub.put(np.full((240, 960, 3), 90, np.uint8))  # 960 px wide: the stream is at most 540 px

        def get(host):
            c = socket.create_connection(("127.0.0.1", port), timeout=5)
            c.sendall(f"GET / HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
            buf = b""
            while not (b"\xff\xd8" in buf and b"\xff\xd9" in buf[buf.index(b"\xff\xd8"):]):
                chunk = c.recv(65536)
                if not chunk:
                    break
                buf += chunk
            c.close()
            return buf

        buf = get(f"127.0.0.1:{port}")
        head, body = buf.split(b"\r\n\r\n", 1)
        assert b"200 OK" in head and b"multipart/x-mixed-replace" in head, head
        assert b"--frame" in body and b"Content-Type: image/jpeg" in body, body[:200]
        jpg = body[body.index(b"\xff\xd8"):body.index(b"\xff\xd9") + 2]
        assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape[1] == 540
        assert b"403" in get(f"evil.example:{port}").split(b"\r\n", 1)[0]  # bad Host: refused
    finally:
        view.close()


def test_stamped_capture_time():
    # frame stamped 0.2 s before it arrived, 0.1 s link latency left in the stamp: captured 0.3 s before arrival
    assert abs(tracker.stamped_capture_time(100.0, 5000.2, 5000.0, 0.1) - 99.7) < 1e-9
    assert tracker.stamped_capture_time(100.0, 5000.0, 5004.0, 0.1) is None   # stamp from the future: clocks off
    assert tracker.stamped_capture_time(100.0, 5010.0, 5000.0, 0.1) is None   # 10 s old: clocks off


def test_mjpeg_source_keeps_capture_time():
    import http.server, threading
    ok, jpg = cv2.imencode(".jpg", np.full((8, 8, 3), 200, np.uint8))
    jpg = jpg.tobytes()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            for extra in (b"X-Capture-Time: 1234.567\r\n", b"", b"X-Capture-Time: 99.5\r\nX-Capture-Source: phone-camera\r\n"):
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n" % len(jpg)
                                 + extra + b"\r\n" + jpg + b"\r\n")

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    src = tracker.MjpegSource(f"http://127.0.0.1:{srv.server_port}/")
    try:
        ok, frame = src.read()
        assert ok and frame.shape == (8, 8, 3) and abs(src.stamp - 1234.567) < 1e-9 and not src.exact  # glasses
        ok, frame = src.read()
        assert ok and src.stamp is None and not src.exact  # a frame without the header: no stamp, so the caller guesses
        ok, frame = src.read()
        assert ok and src.stamp == 99.5 and src.exact  # the phone's own camera: exact capture time
        assert src.read() == (False, None)  # stream ended
    finally:
        src.release()
        srv.server_close()


def test_hand_crop_and_remap():
    M = np.eye(3)  # camera px == screen points: the 100 x 50 screen sits at the frame's top left
    box = tracker.hand_crop(M, 100, 50, 400, 300, None)
    assert box == (0, 0, 130, 85), box  # margins: 30 % sideways, 20 % up (clipped), 70 % down
    assert tracker.hand_crop(M @ np.array([[1, 0, -3], [0, 1, 0], [0, 0, 1.0]]), 100, 50, 400, 300, box) == box  # small move: kept
    assert tracker.hand_crop(None, 100, 50, 400, 300, box) == box  # no M: keep the last box
    from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
    hand = [NormalizedLandmark(x=0.5, y=0.5, z=0.0)]
    tracker.remap_landmarks([hand], box, 400, 300)
    assert abs(hand[0].x - 65 / 400) < 1e-9 and abs(hand[0].y - 42.5 / 300) < 1e-9



def test_double_click_sets_click_state():
    # live: pyautogui.doubleClick sent two clicks apps saw as singles; ours marks the second press as click 2. It is
    # only the second press: the first tap already clicked (a whole 1,2 pair here made double-taps three presses)
    import Quartz
    posted, real_post = [], Quartz.CGEventPost
    Quartz.CGEventPost = lambda tap, ev: posted.append(ev)  # capture instead of moving the real mouse
    try:
        tracker.double_click(300.0, 200.0)
    finally:
        Quartz.CGEventPost = real_post
    kinds = [Quartz.CGEventGetType(e) for e in posted]
    states = [Quartz.CGEventGetIntegerValueField(e, Quartz.kCGMouseEventClickState) for e in posted]
    assert kinds == [Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp], kinds
    assert states == [2, 2], states
    assert all(tuple(Quartz.CGEventGetLocation(e)) == (300.0, 200.0) for e in posted)



def _fake_desktop(W=1512, H=982, seed=3):
    """A screen-like image: light background, window rectangles, title bars and lines of text."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 236, np.uint8)
    for _ in range(9):
        x, y = int(rng.integers(0, W - 400)), int(rng.integers(30, H - 300))
        w, h = int(rng.integers(250, 600)), int(rng.integers(180, 400))
        cv2.rectangle(img, (x, y), (x + w, y + h), [int(c) for c in rng.integers(150, 255, 3)], -1)
        cv2.rectangle(img, (x, y), (x + w, y + 24), (60, 60, 60), -1)
        for k in range(1, h // 28):
            words = " ".join("".join(chr(97 + int(c)) for c in rng.integers(0, 26, int(rng.integers(2, 9)))) for _ in range(5))
            cv2.putText(img, words, (x + 10, y + 24 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    return img


def _camera_view(screen, dst, seed=1):
    """The screen as the glasses see it: perspective into a 720x1280 frame, blur, contrast, a hand, JPEG."""
    H, W = screen.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32(dst))
    cam = cv2.warpPerspective(screen, M, (720, 1280), dst=np.full((1280, 720, 3), 40, np.uint8),
                              borderMode=cv2.BORDER_TRANSPARENT)
    cam = cv2.convertScaleAbs(cv2.GaussianBlur(cam, (0, 0), 1.4), alpha=0.8, beta=25)
    cv2.ellipse(cam, (420, 700), (110, 170), 20, 0, 360, (150, 180, 220), -1)
    cam = cv2.imdecode(cv2.imencode(".jpg", cam, [cv2.IMWRITE_JPEG_QUALITY, 70])[1], cv2.IMREAD_GRAYSCALE)
    return cam, M


def test_screen_matcher_finds_screen_from_content():
    screen = _fake_desktop()
    H, W = screen.shape[:2]
    grid = np.float32([[W * f, H * g] for f in (.2, .5, .8) for g in (.2, .5, .8)])
    for view in ([[90, 420], [650, 380], [690, 760], [60, 800]],              # whole screen in view
                 [[-200, 300], [1100, 260], [1150, 1000], [-250, 1050]]):     # close: only part of it in view
        cam, Mtrue = _camera_view(screen, view)
        m = tracker.ScreenMatcher(W, H, grab=lambda: screen)
        M = m.match(cam, now=0.0)
        assert M is not None, view
        vis = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), Mtrue).reshape(-1, 2)
        inside = (vis[:, 0] >= 0) & (vis[:, 0] < 720) & (vis[:, 1] >= 0) & (vis[:, 1] < 1280)
        err = np.abs(cv2.perspectiveTransform(vis[inside].reshape(-1, 1, 2), M).reshape(-1, 2) - grid[inside]).max()
        print(f"  content match, {inside.sum()} of 9 grid points in view: worst error {err:.1f} pt")
        assert err < 6, err
    blank = np.full((H, W, 3), 236, np.uint8)  # nothing to match on: no guess
    assert tracker.ScreenMatcher(W, H, grab=lambda: blank).match(_camera_view(blank, [[90, 420], [650, 380], [690, 760], [60, 800]])[0], 0.0) is None
    # a screenshot at Retina scale (2x pixels) gives the same points
    big = cv2.resize(screen, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)
    cam, Mtrue = _camera_view(screen, [[90, 420], [650, 380], [690, 760], [60, 800]])
    M = tracker.ScreenMatcher(W, H, grab=lambda: big).match(cam, 0.0)
    c = cv2.perspectiveTransform(np.float32([[[W / 2, H / 2]]]), Mtrue)
    assert np.abs(cv2.perspectiveTransform(c, M).ravel() - (W / 2, H / 2)).max() < 6



def test_picture_of_a_hand_on_screen_is_ignored():
    # live: the tutorial's 3D hand on the screen was detected as a hand. Measured on the screen plane it is small
    M = np.eye(3)  # camera px = screen points
    def hand(x0, y0, span):  # wrist at (x0, y0), middle knuckle `span` px above it, in a 2000 x 2000 frame
        return [SimpleNamespace(x=x0 / 2000, y=(y0 - (span if i == 9 else 0)) / 2000) for i in range(21)]
    real, picture = hand(900, 1500, 480), hand(300, 400, 70)
    assert tracker.real_hands([real, picture], M, 2000, 2000) == [0]
    assert tracker.real_hands([picture], M, 2000, 2000) == []
    assert tracker.real_hands([real, picture], None, 2000, 2000) == [0, 1]  # no screen fix: can't tell, keep both



def test_gesture_drag_drops_at_jittery_stop():
    # live: at the drop the fingertip still read 15-22 pt per frame (jitter, residual motion), then the pull-away
    # (40-60 pt per frame) kept dragging. The stop is under GESTURE_LIFT_SPEED: one up there, nothing after
    def path(t):
        if t < 0.95:
            return (500.0, 300.0)
        if t < 1.6:
            return (500.0 + 750.0 * (t - 0.95), 300.0)            # drags at ~50 pt per frame
        if t < 1.8:
            k = round((t - 1.6) * 15)
            return (1062.0 + (8.0 if k % 2 else -8.0), 300.0)    # stopped, but jittering ~16 pt per frame
        return (1062.0 + 800.0 * (t - 1.8), 300.0 - 500.0 * (t - 1.8))  # pull-away
    timed, eng = _drive(_with_tap(_frames(0.0, 2.4, path, span=lambda t: 500.0 * (1 + 4 * max(t - 1.8, 0.0))), 0.6))
    kinds = [a[0] for _, a in timed]
    assert kinds.count("up") == 1 and "move" not in kinds[kinds.index("up"):], kinds
    up = timed[kinds.index("up")][1]
    assert abs(up[1] - 1062) < 50, up  # dropped at the stop, not along the pull-away
    assert not eng.held


def test_gesture_scroll_ends_at_pause_before_return_stroke():
    # live: a two-finger stroke, a short pause, then the fingers moving back through the air scrolled the page back
    def path(t):
        if t < 0.15:
            return (500.0, 300.0)
        if t < 0.65:
            return (500.0, 300.0 - 700.0 * (t - 0.15))  # stroke up at ~47 pt per frame
        if t < 0.95:
            return (500.0, 300.0 - 350.0)               # pause
        return (500.0, -50.0 + 700.0 * (t - 0.95))      # return stroke in the air, back down
    timed, eng = _drive(_with_tap(_frames(0.0, 1.6, path, two=True, span=lambda t: 500.0 * (1 + 4 * max(t - 0.95, 0.0))), 0.1))
    dys = [(t, a[2]) for t, a in timed if a[0] == "scroll"]
    assert dys and all(dy <= 0 for _, dy in dys), dys  # only the stroke's direction: no scrolling back
    assert max(t for t, _ in dys) <= 0.75, dys[-3:]   # nothing after the pause
    assert eng.state is None and not eng.held



def test_off_screen_placeholders_do_not_make_a_sweep():
    # live: samples (912, 681), (921, 684) then two frames with no screen fix, stored as (0, 0) off screen: it was
    # rejected as "sweeping span=1100". Only on-screen samples count toward the span
    fu = TouchFuser()
    for t, x, y, on in ((0.92, 912, 681, True), (0.98, 921, 684, True), (1.04, 0, 0, False), (1.1, 0, 0, False)):
        fu.add_fingertip(t, x, y, on)
    fu.add_tap(1.0)
    assert len(fu.poll(1.5)) == 1, fu.decisions


def test_gesture_double_tap_scattered_140pt_is_still_a_double_click():
    # live: 2nd taps landed up to ~110 pt from the first (fingertip scatter); 100 pt made them two single clicks
    pos = lambda t: (500.0, 300.0) if t < 1.3 else (520.0, 438.0)  # 139 pt away
    timed, _ = _drive(_with_tap(_frames(0.0, 2.2, pos), 1.0, 1.45))
    assert [a for _, a in timed] == [("click", 500.0, 300.0), ("dclick", 500.0, 300.0)], timed




def test_gesture_held_too_long_then_moved_is_just_a_click():
    # a tap that rests past GESTURE_HOLD_WAIT and only then moves stays its click: no drag
    pos = lambda t: (500.0, 300.0) if t < 2.4 else (500.0 + 600.0 * (t - 2.4), 300.0)
    timed, eng = _drive(_with_tap(_frames(0.0, 3.2, pos), 0.6))
    assert [a[0] for _, a in timed] == ["click"], timed
    assert not eng.held



def test_gesture_drag_survives_slowing_down():
    # live: drags "released too early when my hand doesn't even release": slowing down mid-drag isn't a lift
    def path(t):
        if t < 0.95:
            return (500.0, 300.0)
        if t < 1.4:
            return (500.0 + 750.0 * (t - 0.95), 300.0)           # drags
        if t < 1.65:
            return (837.5 + 100.0 * (t - 1.4), 300.0)            # slows to 100 pt/s for 0.25 s, still on the glass
        return (862.5 + 750.0 * (t - 1.65), 300.0)               # and carries on
    timed, eng = _drive(_with_tap(_frames(0.0, 2.0, path, span=lambda t: 500.0), 0.6))
    kinds = [a[0] for _, a in timed]
    assert kinds.count("down") == 1 and "up" not in kinds, kinds  # still held
    assert sum(1 for t, a in timed if a[0] == "move" and t > 1.7) > 3, kinds  # and still moving after the slow part
    assert eng.held


def test_gesture_drag_released_when_the_hand_lifts():
    # the finger keeps moving, but the hand grows (comes toward the eye): it lifted off, so the drag ends there
    pos = lambda t: (500.0, 300.0) if t < 0.95 else (500.0 + 700.0 * (t - 0.95), 300.0)
    size = lambda t: 500.0 if t < 1.5 else 500.0 * (1 + 3 * (t - 1.5))
    timed, eng = _drive(_with_tap(_frames(0.0, 2.2, pos, span=size), 0.6))
    kinds = [a[0] for _, a in timed]
    assert kinds.count("up") == 1 and "move" not in kinds[kinds.index("up"):], kinds
    t_up = timed[kinds.index("up")][0]
    assert 1.55 <= t_up <= 1.75, t_up  # within a few frames of the lift
    assert not eng.held and any("hand grew" in n for n in eng.notes), eng.notes



def test_gesture_scroll_strokes_only_scroll_one_way():
    # live: scrolling with repeated strokes (stroke, lift, move back, stroke) scrolled back on every return stroke.
    # A scroll keeps its first stroke's direction: return strokes do nothing, each forward stroke scrolls
    def path(t):  # two fingers: 200 pt down at ~670 pt/s, back up, down again, back up
        if t < 0.15:
            return (500.0, 300.0)
        k, u = divmod(t - 0.15, 0.3)
        y = 200.0 * u / 0.3
        return (500.0, 300.0 + (y if k % 2 == 0 else 200.0 - y))
    timed, _ = _drive(_with_tap(_frames(0.0, 1.35, path, two=True), 0.1))
    dys = [a[2] for _, a in timed if a[0] == "scroll"]
    assert dys and all(dy >= 0 for dy in dys), dys  # never scrolled back
    total = sum(dys)
    assert abs(total - 2 * 200 * SCROLL_GAIN) <= 0.3 * 2 * 200 * SCROLL_GAIN, total  # both forward strokes counted



def test_gesture_drag_dropped_when_it_turns_after_a_pause():
    # live: drags stopped for a frame or two, then the pull-away (turning 53-179 deg) was dragged along. Hand size
    # gave no signal (it stays within ~8 %). A pause followed by a sharp turn drops the drag at the pause
    def path(t):
        if t < 0.95:
            return (500.0, 300.0)
        if t < 1.45:
            return (500.0 + 750.0 * (t - 0.95), 300.0)                    # drags right to x=875
        if t < 1.55:
            return (875.0, 300.0)                                         # a short stop
        return (875.0 - 500.0 * (t - 1.55), 300.0 + 700.0 * (t - 1.55))  # pull-away: down and back (~125 deg)
    timed, eng = _drive(_with_tap(_frames(0.0, 2.2, path, span=lambda t: 500.0), 0.6))
    kinds = [a[0] for _, a in timed]
    assert kinds.count("up") == 1 and "move" not in kinds[kinds.index("up"):], kinds
    up = timed[kinds.index("up")][1]
    assert abs(up[1] - 875) < 5 and abs(up[2] - 300) < 5, up  # dropped at the stop
    assert any("turned" in n for n in eng.notes), [n for n in eng.notes if "size" not in n]


if __name__ == "__main__":
    for t in (test_marker_homography_roundtrip, test_homography_custom_targets,
              test_overlay_corner_homography_two_markers, test_collinear_corners_rejected,
              test_pinch_hysteresis, test_map_point_clamped,
              test_screen_tracker_drift, test_screen_tracker_noise_lost,
              test_screen_tracker_scrolling_static_regions, test_screen_tracker_fold_and_jump_rejected,
              test_screen_tracker_realistic_noise,
              test_tap_detector_bursts, test_touch_fuser, test_tap_udp_round_trip,
              test_small_blurred_guided_redetection, test_live_frames_guided_finds_all_four, test_no_false_marker_ids, test_diagonal_pair_trust,
              test_guided_timing_two_missing,
              test_stale_tap_dropped,
              test_ringing_gives_one_event_and_one_click, test_malformed_datagrams_ignored,
              test_drain_at_most_64_per_frame, test_bind_failure_names_port_and_imu_taps,
              test_capture_time_shift, test_hover_moves_cursor_only_without_touch, test_screen_point_edge_tolerance,
              test_touch_decisions_per_reason, test_two_finger_pose, test_two_finger_rule_on_live_poses,
              test_touch_point_settle_and_window, test_two_samples_click_and_no_samples_do_not,
              test_keyboard_samples_off_screen_rejected, test_live_moving_taps_click_at_contact_point,
              test_tracking_only_for_9s_gives_no_source, test_drift_trap_markers_win_after_three_consistent_fits,
              test_hand_covering_marker_one_drops_only_it, test_marker_with_displaced_corners_excluded_by_guard,
              test_one_frame_jump_with_two_markers_keeps_tracker, test_persistent_jump_markers_win_after_drift_frames,
              test_live_frames_no_hand_all_four_used,
              test_gesture_tap_then_withdrawal_is_one_click,
              test_touch_fuser_reports_accepted_downs,
              test_tap_features_window_and_components, test_drain_keeps_valid_feature_fields,
              test_decision_line_has_features_and_hands, test_pointing_hand_nearest_previous_tip,
              test_view_server_mjpeg_and_host_check,
              test_tap_strength_and_knock_gates, test_typing_flag_over_keyboard,
              test_typing_is_logged_not_vetoed, test_tap_flags_validated,
              test_gesture_400_still_taps_one_click_each, test_gesture_hold_drag_down_moves_one_up, test_gesture_short_hold_drags,
              test_gesture_scroll_with_sideways_jitter_stays_vertical, test_gesture_scroll_survives_hand_size_noise,
              test_marker_toggle_shows_when_lost_hides_when_content_holds, test_screen_matcher_ignores_marker_windows,
              test_gesture_double_tap_still_is_two_clicks, test_gesture_double_tap_lift_off_is_a_double_click, test_gesture_double_tap_wobble_is_not_a_drag, test_gesture_taps_one_second_apart_are_two_clicks,
              test_gesture_two_finger_scroll_ends_when_pose_lost,
              test_gesture_drag_stays_held_while_moving, test_gesture_drag_drops_at_pause_before_pull_away, test_gesture_two_finger_swipe_scrolls_no_button,
              test_gesture_stall_releases_held_drag, test_gesture_0p4s_frame_gap_keeps_drag,
              test_gesture_forced_up_after_max, test_gesture_lost_hand_lifts,
              test_gesture_tap_while_dragging_is_ignored, test_release_touch_reaches_mouse_up_on_exception,
              test_stamped_capture_time, test_mjpeg_source_keeps_capture_time, test_hand_crop_and_remap,
              test_double_click_sets_click_state, test_screen_matcher_finds_screen_from_content,
              test_picture_of_a_hand_on_screen_is_ignored, test_gesture_drag_drops_at_jittery_stop,
              test_gesture_scroll_ends_at_pause_before_return_stroke, test_off_screen_placeholders_do_not_make_a_sweep,
              test_gesture_double_tap_scattered_140pt_is_still_a_double_click,
              test_gesture_held_too_long_then_moved_is_just_a_click, test_gesture_drag_survives_slowing_down,
              test_gesture_drag_released_when_the_hand_lifts, test_gesture_scroll_strokes_only_scroll_one_way,
              test_gesture_drag_dropped_when_it_turns_after_a_pause):
        t()
        print(f"{t.__name__} ok")
    print("ALL OK")
