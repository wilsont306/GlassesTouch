"""Camera-free, mouse-free tests for tracker.py. Run: .venv/bin/python test_tracker.py"""
import socket
import time
from types import SimpleNamespace

import cv2
import numpy as np

from imu_taps import TapDetector, tap_datagram, tap_features
from tracker import (ARUCO_DICT, GESTURE_HOLD, GESTURE_MAX, SCROLL_GAIN, TRUST_FRAC, GestureEngine,
                     ViewServer, pointing_hand, typing_flag,
                     release_touch, touch_step,
                     PinchDetector, ScreenTracker, TouchFuser, two_finger,
                     capture_time, choose_overlay_M, format_decision, tracking_age_guard,
                     detect_marker_corners, detect_markers, drain_taps, guided_redetect, homography_from_corners,
                     homography_from_markers, hover_moves_cursor, map_point, marker_extent, marker_png, open_touch_socket,
                     parse_args, screen_point, screen_quad)


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
    tc, thr, exceeded = _true_corners(TILT), TRUST_FRAC * OCAM[0], 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        for pair in ((0, 2), (1, 3)):
            corners = {i: (tc[i] + rng.uniform(-2, 2, (4, 2))).astype(np.float32) for i in pair}
            M_pair = homography_from_corners(corners, OTARGETS)
            d = np.linalg.norm(screen_quad(M_pair, OW, OH) - screen_quad(M_track, OW, OH), axis=1).mean()
            M, source, reset = choose_overlay_M(corners, OTARGETS, M_track, OW, OH, OCAM[0])
            if d > thr:
                exceeded += 1
                assert source == "tracking" and not reset and np.array_equal(M, M_track), (seed, pair, d)
            else:
                assert source == "markers" and np.array_equal(M, M_pair), (seed, pair, d)
    print(f"  diagonal pairs, 600 noisy fits: {exceeded} disagree by more than {thr:.1f} px and keep the tracker M")
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
    assert fu.poll(tap + 0.2) == [] and len(fu.taps) == 1  # not decided before TOUCH_SETTLE
    clicks = fu.poll(tap + 0.3)
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


def _frames(t_from, t_to, pos, on=lambda t: True, two=False):
    """Camera frames at 15 fps, capture times from t_from to t_to: ("frame", t, (x, y), on, two)."""
    out, k = [], round(t_from * 15)
    while k / 15 <= t_to + 1e-9:
        t = round(k / 15, 6)
        out.append(("frame", t, pos(t), on(t), two))
        k += 1
    return out


def _with_tap(events, tap):
    """Insert the tap datagram once its frame would have arrived (tap + 50 ms)."""
    i = next(i for i, e in enumerate(events) if e[0] == "frame" and e[1] >= tap + 0.05)
    return events[:i] + [("tap", tap)] + events[i:]


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
            _, t, tip, on, two = ev
            acts = touch_step(eng, fuser, t, None if tip is None else tip, on, two)
        out += [(t, a) for a in acts]
    return out, eng


def _swipe(dist=200.0, t0=0.1):
    """Finger rests at (500, 300); 50 ms after the tap it moves 300 pt/s until `dist` points, then rests."""
    return lambda t: (500.0 + min(300.0 * max(t - t0 - 0.05, 0.0), dist), 300.0)


def _slow(slow_end, t0=0.1):
    """Fast start (36 pt), then 15 pt/s until capture time slow_end, then rests."""
    def pos(t):
        s_ = max(min(t - t0 - 0.05, slow_end - t0 - 0.05), 0.0)
        return (500.0 + 300.0 * min(s_, 0.12) + 15.0 * max(s_ - 0.12, 0.0), 300.0)
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


def test_gesture_one_finger_drag_down_moves_one_up():
    # tap, still for the 0.4 s hold, then 200 pt: the hold ends in a click, then down, moves, one up
    timed, eng = _drive(_with_tap(_frames(0.0, 2.2, _swipe(t0=0.5)), 0.1))
    kinds = [a[0] for _, a in timed]
    assert kinds[:2] == ["click", "down"] and kinds.count("up") == 1 and kinds[-1] == "up", kinds
    assert kinds.count("move") > 10 and "scroll" not in kinds, kinds
    t_up = next(t for t, a in timed if a[0] == "up")
    assert 1.3 <= t_up <= 1.7, t_up  # the finger rests at 1.22 s; lift-off once the average speed drops
    assert not eng.held


def test_gesture_tap_then_withdrawal_is_one_click():
    def pos(t):  # rests, then withdraws 200 pt in 0.2 s, then rests again
        return (500.0 + 200 * min(max(t - 1.05, 0.0) / 0.2, 1.0), 300.0)
    timed, eng = _drive(_with_tap(_frames(0.0, 2.0, pos), 1.0))
    assert [a[0] for _, a in timed] == ["click"], timed
    assert not eng.held


def test_gesture_slow_drag_stays_held_while_moving():
    timed, _ = _drive(_with_tap(_frames(0.0, 3.2, _slow(2.6, t0=0.5)), 0.1))
    kinds_t = [(t, a[0]) for t, a in timed]
    assert [k for _, k in kinds_t].count("up") == 1, kinds_t
    assert not [t for t, k in kinds_t if k == "up" and t < 2.6], "released while still moving"
    assert sum(1 for t, k in kinds_t if k == "move" and 1.2 <= t <= 2.5) > 10, "no moves during the slow drag"
    t_up = next(t for t, k in kinds_t if k == "up")
    assert 2.6 <= t_up <= 3.0, t_up  # the finger rests at 2.6 s; the average speed drops below 12 pt/s soon after


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


def test_gesture_stall_releases_held_drag():
    timed, eng = _drive(_with_tap(_frames(0.0, 1.0, _slow(3.0, t0=0.5)), 0.1))  # frames stop at 1.0 s
    assert eng.held
    t = 1.0
    while t < 2.0:
        t = round(t + 1 / 30, 6)
        timed += [(t, a) for a in eng.tick(t)]
    ups = [t for t, a in timed if a[0] == "up"]
    assert len(ups) == 1 and 1.6 <= ups[0] <= 1.7, ups


def test_gesture_0p4s_frame_gap_keeps_drag():
    ev = _with_tap(_frames(0.0, 0.8, _slow(3.0, t0=0.5)), 0.1)
    ev += [("tick", round(0.8 + k / 30, 6)) for k in range(1, 13)]  # 0.4 s with no frame, ticks as in run()
    ev += _frames(1.2, 2.5, _slow(3.0, t0=0.5))
    timed, eng = _drive(ev)
    assert not [t for t, a in timed if a[0] == "up" and t < 2.5], "the gap released the drag"
    assert eng.held and sum(1 for t, a in timed if a[0] == "move" and t > 1.2) > 5


def test_gesture_forced_up_after_max():
    timed, _ = _drive(_with_tap(_frames(0.0, 12.0, _slow(12.5, t0=0.5)), 0.1))  # still moving: only the cap can end it
    down = next(t for t, a in timed if a[0] == "down")
    ups = [t for t, a in timed if a[0] == "up"]
    assert len(ups) == 1 and down + GESTURE_MAX <= ups[0] <= down + GESTURE_MAX + 0.2, (down, ups)
    assert timed[-1][1][0] == "up", timed[-3:]  # nothing after the forced release


def test_gesture_lost_hand_lifts():
    ev = _with_tap(_frames(0.0, 0.8, _slow(3.0, t0=0.5)), 0.1)
    ev += _frames(0.8667, 2.0, _slow(3.0, t0=0.5), on=lambda t: False)  # hand gone from 0.87 s
    timed, _ = _drive(ev)
    ups = [t for t, a in timed if a[0] == "up"]
    assert len(ups) == 1 and 1.4 <= ups[0] <= 1.7, ups


def test_gesture_new_touch_down_ends_drag_first():
    path = _slow(2.0, t0=0.5)
    ev = _with_tap(_frames(0.0, 2.3, path), 0.1)
    ev.append(("tap", 2.0))  # the second touch: accepted once frames to 2.25 s are in
    ev += _frames(2.3667, 2.7, lambda t: path(2.0))  # it rests after the tap, so it is a click
    timed, _ = _drive(ev)
    acts = [a for _, a in timed]
    i_up = next(i for i, a in enumerate(acts) if a[0] == "up")
    assert acts.count(("up",) + acts[i_up][1:]) == 1 and i_up < len(acts) - 1, acts
    assert acts[-1][0] == "click" and np.hypot(acts[-1][1] - path(2.0)[0], acts[-1][2] - 300) < 1, acts[-3:]
    assert sum(1 for a in acts if a[0] == "up") == 1, acts


def test_release_touch_reaches_mouse_up_on_exception():
    import types
    import tracker as T
    real_pg, real_perform = T.pyautogui, T.perform_gestures

    class FakePG(types.SimpleNamespace):
        pass
    fake = FakePG(FAILSAFE=True, FailSafeException=real_pg.FailSafeException, ups=0, fail_up=0)

    def mouse_up(*a, **k):
        fake.ups += 1
        if fake.fail_up:
            fake.fail_up -= 1
            raise fake.FailSafeException("parked on a corner")
    fake.mouseUp = mouse_up
    args = types.SimpleNamespace(dry_run=False, touch_debug=False)
    eng = _drive(_with_tap(_frames(0.0, 1.0, _slow(3.0, t0=0.5)), 0.1))[1]  # a drag still held
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
        eng2 = _drive(_with_tap(_frames(0.0, 1.0, _slow(3.0, t0=0.5)), 0.1))[1]
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


def test_tracking_only_for_9s_gives_no_source():
    M = np.eye(3)
    assert tracking_age_guard(M, "tracking", now=9.0, last_marker=0.0) == (None, None)
    assert tracking_age_guard(M, "tracking", now=7.9, last_marker=0.0)[1] == "tracking"
    assert tracking_age_guard(M, "markers", now=9.0, last_marker=0.0)[1] == "markers"


def test_two_finger_pose():
    def pose(index, middle, ring, pinky):  # each finger: (extended?, tip_x offset); wrist at origin
        pts = [(0.0, 0.0)] * 21
        pts[9] = (0.0, -10.0)  # middle MCP: hand size 10
        for k, (pip, tip), (ext, x) in zip((6, 10, 14, 18), ((6, 8), (10, 12), (14, 16), (18, 20)),
                                           (index, middle, ring, pinky)):
            pts[pip] = (x, -6.0)
            pts[tip] = (x, -14.0 if ext else -3.0)
        return pts
    two = pose((True, -1.5), (True, 1.5), (False, 0.5), (False, 0.5))  # index and middle up, close together
    point = pose((True, -1.5), (False, 1.5), (False, 0.5), (False, 0.5))  # pointing: middle folded
    open_hand = pose((True, -3.0), (True, 0.0), (True, 3.0), (True, 4.5))  # all four extended
    spread = pose((True, -4.0), (True, 4.0), (False, 0.5), (False, 0.5))  # V sign, tips 8 apart (> 0.6 * 10)
    assert two_finger(two) is True
    assert two_finger(point) is False and two_finger(open_hand) is False and two_finger(spread) is False


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


def test_typing_flag_over_keyboard():
    M = np.eye(3)  # camera pixels = screen points; frame 1000 x 1000 px, screen 800 pt high
    pointing = _hand_at(0.2, 0.2)
    assert typing_flag([pointing, _hand_at(0.5, 1.2)], 0, 1000, 1000, M, True, 800) is True  # 400 pt under
    assert typing_flag([pointing, _hand_at(0.5, 0.9)], 0, 1000, 1000, M, True, 800) is False  # 100 pt: within 150
    assert typing_flag([pointing, _hand_at(0.6, 0.5)], 0, 1000, 1000, M, True, 800) is False  # on the screen
    assert typing_flag([pointing, _hand_at(0.6, 0.5)], 0, 1000, 1000, M, False, 800) is True  # no fresh M
    assert typing_flag([pointing], 0, 1000, 1000, M, True, 800) is False  # one hand: nothing to check
    assert typing_flag([_hand_at(0.5, 1.2), _hand_at(0.6, 0.5)], 1, 1000, 1000, M, True, 800) is True  # 2nd is pointing


def test_typing_gates_taps_in_window():
    fu = TouchFuser()
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_tap(1.0, 3.5, {})
    fu.add_typing(1.1, True)  # inside the tap window
    fu.poll(1.5)
    assert fu.decisions[0]["outcome"] == "typing" and fu.decisions[0]["typing"] is True
    assert "typing=1" in format_decision(fu.decisions[0])
    fu = TouchFuser()
    _still_finger(fu, 500, 300, 0.0, 2.0)
    fu.add_tap(1.0)
    fu.add_typing(0.9, False)
    fu.add_typing(1.5, True)  # after the window: this tap is not typing
    fu.poll(1.5)
    assert fu.decisions[0]["outcome"] == "CLICK" and fu.decisions[0]["typing"] is False


def test_tap_flags_validated():
    assert parse_args(["--source", "0", "--touch"]).tap_min_acc == 0.004
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
              test_touch_decisions_per_reason, test_two_finger_pose,
              test_touch_point_settle_and_window, test_two_samples_click_and_no_samples_do_not,
              test_keyboard_samples_off_screen_rejected, test_live_moving_taps_click_at_contact_point,
              test_tracking_only_for_9s_gives_no_source, test_drift_trap_markers_win_after_three_consistent_fits,
              test_gesture_tap_then_withdrawal_is_one_click,
              test_touch_fuser_reports_accepted_downs,
              test_tap_features_window_and_components, test_drain_keeps_valid_feature_fields,
              test_decision_line_has_features_and_hands, test_pointing_hand_nearest_previous_tip,
              test_view_server_mjpeg_and_host_check,
              test_tap_strength_and_knock_gates, test_typing_flag_over_keyboard,
              test_typing_gates_taps_in_window, test_tap_flags_validated,
              test_gesture_400_still_taps_one_click_each, test_gesture_one_finger_drag_down_moves_one_up,
              test_gesture_slow_drag_stays_held_while_moving, test_gesture_two_finger_swipe_scrolls_no_button,
              test_gesture_stall_releases_held_drag, test_gesture_0p4s_frame_gap_keeps_drag,
              test_gesture_forced_up_after_max, test_gesture_lost_hand_lifts,
              test_gesture_new_touch_down_ends_drag_first, test_release_touch_reaches_mouse_up_on_exception):
        t()
        print(f"{t.__name__} ok")
    print("ALL OK")
