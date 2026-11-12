#!/usr/bin/env python3
"""Root helper: read the MacBook motion sensor and send screen-tap times to tracker.py --touch.

Run (root-owned copy, see docs): sudo /usr/bin/python3 /usr/local/libexec/glasstouch/imu_taps.py
Standard library + macimu only, so it can run under macOS's root-owned Python.
"""
import argparse
import json
import math
import socket
import statistics
import time
from collections import deque

TAP_BEFORE, TAP_AFTER = 0.05, 0.15  # seconds: a tap's features cover [tap - 0.05, tap + 0.15]


class TapDetector:
    """Taps in a stream of accel magnitudes: a change |d| above k x the rolling median of recent changes.
    A refractory period makes a burst of spikes (or the ringing after a knock) a single event at its first spike."""

    def __init__(self, k=6.0, window=2.0, refractory=0.4, warmup=0.5):
        self.k, self.window, self.refractory, self.warmup = k, window, refractory, warmup
        self.last_mag, self.start, self.base, self.base_t = None, None, None, -1e9
        self.hist = deque()  # (t, |change|) over the last `window` seconds
        self.last_tap, self.last_peak = -1e9, 0.0

    def feed(self, t, mag):
        """Return [t] if this sample starts a tap, else []. The tap's peak (|change| / baseline) is in last_peak."""
        if self.start is None:
            self.start = t
        if self.last_mag is None:
            self.last_mag = mag
            return []
        d = abs(mag - self.last_mag)
        self.last_mag = mag
        self.hist.append((t, d))
        while self.hist[0][0] < t - self.window:
            self.hist.popleft()
        if t - self.start < self.warmup:
            return []
        if t >= self.base_t:  # refresh the baseline every 50 ms
            self.base = max(float(statistics.median(v for _, v in self.hist)), 1e-12)
            self.base_t = t + 0.05
        peak = d / self.base
        if peak > self.k and t - self.last_tap >= self.refractory:
            self.last_tap, self.last_peak = t, peak
            return [t]
        return []


def _norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def tap_features(tap, accel, gyro, lid=()):
    """Motion features of one tap over [tap - 0.05, tap + 0.15] s. accel and gyro: time-ordered (t, x, y, z)
    samples; lid: time-ordered (t, angle). Accel changes are against the previous accel sample; lid_delta is
    measured from the last angle at or before the window start. Features without data are left out."""
    lo, hi = tap - TAP_BEFORE, tap + TAP_AFTER
    f = {}
    g = [s[1:] for s in gyro if lo <= s[0] <= hi]
    if g:
        f["gyro_peak"] = max(_norm(v) for v in g)
        f.update({f"gyro_{k}": max(abs(v[i]) for v in g) for i, k in enumerate("xyz")})
    d = [tuple(accel[i][j] - accel[i - 1][j] for j in (1, 2, 3))
         for i in range(1, len(accel)) if lo <= accel[i][0] <= hi]
    if d:
        f["acc_peak"] = max(_norm(v) for v in d)
        f.update({f"acc_{k}": max(abs(v[i]) for v in d) for i, k in enumerate("xyz")})
    before = [s for s in lid if s[0] <= lo]
    inside = [s[1] for s in lid if lo <= s[0] <= hi]
    if before and inside:
        f["lid_delta"] = max(abs(a - before[-1][1]) for a in inside)
    return f


def tap_datagram(t, peak, feats=None):
    """UDP payload for one tap, read by tracker.py's drain_taps()."""
    return json.dumps({"t": t, "peak": round(peak, 2), **{k: round(v, 3) for k, v in (feats or {}).items()}}).encode()


def _passed(buf, end):
    return bool(buf) and buf[-1][0] >= end


def _trim(buf, cut):
    """Drop samples older than cut, keeping the last one before it (a baseline for lid_delta)."""
    while len(buf) > 1 and buf[1][0] <= cut:
        buf.popleft()


def main(argv=None):
    p = argparse.ArgumentParser(description="Send MacBook motion-sensor taps to tracker.py --touch (needs root).")
    p.add_argument("--k", type=float, default=3.0,
                   help="tap threshold as a multiple of the baseline (default 3; lower catches lighter taps, "
                        "and tracker.py's camera check rejects false ones)")
    p.add_argument("--port", type=int, default=9787, help="UDP port on 127.0.0.1 that tracker.py listens on")
    p.add_argument("--quiet", action="store_true", help="do not print each tap (errors still print); for the LaunchDaemon log")
    a = p.parse_args(argv)
    from macimu import IMU  # imported here so --help and the tests never touch the sensor
    det = TapDetector(k=a.k)
    sock, imu = None, None
    accel, gyro, lid = deque(), deque(), deque()  # sensor history for tap features
    pending = deque()  # (tap time, peak, wall time seen): sent once their window has passed
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        imu = IMU(accel=True, gyro=True, lid=True)
        imu.start()
        print("listening to the motion sensor; Ctrl-C to stop")
        while True:
            # sensor timestamps (mach_absolute_time, the same clock as time.monotonic() on macOS)
            for s in imu.read_accel_timed():
                accel.append(s)
                for tap in det.feed(s.t, math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z)):
                    pending.append((tap, det.last_peak, time.monotonic()))
            gyro.extend(imu.read_gyro_timed())
            now = time.monotonic()
            angle = imu.read_lid()  # a snapshot with no timestamp: stamped when read
            if angle is not None:
                lid.append((now, angle))
            for buf in (accel, gyro):
                if buf:
                    _trim(buf, buf[-1].t - 1.0)
            _trim(lid, now - 1.0)
            # a tap waits for its window in both sensor streams; after 0.5 s (a stalled stream) it goes anyway
            while pending and ((_passed(accel, pending[0][0] + TAP_AFTER) and _passed(gyro, pending[0][0] + TAP_AFTER))
                               or now - pending[0][2] > 0.5):
                tap, peak, _ = pending.popleft()
                feats = tap_features(tap, list(accel), list(gyro), list(lid))
                sock.sendto(tap_datagram(tap, peak, feats), ("127.0.0.1", a.port))
                if not a.quiet:
                    print(f"tap at {tap:.3f}  peak {peak:.1f}  " + " ".join(f"{k}={v:.2f}" for k, v in feats.items()))
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if imu is not None:
            imu.stop()
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    main()
