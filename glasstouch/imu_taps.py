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


def tap_datagram(t, peak):
    """UDP payload for one tap, read by tracker.py's drain_taps()."""
    return json.dumps({"t": t, "peak": round(peak, 2)}).encode()


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
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        imu = IMU(accel=True, gyro=False)
        imu.start()
        print("listening to the motion sensor; Ctrl-C to stop")
        while True:
            # sensor timestamps (mach_absolute_time, the same clock as time.monotonic() on macOS)
            for s in imu.read_accel_timed():
                for tap in det.feed(s.t, math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z)):
                    sock.sendto(tap_datagram(tap, det.last_peak), ("127.0.0.1", a.port))
                    if not a.quiet:
                        print(f"tap at {tap:.3f}  peak {det.last_peak:.1f}")
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
