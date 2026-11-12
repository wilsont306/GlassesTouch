"""Experiment: can screen taps be detected by the MacBook's motion sensor and/or microphone?

Run in Terminal (needs root for the motion sensor):
    sudo .venv/bin/python tap_probe.py

Records 25 s: tap the screen ~10 times (seconds 3-13), then type on the keyboard (seconds 15-23).
Prints spike counts per phase for each signal and saves build/tap_probe.png.
"""
import sys
import threading
import time

import cv2
import numpy as np

DURATION, TAP, TYPE = 25.0, (3, 13), (15, 23)
RATE = 48000


def record_audio(out):
    import sounddevice as sd
    out["audio"] = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()


def record_imu(out, t0):
    from macimu import IMU
    imu = IMU(accel=True, gyro=False)
    imu.start()
    ts, mags = [], []
    end = t0 + DURATION
    while time.monotonic() < end:
        for s in imu.read_accel():
            ts.append(time.monotonic() - t0)
            mags.append(float(np.sqrt(s.x * s.x + s.y * s.y + s.z * s.z)))
        time.sleep(0.002)
    imu.stop()
    out["imu"] = (np.array(ts), np.array(mags))


def envelope(t, x, win=0.02):
    """Max |change| per 20 ms window -> (window times, values)."""
    d = np.abs(np.diff(x, prepend=x[:1]))
    edges = np.arange(0, DURATION + win, win)
    idx = np.digitize(t, edges) - 1
    env = np.zeros(len(edges) - 1)
    np.maximum.at(env, np.clip(idx, 0, len(env) - 1), d)
    return edges[:-1], env


def spikes(t, env, k=6.0):
    """Spikes above k x the median of the quiet first 2.5 s, merged within 150 ms."""
    base = np.median(env[t < 2.5]) + 1e-12
    hits, last = [], -1.0
    for ti, v in zip(t, env):
        if v > k * base and ti - last > 0.15:
            hits.append(ti)
            last = ti
    return np.array(hits), base


def report(name, t, env):
    hits, base = spikes(t, env)
    n_tap = int(((hits >= TAP[0]) & (hits < TAP[1])).sum())
    n_type = int(((hits >= TYPE[0]) & (hits < TYPE[1])).sum())
    print(f"{name:6s} spikes during taps: {n_tap:3d}   during typing: {n_type:3d}   (baseline {base:.3g})")
    return env / (base * 6.0)


def main():
    out, t0 = {}, time.monotonic()
    threads = [threading.Thread(target=record_audio, args=(out,))]
    if "--no-imu" not in sys.argv:
        threads.append(threading.Thread(target=record_imu, args=(out, t0)))
    for th in threads:
        th.start()
    shown = None
    while any(th.is_alive() for th in threads):
        s = time.monotonic() - t0
        msg = ("stay still" if s < TAP[0] else "TAP the screen firmly ~once a second" if s < TAP[1]
               else "stop" if s < TYPE[0] else "TYPE on the keyboard" if s < TYPE[1] else "stop")
        if msg != shown:
            print(f"[{s:4.1f}s] {msg}")
            shown = msg
        time.sleep(0.1)

    rows = []
    a = out["audio"][:, 0]
    ta = np.arange(len(a)) / RATE
    rows.append(("mic",) + envelope(ta, a))
    if "imu" in out and len(out["imu"][0]):
        rows.append(("motion",) + envelope(*out["imu"]))
    else:
        print("motion: no samples (run with sudo, or pass --no-imu)")

    img = np.full((140 * len(rows), 1000, 3), 255, np.uint8)
    for r, (name, t, env) in enumerate(rows):
        norm = report(name, t, env)
        y0 = 140 * r + 130
        for (lo, hi), col in ((TAP, (220, 240, 220)), (TYPE, (220, 220, 245))):
            cv2.rectangle(img, (int(lo * 40), y0 - 120), (int(hi * 40), y0), col, -1)
        pts = np.stack([t * 40, y0 - np.clip(norm, 0, 3) * 35], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, (40, 40, 40), 1)
        cv2.line(img, (0, y0 - 35), (1000, y0 - 35), (0, 0, 200), 1)  # spike threshold
        cv2.putText(img, name, (5, y0 - 105), 0, 0.6, (0, 0, 0), 1)
    cv2.imwrite("../build/tap_probe.png", img)
    print("plot: build/tap_probe.png (green = tap phase, red = typing phase, red line = threshold)")


if __name__ == "__main__":
    main()
