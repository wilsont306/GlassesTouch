"""Tests for panel.py and the tracker's orphan watchdog. Run: .venv/bin/python test_panel.py"""
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

import panel

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKER = os.path.join(HERE, "tracker.py")
STUB_NAME = "gt_panel_test_stub"  # pgrep pattern for the stub

STUB = f'''import signal, sys, time
def bye(*_):
    print("stub got SIGINT", flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, bye)
print("stub start", flush=True)
print("I0000 00:00:0 noise line", flush=True)
print("stats fps=15.0 hand=1/15", flush=True)
while True:
    time.sleep(0.2)
'''


def test_source_validation():
    ok = {
        "10.0.0.5": "http://10.0.0.5:8080/",
        "http://10.0.0.5:8080/": "http://10.0.0.5:8080/",
        "http://phone.local:8080/": "http://phone.local:8080/",
        "  http://10.0.0.5/  ": "http://10.0.0.5:8080/",
        "HTTPS://phone.local:9000": "https://phone.local:9000/",
        "camera:6C707041-05AC-0010-0001-000000000001": "camera:6C707041-05AC-0010-0001-000000000001",
        " camera:0x14100000046d085b ": "camera:0x14100000046d085b",
    }
    for text, want in ok.items():
        assert panel.normalise_source(text) == want, (text, panel.normalise_source(text))
    bad = ["file:///etc/passwd", "http://a b/", "10.0.0.1; rm -rf ~", "http://user:pw@host/",
           "javascript:alert(1)", "10.0.0.256", "http://-bad-/", "http://host:99999/", "http://host/x?y=1",
           "ftp://host/", "http:///path", "", "$(reboot)",
           # inet_aton aliases: a numeric-looking host must be a real dotted quad
           "0x7f.1", "2130706433", "http://0x7f.1/", "http://2130706433/", "http://0x7f.0.0.1/",
           "http://127.0.0.1.5/", "0x7f",
           "http://" + "a" * 3000 + "/",
           "camera:", "camera:a b", "camera:x;reboot", "camera:$(reboot)", "camera:" + "a" * 129, "CAMERA:1"]
    for text in bad:
        try:
            panel.normalise_source(text)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad source {text[:60]!r}")


def test_argv():
    tr = panel.tracker_script()
    py = panel.venv_python()
    s = panel.parse_settings({"source": "10.0.0.5"})  # defaults: touch, markers, debug on; dry run off
    view = ["--view-port", "8767", "--no-window", "--tap-min-acc", "0.003"]  # 0.003: the panel's default strength
    assert panel.build_argv(s) == [py, tr, "--source", "http://10.0.0.5:8080/", *view,
                                   "--show-markers", "--touch", "--touch-debug", "--markers-on-demand"], panel.build_argv(s)
    s = panel.parse_settings({"source": "10.0.0.5", "markers_on_demand": False})  # markers always on
    assert "--show-markers" in panel.build_argv(s) and "--markers-on-demand" not in panel.build_argv(s)
    s = panel.parse_settings({"source": "http://phone.local:8080/", "touch": False, "show_markers": False,
                              "touch_debug": False, "dry_run": True})
    assert panel.build_argv(s) == [py, tr, "--source", "http://phone.local:8080/", *view, "--dry-run"], \
        panel.build_argv(s)
    s = panel.parse_settings({"source": "10.0.0.5", "tap_min_acc": 0.006})  # the panel's value reaches the tracker
    argv = panel.build_argv(s)
    assert argv[argv.index("--tap-min-acc") + 1] == "0.006", argv
    s = panel.parse_settings({"source": "10.0.0.5", "dry_run": True, "marker_size": 200})
    assert panel.build_argv(s) == [py, tr, "--source", "http://10.0.0.5:8080/", *view, "--show-markers",
                                   "--touch", "--touch-debug", "--dry-run", "--markers-on-demand", "--marker-size", "200"], \
        panel.build_argv(s)


def test_settings_ranges():
    for n in (40, 140, 1000):
        assert panel.parse_settings({"marker_size": n})["marker_size"] == n
    for n in (39, 1001, -5, "140", 140.0, True):
        try:
            panel.parse_settings({"marker_size": n})
        except ValueError:
            continue
        raise AssertionError(f"accepted marker_size {n!r}")
    for n in (0, 0.004, 0.05):
        assert panel.parse_settings({"tap_min_acc": n})["tap_min_acc"] == float(n)
    for n in (-0.001, 0.051, "0.004", None, True, float("nan")):
        try:
            panel.parse_settings({"tap_min_acc": n})
        except ValueError:
            continue
        raise AssertionError(f"accepted tap_min_acc {n!r}")
    for key in ("touch", "show_markers", "markers_on_demand", "dry_run", "touch_debug"):
        for v in ("yes", 1, None):
            try:
                panel.parse_settings({key: v})
            except ValueError:
                continue
            raise AssertionError(f"accepted {key}={v!r}")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    h = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    conn.request(method, path, body=data, headers=h)
    r = conn.getresponse()
    text = r.read().decode()
    conn.close()
    return r.status, text


def wait_for(fn, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("timed out")


def pgrep(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.strip()


def stub_running():
    return pgrep(STUB_NAME)


def start_panel(d):
    """Run panel.py with the stub as the tracker and a scratch config file. Returns (proc, port, token)."""
    stub = os.path.join(d, STUB_NAME + ".py")
    with open(stub, "w") as f:
        f.write(STUB)
    port = free_port()
    env = {**os.environ, "GLASSTOUCH_TRACKER_SCRIPT": stub,
           "GLASSTOUCH_PANEL_CONFIG": os.path.join(d, "panel.json")}  # never touch the real build/panel.json
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "panel.py"), "--port", str(port)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def up():
        try:
            return request(port, "GET", "/")[0] == 200
        except OSError:  # not listening yet
            return False
    wait_for(up)
    token = re.search(r'const TOKEN = "([^"]+)"', request(port, "GET", "/")[1]).group(1)
    return proc, port, token


def stop_panel(proc):
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise AssertionError("panel did not exit")
    return proc.returncode


def test_http():
    d = tempfile.mkdtemp(prefix="gt_panel_test_")
    proc, port, token = start_panel(d)
    try:
        status, page = request(port, "GET", "/")
        assert status == 200 and "GlassesTouch" in page and token in page
        assert '"http://127.0.0.1:8767/"' in page  # the preview switches to the tracker's view while it runs
        assert "__TOKEN__" not in page and "__CONFIG__" not in page and 'id="test-go"' in page  # panel.html, filled in
        page_t = json.loads(request(port, "GET", "/status")[1])["page"]  # the open page reloads when this changes
        assert page_t == os.stat(os.path.join(HERE, "panel.html")).st_mtime, page_t
        status, cams = request(port, "GET", "/cameras")
        assert status == 200 and isinstance(json.loads(cams), list), (status, cams)

        good = {"X-Panel-Token": token, "Content-Type": "application/json"}
        body = {"source": "10.0.0.5"}
        assert request(port, "POST", "/start", body)[0] == 403  # no token
        assert request(port, "POST", "/start", body, {"X-Panel-Token": "wrong"})[0] == 403
        assert request(port, "POST", "/start", body, {**good, "Host": "evil.example"})[0] == 403  # bad Host
        assert request(port, "GET", "/", headers={"Host": "evil.example"})[0] == 403
        assert request(port, "POST", "/start", body, {**good, "Origin": "http://evil.example"})[0] == 403
        assert request(port, "POST", "/config", {"source": "file:///etc/passwd"}, good)[0] == 400
        assert request(port, "POST", "/config", {"source": "http://" + "a" * 3000 + "/"}, good)[0] == 400
        assert request(port, "POST", "/start", {"source": "", "touch": True}, good)[0] == 400  # no address
        assert stub_running() == ""  # nothing was started by the refused requests

        # Content-Length: garbage, negative and oversize are refused (400/413), not a crash or a hang
        assert request(port, "POST", "/start", None, {**good, "Content-Length": "abc"})[0] == 400
        assert request(port, "POST", "/start", None, {**good, "Content-Length": "-5"})[0] == 400
        assert request(port, "POST", "/start", None, {**good, "Content-Length": "9999"})[0] == 413
        assert stub_running() == ""

        status, _ = request(port, "POST", "/start", body, good)
        assert status == 200, status
        st = wait_for(lambda: (lambda s: s if s["tracker"]["state"] == "running" and any(
            "stub start" in l for l in s["lines"]) else None)(json.loads(request(port, "GET", "/status")[1])))
        assert "I0000" not in json.dumps(st["lines"]), "noise line not filtered"
        assert st["stats"].startswith("stats "), st["stats"]
        assert request(port, "POST", "/start", body, good)[0] == 409  # one tracker at a time

        t0 = time.monotonic()
        assert request(port, "POST", "/stop", {}, good)[0] == 200
        st = json.loads(request(port, "GET", "/status")[1])
        assert time.monotonic() - t0 < 3, "stop took longer than 3 s"
        assert st["tracker"]["state"] == "stopped" and st["tracker"]["exit_code"] == 0, st["tracker"]
        assert any("stub got SIGINT" in l for l in st["lines"]), st["lines"]
        log = os.path.join(d, "tracker.log")  # beside the test's panel.json, never the real build/
        assert "stub start" in open(log).read() and oct(os.stat(log).st_mode & 0o777) == "0o600", log

        # start again, then close the panel: the stub must not outlive it
        assert request(port, "POST", "/start", body, good)[0] == 200
        wait_for(lambda: stub_running() or None)
    finally:
        proc.send_signal(signal.SIGINT)
        code = stop_panel(proc)
    assert code == 0, code
    assert stub_running() == "", f"stub still running: {stub_running()}"


def test_sighup_stops_tracker():
    d = tempfile.mkdtemp(prefix="gt_panel_test_")
    proc, port, token = start_panel(d)
    try:
        good = {"X-Panel-Token": token, "Content-Type": "application/json"}
        assert request(port, "POST", "/start", {"source": "10.0.0.5"}, good)[0] == 200
        wait_for(lambda: stub_running() or None)
        proc.send_signal(signal.SIGHUP)  # what a closed Terminal window sends
    finally:
        code = stop_panel(proc)
    assert code == 0, code
    assert stub_running() == "", f"stub still running after SIGHUP: {stub_running()}"


def test_orphan_watchdog():
    """A tracker whose parent is SIGKILLed exits on its own within 2 s (its camera open hangs here)."""
    port = free_port()
    listener = socket.socket()  # accepts the TCP connection (queued in the backlog) but never answers
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    marker = f"127.0.0.1:{port}/"  # appears only in the tracker's command line
    d = tempfile.mkdtemp(prefix="gt_panel_test_")
    parent_src = os.path.join(d, "orphan_parent.py")
    with open(parent_src, "w") as f:
        f.write(f"""import subprocess, sys, time
p = subprocess.Popen([{sys.executable!r}, {TRACKER!r}, "--source", "http://{marker}", "--dry-run"],
                     stdin=subprocess.DEVNULL)
print("started", flush=True)
time.sleep(600)
""")
    parent = subprocess.Popen([sys.executable, parent_src], stdout=subprocess.PIPE, text=True)
    try:
        assert parent.stdout.readline().strip() == "started"
        time.sleep(3)  # imports done (~1.2 s), watchdog running, camera open still hanging
        assert pgrep(marker), "tracker exited before the parent was killed"
        parent.kill()
        parent.wait()
        t0 = time.monotonic()
        wait_for(lambda: not pgrep(marker) or None, timeout=2.0)
        assert time.monotonic() - t0 < 2, "orphaned tracker lived on"
    finally:
        subprocess.run(["pkill", "-f", marker], capture_output=True)
        listener.close()


TUTORIAL_IDS = ("tutorial", "tut-voice", "tut-skip", "tut-exit", "tut-n", "tut-h", "tut-hint", "tut-g", "hand3d", "tut-stage",
                "test-go", "paired", "paired-go")


def test_tutorial_in_served_page():
    d = tempfile.mkdtemp(prefix="gt_panel_test_")
    proc, port, _ = start_panel(d)
    try:
        status, page = request(port, "GET", "/")
    finally:
        proc.send_signal(signal.SIGINT)
        stop_panel(proc)
    assert status == 200
    assert 'id="test-area"' not in page  # the on-page touch test is gone; the full-screen tutorial replaces it
    for name in TUTORIAL_IDS:
        assert f'id="{name}"' in page, name
    # no external resources: nothing loads from another host (the phone placeholder is text, not a load)
    assert not re.search(r"(src|href)\s*=\s*[\"']?https?:|url\(\s*[\"']?https?:|@import", page)
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    area = script[script.index("// Tutorial:"):script.index("showConfig(CONFIG);")]
    assert not re.search(r"fetch\(|XMLHttpRequest|sendBeacon|TOKEN", area), "tutorial sends something"
    path = os.path.join(d, "page_script.js")
    with open(path, "w") as f:
        f.write(script)
    r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_pairing():
    """A phone that POSTs the QR's one-time code becomes the source (its own address); wrong, reused or late codes
    change nothing, and the pairing listener closes."""
    import threading
    from urllib.parse import parse_qs, urlparse
    d = tempfile.mkdtemp(prefix="gt_panel_test_")
    os.environ["GLASSTOUCH_PANEL_CONFIG"] = os.path.join(d, "panel.json")
    srv = panel.make_server(0, pair_host="127.0.0.1")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        link = srv.pairing.open()
        q = {k: v[0] for k, v in parse_qs(urlparse(link).query).items()}
        assert link.startswith("cameraaccess://glasstouch-pair?") and q["h"] == "127.0.0.1", link
        port = int(q["p"])
        assert request(port, "GET", "/")[0] == 404  # nothing but POST /pair on this listener
        assert request(port, "POST", "/pair", {"code": "wrong"})[0] == 403
        assert request(port, "POST", "/pair", {"code": 5})[0] == 400
        assert srv.config["source"] == ""
        status, text = request(port, "POST", "/pair", {"code": q["c"]})
        assert status == 200 and json.loads(text)["ok"], text
        assert srv.config["source"] == "http://127.0.0.1:8080/", srv.config
        assert json.load(open(os.environ["GLASSTOUCH_PANEL_CONFIG"]))["source"] == "http://127.0.0.1:8080/"
        st = srv.pairing.status()
        assert not st["open"] and st["last"], st

        def closed():
            try:
                return request(port, "POST", "/pair", {"code": q["c"]})[0] == 410  # still up for a moment: refused
            except OSError:
                return True
        wait_for(closed, timeout=3)

        link = srv.pairing.open()  # a late code is refused
        q = {k: v[0] for k, v in parse_qs(urlparse(link).query).items()}
        srv.pairing.expires = 0
        assert request(int(q["p"]), "POST", "/pair", {"code": q["c"]})[0] == 410
        srv.pairing.close()
        assert qr_ok(link)
    finally:
        srv.pairing.close()
        srv.shutdown()
        srv.server_close()
        del os.environ["GLASSTOUCH_PANEL_CONFIG"]


def qr_ok(link):
    """The panel's QR rows decode back to the link (OpenCV reads its own code, with the page's 4-module quiet zone)."""
    import cv2
    import numpy as np
    rows = panel.qr_rows(link)
    img = np.array([[0 if c == "1" else 255 for c in r] for r in rows], np.uint8)
    img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    img = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    return cv2.QRCodeDetector().detectAndDecode(img)[0] == link


def test_camera_format_pick():
    """The opened camera moves to the most pixels up to 1920x1440 at 24 fps or more, landscape first."""
    import maccam
    iphone = [(1280, 720, 30), (1920, 1080, 60), (1920, 1440, 30), (1920, 1440, 60), (1440, 1920, 60)]  # live list
    assert maccam.pick(iphone) == 3, maccam.pick(iphone)  # 4:3 at 60 fps, landscape over the same-size portrait
    assert maccam.pick([(3840, 2160, 30), (1280, 720, 60)]) == 1  # 4K is over the cap
    assert maccam.pick([(1920, 1080, 15), (1280, 720, 30)]) == 1  # too slow at 1080p
    assert maccam.pick([(1920, 1080, 15)]) is None and maccam.pick([]) is None
    assert maccam.size_of("<AVCaptureDeviceFormat: 'vide'/'420v' 1920x1080, { 1- 30 fps}>") == (1920, 1080)

    class Rate:
        def __init__(self, fps): self.fps = fps
        def maxFrameRate(self): return self.fps

    class Fmt:
        def __init__(self, w, h, fps): self.text, self.fps = f"<'420v' {w}x{h}, {{ 1-{fps} fps}}>", fps
        def __str__(self): return self.text
        def videoSupportedFrameRateRanges(self): return [Rate(self.fps)]

    class Cam:  # a camera whose lock answers the way PyObjC does: a bare bool here, (bool, error) with full metadata
        def __init__(self, lock_answer): self.lock_answer, self.active = lock_answer, None
        def formats(self): return [Fmt(640, 480, 30), Fmt(1920, 1080, 30), Fmt(3840, 2160, 30)]  # 4K over the cap
        def lockForConfiguration_(self, _err): return self.lock_answer
        def setActiveFormat_(self, f): self.active = str(f)

    for answer in (True, (True, None)):
        cam = Cam(answer)
        assert maccam.use_best_format(cam) == (1920, 1080) and "1920x1080" in cam.active, (answer, cam.active)
    for answer in (False, (False, "not authorized")):
        cam = Cam(answer)
        assert maccam.use_best_format(cam) is None and cam.active is None, answer


def test_camera_source():
    """A camera source starts the tracker on that camera's current OpenCV number, with the local-camera latency."""
    cams = [{"id": "BUILTIN-1", "name": "MacBook Pro Camera"}, {"id": "PHONE-2", "name": "iPhone Camera"}]
    real = panel.list_cameras
    panel.list_cameras = lambda: cams
    try:
        argv = panel.build_argv(panel.parse_settings({"source": "camera:PHONE-2"}))
        assert argv[2:4] == ["--source", "1"], argv
        assert argv[argv.index("--camera-latency") + 1] == "0.1", argv
        cams.pop(0)  # the built-in camera went away: the phone is number 0 now
        assert panel.build_argv(panel.parse_settings({"source": "camera:PHONE-2"}))[3] == "0"
        try:
            panel.build_argv(panel.parse_settings({"source": "camera:BUILTIN-1"}))
        except ValueError as e:
            assert "isn't connected" in str(e)
        else:
            raise AssertionError("started on a camera that isn't connected")
        assert "--camera-latency" not in panel.build_argv(panel.parse_settings({"source": "10.0.0.5"}))
    finally:
        panel.list_cameras = real
    real_ids = [cam["id"] for cam in panel.list_cameras()]  # the real list (empty without cameras)
    assert real_ids == sorted(real_ids), real_ids  # OpenCV's numbering: sorted by uniqueID (ASCII ids)
    for uid in real_ids:  # ids the source pattern accepts
        assert panel.normalise_source("camera:" + uid) == "camera:" + uid, uid


if __name__ == "__main__":
    test_source_validation()
    test_argv()
    test_camera_format_pick()
    test_camera_source()
    test_settings_ranges()
    test_http()
    test_sighup_stops_tracker()
    test_orphan_watchdog()
    test_tutorial_in_served_page()
    test_pairing()
    print("ALL OK")
