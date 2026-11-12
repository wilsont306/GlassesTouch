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
    }
    for text, want in ok.items():
        assert panel.normalise_source(text) == want, (text, panel.normalise_source(text))
    bad = ["file:///etc/passwd", "http://a b/", "10.0.0.1; rm -rf ~", "http://user:pw@host/",
           "javascript:alert(1)", "10.0.0.256", "http://-bad-/", "http://host:99999/", "http://host/x?y=1",
           "ftp://host/", "http:///path", "", "$(reboot)",
           # inet_aton aliases: a numeric-looking host must be a real dotted quad
           "0x7f.1", "2130706433", "http://0x7f.1/", "http://2130706433/", "http://0x7f.0.0.1/",
           "http://127.0.0.1.5/", "0x7f",
           "http://" + "a" * 3000 + "/"]
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
                                   "--show-markers", "--touch", "--touch-debug"], panel.build_argv(s)
    s = panel.parse_settings({"source": "http://phone.local:8080/", "touch": False, "show_markers": False,
                              "touch_debug": False, "dry_run": True})
    assert panel.build_argv(s) == [py, tr, "--source", "http://phone.local:8080/", *view, "--dry-run"], \
        panel.build_argv(s)
    s = panel.parse_settings({"source": "10.0.0.5", "tap_min_acc": 0.006})  # the panel's value reaches the tracker
    argv = panel.build_argv(s)
    assert argv[argv.index("--tap-min-acc") + 1] == "0.006", argv
    s = panel.parse_settings({"source": "10.0.0.5", "dry_run": True, "marker_size": 200})
    assert panel.build_argv(s) == [py, tr, "--source", "http://10.0.0.5:8080/", *view, "--show-markers",
                                   "--touch", "--touch-debug", "--dry-run", "--marker-size", "200"], panel.build_argv(s)


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
    for key in ("touch", "show_markers", "dry_run", "touch_debug"):
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


if __name__ == "__main__":
    test_source_validation()
    test_argv()
    test_settings_ranges()
    test_http()
    test_sighup_stops_tracker()
    test_orphan_watchdog()
    print("ALL OK")
