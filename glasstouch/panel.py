#!/usr/bin/env python3
"""Local control panel for tracker.py: QR pairing with the phone, switches, Start/Stop, status and log.

Standard library only. Serves 127.0.0.1 only. Every POST needs the per-run token (X-Panel-Token), a Host
header naming 127.0.0.1 or localhost (DNS rebinding), and an Origin that matches if one is sent.
Pairing: "Pair phone" opens a second listener on the Wi-Fi address for 120 s that accepts one POST /pair with the
one-time code from the QR; the phone's address is the connection's source. Nothing else is served on the Wi-Fi.
Run: .venv/bin/python panel.py [--port 8765]   (GlassesTouch.command does this and opens the browser)

Threat model: the token protects against web pages (CSRF and DNS rebinding), not against other local user
accounts or local processes. The phone stream and the tap datagrams are unauthenticated, so anyone on the same
Wi-Fi or on this machine could inject frames or taps. The tracker's view stream (127.0.0.1:8767) shows the camera
image, which includes the screen. It is local-only with a Host check, and other web pages can embed it but not
read it. Intended for a single-user Mac on a trusted network.
"""
import argparse
import hmac
import ipaddress
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import maccam

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_HEADER = "X-Panel-Token"
VIEW_PORT = 8767  # the tracker's --view-port: its annotated debug frame, shown in the preview while it runs
MARKER_MIN, MARKER_MAX = 40, 1000  # tracker.py refuses < 40; the upper bound is a sanity limit
NOISE = ("I0000", "W0000", "INFO: Created")  # MediaPipe / TF startup chatter, hidden from the log
PROBLEM = re.compile(r"cannot|Accessibility|Screen Recording|rror|Traceback|died|exited")
TAP_MIN_DEFAULT, TAP_MIN_MAX = 0.003, 0.05  # --tap-min-acc (g): the panel's range; tracker's max stays 0.08
DEFAULTS = {"source": "", "touch": True, "show_markers": True, "markers_on_demand": True, "dry_run": False, "touch_debug": True,
            "marker_size": None, "tap_min_acc": TAP_MIN_DEFAULT}
BOOLS = ("touch", "show_markers", "markers_on_demand", "dry_run", "touch_debug")
PAIR_SECONDS = 120  # how long a pairing code works
PAIR_LINK = "cameraaccess://glasstouch-pair?h={host}&p={port}&c={code}"  # the iPhone Camera opens the phone app


def tracker_script():
    # GLASSTOUCH_TRACKER_SCRIPT lets the HTTP test substitute a stub; normal runs use tracker.py
    return os.environ.get("GLASSTOUCH_TRACKER_SCRIPT") or os.path.join(HERE, "tracker.py")


def venv_python():
    p = os.path.join(HERE, ".venv", "bin", "python")
    return p if os.path.exists(p) else sys.executable


def config_path():
    return os.environ.get("GLASSTOUCH_PANEL_CONFIG") or os.path.join(HERE, "..", "build", "panel.json")


def tracker_log_path():
    """The last run's tracker output, beside panel.json: the panel keeps it only in memory, gone once it closes."""
    return os.path.join(os.path.dirname(os.path.abspath(config_path())), "tracker.log")


CAMERA = re.compile(r"camera:([A-Za-z0-9._:-]{1,128})")  # a camera on this Mac, by its AVFoundation uniqueID


def list_cameras():
    """Cameras this Mac sees (built-in, USB, an iPhone through Continuity Camera) as [{"id", "name"}], in the order
    OpenCV numbers them (maccam: live, the system's own order made the iPhone's number open the MacBook camera)."""
    return [{"id": str(d.uniqueID()), "name": str(d.localizedName())} for d in maccam.devices()]


def camera_index(source):
    """'camera:<id>' -> that camera's OpenCV number now (numbers shift as cameras come and go)."""
    uid = CAMERA.fullmatch(source).group(1)
    for i, cam in enumerate(list_cameras()):
        if cam["id"] == uid:
            return i
    raise ValueError("the chosen camera isn't connected: connect it, or pick another under 1 Pair")


def normalise_source(text):
    """A bare IPv4 address or an http(s) URL -> 'http://<host>:<port><path>' (port 8080 and path / by default),
    or 'camera:<id>' for a camera on this Mac, kept as is.
    Raises ValueError for anything else: other schemes, credentials, spaces, shell characters."""
    if len(text) > 2048:
        raise ValueError("address is too long")
    s = text.strip()
    if CAMERA.fullmatch(s):
        return s
    if re.fullmatch(r"[0-9.]+", s):  # bare address: IPv4 only
        try:
            return f"http://{ipaddress.IPv4Address(s)}:8080/"
        except ValueError:
            raise ValueError(f"not an IPv4 address: {s}")
    m = re.fullmatch(r"(https?)://([A-Za-z0-9.-]+)(?::([0-9]{1,5}))?(/[A-Za-z0-9._~/-]*)?", s, re.IGNORECASE)
    if not m:
        raise ValueError("address must be an IP like 10.0.0.5 or a URL like http://phone.local:8080/")
    scheme, host, port, path = m.groups()
    labels = host.split(".")
    # numeric-looking hosts (0x7f.1, 2130706433) are inet_aton aliases: only a real dotted quad may pass
    if re.fullmatch(r"[0-9.]+", host) or labels[-1].isdigit() or labels[-1].lower().startswith("0x"):
        try:
            ipaddress.IPv4Address(host)
        except ValueError:
            raise ValueError(f"not an IPv4 address: {host}")
    elif not all(re.fullmatch(r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)", label) for label in labels):
        raise ValueError(f"bad host name: {host}")
    port = int(port) if port else 8080
    if not 1 <= port <= 65535:
        raise ValueError(f"bad port: {port}")
    return f"{scheme.lower()}://{host}:{port}{path or '/'}"


def check_marker_size(n):
    if isinstance(n, bool) or not isinstance(n, int) or not MARKER_MIN <= n <= MARKER_MAX:
        raise ValueError(f"marker size must be a whole number from {MARKER_MIN} to {MARKER_MAX}")


def parse_settings(body):
    """Validate the panel's settings (from the browser, or the saved file). Empty source is allowed here;
    start() refuses it."""
    if not isinstance(body, dict):
        raise ValueError("settings must be a JSON object")
    s = {**DEFAULTS, **{k: body[k] for k in DEFAULTS if k in body}}
    if not isinstance(s["source"], str):
        raise ValueError("phone address must be text")
    s["source"] = normalise_source(s["source"]) if s["source"].strip() else ""
    for k in BOOLS:
        if not isinstance(s[k], bool):
            raise ValueError(f"{k} must be true or false")
    if s["marker_size"] is not None:
        check_marker_size(s["marker_size"])
    t = s["tap_min_acc"]
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= TAP_MIN_MAX:  # NaN fails this too
        raise ValueError(f"tap strength must be a number from 0 to {TAP_MIN_MAX}")
    s["tap_min_acc"] = float(t)
    return s


def build_argv(s):
    """tracker.py command line for validated settings. A list for Popen, never a shell string."""
    argv = [venv_python(), tracker_script(), "--source", s["source"], "--view-port", str(VIEW_PORT), "--no-window"]
    if CAMERA.fullmatch(s["source"]):  # a camera on this Mac: no capture times in its frames, and far less lag
        argv[3:4] = [str(camera_index(s["source"]))]
        argv += ["--camera-latency", "0.1"]  # shortcut: a guess for a local or Continuity camera; fit it from a session
    argv += ["--tap-min-acc", str(s["tap_min_acc"])]
    for key, flag in (("show_markers", "--show-markers"), ("touch", "--touch"),
                      ("touch_debug", "--touch-debug"), ("dry_run", "--dry-run")):
        if s[key]:
            argv.append(flag)
    if s["show_markers"] and s["markers_on_demand"]:  # only means something with the markers on
        argv.append("--markers-on-demand")
    if s["marker_size"] is not None:
        argv += ["--marker-size", str(s["marker_size"])]
    return argv


def load_config():
    try:
        with open(config_path()) as f:
            return parse_settings(json.load(f))
    except (OSError, ValueError):  # missing or bad file: start from the defaults
        return parse_settings({})


CONFIG_LOCK = threading.Lock()


def save_config(s):
    path = os.path.abspath(config_path())
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    with CONFIG_LOCK:
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".panel-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(s, f, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):  # only left behind when the write or replace failed
                os.remove(tmp)


def helper_running():
    """imu_taps.py runs as root, so only pgrep (not psutil) can see it without sudo."""
    return subprocess.run(["pgrep", "-f", "imu_taps.py"], capture_output=True).returncode == 0


class Tracker:
    """One tracker.py child at a time. Output is read on a thread; Stop sends SIGINT (the tracker releases
    the mouse in its finally), then SIGTERM after 3 s and SIGKILL after 6 s."""

    def __init__(self):
        self.ctl = threading.Lock()   # serialises start and stop
        self.lock = threading.Lock()  # guards lines and stats
        self.proc = None
        self.lines = deque(maxlen=400)
        self.stats = ""

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, argv):
        with self.ctl:
            if self.running():
                raise RuntimeError("tracker is already running")
            with self.lock:
                self.lines.clear()
                self.stats = ""
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # so output reaches the log as it is printed
            try:  # private like panel.json: the log can hold the phone's address
                os.makedirs(os.path.dirname(tracker_log_path()), exist_ok=True)
                logf = open(tracker_log_path(), "w", encoding="utf-8", opener=lambda f, fl: os.open(f, fl, 0o600))
            except OSError:
                logf = None  # tracking still works; only the copy on disk is missing
            self.proc = subprocess.Popen(argv, cwd=HERE, env=env,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            threading.Thread(target=self._read, args=(self.proc, logf), daemon=True).start()

    def _read(self, proc, logf=None):
        for line in proc.stdout:
            line = line.rstrip("\n")
            with self.lock:
                self.lines.append(line)
                if line.startswith("stats "):
                    self.stats = line
            if logf:
                logf.write(line + "\n")
                logf.flush()
        proc.stdout.close()
        if logf:
            logf.close()

    def stop(self):
        with self.ctl:
            p = self.proc
            if p is None or p.poll() is not None:
                return
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()

    def status(self):
        with self.lock:
            shown = [l for l in self.lines if not l.startswith(NOISE)]
            stats = self.stats
        if self.proc is None:
            state, code = "not started", None
        elif self.proc.poll() is None:
            state, code = "running", None
        else:
            state, code = "stopped", self.proc.returncode
        return {"tracker": {"state": state, "exit_code": code}, "helper": helper_running(), "stats": stats,
                "problems": [l for l in shown if PROBLEM.search(l)][-5:], "lines": shown[-60:]}


def lan_ip():
    """This Mac's IPv4 address on the network the default route uses. A UDP connect only picks a route; it sends nothing."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("192.0.2.1", 9))  # TEST-NET-1: never contacted
        return s.getsockname()[0]


def qr_rows(text):
    """QR code modules as strings of '0'/'1', one per row. OpenCV is already in the venv; imported here so the rest
    of the panel stays standard library."""
    import cv2
    img = cv2.QRCodeEncoder.create().encode(text)
    return ["".join("1" if v < 128 else "0" for v in row) for row in img]


class PairHandler(BaseHTTPRequestHandler):
    """The only thing the panel serves on the Wi-Fi: POST /pair {"code": ...} while a pairing code is open."""
    server_version = "GlassesTouchPair"
    timeout = 5

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/pair":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 < n <= 1024:
            self._json(400, {"error": "bad request"})
            return
        try:
            code = json.loads(self.rfile.read(n)).get("code")
        except (ValueError, UnicodeDecodeError, AttributeError):
            code = None
        if not isinstance(code, str) or not 0 < len(code) <= 64:
            self._json(400, {"error": "bad request"})
            return
        status = self.server.pairing.accept(self.server, code, self.client_address[0])
        if status == 200:
            self._json(200, {"ok": True, "mac": socket.gethostname().removesuffix(".local")})
        else:
            self._json(status, {"error": {403: "wrong code", 410: "code expired"}.get(status, "cannot pair")})


class Pairing:
    """One short-lived listener on the Wi-Fi for one phone. It accepts one POST carrying the one-time code from the
    QR, takes the phone's address from the connection's source address, and closes. Wrong or late codes change nothing."""

    def __init__(self, on_paired, host=None):
        self.lock = threading.Lock()
        self.on_paired = on_paired  # called with the phone's IPv4 address; may raise ValueError
        self.host = host  # tests bind to 127.0.0.1; normal runs use lan_ip()
        self.srv = None
        self.code = None
        self.expires = 0.0
        self.last = None  # time.time() of the last successful pairing

    def open(self):
        """Close any open code, open a new one. Returns the link to put in the QR code."""
        with self.lock:
            self._close()
            host = self.host or lan_ip()
            srv = ThreadingHTTPServer((host, 0), PairHandler)
            srv.daemon_threads = True
            srv.pairing = self
            self.srv, self.code = srv, secrets.token_urlsafe(16)
            self.expires = time.monotonic() + PAIR_SECONDS
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            timer = threading.Timer(PAIR_SECONDS, self.close, args=(srv,))
            timer.daemon = True
            timer.start()
            return PAIR_LINK.format(host=host, port=srv.server_port, code=self.code)

    def accept(self, srv, code, peer):
        with self.lock:
            if srv is not self.srv or time.monotonic() > self.expires:
                return 410
            if not hmac.compare_digest(code.encode(), self.code.encode()):
                return 403
            try:
                self.on_paired(peer)
            except ValueError:
                return 400
            self.last = time.time()
            self._close()
            return 200

    def close(self, srv=None):
        with self.lock:
            if srv is None or srv is self.srv:
                self._close()

    def _close(self):  # caller holds the lock
        srv, self.srv, self.code = self.srv, None, None
        if srv is not None:
            # shutdown() waits for serve_forever, so run it on its own thread (we may be in one of srv's handlers)
            threading.Thread(target=lambda: (srv.shutdown(), srv.server_close()), daemon=True).start()

    def status(self):
        with self.lock:
            left = max(0, round(self.expires - time.monotonic())) if self.srv else 0
            return {"open": self.srv is not None and left > 0, "seconds_left": left, "last": self.last}


PAGE_FILE = os.path.join(HERE, "panel.html")  # read on every load: a page change shows without restarting the app


class Handler(BaseHTTPRequestHandler):
    server_version = "GlassesTouchPanel"
    timeout = 10  # a client that promises a body and never sends it must not hold a thread forever

    def log_message(self, *args):  # keep the Terminal quiet
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _allowed(self, post):
        port = self.server.server_port
        if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self._json(403, {"error": "bad Host header"})
            return False
        if not post:
            return True
        origin = self.headers.get("Origin")
        if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            self._json(403, {"error": "bad Origin"})
            return False
        sent = (self.headers.get(TOKEN_HEADER) or "").encode("latin-1", "replace")
        if not hmac.compare_digest(sent, self.server.token.encode()):
            self._json(403, {"error": "missing or wrong token"})
            return False
        return True

    def do_GET(self):
        if not self._allowed(post=False):
            return
        if self.path == "/":
            with open(PAGE_FILE, encoding="utf-8") as f:
                page = f.read().replace("__TOKEN__", json.dumps(self.server.token))
            page = page.replace("__CONFIG__", json.dumps(self.server.config))
            page = page.replace("__VIEW_URL__", json.dumps(f"http://127.0.0.1:{VIEW_PORT}/"))
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif self.path == "/cameras":
            self._json(200, list_cameras())
        elif self.path == "/status":
            self._json(200, {**self.server.tracker.status(), "source": self.server.config["source"],
                             "pairing": self.server.pairing.status(), "page": os.stat(PAGE_FILE).st_mtime})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._allowed(post=True):
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return
        if n < 0:
            self._json(400, {"error": "bad Content-Length"})
            return
        if n > 8192:
            self._json(413, {"error": "request too large"})
            return
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"error": "body must be JSON"})
            return
        try:
            if self.path == "/config":
                s = parse_settings(body)
                save_config(s)
                self.server.config = s
                self._json(200, {"ok": True, "config": s})
            elif self.path == "/start":
                s = parse_settings(body)
                if not s["source"]:
                    raise ValueError("pair the phone or pick a camera first")
                save_config(s)
                self.server.config = s
                self.server.tracker.start(build_argv(s))
                self._json(200, {"ok": True})
            elif self.path == "/stop":
                self.server.tracker.stop()
                self._json(200, {"ok": True})
            elif self.path == "/pair/start":
                link = self.server.pairing.open()
                self._json(200, {"ok": True, "qr": qr_rows(link), "seconds": PAIR_SECONDS})
            elif self.path == "/pair/cancel":
                self.server.pairing.close()
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except RuntimeError as e:
            self._json(409, {"error": str(e)})
        except ImportError:
            self._json(500, {"error": "pairing needs OpenCV: run the panel with .venv/bin/python"})
        except OSError as e:
            what = "pair: not on a network?" if self.path.startswith("/pair") else "start tracker"
            self._json(500, {"error": f"cannot {what} ({e})"})


class Server(ThreadingHTTPServer):
    daemon_threads = True


def make_server(port, tracker=None, pair_host=None):
    srv = Server(("127.0.0.1", port), Handler)
    srv.token = secrets.token_urlsafe(24)  # new every run; only the served page knows it
    srv.tracker = tracker or Tracker()
    srv.config = load_config()

    def paired(phone_ip):
        s = parse_settings({**srv.config, "source": phone_ip})
        save_config(s)
        srv.config = s
    srv.pairing = Pairing(paired, pair_host)
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local control panel for tracker.py (127.0.0.1 only).")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    srv = make_server(a.port)
    print(f"GlassesTouch panel: http://127.0.0.1:{srv.server_port}/  (Ctrl-C to quit)", flush=True)
    for sig in (signal.SIGTERM, signal.SIGHUP):  # SIGHUP: Terminal window closed; both must reach the finally
        signal.signal(sig, lambda *_: sys.exit(0))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.tracker.stop()  # the tracker must not outlive the panel
        srv.pairing.close()
        srv.server_close()


if __name__ == "__main__":
    main()
