#!/usr/bin/env python3
"""Local control panel for tracker.py: phone address, switches, Start/Stop, status and log.

Standard library only. Serves 127.0.0.1 only. Every POST needs the per-run token (X-Panel-Token), a Host
header naming 127.0.0.1 or localhost (DNS rebinding), and an Origin that matches if one is sent.
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
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_HEADER = "X-Panel-Token"
VIEW_PORT = 8767  # the tracker's --view-port: its annotated debug frame, shown in the preview while it runs
MARKER_MIN, MARKER_MAX = 40, 1000  # tracker.py refuses < 40; the upper bound is a sanity limit
NOISE = ("I0000", "W0000", "INFO: Created")  # MediaPipe / TF startup chatter, hidden from the log
PROBLEM = re.compile(r"cannot|Accessibility|rror|Traceback|died|exited")
TAP_MIN_DEFAULT, TAP_MIN_MAX = 0.003, 0.05  # --tap-min-acc (g): the panel's range; tracker's max stays 0.08
DEFAULTS = {"source": "", "touch": True, "show_markers": True, "dry_run": False, "touch_debug": True,
            "marker_size": None, "tap_min_acc": TAP_MIN_DEFAULT}
BOOLS = ("touch", "show_markers", "dry_run", "touch_debug")


def tracker_script():
    # GLASSTOUCH_TRACKER_SCRIPT lets the HTTP test substitute a stub; normal runs use tracker.py
    return os.environ.get("GLASSTOUCH_TRACKER_SCRIPT") or os.path.join(HERE, "tracker.py")


def venv_python():
    p = os.path.join(HERE, ".venv", "bin", "python")
    return p if os.path.exists(p) else sys.executable


def config_path():
    return os.environ.get("GLASSTOUCH_PANEL_CONFIG") or os.path.join(HERE, "..", "build", "panel.json")


def normalise_source(text):
    """A bare IPv4 address or an http(s) URL -> 'http://<host>:<port><path>' (port 8080 and path / by default).
    Raises ValueError for anything else: other schemes, credentials, spaces, shell characters."""
    if len(text) > 2048:
        raise ValueError("address is too long")
    s = text.strip()
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
    argv += ["--tap-min-acc", str(s["tap_min_acc"])]
    for key, flag in (("show_markers", "--show-markers"), ("touch", "--touch"),
                      ("touch_debug", "--touch-debug"), ("dry_run", "--dry-run")):
        if s[key]:
            argv.append(flag)
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
            self.proc = subprocess.Popen(argv, cwd=HERE, env=env,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            line = line.rstrip("\n")
            with self.lock:
                self.lines.append(line)
                if line.startswith("stats "):
                    self.stats = line
        proc.stdout.close()

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


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GlassesTouch</title>
<style>
:root { color-scheme: light dark; --bg:#f5f5f7; --fg:#1d1d1f; --card:#ffffff; --line:#d2d2d7; --muted:#6e6e73;
        --accent:#0a6cdb; --bad:#c62828; --good:#2e7d32; --mono: ui-monospace, Menlo, monospace; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1c1c1e; --fg:#f2f2f7; --card:#2c2c2e; --line:#48484a; --muted:#a1a1a6;
          --accent:#0a84ff; --bad:#ff6961; --good:#30d158; } }
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin:0; font: 15px -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--fg); }
main { max-width: 960px; margin: 0 auto; padding: 20px; display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }
@media (max-width: 720px) { main { grid-template-columns: 1fr; } }
h1 { margin: 0; font-size: 20px; grid-column: 1 / -1; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; min-width: 0; }
section.wide { grid-column: 1 / -1; }
h2 { margin: 0 0 10px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 6px 0; }
.hint { color: var(--muted); font-size: 13px; margin: 6px 0 0; }
input[type=text], input[type=number] { font: inherit; color: var(--fg); background: var(--bg);
        border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; }
input[type=text] { width: 100%; }
input[type=number] { width: 90px; }
input[type=checkbox] { width: 18px; height: 18px; accent-color: var(--accent); }
button { font: inherit; padding: 8px 18px; border-radius: 6px; border: 1px solid var(--line);
         background: var(--card); color: var(--fg); cursor: pointer; }
button.go { background: var(--accent); border-color: transparent; color: #fff; }
button:disabled { opacity: .45; cursor: default; }
img#preview { width: 100%; min-height: 140px; border-radius: 6px; background: #000; display: block; }
pre { margin: 8px 0 0; height: 300px; overflow: auto; padding: 8px; font: 12px/1.4 var(--mono);
      background: var(--bg); border: 1px solid var(--line); border-radius: 6px; white-space: pre-wrap; word-break: break-all; }
.problem { color: var(--bad); font-weight: 600; margin: 4px 0; }
.good { color: var(--good); }
#msg { color: var(--bad); min-height: 1.3em; margin: 8px 0 0; }
h3 { margin: 14px 0 6px; font-size: 13px; color: var(--muted); font-weight: 600; }
.targets { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.target { min-width: 120px; min-height: 80px; display: flex; flex-direction: column; align-items: center;
          justify-content: center; gap: 6px; padding: 8px; }
.target.flash { background: var(--accent); border-color: transparent; color: #fff; }
.scroll-x { overflow-x: auto; }
.field { position: relative; height: 220px; min-width: 500px; background: var(--bg); border: 1px dashed var(--line);
         border-radius: 6px; overflow: hidden; }
.box { position: absolute; left: 20px; top: 65px; width: 90px; height: 90px; padding: 4px; touch-action: none;
       user-select: none; cursor: grab; font-size: 13px; }
.list { position: relative; height: 260px; overflow-y: auto; background: var(--bg); border: 1px solid var(--line);
        border-radius: 6px; }
.list div { height: 36px; line-height: 36px; padding: 0 10px; border-bottom: 1px solid var(--line); }
.strip { overflow-x: auto; white-space: nowrap; background: var(--bg); border: 1px solid var(--line); border-radius: 6px; }
.strip span { display: inline-block; width: 120px; height: 60px; line-height: 60px; text-align: center;
              border-right: 1px solid var(--line); }
</style></head>
<body><main>
<h1>GlassesTouch</h1>

<section>
  <h2>Phone</h2>
  <input type="text" id="source" placeholder="10.0.0.5 or http://phone.local:8080/"
         spellcheck="false" autocapitalize="off" autocomplete="off">
  <p class="hint">The phone's stream address. Saved when you leave the field.</p>
  <img id="preview" alt="Live preview from the phone" hidden>
  <p id="nopreview" class="hint">Enter the phone's address to see the preview.</p>
</section>

<section>
  <h2>Options</h2>
  <label class="row">Touch: click on taps <input type="checkbox" id="touch"></label>
  <label class="row">Show markers in screen corners <input type="checkbox" id="show_markers"></label>
  <label class="row">Test mode: dry run, no mouse <input type="checkbox" id="dry_run"></label>
  <label class="row">Debug: one line per tap <input type="checkbox" id="touch_debug"></label>
  <label class="row">Marker size (points) <input type="number" id="marker_size" min="40" max="1000"
         placeholder="140"></label>
  <label class="row">Tap strength (min) <input type="number" id="tap_min_acc" min="0" max="0.05" step="0.001"></label>
</section>

<section class="wide">
  <h2>Tracker</h2>
  <p><button class="go" id="start">Start</button> <button id="stop" disabled>Stop</button></p>
  <p id="state">Tracker: checking...</p>
  <p id="helper"></p>
  <p id="stats" class="hint"></p>
  <div id="problems"></div>
  <pre id="log" aria-label="Tracker output"></pre>
  <p id="msg" role="alert"></p>
</section>

<section class="wide" id="test-area" aria-labelledby="test-h">
  <h2 id="test-h">Test area</h2>
  <p class="hint">Touch these with the glasses to check clicks, double-clicks, drags and scrolling. Nothing here
    leaves this page.</p>
  <div class="targets">
    <button type="button" class="target" id="t-a"><span>Click target A</span><span id="n-a">Clicks: 0</span></button>
    <button type="button" class="target" id="t-b"><span>Click target B</span><span id="n-b">Clicks: 0</span></button>
    <button type="button" class="target" id="t-c"><span>Click target C</span><span id="n-c">Clicks: 0</span></button>
    <button type="button" class="target" id="t-dbl"><span>Double-click target</span><span id="n-dbl">Double-clicks: 0</span></button>
  </div>

  <h3>Drag</h3>
  <div class="scroll-x">
    <div id="field" class="field" role="group" aria-label="Drag field">
      <button type="button" id="box" class="box" aria-label="Draggable box">Drag me</button>
    </div>
  </div>
  <p id="n-drag">Drags: 0</p>
  <p id="drop" class="hint">Last drop: none yet</p>

  <h3>Scroll</h3>
  <div id="list" class="list" role="region" aria-label="Scrollable list of 100 rows" tabindex="0"></div>
  <p>Top row: <span id="top">1</span></p>
  <div id="strip" class="strip" role="region" aria-label="Horizontal strip of 40 cells" tabindex="0"></div>

  <p>Last touch: <span id="last">none yet</span></p>
  <p><button type="button" id="reset">Reset test area</button></p>
</section>
</main>
<script>
const TOKEN = __TOKEN__;
const CONFIG = __CONFIG__;
const VIEW_URL = __VIEW_URL__;
const BOOL_KEYS = ["touch", "show_markers", "dry_run", "touch_debug"];
const $ = id => document.getElementById(id);

function setPreview(src) {
  const img = $("preview"), hint = $("nopreview");
  if (!src) {
    img.hidden = true;
    img.removeAttribute("src");
    hint.textContent = "Enter the phone's address to see the preview.";
    hint.hidden = false;
  } else if (img.getAttribute("src") !== src || img.hidden) {
    hint.hidden = true;
    img.hidden = false;
    img.src = src;
  }
}
let phoneSrc = "", trackerRunning = false;
function updatePreview() {  // the tracker's view while it runs, else the phone's stream
  setPreview(trackerRunning ? VIEW_URL : phoneSrc);
}
$("preview").onerror = () => {  // no broken-image icon: say what to check instead
  $("preview").hidden = true;
  $("nopreview").textContent = trackerRunning
    ? "No picture from the tracker yet. Check that it is running."
    : "No picture from the phone. Check the address and that the stream is on.";
  $("nopreview").hidden = false;
};

function showConfig(c) {
  $("source").value = c.source;
  BOOL_KEYS.forEach(k => { $(k).checked = c[k]; });
  $("marker_size").value = c.marker_size === null ? "" : c.marker_size;
  $("tap_min_acc").value = c.tap_min_acc;
  phoneSrc = c.source;
  updatePreview();
}

function settings() {
  const s = { source: $("source").value };
  BOOL_KEYS.forEach(k => { s[k] = $(k).checked; });
  const m = $("marker_size").value.trim();
  s.marker_size = m === "" ? null : (/^\d+$/.test(m) ? Number(m) : m);  // a non-number goes through and is refused
  const t = $("tap_min_acc").value.trim();
  s.tap_min_acc = t === "" ? null : Number(t);  // null (empty or not a number) is refused by the server
  return s;
}

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Panel-Token": TOKEN },
                                body: JSON.stringify(body || {}) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}

function fail(e) { $("msg").textContent = e.message; }

async function saveConfig() {
  try { $("msg").textContent = ""; showConfig((await post("/config", settings())).config); } catch (e) { fail(e); }
}

$("start").onclick = async () => {
  try { $("msg").textContent = ""; await post("/start", settings()); } catch (e) { fail(e); }
  poll();
};
$("stop").onclick = async () => {
  try { $("msg").textContent = ""; await post("/stop"); } catch (e) { fail(e); }
  poll();
};
$("source").addEventListener("change", saveConfig);
BOOL_KEYS.concat(["marker_size", "tap_min_acc"]).forEach(k => $(k).addEventListener("change", saveConfig));

async function poll() {
  let s;
  try { s = await (await fetch("/status", { cache: "no-store" })).json(); } catch (e) { return; }
  const t = s.tracker;
  $("state").textContent = t.state === "stopped" ? `Tracker: stopped (exit code ${t.exit_code})`
                         : "Tracker: " + t.state;
  $("state").className = t.state === "running" ? "good" : "";
  trackerRunning = t.state === "running";
  updatePreview();
  $("helper").textContent = "Touch helper (imu_taps.py, root): " + (s.helper ? "running" : "not running");
  $("helper").className = s.helper ? "good" : "";
  $("stats").textContent = s.stats;
  $("problems").replaceChildren(...s.problems.map(text => {
    const d = document.createElement("div");
    d.className = "problem";
    d.textContent = text;
    return d;
  }));
  const log = $("log");
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = s.lines.join("\n");
  if (atEnd) log.scrollTop = log.scrollHeight;
  $("start").disabled = t.state === "running";
  $("stop").disabled = t.state !== "running";
}

// Test area: purely on this page. No requests, no token.
const ta = { a: 0, b: 0, c: 0, dbl: 0, drags: 0 };
const taSection = $("test-area"), field = $("field"), box = $("box"), list = $("list"), strip = $("strip");
function showCounts() {
  $("n-a").textContent = "Clicks: " + ta.a;
  $("n-b").textContent = "Clicks: " + ta.b;
  $("n-c").textContent = "Clicks: " + ta.c;
  $("n-dbl").textContent = "Double-clicks: " + ta.dbl;
  $("n-drag").textContent = "Drags: " + ta.drags;
}
function flash(el) {
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 250);
}
["a", "b", "c"].forEach(k => {
  $("t-" + k).addEventListener("click", () => { ta[k] += 1; showCounts(); flash($("t-" + k)); });
});
$("t-dbl").addEventListener("dblclick", () => { ta.dbl += 1; showCounts(); flash($("t-dbl")); });

let drag = null;  // the box's pointer: {id, dx, dy, moved, x, y} while it is down
box.addEventListener("pointerdown", e => {
  box.setPointerCapture(e.pointerId);
  const b = box.getBoundingClientRect();
  drag = { id: e.pointerId, dx: e.clientX - b.left, dy: e.clientY - b.top, moved: false };
});
box.addEventListener("pointermove", e => {
  if (!drag || e.pointerId !== drag.id) return;
  const f = field.getBoundingClientRect();  // keep the box inside the field
  drag.x = Math.min(Math.max(e.clientX - drag.dx - f.left - field.clientLeft, 0), field.clientWidth - box.offsetWidth);
  drag.y = Math.min(Math.max(e.clientY - drag.dy - f.top - field.clientTop, 0), field.clientHeight - box.offsetHeight);
  box.style.left = drag.x + "px";
  box.style.top = drag.y + "px";
  drag.moved = true;
});
function endDrag(e) {
  if (!drag || e.pointerId !== drag.id) return;
  if (drag.moved) {
    ta.drags += 1;
    showCounts();
    $("drop").textContent = `Last drop: x ${Math.round(drag.x)}, y ${Math.round(drag.y)} px from the field's top left`;
  }
  drag = null;
}
box.addEventListener("pointerup", endDrag);
box.addEventListener("pointercancel", endDrag);

for (let i = 1; i <= 100; i++) {
  const row = document.createElement("div");
  row.textContent = "Row " + i;
  list.appendChild(row);
}
for (let i = 1; i <= 40; i++) {
  const cell = document.createElement("span");
  cell.textContent = "Cell " + i;
  strip.appendChild(cell);
}
function topRow() {  // the first row whose bottom edge is below the top of the list
  const rows = list.children;
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].offsetTop + rows[i].offsetHeight > list.scrollTop) return i + 1;
  }
  return rows.length;
}
list.addEventListener("scroll", () => { $("top").textContent = topRow(); });
taSection.addEventListener("pointerdown", e => {
  $("last").textContent = `x ${Math.round(e.pageX)}, y ${Math.round(e.pageY)}`;
});
$("reset").addEventListener("click", () => {
  Object.assign(ta, { a: 0, b: 0, c: 0, dbl: 0, drags: 0 });
  showCounts();
  box.style.left = "";
  box.style.top = "";
  list.scrollTop = 0;
  strip.scrollLeft = 0;
  $("top").textContent = 1;
  $("drop").textContent = "Last drop: none yet";
  $("last").textContent = "none yet";
});
showCounts();

showConfig(CONFIG);
poll();
setInterval(poll, 1000);
</script>
</body></html>
"""


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
            page = PAGE.replace("__TOKEN__", json.dumps(self.server.token))
            page = page.replace("__CONFIG__", json.dumps(self.server.config))
            page = page.replace("__VIEW_URL__", json.dumps(f"http://127.0.0.1:{VIEW_PORT}/"))
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif self.path == "/status":
            self._json(200, self.server.tracker.status())
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
                    raise ValueError("enter the phone address first")
                save_config(s)
                self.server.config = s
                self.server.tracker.start(build_argv(s))
                self._json(200, {"ok": True})
            elif self.path == "/stop":
                self.server.tracker.stop()
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except RuntimeError as e:
            self._json(409, {"error": str(e)})
        except OSError as e:
            self._json(500, {"error": f"cannot start tracker: {e}"})


class Server(ThreadingHTTPServer):
    daemon_threads = True


def make_server(port, tracker=None):
    srv = Server(("127.0.0.1", port), Handler)
    srv.token = secrets.token_urlsafe(24)  # new every run; only the served page knows it
    srv.tracker = tracker or Tracker()
    srv.config = load_config()
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
        srv.server_close()


if __name__ == "__main__":
    main()
