#!/bin/bash
# Installs GlassesTouch as a Mac app: /Applications/GlassesTouch.app opens the control panel in its own window
# (glasstouch/app.py). It runs a copy of glasstouch/ and its .venv kept in ~/Library/Application Support/GlassesTouch,
# so it works wherever this project folder is; rerun this after changing the code. Settings and logs go to build/
# beside that copy. macOS asks for Accessibility and Screen Recording for "GlassesTouch" the first time it needs them.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/glasstouch/.venv/bin/python"
APP="/Applications/GlassesTouch.app"
SUPPORT="$HOME/Library/Application Support/GlassesTouch"
[ -x "$PY" ] || { echo "missing $PY: set up glasstouch/.venv first (see README)"; exit 1; }

# the code and its Python live outside the bundle, so a code change leaves the bundle (and the signature macOS files
# the app's permissions under) the same
mkdir -p "$SUPPORT/build"
rsync -a --delete --exclude '/__pycache__' --exclude '/testdata' --exclude '/test_*.py' "$ROOT/glasstouch/" "$SUPPORT/glasstouch/"
if [ -f "$ROOT/build/panel.json" ] && [ ! -f "$SUPPORT/build/panel.json" ]; then
  cp -p "$ROOT/build/panel.json" "$SUPPORT/build/"  # first install: keep the settings from in-place runs
fi
"$SUPPORT/glasstouch/.venv/bin/python" -c "import AppKit, cv2, mediapipe" || { echo "the copied .venv does not run"; exit 1; }

APP_TMP="$(mktemp -d)/GlassesTouch.app"  # built aside, then swapped in
mkdir -p "$APP_TMP/Contents/MacOS" "$APP_TMP/Contents/Resources"

# icon: a pair of glasses on a rounded square, drawn with OpenCV, then every size macOS wants
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
"$PY" - "$ICONSET/icon_512x512@2x.png" <<'PYEOF'
import sys, cv2, numpy as np
n = 1024
rgb = np.full((n, n, 3), (238, 242, 246), np.uint8)                       # paper (BGR)
alpha = np.zeros((n, n), np.uint8)                                         # rounded square
cv2.rectangle(alpha, (280, 100), (744, 924), 255, -1)
cv2.rectangle(alpha, (100, 280), (924, 744), 255, -1)
for cx, cy in ((280, 280), (744, 280), (744, 744), (280, 744)):
    cv2.circle(alpha, (cx, cy), 180, 255, -1, cv2.LINE_AA)
frame = (26, 28, 30)
for x0, lens in ((170, (40, 40, 40)), (548, (88, 108, 52))):               # left lens: the camera; right: tinted
    pts = np.int32([[x0, 380], [x0 + 306, 360], [x0 + 290, 560], [x0 + 230, 640], [x0 + 70, 640], [x0 + 16, 560]])
    cv2.fillPoly(rgb, [pts], lens, cv2.LINE_AA)
    cv2.polylines(rgb, [pts], True, frame, 46, cv2.LINE_AA)
cv2.line(rgb, (476, 395), (548, 395), frame, 46, cv2.LINE_AA)            # bridge
cv2.circle(rgb, (200, 395), 22, (255, 255, 255), -1, cv2.LINE_AA)       # capture LED
cv2.imwrite(sys.argv[1], np.dstack([rgb, alpha]))
PYEOF
for s in 16 32 128 256 512; do
  sips -z $s $s "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP_TMP/Contents/Resources/AppIcon.icns"
cp "$ICONSET/icon_512x512@2x.png" "$APP_TMP/Contents/Resources/AppIcon.png"  # app.py sets the Dock icon from this

cat > "$APP_TMP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>GlassesTouch</string>
  <key>CFBundleDisplayName</key><string>GlassesTouch</string>
  <key>CFBundleIdentifier</key><string>local.glasstouch.panel</string>
  <key>CFBundleExecutable</key><string>GlassesTouch</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key><string>GlassesTouch watches the screen and your hand through a camera you point at the screen.</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict>
</plist>
PLIST

# launcher: a compiled program, not a script. macOS files permissions (Accessibility, Camera) under the program the
# app starts as: a script made that /bin/bash (any bash script would share the grant) and could start under Rosetta.
# No paths are compiled in, so every build is the same program and macOS keeps the permissions across reinstalls.
cat > "$(dirname "$ICONSET")/launcher.c" <<'CEOF'
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
/* GlassesTouch.app's program (built by make_app.sh): runs the panel window, app.py, from the installed copy */
int main(void) {
    char exe[PATH_MAX], icon[PATH_MAX], code[PATH_MAX], py[PATH_MAX];
    uint32_t n = sizeof exe;
    const char *home = getenv("HOME");
    if (_NSGetExecutablePath(exe, &n) == 0) {
        snprintf(icon, sizeof icon, "%s/../Resources/AppIcon.png", dirname(exe));
        setenv("GLASSTOUCH_ICON", icon, 1);
    }
    if (!home) { fputs("GlassesTouch: HOME is not set\n", stderr); return 1; }
    snprintf(code, sizeof code, "%s/Library/Application Support/GlassesTouch/glasstouch", home);
    snprintf(py, sizeof py, "%s/.venv/bin/python", code);
    if (chdir(code) != 0) { perror("GlassesTouch: installed code (run make_app.sh)"); return 1; }
    execl(py, py, "app.py", (char *)NULL);
    perror("GlassesTouch: installed .venv python (run make_app.sh)");  /* only if the exec failed */
    return 1;
}
CEOF
clang -O2 -Wall -Werror -o "$APP_TMP/Contents/MacOS/GlassesTouch" "$(dirname "$ICONSET")/launcher.c"
codesign --force --sign - "$APP_TMP" >/dev/null 2>&1 || true  # ad-hoc: lets macOS remember the app's permissions

rm -rf "$APP" "$ROOT/GlassesTouch.app"  # the old in-project build has the same bundle id: keep one app
mv "$APP_TMP" "$APP"
echo "installed $APP (code in $SUPPORT/glasstouch)"
