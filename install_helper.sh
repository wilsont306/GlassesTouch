#!/bin/bash
# Install the root motion-sensor helper (glasstouch/imu_taps.py + macimu) as a LaunchDaemon, so it always runs.
#   sudo ./install_helper.sh              install or update (run again after imu_taps.py changes)
#   sudo ./install_helper.sh --uninstall  remove the daemon and its files
# DRY_RUN=1 ./install_helper.sh  prints every command instead of running it (no root needed) and writes the
# plist to a temp file, whose path it prints, so it can be checked with plutil -lint.
# Order: unload the daemon, copy into a root-owned staging dir, verify it, swap it into place, load the daemon.
# The daemon is KeepAlive, so it is unloaded first: it can't restart on half-copied files.
set -euo pipefail

LABEL=com.glasstouch.imutaps
REPO="$(cd "$(dirname "$0")" && pwd)"
SRC="$REPO/glasstouch"
DEST=/usr/local/libexec/glasstouch
PARENT=$(dirname "$DEST")
PLIST=/Library/LaunchDaemons/$LABEL.plist
LOG=/var/log/glasstouch-imutaps.log
DRY_RUN=${DRY_RUN:-0}
STAGE=""
OLD=""

run() {
  if [ "$DRY_RUN" = 1 ]; then printf '+ %s\n' "$*"; else "$@"; fi
}

die() {
  echo "$*" >&2
  exit 1
}

# on any exit: remove the stage; if the swap was half done, put the old tree back; reload the old daemon
cleanup() {
  local rc=$?
  if [ "$DRY_RUN" != 1 ]; then
    [ -n "$STAGE" ] && rm -rf "$STAGE"
    if [ -n "$OLD" ] && [ -e "$OLD" ] && [ ! -e "$DEST" ]; then mv "$OLD" "$DEST"; fi
    if [ "$rc" -ne 0 ] && [ -f "$PLIST" ]; then
      echo "install failed (exit $rc); reloading the previous helper if it is still there" >&2
      launchctl bootstrap system "$PLIST" 2>/dev/null || true
    fi
  fi
  exit "$rc"
}

if [ "$DRY_RUN" != 1 ] && [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo: sudo ./install_helper.sh" >&2
  exit 1
fi

if [ "${1:-}" = "--uninstall" ]; then
  run launchctl bootout "system/$LABEL" || true  # fails if it was not loaded: fine
  run rm -f "$PLIST"
  run rm -rf "$DEST"
  echo "uninstalled $LABEL"
  exit 0
fi

MACIMU=$(ls -d "$SRC"/.venv/lib/python3*/site-packages/macimu 2>/dev/null | head -n 1 || true)
if [ ! -f "$SRC/imu_taps.py" ] || [ -z "$MACIMU" ]; then
  die "missing source: need $SRC/imu_taps.py and the venv's macimu package (run pip install -r requirements.txt)"
fi

# a symlink in the sources would let user-writable files be run as root (or root files be copied out): refuse
if [ -L "$SRC/imu_taps.py" ] || [ -n "$(find "$MACIMU" -type l -print -quit)" ]; then
  die "refusing: symlink in $SRC/imu_taps.py or $MACIMU"
fi

trap cleanup EXIT
run launchctl bootout "system/$LABEL" || true  # not loaded yet on first install: fine

run mkdir -p "$PARENT"
if [ "$DRY_RUN" = 1 ]; then
  run mktemp -d "$PARENT/.glasstouch-stage.XXXXXX"
  STAGE="$PARENT/.glasstouch-stage.DRYRUN"
else
  STAGE=$(mktemp -d "$PARENT/.glasstouch-stage.XXXXXX")  # root-owned, mode 700, in the same directory as DEST
fi
run cp -P "$SRC/imu_taps.py" "$STAGE/"
run cp -RP "$MACIMU" "$STAGE/"  # -P: copy symlinks as links, never follow them into root's copy
run find "$STAGE/macimu" -name __pycache__ -type d -prune -exec rm -rf {} +

# re-check the staged copy (what gets run as root), not the sources
if [ "$DRY_RUN" = 1 ]; then
  echo "+ verify $STAGE: imu_taps.py and macimu/ present, no symlinks, FIFOs or devices"
else
  [ -f "$STAGE/imu_taps.py" ] && [ ! -L "$STAGE/imu_taps.py" ] || die "staged imu_taps.py is not a regular file"
  [ -d "$STAGE/macimu" ] || die "staged macimu is missing"
  [ -z "$(find "$STAGE" ! -type f ! -type d)" ] || die "staged tree has a symlink, FIFO or device: refusing"
fi

run chown -R root:wheel "$STAGE"
run chmod -R go-w "$STAGE"

# swap: the old tree moves aside, the staged one moves in (a rename on the same volume), then the old one goes
if [ -e "$DEST" ]; then
  OLD="$PARENT/.glasstouch-old.$$"
  run mv "$DEST" "$OLD"
fi
run mv "$STAGE" "$DEST"
STAGE=""
if [ -n "$OLD" ]; then
  run rm -rf "$OLD"
  OLD=""
fi

write_plist() {
  cat > "$1" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>$DEST/imu_taps.py</string>
    <string>--quiet</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
}

if [ "$DRY_RUN" = 1 ]; then
  TMP_PLIST=$(mktemp -t glasstouch-imutaps)
  write_plist "$TMP_PLIST"
  echo "dry run: plist written to $TMP_PLIST (would install to $PLIST)"
  printf '+ write %s\n' "$PLIST"
else
  write_plist "$PLIST"
fi
run chown root:wheel "$PLIST"
run chmod 644 "$PLIST"

run launchctl bootstrap system "$PLIST"
if [ "$DRY_RUN" = 1 ]; then echo "dry run: nothing installed"; else echo "installed $LABEL (log: $LOG)"; fi
