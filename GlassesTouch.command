#!/bin/bash
# Double-click this in Finder. Finder opens it in Terminal, and the tracker started by the panel inherits
# Terminal's Accessibility permission (macOS needs it to move the mouse). Keep this window open while you use it.
cd "$(dirname "$0")/glasstouch" || exit 1
( sleep 1.5; open "http://127.0.0.1:8765/" ) &  # give the panel a moment to start listening
exec .venv/bin/python panel.py --port 8765
