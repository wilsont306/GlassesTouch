# GlassesTouch

Use Ray-Ban Meta glasses to make a MacBook screen act like a touch screen: the glasses' camera watches the screen and your hand, your index fingertip moves the cursor, and a thumb–index pinch holds the click.

- `glasstouch/`: Mac-side tracker (Python). Finds four ArUco markers at the screen corners, tracks the hand with MediaPipe, and drives the cursor with pyautogui.
- `GlassesCam/`: iPhone app, based on Meta's `CameraAccess` sample from the Wearables Device Access Toolkit. It connects to the glasses and shows the live camera view. Next step: forward frames to the Mac over Wi-Fi.

## Tracker

```bash
cd glasstouch
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python test_tracker.py                        # self-test, prints ALL OK
.venv/bin/python tracker.py --make-markers markers.png  # print, stick at screen corners
.venv/bin/python tracker.py --region x,y,w,h --dry-run  # capture a video-call window
.venv/bin/python tracker.py --source http://<phone-ip>:<port>/  # network stream
```

Needs Accessibility and Screen Recording permission for your terminal. Press `q` in the debug window to quit.

## Touch (tap the screen)

`imu_taps.py` reads the MacBook's motion sensor (needs root) and sends tap times to `tracker.py --touch`, which clicks only when the camera sees your fingertip still on the screen at that moment.

1. Once, and again after `imu_taps.py` changes: `sudo ./install_helper.sh` (from the repo root). It copies `imu_taps.py` and `macimu` to a root-owned `/usr/local/libexec/glasstouch` (so nothing your user account can edit runs as root) and loads it as a LaunchDaemon that always runs. Remove it with `sudo ./install_helper.sh --uninstall`.
2. Double-click `GlassesTouch.command`. It opens Terminal and the control panel in your browser (`http://127.0.0.1:8765/`). Enter the phone's address, set the switches, press Start. Stop ends the tracker and releases the mouse. Keep the Terminal window open while you use it: the tracker inherits its Accessibility permission.

Manual fallback (no panel): `cd glasstouch && .venv/bin/python tracker.py --source http://<phone-ip>:8080/ --show-markers --touch`. The helper must be running (step 1), or run it by hand with `sudo /usr/bin/python3 /usr/local/libexec/glasstouch/imu_taps.py --quiet`.

## GlassesCam

Needs Xcode 26.4+, an iPhone on iOS 17.2+, and Developer Mode turned on in the Meta AI app (no Meta developer credentials needed for that). Open `GlassesCam/CameraAccess.xcodeproj`, choose your iPhone, and run.
