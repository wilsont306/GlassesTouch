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
2. Double-click `GlassesTouch.command`. It opens Terminal and the control panel in your browser (`http://127.0.0.1:8765/`). Press **Pair phone**, scan the QR code with the iPhone Camera and tap the banner: the phone app sends its address to the Mac (the code works once, for 2 minutes). The address stays blurred in the panel and on the phone until you press Show; you can still type it under "Enter the address yourself". Set the switches and press Start tracking; the switches and the tracker's activity log are under the gear button, top right. Stop ends the tracker and releases the mouse. Keep the Terminal window open while you use it: the tracker inherits its Accessibility permission. Give Terminal **Screen Recording** permission too (System Settings > Privacy & Security): the tracker then also finds the screen by matching what it shows, so the markers don't all need to be in view. Without it, the markers alone are used.

No glasses? Point a camera at the screen instead: an iPhone on a stand (it shows up on the Mac as a Continuity Camera when it uses the same Apple ID) or a USB webcam, placed so the screen and its markers are in view. In the panel, under **1 Pair**, open "Use a camera pointed at the screen instead" and choose the camera, then Start tracking; Practice mode shows what it sees without moving the mouse. macOS asks once to let the app running the tracker (Terminal or GlassesTouch) use the camera; allow it and start again.

Or use the panel as a Mac app instead of Terminal and a browser tab: run `./make_app.sh` (from the repo root). It installs **GlassesTouch** in Applications (open it from Launchpad, Spotlight or the Dock) and copies `glasstouch/` and its `.venv` to `~/Library/Application Support/GlassesTouch`, so the app does not need this folder; run `./make_app.sh` again after changing the code (quit the app first). It still needs Homebrew's python@3.12, which the `.venv` is built on. The app opens the panel in its own window; closing the window or Cmd-Q stops the tracker. The app has its own permissions: the first time you press Start tracking, macOS asks to give GlassesTouch **Accessibility**; turn it on in System Settings > Privacy & Security > Accessibility, quit the app and open it again. Turn it on under **Screen Recording** (and **Camera** for a camera source) too. Reinstalling builds the same app, so the permissions stay. Settings and the last tracker run's output (`tracker.log`) are kept in `~/Library/Application Support/GlassesTouch/build/`; running the panel from this folder uses `build/` here.

Screen markers: by default they stay hidden and appear only when the tracker can't find the screen from what it shows (a plain page, a blank window), then fade out once it can again (settings: **Screen markers** on, **Hide markers until needed** on; needs Screen Recording for GlassesTouch). Turn **Hide markers until needed** off to keep them showing, or **Screen markers** off to never show them.

Manual fallback (no panel): `cd glasstouch && .venv/bin/python tracker.py --source http://<phone-ip>:8080/ --show-markers --markers-on-demand --touch`. The helper must be running (step 1), or run it by hand with `sudo /usr/bin/python3 /usr/local/libexec/glasstouch/imu_taps.py --quiet`.

## GlassesCam

Needs Xcode 26.4+, an iPhone on iOS 17.2+, and Developer Mode turned on in the Meta AI app (no Meta developer credentials needed for that). Open `GlassesCam/CameraAccess.xcodeproj`, choose your iPhone, and run.

## Stopping

Fold the glasses' arms (or take them off and fold them): the stream stops and the tracker stops itself within about 3 s, releasing the mouse. The panel's Stop button does the same. The capture button on the glasses does nothing during a stream. To start again: Start Session → Preview in the phone app, then Start in the panel.

## License

MIT for GlassesTouch's own code (see `LICENSE`). `GlassesCam/` and `GlassesCamBT/` are built on Meta's Wearables Device Access Toolkit sample app and stay under Meta's terms (the `LICENSE` file in each folder).
