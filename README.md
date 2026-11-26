# GlassesTouch

Turn a MacBook screen into a touchscreen. A camera watches the screen and your hand: Ray-Ban Meta glasses you wear, or any camera pointed at the screen. Your index fingertip moves the cursor, and the MacBook's built-in motion sensor feels the moment your finger hits the glass, so tapping the screen clicks.

| Gesture | Does |
|---|---|
| Tap | Click |
| Two quick taps | Double-click |
| Tap, keep the finger down a moment, slide | Drag |
| Two fingers together, slide | Scroll |

The app's tutorial, with a voice that reads each step, walks you through all four.

## What's in here

- `glasstouch/`: the Mac side (Python). `tracker.py` finds the screen in the camera view (from what the screen shows, or from four markers it puts in the corners when that fails), tracks your hand with MediaPipe, and moves and clicks the mouse. `panel.py` and `panel.html` are the control panel, `app.py` shows it as a Mac app, and `imu_taps.py` reads the motion sensor.
- `GlassesCamBT/`: the iPhone app. It streams the glasses' camera (or the phone's own) to the Mac over Wi-Fi and pairs with the Mac by QR code. Built on Meta's `CameraAccess` sample from the Wearables Device Access Toolkit.
- `GlassesCam/`: the first version of the iPhone app, which only shows the glasses' view.

## What you need

- An Apple Silicon MacBook: taps are felt through its built-in motion sensor, read with [macimu](https://github.com/olvvier/apple-silicon-accelerometer).
- Homebrew's Python 3.12 (`brew install python@3.12`) and the Xcode command line tools (`xcode-select --install`).
- A camera that sees the screen, one of:
  - Ray-Ban Meta glasses, through the iPhone app (below);
  - an iPhone on a stand as a Continuity Camera (signed in to the same Apple ID as the Mac), or a USB webcam.

## Quick start

```bash
git clone https://github.com/wilsont306/GlassesTouch.git
cd GlassesTouch/glasstouch
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python test_tracker.py   # self-test, prints ALL OK
cd ..
sudo ./install_helper.sh           # the tap sensor helper, once
./make_app.sh                      # installs GlassesTouch in Applications
```

Open **GlassesTouch**. Under **1 Pair**, press **Pair phone** and scan the QR code with the iPhone (for the glasses), or open "Use a camera pointed at the screen instead" and choose a camera. Press **Start tracking**. The first time, macOS asks to give GlassesTouch **Accessibility** (to move and click the mouse): turn it on in System Settings > Privacy & Security > Accessibility, quit the app and open it again. Turn it on under **Screen Recording** (to find the screen from what it shows) and, for a camera source, **Camera** too. Then start the tutorial. On its first run the tracker downloads MediaPipe's hand model (7.5 MB).

## The tap sensor helper

`imu_taps.py` reads the MacBook's motion sensor (which needs root) and sends tap times to the tracker, which clicks only when the camera sees your fingertip still on the screen at that moment. `sudo ./install_helper.sh` copies `imu_taps.py` and the venv's `macimu` package to a root-owned `/usr/local/libexec/glasstouch` (so nothing your user account can edit runs as root) and loads it as a LaunchDaemon that always runs. Run it again after `imu_taps.py` changes; remove it with `sudo ./install_helper.sh --uninstall`.

## The app

`./make_app.sh` installs **GlassesTouch** in Applications (open it from Launchpad, Spotlight or the Dock) and copies `glasstouch/` and its `.venv` to `~/Library/Application Support/GlassesTouch`, so the app does not need the repo folder. Run it again after changing the code (quit the app first). The app still needs Homebrew's python@3.12, which the `.venv` is built on. Reinstalling builds the same app, so the permissions stay. Settings and the last tracker run's output (`tracker.log`) are kept in `~/Library/Application Support/GlassesTouch/build/`.

The app shows the control panel in its own window; closing the window or Cmd-Q stops the tracker. Pairing: the QR code works once, for 2 minutes, and the phone app sends its address to the Mac. The address stays blurred in the panel and on the phone until you press Show; you can also type it under "Type the address instead". The switches and the tracker's activity log are under the gear button, top right. Stop ends the tracker and releases the mouse.

Without the app: double-click `GlassesTouch.command`. It opens Terminal and the same panel in your browser (`http://127.0.0.1:8765/`). Keep the Terminal window open, and give Terminal the Accessibility and Screen Recording permissions instead (the tracker inherits them).

Screen markers: by default they stay hidden and appear only when the tracker can't find the screen from what it shows (a plain page, a blank window), then fade out once it can again (settings: **Screen markers** on, **Hide markers until needed** on; needs Screen Recording). Turn **Hide markers until needed** off to keep them showing, or **Screen markers** off to never show them.

## iPhone app

Needs Xcode 26.4+, an iPhone on iOS 17.2+, and Developer Mode turned on in the Meta AI app (no Meta developer credentials needed for that). Open `GlassesCamBT/CameraAccess.xcodeproj`, set your own team and a bundle identifier of your own under Signing & Capabilities, choose your iPhone and run. The first time, trust your developer account on the phone (Settings > General > VPN & Device Management). A free Apple account works, but its apps stop opening after 7 days: run it from Xcode again.

No glasses? The app can send the phone's own camera instead: put the phone on a stand, pointed at the screen.

## Command line

```bash
cd glasstouch
.venv/bin/python tracker.py --source http://<phone-ip>:8080/ --show-markers --markers-on-demand --touch
.venv/bin/python tracker.py --help
```

Needs the Accessibility and Screen Recording permissions for your terminal. The tap helper must be running, or run it by hand with `sudo /usr/bin/python3 /usr/local/libexec/glasstouch/imu_taps.py --quiet`.

## Stopping

Fold the glasses' arms (or take them off and fold them): the stream stops and the tracker stops itself within about 3 s, releasing the mouse. The panel's Stop button does the same. The capture button on the glasses does nothing during a stream. To start again: Start Session → Preview in the phone app, then Start in the panel.

## License

MIT for GlassesTouch's own code (see `LICENSE`). `GlassesCam/` and `GlassesCamBT/` are built on Meta's Wearables Device Access Toolkit sample app and stay under Meta's terms (the `LICENSE` file in each folder).
