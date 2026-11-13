# Decisions

- **Video path: a phone app using Meta's DAT kit, not a web app.** Per Meta's docs, only the Device Access Toolkit can use the glasses' camera. Glasses Web Apps run only on Ray-Ban Display glasses and have no camera access. A WhatsApp or Messenger video call plus `--region` screen capture is the fallback that needs no code.
- **Start from Meta's CameraAccess sample** instead of writing a new app.
- **Use SDK 0.5.0 (`GlassesCamBT/`) for now:** it's the last version that streams over Bluetooth, which works on a free Apple account. Move to 1.x (`GlassesCam/`) if a paid membership is added or 0.5.0 stops working.
- **Screen location from 4 printed ArUco markers** (DICT_4X4_50, ids 0–3 = TL, TR, BR, BL) and a per-frame homography.
- **On-screen overlay markers (`--show-markers`) instead of printed ones:** exact known positions, no printing; any 2 markers suffice (corner-based RANSAC homography).
- **Track only static screen regions between marker sightings** (marker windows + bezel), because tracking the whole screen follows scrolling content.
- **Touch = camera for *where*, tap sensor for *when*:** a single camera can't tell touching from hovering. Choosing between mic and motion sensor by experiment (`tap_probe.py`).
- **Click by pinch with hysteresis** (enter below 0.30, exit above 0.45). The DAT experimental Inputs module (capture button or touchpad) is a possible, more reliable alternative.
- **Developer Mode in the Meta AI app** for testing; Developer Center release channels are only needed for sharing.
- **hvc1 codec on the phone** so the stream keeps running with the app backgrounded or the phone locked; costs one resolution step (504x896).
- **Touch mode = touchscreen semantics:** no hover cursor; click only on camera-gated motion-sensor taps.
- **Touchscreen gestures (user choice):** one finger: tap = click, touch-and-move = drag; index+middle together: touch-and-move = scroll (like a Mac trackpad). Touch-down comes from motion-sensor taps; lift-off must come from the camera (finger stops, leaves the screen or the hand pulls back), so gesture ends are heuristic.
- **Local web panel instead of a native app:** stdlib-only, launched by a .command file so the tracker inherits Terminal's Accessibility permission; the root helper runs as a LaunchDaemon so nothing needs sudo per session.
- **Hold-to-drag:** one-finger drag = touch, hold 0.35 s, move (click, then down/moves/up), because the withdrawal after a tap looks like movement to a single camera. Two-finger scroll starts immediately. Clicks land ~0.35 s after the touch.
- **Gestures v3 (replaces hold-to-drag):** tap = click immediately; drag = double-tap-and-drag (second tap 0.4–0.9 s after a click, within 60 pt, then move); two fingers = scroll, ending as soon as the pose breaks or the fingertip leaves the screen. Reason: with no lift-off sensor, "tap, rest, pull away" looked like hold-then-drag and produced phantom drags; a second sensed tap can't be faked by pulling away. Taps < 0.4 s apart are dropped as ringing (helper refractory and TOUCH_MIN_GAP), hence the 0.4 s lower bound.
- **(Reverted: the shift felt "a lot worse" live, TIP_DY = 0.) Calibration = a constant fingertip shift (TIP_DY), not a fitted warp:** one session's 10 points gave 54 → 43 pt with a shift and 38 pt with an affine; the remaining error is scatter, so the extra parameters would mostly fit noise. Re-run calibrate.html if the glasses fit or seating position changes.
