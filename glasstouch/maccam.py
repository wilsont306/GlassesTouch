"""Cameras on this Mac, as OpenCV sees them (shared by panel.py's camera list and tracker.py's camera source).

OpenCV's macOS backend (cap_avfoundation_mac.mm) numbers cameras by joining AVCaptureDevice devicesWithMediaType:
video and muxed and sorting them by uniqueID, and opens each with the session preset Medium (about 480x360), never
changing the device's format; a CAP_PROP_FRAME_WIDTH/HEIGHT request snaps back to that format. So the camera list
must be built the same way, and the tracker switches the opened camera to a bigger format itself.
"""
import functools
import re

MAX_AREA = 1920 * 1440  # the iPhone's 4:3 format (its whole sensor, the widest view); 4K slows marker detection
MIN_FPS = 24


def devices():
    """AVCaptureDevice objects in OpenCV's camera-number order ([] without PyObjC)."""
    try:
        import objc
    except ImportError:
        return []
    objc.loadBundle("AVFoundation", {}, bundle_path="/System/Library/Frameworks/AVFoundation.framework")
    dev = objc.lookUpClass("AVCaptureDevice")
    found = list(dev.devicesWithMediaType_("vide")) + list(dev.devicesWithMediaType_("muxx"))  # Video, Muxed
    return sorted(found, key=functools.cmp_to_key(lambda a, b: a.uniqueID().compare_(b.uniqueID())))


def size_of(fmt):
    """(width, height) of an AVCaptureDeviceFormat, from its description ('... 1920x1080, { 1- 30 fps}...').
    shortcut: parses the description instead of calling CMVideoFormatDescriptionGetDimensions (no CoreMedia
    bindings installed); upgrade if a macOS release changes the description's text."""
    m = re.search(r"\b(\d{2,5})x(\d{2,5})\b", str(fmt))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def pick(specs):
    """Index of the best of [(width, height, max_fps)]: the most pixels up to MAX_AREA at MIN_FPS or more,
    landscape before portrait; None if none qualifies."""
    ok = [(w * h, w >= h, fps, i) for i, (w, h, fps) in enumerate(specs) if 0 < w * h <= MAX_AREA and fps >= MIN_FPS]
    return max(ok)[3] if ok else None


def use_best_format(device):
    """Switch an opened camera to its best format and keep the device locked, so the session's Medium preset
    can't switch it back (macOS resets a custom format when the lock is released). Returns the new (w, h) or None."""
    fmts = list(device.formats())
    specs = [(*size_of(f), max((r.maxFrameRate() for r in f.videoSupportedFrameRateRanges()), default=0))
             for f in fmts]
    i = pick(specs)
    if i is None:
        return None
    locked = device.lockForConfiguration_(None)  # a bare BOOL here (AVFoundation loaded without PyObjC's metadata);
    if not (locked[0] if isinstance(locked, tuple) else locked):  # (BOOL, NSError) where pyobjc-framework-AVFoundation is
        return None
    device.setActiveFormat_(fmts[i])  # held locked for the process's life on purpose (see above)
    return specs[i][:2]
