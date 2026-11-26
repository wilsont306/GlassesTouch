#!/usr/bin/env python3
"""GlassesTouch.app's program: the control panel in its own window instead of a browser tab.

Runs panel.py's server in this process and shows it in a WebKit view. Closing the window (or Cmd-Q) quits and stops
the tracker, as closing the Terminal window does for GlassesTouch.command. If the panel is already running (from
GlassesTouch.command), the window just shows that one. make_app.sh builds the .app that starts this file.
"""
import os
import signal
import threading

import AppKit
import Foundation
import objc

import panel

PORT = 8765
URL = f"http://127.0.0.1:{PORT}/"

# WebKit is part of macOS; load it through the bridge we have instead of adding pyobjc-framework-WebKit
objc.loadBundle("WebKit", globals(), bundle_path="/System/Library/Frameworks/WebKit.framework")
WKWebView = objc.lookUpClass("WKWebView")
WKWebViewConfiguration = objc.lookUpClass("WKWebViewConfiguration")


def start_panel():
    """The panel server on a thread, or None if one already listens on PORT (then the window shows that one)."""
    try:
        srv = panel.make_server(PORT)
    except OSError:
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def stop_panel(srv):
    if srv is not None:
        srv.tracker.stop()  # the tracker must not outlive the app (it holds the mouse while dragging)
        srv.pairing.close()
        srv.shutdown()
        srv.server_close()


def build_menu(app):
    """App menu (Quit) and Edit menu: without the Edit menu, Cmd-C/V/X/A don't reach the page's text fields."""
    bar = AppKit.NSMenu.alloc().init()
    for title, items in (("GlassesTouch", [("Quit GlassesTouch", "terminate:", "q")]),
                         ("Edit", [("Cut", "cut:", "x"), ("Copy", "copy:", "c"), ("Paste", "paste:", "v"),
                                   ("Select All", "selectAll:", "a")])):
        top = AppKit.NSMenuItem.alloc().init()
        menu = AppKit.NSMenu.alloc().initWithTitle_(title)
        for name, action, key in items:
            menu.addItemWithTitle_action_keyEquivalent_(name, action, key)
        top.setSubmenu_(menu)
        bar.addItem_(top)
    app.setMainMenu_(bar)


class Delegate(AppKit.NSObject):
    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True

    def applicationWillTerminate_(self, note):
        stop_panel(self.srv)
        self.srv = None


def main():
    os.environ["GLASSTOUCH_APP"] = "GlassesTouch"  # the tracker names the app permissions go to
    srv = start_panel()
    # the process runs Homebrew's Python.app, so name it ourselves (menu bar, Dock) before the app object exists
    Foundation.NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "GlassesTouch"
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
    icon = os.environ.get("GLASSTOUCH_ICON")  # set by the .app's launcher
    if icon and os.path.exists(icon):
        app.setApplicationIconImage_(AppKit.NSImage.alloc().initWithContentsOfFile_(icon))
    delegate = Delegate.alloc().init()
    delegate.srv = srv
    app.setDelegate_(delegate)
    build_menu(app)

    config = WKWebViewConfiguration.alloc().init()
    prefs = config.preferences()
    if prefs.respondsToSelector_("setElementFullscreenEnabled:"):  # the tutorial goes full screen (macOS 12.3+)
        prefs.setElementFullscreenEnabled_(True)
    frame = AppKit.NSMakeRect(0, 0, 1280, 860)
    web = WKWebView.alloc().initWithFrame_configuration_(frame, config)
    web.loadRequest_(Foundation.NSURLRequest.requestWithURL_(Foundation.NSURL.URLWithString_(URL)))
    style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
             | AppKit.NSWindowStyleMaskMiniaturizable | AppKit.NSWindowStyleMaskResizable)
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, style, AppKit.NSBackingStoreBuffered, False)
    window.setTitle_("GlassesTouch")
    window.setContentView_(web)
    window.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorFullScreenPrimary)
    window.center()
    window.setFrameAutosaveName_("GlassesTouchMain")  # reopens where it was left
    window.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):  # quit cleanly from a shell too
        signal.signal(sig, lambda *_: app.terminate_(None))
    # Python runs signal handlers only when it gets control, which Cocoa's run loop never gives it: wake it twice a second
    Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.5, True, lambda t: None)
    try:
        app.run()
    finally:
        stop_panel(delegate.srv)


if __name__ == "__main__":
    main()
