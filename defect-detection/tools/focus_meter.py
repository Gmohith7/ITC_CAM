"""
Live focus / exposure meter for the Pi camera — no OCR, just frame quality.

Why this exists: the detector reads a frame fine when it is sharp and well-lit
(test_images/ref_batch_dark.jpg measures sharp~83, bright~64 and OCRs cleanly).
Real Pi captures have measured sharp~34 / bright~37 — too soft and too dark, so
OCR gets only noise. Tune distance, lighting and focus with this FIRST, until
the numbers are in range, before running the full detector.

Usage:
    python tools/focus_meter.py          # live meter (uses configured AF_MODE)
    python tools/focus_meter.py --sweep  # sweep manual LensPosition, find sharpest

Targets:  sharpness >= ~80 (FOCUS:OK),  brightness ~50-120.
Press Ctrl+C to stop.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from camera.capture import CameraCapture
from model.inference import _frame_quality, _to_gray


def live(cam):
    """Print brightness + sharpness ~6×/sec so focus/light can be tuned by hand."""
    print("[focus] live meter — adjust distance / lighting / focus and watch "
          "'sharp' climb. Target sharp>=80. Ctrl+C to stop.")
    try:
        while True:
            frame = cam.get_frame()
            b, s = _frame_quality(_to_gray(frame))
            focus = "OK  " if s >= config.FOCUS_MIN_SHARPNESS else "SOFT"
            bar = "#" * min(int(s / 4), 50)
            print(f"bright={b:6.1f}  sharp={s:7.1f}  [{focus}]  {bar}")
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\n[focus] stopped.")


def sweep(cam):
    """
    Drive the lens through its full manual range and report the LensPosition
    that gives the sharpest image of whatever is in view — i.e. the value to
    lock into .env for this bench distance. Continuous AF won't lock on
    low-contrast dark cardboard, so a fixed manual focus is more reliable.
    """
    if getattr(cam, "_mode", None) != "picamera2":
        print("[focus] --sweep needs the Pi camera (picamera2). Aborting.")
        return
    from libcamera import controls

    print("[focus] sweeping manual LensPosition 0.5 .. 12.0 — hold the product "
          "still at the working distance...")
    results = []
    lp = 0.5
    while lp <= 12.01:
        cam.cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                              "LensPosition": lp})
        time.sleep(0.6)  # let the lens settle
        best = 0.0
        for _ in range(3):
            _, s = _frame_quality(_to_gray(cam.get_frame()))
            best = max(best, s)
            time.sleep(0.1)
        dist = 100.0 / lp if lp > 0 else float("inf")
        results.append((best, lp, dist))
        print(f"  LensPosition={lp:5.2f} (~{dist:5.1f} cm)  sharp={best:7.1f}")
        lp += 0.5

    results.sort(reverse=True)
    best_s, best_lp, best_dist = results[0]
    print("\n[focus] SHARPEST:")
    print(f"  LensPosition={best_lp:.2f}  (~{best_dist:.0f} cm)  sharp={best_s:.1f}")
    print(f"  -> set in .env:   AF_MODE=manual   LENS_POSITION={best_lp:.2f}")
    if best_s < config.FOCUS_MIN_SHARPNESS:
        print("  WARNING: even the sharpest position is below target — the issue "
              "is likely lighting/distance, not focus. Add light and re-run.")


def main():
    cam = CameraCapture()
    try:
        sweep(cam) if "--sweep" in sys.argv else live(cam)
    finally:
        cam.release()


if __name__ == "__main__":
    main()
