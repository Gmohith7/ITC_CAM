"""
Run this on the Pi to diagnose exactly what picamera2 gives us:
    python tools/diagnose_camera.py

Prints: array shape, dtype, channel order (by measuring known-colour objects),
and saves test frames in all formats so you can inspect them visually.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cam_diag")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_format(fmt: str, label: str):
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cfg = cam.create_video_configuration(
            main={"size": (640, 480), "format": fmt},
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(1.0)  # let AGC settle
        raw = cam.capture_array("main")
        cam.stop()
        cam.close()

        print(f"\n[{label}] format={fmt}")
        print(f"  shape={raw.shape}  dtype={raw.dtype}")
        print(f"  channel means: {[round(float(raw[...,i].mean()),1) for i in range(raw.shape[2] if raw.ndim==3 else 1)]}")

        # Save the raw array as-is interpreted as BGR (OpenCV default)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_raw_as_bgr.jpg"), raw)

        # Save with R and B swapped
        if raw.ndim == 3 and raw.shape[2] >= 3:
            swapped = raw[..., :3][..., ::-1].copy()
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_rb_swapped.jpg"), swapped)

        # If 4-channel, try slices [0:3] and [1:4]
        if raw.ndim == 3 and raw.shape[2] == 4:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_ch0-2.jpg"), raw[..., 0:3])
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_ch1-3.jpg"), raw[..., 1:4])
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{label}_ch1-3_swapped.jpg"),
                        raw[..., 1:4][..., ::-1])

    except Exception as e:
        print(f"[{label}] FAILED: {e}")

test_format("BGR888",   "bgr888")
test_format("RGB888",   "rgb888")
try:
    test_format("XRGB8888", "xrgb8888")
except Exception as e:
    print(f"XRGB8888 not supported: {e}")
try:
    test_format("XBGR8888", "xbgr8888")
except Exception as e:
    print(f"XBGR8888 not supported: {e}")

print(f"\nDiagnostic images saved to: {OUTPUT_DIR}")
print("Copy them off the Pi (scp) and check which one looks correct.")
