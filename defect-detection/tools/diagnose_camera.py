#!/usr/bin/env python3
"""
Run this ON THE PI with the SYSTEM python (not the venv) so picamera2 is found:

    python3 tools/diagnose_camera.py

picamera2 is installed system-wide via apt, not inside the venv.
This script captures one frame in every common format, saves JPEGs, and
prints the array shape so you can see exactly what the hardware gives you.
"""
import sys, os, time

# Ensure system dist-packages are on the path (picamera2 lives there)
for p in ("/usr/lib/python3/dist-packages",):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam_diag")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def grab(fmt: str):
    from picamera2 import Picamera2
    cam = Picamera2()
    try:
        cfg = cam.create_video_configuration(
            main={"size": (640, 480), "format": fmt}
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(1.5)   # let AGC/AWB settle
        raw = cam.capture_array("main")
    finally:
        cam.stop()
        cam.close()
    return raw


def save(name: str, arr: np.ndarray):
    path = os.path.join(OUTPUT_DIR, f"{name}.jpg")
    # cv2.imwrite expects BGR; we write as-is so the file reflects raw bytes
    cv2.imwrite(path, arr)
    print(f"  saved {path}")


FORMATS = ["XRGB8888", "XBGR8888", "RGB888", "BGR888"]

for fmt in FORMATS:
    print(f"\n── {fmt} ──")
    try:
        raw = grab(fmt)
        print(f"  shape={raw.shape}  dtype={raw.dtype}")
        if raw.ndim == 3:
            means = [round(float(raw[:,:,i].mean()), 1) for i in range(raw.shape[2])]
            print(f"  channel means (as-is): {means}")

        # Save the raw array interpreted as BGR (what cv2 does by default)
        save(f"{fmt}_raw_as_bgr", raw)

        if raw.ndim == 3 and raw.shape[2] == 4:
            # Try all four 3-channel slices / orderings
            save(f"{fmt}_ch0-2",          raw[:,:,0:3])
            save(f"{fmt}_ch1-3",          raw[:,:,1:4])
            save(f"{fmt}_ch0-2_reversed", raw[:,:,2::-1])
            save(f"{fmt}_ch1-3_reversed", raw[:,:,3:0:-1])

        if raw.ndim == 3 and raw.shape[2] == 3:
            save(f"{fmt}_reversed",       raw[:,:,::-1])

    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\nDone. Images in: {OUTPUT_DIR}")
print("The file whose name ends in the correct-looking image = that's your format.")
