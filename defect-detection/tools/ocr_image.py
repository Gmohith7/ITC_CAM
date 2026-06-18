"""
Run the FULL detection pipeline on a single saved image, with OCR debug on.

This closes the Pi debugging loop without needing the camera or a 60-second
hold: capture a frame on the Pi (run.py with OCR_DEBUG_IMAGES=true dumps
data/debug/fNNNNN_raw.jpg), then feed that exact frame through the exact
pipeline here and read the per-pass + per-frame diagnostics.

Usage:
    python tools/ocr_image.py path/to/frame.jpg
    python tools/ocr_image.py data/debug/f00030_raw.jpg

Prints the [OCR]/[FRAME] debug lines and the final OK/DEFECT verdict.
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.OCR_DEBUG = True   # force per-pass + per-frame diagnostics on

from model.inference import BatchCodeDetector


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    path = sys.argv[1]
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"[ocr_image] Could not read image: {path}")
        sys.exit(1)

    # Pipeline expects RGB (run.py feeds RGB frames from the camera).
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"[ocr_image] {path}  shape={rgb.shape}")

    detector = BatchCodeDetector()
    if detector._engine is None:
        print("[ocr_image] OCR engine not available — pip install rapidocr_onnxruntime")
        sys.exit(1)

    label, confidence, is_defect, regions, text = detector.predict(rgb)
    print("\n" + "=" * 60)
    print(f"RESULT: {label}  confidence={confidence:.0%}  regions={len(regions)}")
    print(f"BEST TEXT: {text!r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
