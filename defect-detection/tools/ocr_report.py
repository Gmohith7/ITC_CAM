"""
Evaluate OCR + batch-code detection on test_images/.

Usage:
    python tools/ocr_report.py
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference import BatchCodeDetector

TEST_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test_images",
)

KNOWN_OK = [
    "IMG-20251110-WA0010.jpg",
    "IMG-20251110-WA0011.jpg",
    "IMG-20251110-WA0015.jpg",
    "IMG-20251110-WA0017.jpg",
    "IMG-20251110-WA0016.jpg",
    "IMG-20251110-WA0019.jpg",
]


def main():
    detector = BatchCodeDetector()
    if not detector._tesseract_ok:
        print("[OCR Report] Tesseract not available. Exiting.")
        sys.exit(1)

    files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    ok_hits = 0
    ok_total = 0
    print("[OCR Report] Running on test_images/ ...")

    for filename in files:
        path = os.path.join(TEST_IMAGES_DIR, filename)
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"[OCR Report] Skipped {filename} (read error)")
            continue

        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        label, confidence, is_defect, regions, text = detector.predict(frame)
        first_line = text.split("\n")[0][:80] if text else "(no text)"

        is_known_ok = filename in KNOWN_OK
        if is_known_ok:
            ok_total += 1
            if label == "OK":
                ok_hits += 1

        tag = "OK" if is_known_ok else "UNK"
        print(f"{filename:40s} [{tag}]  label={label:7s}  conf={confidence:.0%}  text='{first_line}'")

    if ok_total:
        hit_rate = ok_hits / ok_total * 100
        print(f"\n[OCR Report] Known-OK hit rate: {ok_hits}/{ok_total} ({hit_rate:.1f}%)")


if __name__ == "__main__":
    main()
