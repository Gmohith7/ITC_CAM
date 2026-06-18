"""
Smoke-test the detector against the real product images in test_images/.
Run with: pytest tests/test_on_real_images.py -v -s

Requires Tesseract to be installed. Skipped automatically if not available.
"""

import sys
import os
import pytest
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test_images"
)

# Images we know have a batch code — all should return OK
KNOWN_OK = [
    "IMG-20251110-WA0010.jpg",  # white sticker, blue wrapper
    "IMG-20251110-WA0011.jpg",  # white sticker, pink wrapper
    "IMG-20251110-WA0015.jpg",  # white sticker, dark wrapper
    "IMG-20251110-WA0017.jpg",  # direct print on BLACK packaging (hardest case)
    "IMG-20251110-WA0016.jpg",  # direct print on dark packaging
    "IMG-20251110-WA0019.jpg",  # direct print on dark packaging
]


@pytest.fixture(scope="module")
def detector():
    from model.inference import BatchCodeDetector
    det = BatchCodeDetector()
    if det._engine is None:
        pytest.skip("RapidOCR not installed — skipping real-image tests")
    return det


@pytest.mark.parametrize("filename", KNOWN_OK)
def test_known_ok_image(detector, filename):
    path = os.path.join(TEST_IMAGES_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"Image not found: {path}")

    bgr = cv2.imread(path)
    assert bgr is not None, f"Could not read {filename}"
    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    label, confidence, is_defect, regions, text = detector.predict(frame_rgb)

    first_line = text.split('\n')[0][:80] if text else "(no text)"
    print(f"\n{filename}: label={label}  conf={confidence:.0%}  text={first_line!r}")

    assert label == "OK", (
        f"{filename} should be OK but got DEFECT. "
        f"Confidence={confidence:.2f}, OCR='{first_line}'"
    )
    assert confidence > 0.0
