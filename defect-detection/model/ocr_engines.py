"""
RapidOCR (PP-OCR on onnxruntime) backend for the batch-code detector.

Runs PP-OCR detection + recognition on onnxruntime — accurate and reliable on
Raspberry Pi / ARM, where paddlepaddle's native inference segfaults. onnxruntime
uses all CPU cores by default (do NOT set OMP_NUM_THREADS=1 — it throttles
inference to a single core and was the cause of ~10 s/frame latency).

The model is loaded once and reused for every frame.

Install on the Pi venv:  pip install rapidocr_onnxruntime
"""

import cv2
import numpy as np


def _extract_rapid_lines(out) -> list:
    """Pull recognised text strings from a RapidOCR result across its API versions."""
    if out is None:
        return []
    # rapidocr (new package): result object exposing .txts
    txts = getattr(out, "txts", None)
    if txts:
        return [str(t) for t in txts]
    # rapidocr_onnxruntime: returns (result, elapse); result = [[box, text, score], ...]
    result = out[0] if isinstance(out, tuple) else out
    if not result:
        return []
    lines = []
    for entry in result:
        try:
            lines.append(str(entry[1]))   # [box, text, score]
        except Exception:
            continue
    return lines


class RapidOCREngine:
    """
    Thin wrapper over RapidOCR. `read()` returns all recognised text as a single
    newline-joined string, ready to feed into _evaluate().
    """

    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception:
            from rapidocr import RapidOCR          # newer renamed package
        self._ocr = RapidOCR()
        # Warm up so first-call latency is paid here, not mid-stream.
        try:
            self.read(np.zeros((64, 192, 3), dtype=np.uint8))
        except Exception:
            pass

    def read(self, image_rgb: np.ndarray) -> str:
        if image_rgb.ndim == 3 and image_rgb.shape[2] >= 3:
            bgr = cv2.cvtColor(image_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            bgr = image_rgb
        try:
            out = self._ocr(bgr)
        except Exception:
            return ""
        return "\n".join(_extract_rapid_lines(out))
