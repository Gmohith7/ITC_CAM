"""
Pluggable OCR backends for the batch-code detector.

Tesseract (the default, implemented inline in inference.py) needs heavy
binarisation + a PSM sweep to cope with low-contrast embossed print. PaddleOCR
detects AND recognises text in a single pass directly on the raw frame, and is
substantially more robust on the dark-cardboard ITC codes — no preprocessing
variants required.

Install on the Pi (inside the venv):
    pip install paddlepaddle paddleocr
First run downloads the detection/recognition models (needs internet once).
"""

import cv2
import numpy as np


def _extract_lines(result) -> list:
    """
    Pull recognised text strings out of a PaddleOCR result, tolerating the
    several result shapes across PaddleOCR 2.x / 3.x.
    """
    lines = []
    if not result:
        return lines

    # PaddleOCR 3.x: ocr.predict() -> list of result objects/dicts with
    # a 'rec_texts' list.
    for item in result:
        rec_texts = None
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts")
        elif hasattr(item, "get"):
            try:
                rec_texts = item.get("rec_texts")
            except Exception:
                rec_texts = None
        if rec_texts:
            lines.extend(str(t) for t in rec_texts)
    if lines:
        return lines

    # PaddleOCR 2.x: ocr.ocr() -> [ per_image ] where per_image is a list of
    # [box, (text, conf)] entries (or sometimes the per-image list directly).
    page = result[0] if (len(result) == 1 and isinstance(result[0], list)) else result
    for entry in page:
        try:
            txt = entry[1][0]            # [box, (text, conf)]
            if txt:
                lines.append(str(txt))
        except Exception:
            continue
    return lines


class PaddleOCREngine:
    """
    Thin wrapper over PaddleOCR. The model is loaded once and reused for every
    frame (loading is the slow part). `read()` returns all recognised text as a
    single newline-joined string, ready to feed into _evaluate().
    """

    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR

        # use_angle_cls/textline orientation off: batch codes are upright, so we
        # skip the orientation classifier for speed. Arg names differ by version
        # — fall back progressively to the lowest-common-denominator constructor.
        self._ocr = None
        for kwargs in (
            dict(use_angle_cls=False, lang=lang, show_log=False),
            dict(use_angle_cls=False, lang=lang),
            dict(lang=lang),
        ):
            try:
                self._ocr = PaddleOCR(**kwargs)
                break
            except TypeError:
                continue
        if self._ocr is None:
            self._ocr = PaddleOCR()

    def _infer(self, bgr: np.ndarray):
        # 2.x uses .ocr(img, cls=False); 3.x prefers .predict(img).
        try:
            return self._ocr.ocr(bgr, cls=False)
        except TypeError:
            try:
                return self._ocr.ocr(bgr)
            except Exception:
                pass
        except Exception:
            pass
        try:
            return self._ocr.predict(bgr)
        except Exception:
            return None

    def read(self, image_rgb: np.ndarray) -> str:
        """Recognise text in `image_rgb` (H×W×3 RGB) → newline-joined string."""
        if image_rgb.ndim == 3 and image_rgb.shape[2] >= 3:
            bgr = cv2.cvtColor(image_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            bgr = image_rgb
        result = self._infer(bgr)
        return "\n".join(_extract_lines(result))
