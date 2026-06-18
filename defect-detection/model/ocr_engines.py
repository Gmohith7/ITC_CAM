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

import os

# Limit native threading BEFORE paddle is imported — multithreaded paddle on ARM
# (Pi) is a common SIGSEGV source, especially when called from a worker thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
# Skip PaddleOCR's online model-source connectivity check (faster, offline-safe).
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

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

        # Orientation classifier off: batch codes are upright (saves time). Arg
        # names differ across versions (3.x = use_textline_orientation, 2.x =
        # use_angle_cls, and 3.x raises its OWN error type for unknown args, not
        # TypeError) — so try the variants and catch broadly.
        # Strip the 3.x pipeline to detect+recognise ONLY. The default doc
        # orientation (PP-LCNet_doc_ori) and especially doc unwarping (UVDoc)
        # models are unneeded for a flat, upright code AND are a known ARM/Pi
        # segfault source — disable them.
        self._ocr = None
        last_err = None
        for kwargs in (
            dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                 use_textline_orientation=False, lang=lang),   # PaddleOCR 3.x
            dict(use_textline_orientation=False, lang=lang),   # 3.x (older)
            dict(use_angle_cls=False, lang=lang),              # PaddleOCR 2.x
            dict(lang=lang),                                   # minimal
            dict(),                                            # bare
        ):
            try:
                self._ocr = PaddleOCR(**kwargs)
                break
            except Exception as e:
                last_err = e
                continue
        if self._ocr is None:
            raise RuntimeError(f"could not construct PaddleOCR ({last_err})")

        # Warm up on the MAIN thread with a tiny dummy frame: pays first-call
        # latency upfront and makes any inference crash deterministic here rather
        # than mid-stream in the OCR worker thread.
        try:
            self.read(np.zeros((64, 192, 3), dtype=np.uint8))
        except Exception:
            pass

    def _infer(self, bgr: np.ndarray):
        # 3.x prefers .predict(img); 2.x uses .ocr(img). Try both, catch broadly.
        for call in (
            lambda: self._ocr.predict(bgr),
            lambda: self._ocr.ocr(bgr),
        ):
            try:
                return call()
            except Exception:
                continue
        return None

    def read(self, image_rgb: np.ndarray) -> str:
        """Recognise text in `image_rgb` (H×W×3 RGB) → newline-joined string."""
        if image_rgb.ndim == 3 and image_rgb.shape[2] >= 3:
            bgr = cv2.cvtColor(image_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            bgr = image_rgb
        result = self._infer(bgr)
        return "\n".join(_extract_lines(result))


def _extract_rapid_lines(out) -> list:
    """Pull text strings from a RapidOCR result across its API versions."""
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
    PP-OCR detection + recognition running on onnxruntime (via RapidOCR) instead
    of paddlepaddle. Same models/accuracy, but no paddle native code — the
    reliable choice on Raspberry Pi / ARM where paddlepaddle inference segfaults.

    Install on the Pi venv:  pip install rapidocr_onnxruntime
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
