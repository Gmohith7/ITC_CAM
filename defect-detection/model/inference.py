"""
Batch code detector — ITC product packaging, RapidOCR (PP-OCR on onnxruntime).

ITC batch code block structure (all products observed):
───────────────────────────────────────────────────────
  Batch No.:   HH:MM  XXXXXB11          ← time + alphanumeric code
  PKD.:        DD/MM/YY                  ← packed date
  Use By:      DD/MM/YY                  ← expiry date
  MRP Rs. incl. of all taxes/(Rs. per g) ← price line
  NN.NN/(N.NN)

We care ONLY about this batch code block. RapidOCR recognises all text on the
frame; the evidence gate (see _evaluate) then requires the block's signature —
two dates (PKD + Use By) plus a corroborating signal (time / batch code / label)
— so other pack text (ingredients, nutrition, FSSAI) can never trigger a false
OK. With no OCR engine available, detection always returns DEFECT.
"""

import re
import os
import time
import subprocess
from collections import namedtuple

import cv2
import numpy as np
import config

# ── Structural patterns ───────────────────────────────────────────────────────

# Dates: DD/MM/YY or DD/MM/YYYY.
# OCR misreads the '/' separator in many ways on direct print: '1' (narrow
# stroke), '9' (top serif), '4' (strokes merge), etc. The class [/\-\.|l1-9]
# accepts any digit 1-9 as a separator substitute so "24102127" (24/02/27),
# "24902127" (sep=9) and "31408126" (sep=4) all match. Day [0-3]\d, month
# [0-1]\d to suppress false matches on arbitrary digit strings.
_PAT_DATE = re.compile(
    r'\b[0-3]\d[/\-\.|l1-9]?[0-1]\d[/\-\.|l1-9]?\d{2,4}\b',
    re.IGNORECASE,
)

# ITC-specific label keywords that co-occur with the batch block.
_PAT_KEYWORD = re.compile(
    r'\b(batch[\s\.\-]*no|batch[\s\.\-]*code|lot[\s\.\-]*no|'
    r'pkd|use[\s]*by|mfd|mfg|best[\s]*before|expiry|exp|mrp)\b',
    re.IGNORECASE,
)

# Time printed before the batch code: HH:MM or HH MM (OCR sometimes drops ':').
# Minutes: OCR often misreads '0' as 'O' at small sizes, so [0-5O][0-9O] accepts
# both — '07 O4' and '07:04' both match → 07:04.
_PAT_TIME = re.compile(r'\b([01]?\d|2[0-3])[: ][0-5O][0-9O]\b')

# Alphanumeric batch code: uppercase letters + digits, 4-10 chars (01B11, 09A11).
_PAT_BATCH_CODE = re.compile(
    r'\b(?=[A-Z0-9]{4,10}\b)(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{4,10}\b'
)

# MRP price line: NN.NN/(N.NN) — unique to the ITC batch block.
_PAT_MRP = re.compile(r'\d+\.\d{2}\s*/\s*\(\s*\d+\.\d{2,3}\s*\)')

# ── Scoring ───────────────────────────────────────────────────────────────────

_THRESHOLD = config.DETECTION_THRESHOLD   # default 0.55

# Full per-result breakdown — carries *why* a result scored what it did, so the
# debug log can show exactly which evidence was present and which gate failed.
_Evidence = namedtuple(
    "_Evidence",
    "score has_keyword has_time n_dates has_batch has_mrp dates time keyword",
)


def _evaluate(text: str) -> "_Evidence":
    """
    Evidence-based confidence score in [0.0, 1.0] plus the supporting matches.

    Block-specific gate: the ITC batch code block ALWAYS carries two dates (PKD +
    Use By), so we require >=2 dates AND one corroborating block signal (printed
    time, the alphanumeric batch code, or a block label). This locks detection
    onto the batch block — stray text elsewhere on the pack (ingredients, a lone
    date or MRP line) can never trigger a false OK. score is 0.0 otherwise.
    """
    kw_m   = _PAT_KEYWORD.search(text)
    time_m = _PAT_TIME.search(text)
    dates  = _PAT_DATE.findall(text)
    has_batch = bool(_PAT_BATCH_CODE.search(text))
    has_mrp   = bool(_PAT_MRP.search(text))

    has_keyword = bool(kw_m)
    has_time    = bool(time_m)
    n_dates     = len(dates)

    is_block = n_dates >= 2 and (has_time or has_batch or has_keyword)
    if is_block:
        score = 0.40                           # base
        score += 0.20 * min(n_dates, 2)        # +0.40 for PKD + Use By
        if has_time:
            score += 0.10
        if has_batch:
            score += 0.10
        if has_mrp:
            score += 0.10
        score = min(round(score, 2), 1.0)
    else:
        score = 0.0

    return _Evidence(
        score, has_keyword, has_time, n_dates, has_batch, has_mrp,
        dates,
        time_m.group(0) if time_m else None,
        kw_m.group(0) if kw_m else None,
    )


def _score_text(text: str) -> float:
    """Confidence score in [0.0, 1.0] — thin wrapper over _evaluate()."""
    return _evaluate(text).score


# ── Frame quality (diagnostics) ───────────────────────────────────────────────

def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _frame_quality(gray: np.ndarray) -> tuple:
    """
    (brightness, sharpness) for a grayscale frame — pure diagnostics.

      brightness : mean pixel value 0-255. Too low = underexposed / lens cap;
                   too high = blown-out glare. OCR needs the text in between.
      sharpness  : variance of the Laplacian (focus measure). High = crisp /
                   in focus; low = blurred. If OCR is noise AND sharpness is
                   low, the lens — not the logic — is the problem.
    """
    brightness = float(np.mean(gray))
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _startup_diagnostics() -> None:
    """
    Print the running code version + active config once at startup, so a stale
    Pi run (old code) is obvious at the top of output.txt.
    """
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        if subprocess.run(["git", "diff", "--quiet"], cwd=repo).returncode != 0:
            commit += "+dirty"
    except Exception:
        commit = "unknown"

    print(f"[Detector] code commit={commit} | engine=rapidocr | "
          f"threshold={_THRESHOLD} | ocr_max_side={config.OCR_MAX_SIDE}")
    print(f"[Detector] config: DARK_FRAME_THRESHOLD={config.DARK_FRAME_THRESHOLD} "
          f"FOCUS_MIN_SHARPNESS={config.FOCUS_MIN_SHARPNESS} "
          f"OCR_DEBUG={config.OCR_DEBUG} OCR_DEBUG_IMAGES={config.OCR_DEBUG_IMAGES}")


def _print_frame_header(frame_n, w, h, brightness, sharpness) -> None:
    """
    Printed BEFORE inference so focus/exposure is captured even if the run is
    killed mid-frame. Low sharp ⇒ fix the lens/light, not the OCR logic.
    """
    focus = "OK" if sharpness >= config.FOCUS_MIN_SHARPNESS else "SOFT"
    print(f"[FRAME #{frame_n}] {w}x{h} bright={brightness:.1f} sharp={sharpness:.0f} "
          f"FOCUS:{focus} — OCR running")


def _print_frame_summary(frame_n, w, h, brightness, sharpness,
                         ev: "_Evidence", text: str, ocr_ms=None) -> None:
    """
    One consolidated line per processed frame — the headline diagnostic: focus,
    inference time, score, which evidence survived (kw/time/dates/batch/mrp),
    and the best text. When detection fails this line says *why*.
    """
    focus   = "OK" if sharpness >= config.FOCUS_MIN_SHARPNESS else "SOFT"
    dates_s = ",".join(ev.dates[:3]) if ev.dates else "-"
    ms      = f" ocr_ms={ocr_ms:.0f}" if ocr_ms is not None else ""
    preview = " ".join(text.split())[:120] or "(no text)"
    print(f"[FRAME #{frame_n}] {w}x{h} bright={brightness:.1f} sharp={sharpness:.0f} "
          f"FOCUS:{focus} |{ms} score={ev.score:.2f} | "
          f"kw={int(ev.has_keyword)} time={ev.time or '-'} "
          f"dates={ev.n_dates}[{dates_s}] batch={int(ev.has_batch)} "
          f"mrp={int(ev.has_mrp)} | best={preview!r}")


def _dump_debug_images(frame_rgb: np.ndarray, frame_n: int) -> None:
    """Save the raw frame so framing/focus can be judged by eye (OCR_DEBUG_IMAGES)."""
    try:
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        path = os.path.join(config.DEBUG_DIR, f"f{frame_n:05d}_raw.jpg")
        cv2.imwrite(path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        print(f"[debug-img] wrote {path}")
    except Exception as exc:
        if config.OCR_DEBUG:
            print(f"[debug-img] dump failed: {exc}")


# ── Main detector ─────────────────────────────────────────────────────────────

class BatchCodeDetector:
    """
    Detects whether an ITC product batch code block is visible in the frame.

    Returns (label, confidence, is_defect, regions, ocr_text):
        label      : "OK" or "DEFECT"
        confidence : float [0.0, 1.0]  — 0.0 when DEFECT
        is_defect  : bool
        regions    : always [] (kept for API compatibility with the HUD)
        ocr_text   : recognised text (for debug / dashboard)
    """

    def __init__(self):
        self._frame_n = 0
        self._engine = self._init_engine()
        _startup_diagnostics()

    def _init_engine(self):
        try:
            from model.ocr_engines import RapidOCREngine
            eng = RapidOCREngine()
            print("[Detector] RapidOCR (onnxruntime) engine ready.")
            return eng
        except Exception as e:
            print(f"[Detector] RapidOCR unavailable ({e}) — will always return "
                  f"DEFECT. Install with: pip install rapidocr_onnxruntime")
            return None

    def predict(self, frame_rgb: np.ndarray) -> tuple:
        """
        Batch code detection on `frame_rgb` (H×W×3 RGB uint8). RapidOCR detects +
        recognises text in one pass; the evidence gate decides OK/DEFECT. With no
        engine, always returns DEFECT — never a false positive.
        """
        if self._engine is None:
            return "DEFECT", 0.0, True, [], "OCR unavailable — install rapidocr_onnxruntime"

        self._frame_n += 1
        h_img, w_img = frame_rgb.shape[:2]
        brightness, sharpness = _frame_quality(_to_gray(frame_rgb))
        if config.OCR_DEBUG:
            _print_frame_header(self._frame_n, w_img, h_img, brightness, sharpness)

        # Downscale before inference: fewer/smaller text boxes → much faster →
        # lower lag. The large batch digits stay readable at the reduced size.
        ocr_frame = frame_rgb
        longest = max(h_img, w_img)
        if config.OCR_MAX_SIDE and longest > config.OCR_MAX_SIDE:
            s = config.OCR_MAX_SIDE / longest
            ocr_frame = cv2.resize(frame_rgb, (int(w_img * s), int(h_img * s)),
                                   interpolation=cv2.INTER_AREA)

        t0 = time.monotonic()
        text = self._engine.read(ocr_frame)
        ocr_ms = (time.monotonic() - t0) * 1000.0
        ev = _evaluate(text)
        if config.OCR_DEBUG:
            _print_frame_summary(self._frame_n, w_img, h_img, brightness, sharpness,
                                 ev, text, ocr_ms=ocr_ms)
        if config.OCR_DEBUG_IMAGES and self._frame_n % config.DEBUG_IMAGE_EVERY == 0:
            _dump_debug_images(frame_rgb, self._frame_n)

        found = ev.score >= _THRESHOLD
        return (
            "OK" if found else "DEFECT",
            ev.score if found else 0.0,
            not found,
            [],
            text.strip(),
        )
