"""
Batch code detector using OpenCV + Tesseract OCR.

Detection strategy
──────────────────
Rather than matching the batch code number itself (which varies wildly across
products — "01B11", "BHB241025H", "0606 A11", etc.), we match the structural
evidence that a batch code block is present:

  • Label keywords  — "BATCH NO", "PKD", "USE BY", "MFD", "PACKED"
  • Dates           — DD/MM/YY or DD/MM/YYYY (always present: PKD + USE BY)
  • Time stamp      — HH:MM at the start of a line (precedes the code on most products)
  • MRP price line  — NN.NN/(N.NN) pattern (present on most sticker labels)

Confidence is additive across evidence types, capped at 1.0.
DEFECT always returns 0.0.

Two-stage scan
──────────────
Stage 1 — white sticker regions (coloured wrappers): crop + OCR each bright rect.
Stage 2 — full-frame OCR with multiple contrast variants (dark/direct-print packaging).
"""

import re
import shutil
import platform
import cv2
import numpy as np
import config

# ── Structural patterns — no hardcoded values ────────────────────────────────

# Date in any reasonable separator form: DD/MM/YY, DD/MM/YYYY, DD-MM-YY etc.
_PAT_DATE = re.compile(r'\b\d{2}[/\\\-\.]\d{2}[/\\\-\.]\d{2,4}\b')

# Time stamp: HH:MM or HH MM (OCR sometimes drops the colon)
_PAT_TIME = re.compile(r'\b\d{1,2}[: ]\d{2}\b')

# Label keywords that appear next to batch info
_PAT_KEYWORDS = re.compile(
    r'\b(batch\s*no|batch\s*code|lot\s*no|lot|pkd|use\s*by|mfd|mfg|packed\s*on|'
    r'manufacturing|best\s*before|expiry|exp)\b',
    re.IGNORECASE
)

# MRP price line: digits.digits/(digits.digits) — unique to these labels
_PAT_MRP_LINE = re.compile(r'\d+\.\d{2}\s*/\s*\(\s*\d+\.\d{2}\s*\)')

# Weights: how much each evidence type contributes to confidence
# PSM modes: 6=single block, 7=single text line, 11=sparse text, 3=fully automatic
_EVIDENCE = [
    (_PAT_DATE,     0.35, 2),   # up to 2 dates (PKD + USE BY) = 0.70
    (_PAT_KEYWORDS, 0.20, 2),   # up to 2 keyword hits = 0.40
    (_PAT_TIME,     0.20, 1),
    (_PAT_MRP_LINE, 0.15, 1),
]
_OCR_PSMS = (6, 7, 11, 3)


# ── Scoring ──────────────────────────────────────────────────────────────────

def _score_text(text: str) -> float:
    """
    Return a confidence score in [0.0, 1.0] based on structural evidence
    found in `text`. No specific codes or values are matched.
    """
    score = 0.0
    for pat, weight, max_count in _EVIDENCE:
        hits = len(pat.findall(text))
        if hits:
            score += weight * min(hits, max_count)
    return min(round(score, 2), 1.0)


# ── Image preprocessing ───────────────────────────────────────────────────────

def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _preprocessing_variants(gray: np.ndarray):
    """
    Yield binarised grayscale images to try for OCR.
    Covers dark-on-light, light-on-dark, and low-contrast text.
    """
    h, w = gray.shape
    if h < config.OCR_MIN_HEIGHT:
        scale = config.OCR_MIN_HEIGHT / h
        gray = cv2.resize(gray, (int(w * scale), config.OCR_MIN_HEIGHT),
                          interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    # 1. Otsu — dark text on light background (sticker labels)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu

    # 2. Inverted Otsu — light/gold text on dark background (direct-print packaging)
    yield cv2.bitwise_not(otsu)

    # 3. CLAHE + Otsu — lifts low-contrast or faded print
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, clahe_bin = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield clahe_bin

    # 4. Inverted CLAHE — low-contrast light-on-dark
    yield cv2.bitwise_not(clahe_bin)

    # 5. Adaptive threshold — uneven lighting
    adapt = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    yield adapt

    # 6. Inverted adaptive threshold
    yield cv2.bitwise_not(adapt)


def _ocr_best(image: np.ndarray, psm_list: tuple) -> tuple:
    """
    Run Tesseract across all preprocessing variants and PSM modes.
    Return (text, score) for the variant with the highest evidence score.
    """
    import pytesseract
    gray = _to_gray(image)
    best_text, best_score = "", 0.0
    for variant in _preprocessing_variants(gray):
        for psm in psm_list:
            cfg = f'--psm {psm} --oem 3'
            try:
                text = pytesseract.image_to_string(variant, config=cfg)
                score = _score_text(text)
                if score > best_score:
                    best_score, best_text = score, text
            except Exception:
                continue
    return best_text, best_score


# ── White label region finder ─────────────────────────────────────────────────

def _find_white_label_regions(frame_rgb: np.ndarray) -> list:
    """
    Detect white/light rectangular sticker labels on coloured packaging.
    Returns list of (x, y, w, h) sorted by area descending.
    """
    gray = _to_gray(frame_rgb)
    _, thresh = cv2.threshold(gray, config.WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.MORPH_KERNEL_W, config.MORPH_KERNEL_H)
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = frame_rgb.shape[:2]
    frame_area = w_img * h_img

    regions = []
    pad = config.REGION_PADDING
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / max(h, 1)
        if frame_area * 0.003 < area < frame_area * 0.60 and 0.3 < aspect < 12.0:
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            regions.append((x1, y1, x2 - x1, y2 - y1))

    regions.sort(key=lambda r: r[2] * r[3], reverse=True)
    return regions[:6]


# ── Main detector ─────────────────────────────────────────────────────────────

class BatchCodeDetector:
    """
    Detects whether a printed batch code block is visible in the frame.

    Returns (label, confidence, is_defect, regions, ocr_text):
        label      : "OK" or "DEFECT"
        confidence : float [0.0, 1.0]  — 0.0 when DEFECT
        is_defect  : True when no batch code evidence found
        regions    : list of (x, y, w, h) bounding boxes for overlay
        ocr_text   : best OCR text (first line shown on HUD)
    """

    def __init__(self):
        self._tesseract_ok = self._init_tesseract()

    def _init_tesseract(self) -> bool:
        try:
            import pytesseract
            if platform.system() == "Windows":
                # Prefer PATH lookup; fall back to config value.
                exe = shutil.which("tesseract") or config.TESSERACT_CMD
                pytesseract.pytesseract.tesseract_cmd = exe
            pytesseract.get_tesseract_version()
            print("[Detector] Tesseract OCR ready.")
            return True
        except Exception as e:
            print(f"[Detector] Tesseract unavailable ({e}) — region-only fallback.")
            return False

    def predict(self, frame_rgb: np.ndarray) -> tuple:
        """
        Run the two-stage batch code detection on `frame_rgb`.

        Returns:
            (label, confidence, is_defect, regions, ocr_text)
        """
        regions = _find_white_label_regions(frame_rgb)

        if not self._tesseract_ok:
            found = len(regions) > 0
            conf = 0.60 if found else 0.0
            return ("OK" if found else "DEFECT"), conf, not found, regions, "OCR unavailable"

        best_text, best_score = "", 0.0
        hit_regions = []

        # ── Stage 1: OCR white sticker regions ──────────────────────────────
        for (x, y, w, h) in regions:
            crop = frame_rgb[y:y+h, x:x+w]
            text, score = _ocr_best(crop, _OCR_PSMS)
            if score > best_score:
                best_score, best_text = score, text
            if score >= config.DETECTION_THRESHOLD:
                hit_regions.append((x, y, w, h))

        if best_score >= config.DETECTION_THRESHOLD:
            return "OK", best_score, False, hit_regions or regions, best_text.strip()

        # ── Stage 2: full-frame OCR for direct-print packaging ───────────────
        h_img, w_img = frame_rgb.shape[:2]
        scale = min(1.0, 1200 / max(w_img, h_img))
        small = (cv2.resize(frame_rgb, (int(w_img * scale), int(h_img * scale)),
                            interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else frame_rgb)

        full_text, full_score = _ocr_best(small, _OCR_PSMS)
        if full_score > best_score:
            best_score, best_text = full_score, full_text

        found = best_score >= config.DETECTION_THRESHOLD
        return (
            "OK" if found else "DEFECT",
            best_score if found else 0.0,
            not found,
            regions,
            best_text.strip(),
        )
