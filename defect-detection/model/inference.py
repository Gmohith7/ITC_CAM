"""
Batch code detector — ITC product packaging, OpenCV + Tesseract OCR.

ITC batch code block structure (all products observed):
───────────────────────────────────────────────────────
  Batch No.:   HH:MM  XXXXXB11          ← time + alphanumeric code
  PKD.:        DD/MM/YY                  ← packed date
  Use By:      DD/MM/YY                  ← expiry date
  MRP Rs. incl. of all taxes/(Rs. per g) ← price line
  NN.NN/(N.NN)

Two physical formats:
  • White sticker on dark/foil packaging  (Stage 1: sticker finder)
  • Direct-print on coloured cardboard box (Stage 2: full-frame OCR)

Detection is STRICT: we require co-occurrence of a label keyword AND a
date in the same OCR result. A date alone, or a keyword alone, is not
enough — this eliminates false positives from faces, laptops, shelves.

The region-only fallback (no Tesseract) always returns DEFECT.
"""

import re
import shutil
import platform
import cv2
import numpy as np
import config

# ── Structural patterns ───────────────────────────────────────────────────────

# Dates: DD/MM/YY, DD/MM/YYYY, DD-MM-YY, DD.MM.YY — always 2-digit day/month
_PAT_DATE = re.compile(r'\b\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}\b')

# ITC-specific label keywords that bracket the batch block
_PAT_KEYWORD = re.compile(
    r'\b(batch[\s\.\-]*no|batch[\s\.\-]*code|lot[\s\.\-]*no|'
    r'pkd|use[\s]*by|mfd|mfg|best[\s]*before|expiry|exp)\b',
    re.IGNORECASE,
)

# Time printed before the batch code: HH:MM or HH MM (OCR drops colon sometimes)
_PAT_TIME = re.compile(r'\b([01]?\d|2[0-3])[: ][0-5]\d\b')

# Alphanumeric batch code: digits + letters combo like 01B11, 06B11, 2007B11, 02A11
# Must contain at least one letter and at least one digit, 4-10 chars
_PAT_BATCH_CODE = re.compile(r'\b(?=[A-Z0-9]{4,10}\b)(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{4,10}\b')

# MRP line: NN.NN/(N.NN) — unique to ITC sticker labels
_PAT_MRP = re.compile(r'\d+\.\d{2}\s*/\s*\(\s*\d+\.\d{2,3}\s*\)')

# ── Scoring ───────────────────────────────────────────────────────────────────
#
# Scoring requires BOTH a keyword AND a date to co-occur (AND-gate at 0.55).
# Additional evidence (time, batch code, MRP) adds on top.
# Threshold is at 0.55 so a lone date or lone keyword never triggers OK.

_THRESHOLD = config.DETECTION_THRESHOLD   # default 0.55 (set in config)


def _score_text(text: str) -> float:
    """
    Evidence-based confidence score in [0.0, 1.0].

    Requires keyword + date co-occurrence as the base condition.
    Everything else is bonus evidence. Returns 0.0 if the base fails.
    """
    has_keyword = bool(_PAT_KEYWORD.search(text))
    dates       = _PAT_DATE.findall(text)
    n_dates     = len(dates)

    # Hard gate: must have at least one label keyword AND one date
    if not has_keyword or n_dates == 0:
        return 0.0

    score = 0.40                           # base: keyword present
    score += 0.20 * min(n_dates, 2)        # up to 2 dates (PKD + Use By) = +0.40
    if _PAT_TIME.search(text):
        score += 0.10                      # time before batch code
    if _PAT_BATCH_CODE.search(text):
        score += 0.10                      # alphanumeric code found
    if _PAT_MRP.search(text):
        score += 0.10                      # MRP price line

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
    Yield binarised images for OCR, tuned for ITC packaging:
      - White sticker (dark text on white): Otsu + adaptive
      - Direct-print gold/white on dark: inverted Otsu + CLAHE
      - Faded/low-contrast: CLAHE + adaptive
    """
    h, w = gray.shape
    if h < config.OCR_MIN_HEIGHT:
        scale = config.OCR_MIN_HEIGHT / h
        gray  = cv2.resize(gray, (int(w * scale), config.OCR_MIN_HEIGHT),
                           interpolation=cv2.INTER_CUBIC)

    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # 1. Otsu — dark text on white sticker
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu

    # 2. Inverted Otsu — white/gold text on dark background (direct-print box)
    yield cv2.bitwise_not(otsu)

    # 3. CLAHE + Otsu — low-contrast / faded print
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(gray)
    _, clahe_bin = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield clahe_bin

    # 4. Inverted CLAHE — low-contrast light-on-dark
    yield cv2.bitwise_not(clahe_bin)

    # 5. Adaptive threshold — uneven lighting / curved packaging
    adapt = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    yield adapt

    # 6. Inverted adaptive
    yield cv2.bitwise_not(adapt)


# PSM modes used: 6=single block, 11=sparse text, 4=single column, 3=auto
_OCR_PSMS = (6, 11, 4, 3)


def _ocr_best(image: np.ndarray) -> tuple:
    """
    Run Tesseract across all preprocessing variants and PSM modes.
    Returns (text, score) for the variant with the highest evidence score.
    """
    import pytesseract
    gray = _to_gray(image)
    best_text, best_score = "", 0.0
    for variant in _preprocessing_variants(gray):
        for psm in _OCR_PSMS:
            cfg = f'--psm {psm} --oem 3'
            try:
                text  = pytesseract.image_to_string(variant, config=cfg)
                score = _score_text(text)
                if score > best_score:
                    best_score, best_text = score, text
            except Exception:
                continue
    return best_text, best_score


# ── Sticker / label region finder ────────────────────────────────────────────

def _find_label_regions(frame_rgb: np.ndarray) -> list:
    """
    Find candidate regions that could contain the batch code block.

    Two strategies:
      A. White sticker on dark packaging — threshold on brightness
      B. Lighter rectangular patch on a colour box — variance-based

    Returns list of (x, y, w, h) sorted by area descending (largest first).
    """
    gray     = _to_gray(frame_rgb)
    h_img, w_img = gray.shape[:2]
    frame_area   = w_img * h_img

    # ── Strategy A: bright white sticker ────────────────────────────────────
    _, thresh = cv2.threshold(gray, config.WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel_a  = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.MORPH_KERNEL_W, config.MORPH_KERNEL_H)
    )
    closed_a = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_a)

    # ── Strategy B: any high-contrast rectangular region (colour boxes) ──────
    # Canny edges → dilate → find rectangles
    edges    = cv2.Canny(gray, 40, 120)
    kernel_b = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))
    closed_b = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_b)

    # Merge both masks
    combined = cv2.bitwise_or(closed_a, closed_b)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    pad = config.REGION_PADDING
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area   = rw * rh
        aspect = rw / max(rh, 1)
        # Filter by size and aspect ratio — batch block is wider than tall
        if frame_area * 0.003 < area < frame_area * 0.55 and 0.25 < aspect < 14.0:
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + rw + pad)
            y2 = min(h_img, y + rh + pad)
            regions.append((x1, y1, x2 - x1, y2 - y1))

    # Deduplicate heavily overlapping regions (IoU > 0.5 → keep larger)
    regions = _nms_regions(regions)
    regions.sort(key=lambda r: r[2] * r[3], reverse=True)
    return regions[:8]


def _iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms_regions(regions: list) -> list:
    """Remove duplicate / heavily overlapping candidate regions."""
    regions = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)
    keep = []
    for r in regions:
        if all(_iou(r, k) < 0.5 for k in keep):
            keep.append(r)
    return keep


# ── Main detector ─────────────────────────────────────────────────────────────

class BatchCodeDetector:
    """
    Detects whether an ITC product batch code block is visible in the frame.

    Returns (label, confidence, is_defect, regions, ocr_text):
        label      : "OK" or "DEFECT"
        confidence : float [0.0, 1.0]  — 0.0 when DEFECT
        is_defect  : True when no batch code evidence found
        regions    : list of (x, y, w, h) confirmed hit boxes (empty if DEFECT)
        ocr_text   : best OCR text (for HUD display)
    """

    def __init__(self):
        self._tesseract_ok = self._init_tesseract()

    def _init_tesseract(self) -> bool:
        try:
            import pytesseract
            if platform.system() == "Windows":
                exe = shutil.which("tesseract") or config.TESSERACT_CMD
                pytesseract.pytesseract.tesseract_cmd = exe
            pytesseract.get_tesseract_version()
            print("[Detector] Tesseract OCR ready.")
            return True
        except Exception as e:
            print(f"[Detector] Tesseract unavailable ({e}) — will always return DEFECT.")
            return False

    def predict(self, frame_rgb: np.ndarray) -> tuple:
        """
        Run two-stage batch code detection on `frame_rgb` (H×W×3 RGB uint8).
        Without Tesseract, always returns DEFECT — never a false positive.
        """
        # Without OCR we cannot confirm anything — strict DEFECT
        if not self._tesseract_ok:
            return "DEFECT", 0.0, True, [], "OCR unavailable — install Tesseract"

        candidates = _find_label_regions(frame_rgb)

        best_text, best_score = "", 0.0
        hit_regions = []

        # ── Stage 1: OCR each candidate region ──────────────────────────────
        for (x, y, rw, rh) in candidates:
            crop = frame_rgb[y:y + rh, x:x + rw]
            text, score = _ocr_best(crop)
            if score > best_score:
                best_score, best_text = score, text
            if score >= _THRESHOLD:
                hit_regions.append((x, y, rw, rh))

        if best_score >= _THRESHOLD:
            return "OK", best_score, False, hit_regions, best_text.strip()

        # ── Stage 2: full-frame OCR (direct-print on coloured cardboard) ────
        h_img, w_img = frame_rgb.shape[:2]
        scale = min(1.0, 1280 / max(w_img, h_img))
        small = (cv2.resize(frame_rgb,
                            (int(w_img * scale), int(h_img * scale)),
                            interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else frame_rgb)

        full_text, full_score = _ocr_best(small)
        if full_score > best_score:
            best_score, best_text = full_score, full_text

        found = best_score >= _THRESHOLD
        return (
            "OK"     if found else "DEFECT",
            best_score if found else 0.0,
            not found,
            hit_regions if found else [],   # empty list = no boxes drawn when DEFECT
            best_text.strip(),
        )
