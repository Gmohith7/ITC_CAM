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
  • White sticker on dark/foil packaging  (Stage 2: region crop OCR)
  • Direct-print on coloured cardboard box (Stage 1: full-frame OCR)

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

# Dates: DD/MM/YY or DD/MM/YYYY.
# On direct-print dark cardboard Tesseract misreads the '/' separator in many
# ways: '1' (narrow stroke), '9' (top serif), '4' (two strokes merge), etc.
# The class [/\-\.|l1-9] accepts any digit 1-9 as a separator substitute so
# "24102127" (24/02/27), "24902127" (sep=9), and "31408126" (sep=4) all match.
# Day is [0-3]\d and month is [0-1]\d to suppress false matches on arbitrary
# digit strings.
_PAT_DATE = re.compile(
    r'\b[0-3]\d[/\-\.|l1-9]?[0-1]\d[/\-\.|l1-9]?\d{2,4}\b',
    re.IGNORECASE,
)

# ITC-specific label keywords that co-occur with the batch block.
# 'mrp' is added as a short, unambiguous fallback — "MRP Rs." always appears
# in the block and is more likely to survive OCR noise than multi-word labels.
_PAT_KEYWORD = re.compile(
    r'\b(batch[\s\.\-]*no|batch[\s\.\-]*code|lot[\s\.\-]*no|'
    r'pkd|use[\s]*by|mfd|mfg|best[\s]*before|expiry|exp|mrp)\b',
    re.IGNORECASE,
)

# Time printed before the batch code: HH:MM or HH MM (OCR sometimes drops ':')
_PAT_TIME = re.compile(r'\b([01]?\d|2[0-3])[: ][0-5]\d\b')

# Alphanumeric batch code: mix of uppercase letters + digits, 4-10 chars
# e.g. 01B11, 06B11, 09A11, 2007B11
_PAT_BATCH_CODE = re.compile(
    r'\b(?=[A-Z0-9]{4,10}\b)(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{4,10}\b'
)

# MRP price line: NN.NN/(N.NN) — unique to ITC batch sticker labels
_PAT_MRP = re.compile(r'\d+\.\d{2}\s*/\s*\(\s*\d+\.\d{2,3}\s*\)')

# ── Scoring ───────────────────────────────────────────────────────────────────

_THRESHOLD = config.DETECTION_THRESHOLD   # default 0.55


def _score_text(text: str) -> float:
    """
    Evidence-based confidence score in [0.0, 1.0].

    Hard AND-gate — two valid paths:
      A: label keyword (batch/pkd/use by/mrp/…) + at least one date.
      B: printed time (HH:MM) + at least two dates.
         Covers dark cardboard where all label text is garbled but the
         numeric block (time, PKD date, Use By date) still reads through.
    Returns 0.0 if neither path is satisfied.
    """
    has_keyword = bool(_PAT_KEYWORD.search(text))
    has_time    = bool(_PAT_TIME.search(text))
    dates       = _PAT_DATE.findall(text)
    n_dates     = len(dates)

    if not ((has_keyword and n_dates >= 1) or (has_time and n_dates >= 2)):
        return 0.0

    score = 0.40                           # base
    score += 0.20 * min(n_dates, 2)        # up to +0.40 for PKD + Use By
    if has_time:
        score += 0.10
    if _PAT_BATCH_CODE.search(text):
        score += 0.10
    if _PAT_MRP.search(text):
        score += 0.10

    return min(round(score, 2), 1.0)


# ── Image preprocessing ───────────────────────────────────────────────────────

def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _min_gb(frame: np.ndarray) -> np.ndarray:
    """
    Per-pixel minimum of the G and B channels (RGB input).

    For white text (255, 255, 255) on dark red/maroon (~120, 20, 30):
      standard luminance gray ≈ 58,  min(G, B) ≈ 20.
    The dark-red background becomes near-black, giving a contrast ratio of
    ~12:1 vs ~4:1 with standard gray — Otsu thresholding is much cleaner.
    Falls back to standard gray for grayscale or single-channel inputs.
    """
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return np.minimum(frame[:, :, 1], frame[:, :, 2])
    return _to_gray(frame)


def _preprocessing_variants(gray: np.ndarray):
    """
    Yield binarised variants for Tesseract, tuned for ITC packaging:
      1/2  Otsu + inverted  (dark-on-white sticker / white-on-dark direct-print)
      3/4  CLAHE + inverted (low-contrast or faded print)
      5/6  Adaptive + inverted (uneven lighting / curved packaging surface)
    """
    h, w = gray.shape
    if h < config.OCR_MIN_HEIGHT:
        scale = config.OCR_MIN_HEIGHT / h
        gray  = cv2.resize(gray, (int(w * scale), config.OCR_MIN_HEIGHT),
                           interpolation=cv2.INTER_CUBIC)

    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu
    yield cv2.bitwise_not(otsu)

    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, clahe_bin = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield clahe_bin
    yield cv2.bitwise_not(clahe_bin)

    adapt = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    yield adapt
    yield cv2.bitwise_not(adapt)


# PSM 11 (sparse text) is best for full-frame packaging — finds text anywhere
# without layout assumptions. PSM 3 (auto) is a complementary fallback.
# Region crops also try PSM 6 (single block) and PSM 4 (single column).
_OCR_PSMS_FRAME = (11, 3)
_OCR_PSMS_CROP  = (6, 11, 4, 3)


def _ocr_run(image: np.ndarray, psms: tuple) -> tuple:
    """
    Run Tesseract on `image` across all preprocessing variants and given PSMs.
    Returns (text, score) for the highest-scoring combination found.
    Exits early the moment a score at or above the detection threshold is reached.

    Two grayscale sources are tried per image:
      • Standard luminance gray
      • min(G, B) channel — far better contrast for white text on coloured backgrounds
    """
    import pytesseract

    std_gray = _to_gray(image)
    mg_gray  = _min_gb(image)
    # Skip the second source if it's identical (grayscale input)
    grays = [std_gray, mg_gray] if not np.array_equal(std_gray, mg_gray) else [std_gray]

    best_text, best_score = "", 0.0

    for g_idx, gray in enumerate(grays):
        for v_idx, variant in enumerate(_preprocessing_variants(gray)):
            for psm in psms:
                cfg = f'--psm {psm} --oem 3'
                try:
                    text  = pytesseract.image_to_string(variant, config=cfg)
                    score = _score_text(text)
                    if config.OCR_DEBUG and text.strip():
                        print(f"[OCR] g={g_idx} v={v_idx} psm={psm} "
                              f"score={score:.2f} | {text.strip()[:120]!r}")
                    if score > best_score:
                        best_score, best_text = score, text
                    if best_score >= _THRESHOLD:
                        return best_text, best_score
                except Exception:
                    continue

    return best_text, best_score


# ── Sticker / label region finder ────────────────────────────────────────────

def _find_label_regions(frame_rgb: np.ndarray) -> list:
    """
    Find candidate regions that could contain the batch code block.

    Strategy A: bright white sticker on dark packaging (brightness threshold).
    Strategy B: any high-contrast rectangular region (Canny edges).

    Returns list of (x, y, w, h) sorted largest-first, capped at 8 candidates.
    """
    gray         = _to_gray(frame_rgb)
    h_img, w_img = gray.shape[:2]
    frame_area   = w_img * h_img

    _, thresh = cv2.threshold(gray, config.WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel_a  = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.MORPH_KERNEL_W, config.MORPH_KERNEL_H)
    )
    closed_a = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_a)

    edges    = cv2.Canny(gray, 40, 120)
    kernel_b = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))
    closed_b = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_b)

    contours, _ = cv2.findContours(
        cv2.bitwise_or(closed_a, closed_b), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    pad = config.REGION_PADDING
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area   = rw * rh
        aspect = rw / max(rh, 1)
        if frame_area * 0.003 < area < frame_area * 0.55 and 0.25 < aspect < 14.0:
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + rw + pad)
            y2 = min(h_img, y + rh + pad)
            regions.append((x1, y1, x2 - x1, y2 - y1))

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
        is_defect  : bool
        regions    : list of (x, y, w, h) confirmed hit boxes (empty if DEFECT)
        ocr_text   : best OCR text seen (for HUD subtitle / debug)
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
        Two-stage batch code detection on `frame_rgb` (H×W×3 RGB uint8).
        Without Tesseract, always returns DEFECT — never a false positive.

        Stage 1 — Full-frame OCR at two resolutions:
          • Full resolution (e.g. 1920×1080): small label text ("Batch No.:",
            "PKD.:") lands at ~25 px — Tesseract's viable lower floor.
          • 1280 px max-dim: the large value text (dates, batch code) becomes
            ~53 px — a complementary pass in case full-res misses it.
          Uses only PSM 11/3 (sparse text / auto) — best for unstructured
          packaging layouts.  Exits as soon as either scale clears the threshold.

        Stage 2 — Region crop OCR:
          Fallback for white-sticker packaging.  Each detected region is
          cropped and re-OCR'd at higher relative resolution using the full
          set of PSM modes.
        """
        if not self._tesseract_ok:
            return "DEFECT", 0.0, True, [], "OCR unavailable — install Tesseract"

        best_text, best_score = "", 0.0
        h_img, w_img = frame_rgb.shape[:2]

        # ── Stage 1: full-frame OCR ──────────────────────────────────────────
        for max_dim in (max(h_img, w_img), 1280):
            scale = min(1.0, max_dim / max(h_img, w_img))
            candidate = (
                cv2.resize(frame_rgb, (int(w_img * scale), int(h_img * scale)),
                           interpolation=cv2.INTER_AREA)
                if scale < 1.0 else frame_rgb
            )
            text, score = _ocr_run(candidate, _OCR_PSMS_FRAME)
            if score > best_score:
                best_score, best_text = score, text
            if best_score >= _THRESHOLD:
                return "OK", best_score, False, [], best_text.strip()

        # ── Stage 2: region crop OCR (white sticker on dark packaging) ───────
        candidates = _find_label_regions(frame_rgb)
        hit_regions = []
        for (x, y, rw, rh) in candidates:
            crop = frame_rgb[y:y + rh, x:x + rw]
            text, score = _ocr_run(crop, _OCR_PSMS_CROP)
            if score > best_score:
                best_score, best_text = score, text
            if score >= _THRESHOLD:
                hit_regions.append((x, y, rw, rh))

        found = best_score >= _THRESHOLD
        return (
            "OK"     if found else "DEFECT",
            best_score if found else 0.0,
            not found,
            hit_regions if found else [],
            best_text.strip(),
        )
