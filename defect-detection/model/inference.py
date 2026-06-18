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
import os
import shutil
import platform
import subprocess
from collections import namedtuple

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

# Time printed before the batch code: HH:MM or HH MM (OCR sometimes drops ':').
# Minutes digits: Tesseract often misreads '0' as 'O' (letter O) at small sizes,
# so [0-5O][0-9O] accepts both.  '07 O4' and '07:04' both match → 07:04.
_PAT_TIME = re.compile(r'\b([01]?\d|2[0-3])[: ][0-5O][0-9O]\b')

# Alphanumeric batch code: mix of uppercase letters + digits, 4-10 chars
# e.g. 01B11, 06B11, 09A11, 2007B11
_PAT_BATCH_CODE = re.compile(
    r'\b(?=[A-Z0-9]{4,10}\b)(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{4,10}\b'
)

# MRP price line: NN.NN/(N.NN) — unique to ITC batch sticker labels
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

    Hard AND-gate — two valid paths:
      A: label keyword (batch/pkd/use by/mrp/…) + at least one date.
      B: printed time (HH:MM) + at least two dates.
         Covers dark cardboard where all label text is garbled but the
         numeric block (time, PKD date, Use By date) still reads through.
    score is 0.0 if neither path is satisfied.
    """
    kw_m   = _PAT_KEYWORD.search(text)
    time_m = _PAT_TIME.search(text)
    dates  = _PAT_DATE.findall(text)
    has_batch = bool(_PAT_BATCH_CODE.search(text))
    has_mrp   = bool(_PAT_MRP.search(text))

    has_keyword = bool(kw_m)
    has_time    = bool(time_m)
    n_dates     = len(dates)

    if (has_keyword and n_dates >= 1) or (has_time and n_dates >= 2):
        score = 0.40                           # base
        score += 0.20 * min(n_dates, 2)        # up to +0.40 for PKD + Use By
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


def _richness(ev: "_Evidence") -> int:
    """
    Tie-break rank for results that all score 0.0, so the per-frame debug
    summary surfaces the *most promising* fragment (e.g. one with a date over
    one with pure noise). Purely diagnostic — never affects the OK/DEFECT call.
    """
    return (ev.n_dates * 3 + ev.has_time * 2 + ev.has_keyword * 2
            + ev.has_batch + ev.has_mrp)


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


def _frame_quality(gray: np.ndarray) -> tuple:
    """
    (brightness, sharpness) for a grayscale frame — pure diagnostics.

      brightness : mean pixel value 0-255. Too low = underexposed / lens cap;
                   too high = blown-out glare. OCR needs the text in between.
      sharpness  : variance of the Laplacian, the standard focus measure.
                   High = crisp edges (in focus); low = blurred / out of focus.
                   If OCR is pure noise AND sharpness is low, the lens — not the
                   threshold logic — is the problem.
    """
    brightness = float(np.mean(gray))
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness


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


# Stage 1 (full-frame): PSM 11 only — sparse-text mode is best for unstructured
# packaging; PSM 3 rarely adds signal and doubles the call count.
# Stage 1 also caps at 4 preprocessing variants (skips adaptive — slowest,
# rarely needed on a full 1920×1080 frame).
# Stage 2 (region crops): full PSM set + all 6 variants since crops are small
# and run much faster than full-frame calls.
_OCR_PSMS_FRAME   = (11,)
_FRAME_MAX_VARS   = 4        # Otsu ×2 + CLAHE ×2; adaptive (v=4,5) skipped
_OCR_PSMS_CROP    = (11, 6)  # sparse + block; dropped 4/3 — too slow for live
# Stage-2 latency caps: the no-code path must stay responsive. Worst case is
# now _STAGE2_MAX_REGIONS × 2 grays × _STAGE2_MAX_VARS × len(_OCR_PSMS_CROP).
_STAGE2_MAX_REGIONS = 3
_STAGE2_MAX_VARS    = 2      # Otsu + inverted Otsu only


def _ocr_run(image: np.ndarray, psms: tuple, max_vars: int = 0) -> tuple:
    """
    Run Tesseract on `image` across preprocessing variants and given PSMs.
    Returns (text, score, evidence) for the best combination found.
    Exits early the moment a score at or above the detection threshold is reached.

    "Best" is ranked by (score, _richness) so that when every pass scores 0.0
    we still return the *most promising* fragment (e.g. one that found a date)
    instead of an empty string — critical for diagnosing why nothing passed.

    Two grayscale sources are tried per image (min_gb first):
      • g=0  min(G, B) channel — ~12:1 contrast for white text on dark red/maroon
      • g=1  Standard luminance gray — fallback

    max_vars: cap on preprocessing variants per gray source (0 = no cap).
              Set to _FRAME_MAX_VARS for full-frame calls to skip slow
              adaptive variants and keep Stage 1 under ~16 Tesseract calls.
    """
    import pytesseract

    std_gray = _to_gray(image)
    mg_gray  = _min_gb(image)
    # min_gb first: for ITC dark red/maroon packaging it gives ~12:1 contrast
    # vs ~4:1 with standard gray, so it reaches threshold faster → earlier exit.
    # Skip the second source if it's identical (grayscale or single-channel input).
    grays = [mg_gray, std_gray] if not np.array_equal(std_gray, mg_gray) else [std_gray]

    best_text, best_ev = "", _evaluate("")
    best_rank = (best_ev.score, _richness(best_ev))

    for g_idx, gray in enumerate(grays):
        for v_idx, variant in enumerate(_preprocessing_variants(gray)):
            if max_vars and v_idx >= max_vars:
                break
            for psm in psms:
                cfg = f'--psm {psm} --oem 3'
                try:
                    text = pytesseract.image_to_string(variant, config=cfg)
                    ev   = _evaluate(text)
                    if config.OCR_DEBUG:
                        flags = (f"kw={int(ev.has_keyword)} t={int(ev.has_time)} "
                                 f"d={ev.n_dates} bc={int(ev.has_batch)} "
                                 f"mrp={int(ev.has_mrp)}")
                        # Collapse whitespace so the raw \n\n soup is readable.
                        preview = " ".join(text.split())[:90] or "(empty)"
                        print(f"[OCR] g={g_idx} v={v_idx} psm={psm} "
                              f"score={ev.score:.2f} {flags} | {preview!r}")
                    rank = (ev.score, _richness(ev))
                    if rank > best_rank:
                        best_rank, best_text, best_ev = rank, text, ev
                    if ev.score >= _THRESHOLD:
                        return best_text, best_ev.score, best_ev
                except Exception as exc:
                    if config.OCR_DEBUG:
                        print(f"[OCR] g={g_idx} v={v_idx} psm={psm} ERROR: {exc}")
                    continue

    return best_text, best_ev.score, best_ev


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


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _startup_diagnostics() -> None:
    """
    Print the running code version + active OCR config once at startup.

    A stale-run (Pi running old code) is the single easiest way to waste a debug
    cycle. Logging the git commit and the live PSM/threshold settings makes it
    obvious at the top of output.txt exactly what produced the run below.
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

    print(f"[Detector] code commit={commit} | engine={config.OCR_ENGINE} | "
          f"threshold={_THRESHOLD} | frame_psms={_OCR_PSMS_FRAME} "
          f"frame_max_vars={_FRAME_MAX_VARS} crop_psms={_OCR_PSMS_CROP}")
    print(f"[Detector] config: WHITE_THRESHOLD={config.WHITE_THRESHOLD} "
          f"OCR_MIN_HEIGHT={config.OCR_MIN_HEIGHT} "
          f"DARK_FRAME_THRESHOLD={config.DARK_FRAME_THRESHOLD} "
          f"FOCUS_MIN_SHARPNESS={config.FOCUS_MIN_SHARPNESS} "
          f"OCR_DEBUG={config.OCR_DEBUG} OCR_DEBUG_IMAGES={config.OCR_DEBUG_IMAGES}")


def _print_frame_header(frame_n, w, h, brightness, sharpness) -> None:
    """
    Printed BEFORE the OCR passes so focus/exposure is captured even if the run
    is killed mid-frame. brightness/sharpness here is the single most important
    triage signal: if sharp is low, fix the lens before touching OCR logic.
    """
    focus = "OK" if sharpness >= config.FOCUS_MIN_SHARPNESS else "SOFT"
    print(f"[FRAME #{frame_n}] {w}x{h} bright={brightness:.1f} sharp={sharpness:.0f} "
          f"FOCUS:{focus} — OCR passes follow")


def _print_frame_summary(frame_n, w, h, brightness, sharpness, stage,
                         ev: "_Evidence", text: str, n_regions=None) -> None:
    """
    One consolidated line per processed frame — the headline diagnostic.

    Tells us, at a glance: was the frame in focus and exposed (sharp/bright),
    how far the best OCR pass got (score), which evidence survived (kw/time/
    dates/batch/mrp), and the actual best text. When detection fails this line
    says *why*: blurry vs. dark vs. "read fine but only 1 date so gate B failed".
    """
    focus   = "OK" if sharpness >= config.FOCUS_MIN_SHARPNESS else "SOFT"
    dates_s = ",".join(ev.dates[:3]) if ev.dates else "-"
    reg     = f" regions={n_regions}" if n_regions is not None else ""
    preview = " ".join(text.split())[:120] or "(no text)"
    print(f"[FRAME #{frame_n}] {w}x{h} bright={brightness:.1f} sharp={sharpness:.0f} "
          f"FOCUS:{focus} | stage={stage}{reg} score={ev.score:.2f} | "
          f"kw={int(ev.has_keyword)} time={ev.time or '-'} "
          f"dates={ev.n_dates}[{dates_s}] batch={int(ev.has_batch)} "
          f"mrp={int(ev.has_mrp)} | best={preview!r}")


def _dump_debug_images(frame_rgb: np.ndarray, frame_n: int) -> None:
    """
    Save the raw frame + both gray sources + the binarised variants Tesseract
    sees, so legibility can be judged by eye (is the text even there / sharp /
    surviving the threshold?). Gated by OCR_DEBUG_IMAGES; throttled by caller.
    """
    try:
        os.makedirs(config.DEBUG_DIR, exist_ok=True)
        stamp = f"f{frame_n:05d}"
        cv2.imwrite(os.path.join(config.DEBUG_DIR, f"{stamp}_raw.jpg"),
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        for name, gray in (("mingb", _min_gb(frame_rgb)), ("std", _to_gray(frame_rgb))):
            cv2.imwrite(os.path.join(config.DEBUG_DIR, f"{stamp}_{name}.jpg"), gray)
            for v_idx, variant in enumerate(_preprocessing_variants(gray)):
                if v_idx >= _FRAME_MAX_VARS:
                    break
                cv2.imwrite(
                    os.path.join(config.DEBUG_DIR, f"{stamp}_{name}_v{v_idx}.jpg"),
                    variant,
                )
        print(f"[debug-img] wrote frame #{frame_n} set to {config.DEBUG_DIR}")
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
        regions    : list of (x, y, w, h) confirmed hit boxes (empty if DEFECT)
        ocr_text   : best OCR text seen (for HUD subtitle / debug)
    """

    def __init__(self):
        self._frame_n = 0
        self._paddle = None
        self._tesseract_ok = False

        # PaddleOCR is the accurate on-device path; Tesseract is the fast default
        # and the fallback if Paddle is requested but unavailable.
        if config.OCR_ENGINE == "paddle":
            self._paddle = self._init_paddle()
        if self._paddle is None:
            self._tesseract_ok = self._init_tesseract()

        _startup_diagnostics()

    def _init_paddle(self):
        try:
            from model.ocr_engines import PaddleOCREngine
            eng = PaddleOCREngine()
            print("[Detector] PaddleOCR engine ready.")
            return eng
        except Exception as e:
            print(f"[Detector] PaddleOCR unavailable ({e}); falling back to Tesseract. "
                  f"Install with: pip install paddlepaddle paddleocr")
            return None

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
        Batch code detection on `frame_rgb` (H×W×3 RGB uint8). Dispatches to the
        active OCR engine. With no working engine, always returns DEFECT — never
        a false positive.
        """
        if self._paddle is not None:
            return self._predict_paddle(frame_rgb)
        if not self._tesseract_ok:
            return "DEFECT", 0.0, True, [], "OCR unavailable — install an OCR engine"
        return self._predict_tesseract(frame_rgb)

    def _predict_paddle(self, frame_rgb: np.ndarray) -> tuple:
        """
        Single-pass detection with PaddleOCR — it detects + recognises text on
        the raw frame, so no binarisation/PSM sweep is needed. Recognised text
        is scored by the same evidence gate as Tesseract.
        """
        self._frame_n += 1
        h_img, w_img = frame_rgb.shape[:2]
        brightness, sharpness = _frame_quality(_to_gray(frame_rgb))
        if config.OCR_DEBUG:
            _print_frame_header(self._frame_n, w_img, h_img, brightness, sharpness)

        text = self._paddle.read(frame_rgb)
        ev   = _evaluate(text)
        if config.OCR_DEBUG:
            _print_frame_summary(self._frame_n, w_img, h_img, brightness, sharpness,
                                 "paddle", ev, text)
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

    def _predict_tesseract(self, frame_rgb: np.ndarray) -> tuple:
        """
        Two-stage Tesseract detection.

        Stage 1 — Full-frame OCR (PSM 11, 4 variants), single full-res pass by
          default (OCR_MULTISCALE adds a 1280px pass). Exits as soon as the
          threshold is cleared.
        Stage 2 — Region crop OCR fallback for white-sticker packaging,
          hard-capped (_STAGE2_MAX_REGIONS) so the no-code path stays responsive.
        """
        self._frame_n += 1
        h_img, w_img = frame_rgb.shape[:2]
        brightness, sharpness = _frame_quality(_to_gray(frame_rgb))
        if config.OCR_DEBUG:
            _print_frame_header(self._frame_n, w_img, h_img, brightness, sharpness)

        best_text, best_ev = "", _evaluate("")
        best_rank = (best_ev.score, _richness(best_ev))

        def _consider(text, ev):
            nonlocal best_text, best_ev, best_rank
            rank = (ev.score, _richness(ev))
            if rank > best_rank:
                best_rank, best_text, best_ev = rank, text, ev

        # ── Stage 1: full-frame OCR ──────────────────────────────────────────
        # Single full-res pass by default; second 1280px pass only if requested.
        scales = [max(h_img, w_img)]
        if config.OCR_MULTISCALE:
            scales.append(1280)
        for max_dim in scales:
            scale = min(1.0, max_dim / max(h_img, w_img))
            candidate = (
                cv2.resize(frame_rgb, (int(w_img * scale), int(h_img * scale)),
                           interpolation=cv2.INTER_AREA)
                if scale < 1.0 else frame_rgb
            )
            text, score, ev = _ocr_run(candidate, _OCR_PSMS_FRAME, max_vars=_FRAME_MAX_VARS)
            _consider(text, ev)
            if best_ev.score >= _THRESHOLD:
                if config.OCR_DEBUG:
                    _print_frame_summary(self._frame_n, w_img, h_img, brightness,
                                         sharpness, 1, best_ev, best_text)
                return "OK", best_ev.score, False, [], best_text.strip()

        # ── Stage 2: region crop OCR (white sticker on dark packaging) ───────
        # Only runs on negative frames (Stage 1 returns early on success) and is
        # hard-capped so the no-code path can't freeze the live indicator.
        candidates = (
            _find_label_regions(frame_rgb)[:_STAGE2_MAX_REGIONS]
            if config.STAGE2_ENABLED else []
        )
        hit_regions = []
        for (x, y, rw, rh) in candidates:
            crop = frame_rgb[y:y + rh, x:x + rw]
            text, score, ev = _ocr_run(crop, _OCR_PSMS_CROP, max_vars=_STAGE2_MAX_VARS)
            _consider(text, ev)
            if score >= _THRESHOLD:
                hit_regions.append((x, y, rw, rh))

        found = best_ev.score >= _THRESHOLD

        if config.OCR_DEBUG:
            _print_frame_summary(self._frame_n, w_img, h_img, brightness, sharpness,
                                 2, best_ev, best_text, n_regions=len(candidates))
        if config.OCR_DEBUG_IMAGES and self._frame_n % config.DEBUG_IMAGE_EVERY == 0:
            _dump_debug_images(frame_rgb, self._frame_n)

        return (
            "OK"     if found else "DEFECT",
            best_ev.score if found else 0.0,
            not found,
            hit_regions if found else [],
            best_text.strip(),
        )
