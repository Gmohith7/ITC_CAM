"""
Unit tests for the defect detection pipeline.
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── draw_result overlay ───────────────────────────────────────────────────────

def test_draw_result_ok():
    from preprocessing.preprocess import draw_result
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_result(frame.copy(), "OK", 0.80, defect=False)
    assert out.shape == frame.shape


def test_draw_result_defect():
    from preprocessing.preprocess import draw_result
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_result(frame.copy(), "DEFECT", 0.0, defect=True,
                      regions=[], ocr_text="")
    assert out.shape == frame.shape


def test_draw_result_scanning():
    from preprocessing.preprocess import draw_result
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_result(frame.copy(), "SCANNING", 0.0, scanning=True)
    assert out.shape == frame.shape


def test_draw_result_grayscale_input():
    """Grayscale frames are converted to RGB before drawing."""
    from preprocessing.preprocess import draw_result
    frame = np.zeros((480, 640), dtype=np.uint8)
    out = draw_result(frame.copy(), "SCANNING", 0.0, scanning=True)
    assert out.ndim == 3 and out.shape[2] == 3


def test_draw_result_high_res_stays_in_bounds():
    """Badge and text must not exceed frame bounds at 1080p."""
    from preprocessing.preprocess import draw_result
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = draw_result(frame.copy(), "OK", 0.92,
                      regions=[(100, 100, 400, 200)],
                      ocr_text="Batch No.: 14:23 01A11\nPKD: 19/01/26")
    assert out.shape == frame.shape


# ── Scoring — strict AND-gate logic ──────────────────────────────────────────

def test_score_empty_is_zero():
    from model.inference import _score_text
    assert _score_text("") == 0.0


def test_score_date_alone_is_zero():
    """A date alone must NOT pass — no keyword co-occurrence."""
    from model.inference import _score_text
    assert _score_text("28/09/25") == 0.0


def test_score_keyword_alone_is_zero():
    """A keyword alone must NOT pass — no date co-occurrence."""
    from model.inference import _score_text
    assert _score_text("Batch No.") == 0.0
    assert _score_text("PKD") == 0.0


def test_score_keyword_plus_two_dates_passes():
    """Keyword + the block's two dates (PKD + Use By) is a passing combination."""
    from model.inference import _score_text
    assert _score_text("Batch No.\n28/09/25\n24/06/26") >= 0.55


def test_score_keyword_plus_single_date_is_zero():
    """Only the batch block matters: a label + ONE date is not enough — the real
    block always prints both PKD and Use By dates. Guards against false OK on a
    stray date elsewhere on the pack."""
    from model.inference import _score_text
    assert _score_text("Batch No.\n28/09/25") == 0.0
    assert _score_text("MRP 24/02/27") == 0.0


def test_score_full_itc_block_high():
    """Full realistic ITC sticker text must score ≥ 0.80."""
    from model.inference import _score_text
    sample = "Batch No.: 14:47 01B11\nPKD.: 28/09/25\nUse By: 24/06/26\n44.00/(0.64)"
    assert _score_text(sample) >= 0.80


def test_score_never_exceeds_one():
    from model.inference import _score_text
    sample = "Batch No. PKD Use By\n28/09/25\n24/06/26\n14:47\n44.00/(0.64)"
    assert _score_text(sample) <= 1.0


def test_score_random_text_is_zero():
    from model.inference import _score_text
    assert _score_text("The quick brown fox jumps over the lazy dog") == 0.0


def test_score_garbled_date_slash_as_nine():
    """'/' read as '9' on dark cardboard: 24/02/27 -> 24902127 still matches."""
    from model.inference import _score_text
    assert _score_text("PKD 24902127\n31105126") >= 0.55


def test_score_garbled_date_slash_as_one():
    """'/' read as '1': 31/05/26 -> 31105126 still matches."""
    from model.inference import _score_text
    assert _score_text("PKD 31105126\n24902127") >= 0.55


def test_score_mrp_keyword_plus_two_dates_passes():
    """'mrp' is accepted as a block keyword; needs the block's two dates."""
    from model.inference import _score_text
    assert _score_text("MRP 24/02/27\n31/05/26") >= 0.55


def test_score_mrp_alone_is_zero():
    """'mrp' alone (no date) must not pass the AND-gate."""
    from model.inference import _score_text
    assert _score_text("MRP Rs. incl. of all taxes") == 0.0


def test_score_date_alone_with_relaxed_regex_still_zero():
    """Even with relaxed separators, a date without keyword must score 0."""
    from model.inference import _score_text
    assert _score_text("24902127") == 0.0


def test_score_garbled_date_slash_as_four():
    """'/' read as '4': 31/08/26 -> 31408126 still matches."""
    from model.inference import _score_text
    assert _score_text("PKD 31408126\n24102127") >= 0.55


def test_score_time_plus_two_dates_no_keyword():
    """Time + 2 dates passes with zero readable keywords (dark cardboard).

    Mirrors real OCR output: '07 04 ODAN\\n31408126\\n24102127' where all
    label text (Batch No., PKD., Use By) is garbled but the numeric block
    (time + both dates) survives.
    """
    from model.inference import _score_text
    assert _score_text("07 04 ODAN\n31408126\n24102127") >= 0.55


def test_score_time_plus_one_date_is_zero():
    """Time + only 1 date must NOT pass — too ambiguous without keyword."""
    from model.inference import _score_text
    assert _score_text("14:47\n28/09/25") == 0.0


def test_score_time_with_letter_O_in_minutes():
    """'O' (letter O) misread as '0' in minutes: '07 O4' must match as 07:04."""
    from model.inference import _score_text
    # '07 O4' = time, '24102127' + '31408126' = two dates → time+2dates gate
    assert _score_text("07 O4\n24102127\n31408126") >= 0.55


# ── Batch code detector ───────────────────────────────────────────────────────

def test_detector_blank_frame_is_defect():
    from model.inference import BatchCodeDetector
    det = BatchCodeDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    label, confidence, is_defect, regions, text = det.predict(blank)
    assert label == "DEFECT"
    assert is_defect is True
    assert confidence == 0.0


def test_detector_no_tesseract_is_strict_defect():
    """Without Tesseract, detector must always return DEFECT (no false OK)."""
    from model.inference import BatchCodeDetector
    det = BatchCodeDetector()
    det._tesseract_ok = False   # simulate missing Tesseract
    white = np.full((480, 640, 3), 255, dtype=np.uint8)
    label, confidence, is_defect, regions, text = det.predict(white)
    assert label == "DEFECT"
    assert is_defect is True
    assert confidence == 0.0
    assert regions == []


def test_detector_returns_empty_regions_on_defect():
    """DEFECT must return empty regions list so no boxes are drawn."""
    from model.inference import BatchCodeDetector
    det = BatchCodeDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    label, confidence, is_defect, regions, text = det.predict(blank)
    assert regions == []


# ── Logging ───────────────────────────────────────────────────────────────────

def test_log_detection_creates_csv(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "LOG_SNAPSHOTS", False)
    import importlib
    import defect_logging.logger as lg_mod
    importlib.reload(lg_mod)
    lg_mod.log_detection("DEFECT", 0.0)
    csv_path = tmp_path / "detections.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 2  # header + 1 row


def test_log_detection_saves_snapshot(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "LOG_SNAPSHOTS", True)
    import importlib
    import defect_logging.logger as lg_mod
    importlib.reload(lg_mod)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    lg_mod.log_detection("DEFECT", 0.0, frame=frame)
    snapshots = list((tmp_path / "snapshots").glob("*.jpg"))
    assert len(snapshots) == 1


def test_log_detection_no_snapshot_when_disabled(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "LOG_SNAPSHOTS", False)
    import importlib
    import defect_logging.logger as lg_mod
    importlib.reload(lg_mod)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    lg_mod.log_detection("DEFECT", 0.0, frame=frame)
    assert not (tmp_path / "snapshots").exists()


# ── Alert system ──────────────────────────────────────────────────────────────

def test_alert_no_crash_without_gpio():
    from alerts.alert import AlertSystem
    a = AlertSystem()
    a.trigger(duration=0)
    a.clear()


def test_alert_trigger_is_nonblocking():
    import time
    from alerts.alert import AlertSystem
    a = AlertSystem()
    start = time.monotonic()
    a.trigger(duration=2.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"trigger() blocked for {elapsed:.2f}s"


def test_alert_debounce():
    from alerts.alert import AlertSystem
    a = AlertSystem()
    a.trigger(duration=5.0)
    a.trigger(duration=5.0)  # must not raise or double-lock
    a.clear()


# ── Config sanity ─────────────────────────────────────────────────────────────

def test_config_values():
    import config
    assert config.FRAME_RATE > 0
    assert isinstance(config.TESSERACT_CMD, str)
    assert isinstance(config.LOG_DIR, str)
    assert 0 < config.WHITE_THRESHOLD < 256
    assert config.OCR_MIN_HEIGHT > 0
    assert 0.0 < config.DETECTION_THRESHOLD < 1.0
    assert config.DARK_FRAME_THRESHOLD >= 0
    assert config.REGION_PADDING >= 0
    assert config.ALERT_DURATION_S >= 0


def test_grayscale_mode_is_false_by_default():
    """Color mode must be on by default — grayscale was the old broken default."""
    import config
    assert config.GRAYSCALE_MODE is False
