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


def test_score_keyword_plus_date_passes():
    """Keyword + date is the minimum passing combination."""
    from model.inference import _score_text
    assert _score_text("Batch No.\n28/09/25") >= 0.55


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
