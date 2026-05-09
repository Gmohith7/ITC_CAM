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
                      regions=[(10, 10, 100, 50)], ocr_text="11:05 04A11")
    assert out.shape == frame.shape


# ── Batch code detector ───────────────────────────────────────────────────────

def test_detector_blank_frame_is_defect():
    from model.inference import BatchCodeDetector
    det = BatchCodeDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    label, confidence, is_defect, regions, text = det.predict(blank)
    assert label == "DEFECT"
    assert is_defect is True
    assert confidence == 0.0   # DEFECT always returns 0.0


def test_detector_white_frame_finds_region():
    from model.inference import BatchCodeDetector
    det = BatchCodeDetector()
    white = np.full((480, 640, 3), 255, dtype=np.uint8)
    label, confidence, is_defect, regions, text = det.predict(white)
    assert isinstance(label, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert isinstance(is_defect, bool)


def test_confidence_score_range():
    from model.inference import _score_text
    # Empty / irrelevant text
    assert _score_text("") == 0.0
    assert _score_text("random unrelated text") == 0.0

    # Single date alone is enough to pass the threshold
    assert _score_text("28/09/25") >= 0.35

    # Two dates (PKD + USE BY) is strong evidence
    assert _score_text("24/10/25\n21/04/26") >= 0.60

    # Keyword + date is strong
    assert _score_text("Batch No.\n24/10/25") >= 0.50

    # Full realistic sticker text scores high
    assert _score_text("Batch No.:\n14:47\n28/09/25\n24/06/26\n44.00/(0.64)") >= 0.80

    # Score never exceeds 1.0
    assert _score_text("Batch No. PKD Use By\n28/09/25\n24/06/26\n14:47\n44.00/(0.64)") <= 1.0


# ── Logging ───────────────────────────────────────────────────────────────────

def test_log_detection_creates_csv(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))
    import importlib
    import defect_logging.logger as lg_mod
    importlib.reload(lg_mod)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    lg_mod.log_detection(frame, "DEFECT", 0.0)
    csv_path = tmp_path / "detections.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 2  # header + 1 row


# ── Alert system (no GPIO) ────────────────────────────────────────────────────

def test_alert_no_crash_without_gpio():
    from alerts.alert import AlertSystem
    a = AlertSystem()
    a.trigger(duration=0)
    a.clear()


# ── Config sanity ─────────────────────────────────────────────────────────────

def test_config_values():
    import config
    assert config.FRAME_RATE > 0
    assert isinstance(config.TESSERACT_CMD, str)
    assert isinstance(config.LOG_DIR, str)
