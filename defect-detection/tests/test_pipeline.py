"""
Unit tests for the defect detection pipeline.
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Preprocessing ────────────────────────────────────────────────────────────

def test_preprocess_output_shape():
    from preprocessing.preprocess import preprocess_frame
    import config
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_frame(dummy)
    h, w = config.INFERENCE_SIZE
    assert result.shape == (1, h, w, 3), f"Unexpected shape: {result.shape}"


def test_preprocess_normalisation_range():
    from preprocessing.preprocess import preprocess_frame
    dummy = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = preprocess_frame(dummy)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_draw_result_does_not_crash():
    from preprocessing.preprocess import draw_result
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_result(frame.copy(), "OK", 0.95, defect=False)
    assert out.shape == frame.shape

    out_defect = draw_result(frame.copy(), "DEFECT", 0.88, defect=True)
    assert out_defect.shape == frame.shape


# ── Inference (dummy mode) ────────────────────────────────────────────────────

def test_inference_dummy_returns_valid_output():
    import config
    config.MODEL_PATH = "nonexistent_model.tflite"
    from model.inference import DefectInference
    inf = DefectInference()
    dummy_tensor = np.random.rand(1, 224, 224, 3).astype(np.float32)
    label, confidence, is_defect = inf.predict(dummy_tensor)
    assert label in inf.labels
    assert 0.0 <= confidence <= 1.0
    assert isinstance(is_defect, bool)


# ── Logging ───────────────────────────────────────────────────────────────────

def test_log_detection_creates_csv(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path))

    # Re-import after patching so module-level init uses tmp_path
    import importlib
    import defect_logging.logger as lg_mod
    importlib.reload(lg_mod)

    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    lg_mod.log_detection(frame, "DEFECT", 0.92)

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
    assert 0.0 < config.CONFIDENCE_THRESHOLD <= 1.0
    assert len(config.INFERENCE_SIZE) == 2
