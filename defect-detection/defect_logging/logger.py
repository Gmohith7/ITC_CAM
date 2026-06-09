import csv
import os
import sys
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(config.LOG_DIR, "detections.csv")
SNAPSHOT_DIR = os.path.join(config.LOG_DIR, "snapshots")

_csv_lock = threading.Lock()

if not os.path.isfile(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "label", "confidence", "snapshot"])


def log_detection(label: str, confidence: float, frame: Optional[np.ndarray] = None):
    """
    Append a detection event to detections.csv.

    If `frame` is provided and LOG_SNAPSHOTS is enabled, saves a JPEG alongside
    the CSV row so defects have a visual audit trail.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_path = ""

    if frame is not None and config.LOG_SNAPSHOTS:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        fname = f"{timestamp}_{label}.jpg"
        snapshot_path = os.path.join(SNAPSHOT_DIR, fname)
        _save_snapshot(frame, snapshot_path)

    with _csv_lock:
        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, label, f"{confidence:.4f}", snapshot_path])


def _save_snapshot(frame: np.ndarray, path: str):
    """Convert frame to BGR and write as JPEG. Silently ignores errors."""
    try:
        if frame.ndim == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr = frame
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception as e:
        print(f"[Logger] Snapshot save failed: {e}")
