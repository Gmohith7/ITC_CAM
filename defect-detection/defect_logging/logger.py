import csv
import os
import sys
import cv2
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(config.LOG_DIR, "detections.csv")

if not os.path.isfile(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "label", "confidence", "image_path"])


def log_detection(frame: np.ndarray, label: str, confidence: float):
    """Save a JPEG snapshot and append a row to detections.csv."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = os.path.join(config.LOG_DIR, f"{timestamp}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([timestamp, label, f"{confidence:.4f}", img_path])

    print(f"[Log] {label} ({confidence:.2f}) logged → {img_path}")
