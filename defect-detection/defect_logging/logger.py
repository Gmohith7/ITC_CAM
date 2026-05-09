import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(config.LOG_DIR, "detections.csv")

if not os.path.isfile(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "label", "confidence"])


def log_detection(label: str, confidence: float):
    """Append a detection event row to detections.csv. No image capture."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([timestamp, label, f"{confidence:.4f}"])
