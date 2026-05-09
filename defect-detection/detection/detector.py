import sys
import os
import time
import threading
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from camera.capture import CameraCapture
from preprocessing.preprocess import draw_result
from model.inference import BatchCodeDetector
from alerts.alert import AlertSystem


class _DetectionState:
    """Shared state between the display loop and the OCR worker thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self.label = "SCANNING"
        self.confidence = 0.0
        self.is_defect = False
        self.scanning = True   # True until first OCR result arrives
        self.regions = []
        self.text = ""

    def update(self, label, confidence, is_defect, regions, text):
        with self._lock:
            self.label = label
            self.confidence = confidence
            self.is_defect = is_defect
            self.scanning = False
            self.regions = regions
            self.text = text

    def snapshot(self):
        with self._lock:
            return (self.label, self.confidence, self.is_defect,
                    self.scanning, self.regions[:], self.text)


def _ocr_worker(camera: CameraCapture, detector: BatchCodeDetector,
                alerts: AlertSystem, state: _DetectionState,
                stop_event: threading.Event):
    """
    OCR runs in its own thread — never blocks the display loop.
    Uses get_frame_ref() (zero copy) since it only reads the frame.
    """
    while not stop_event.is_set():
        try:
            frame = camera.get_frame_ref()
            if frame is None:
                time.sleep(0.02)
                continue

            # Skip frames that are too dark (camera still stabilising)
            if np.mean(frame) < 8:
                time.sleep(0.05)
                continue

            # Work on a copy so the capture thread can replace _last_frame freely
            frame_copy = frame.copy()

            label, confidence, is_defect, regions, text = detector.predict(frame_copy)
            state.update(label, confidence, is_defect, regions, text)

            if is_defect:
                alerts.trigger()
            else:
                alerts.clear()

        except Exception as e:
            print(f"[OCR worker] {e}")
            time.sleep(0.1)


def run(headless: bool = False):
    camera = CameraCapture()
    detector = BatchCodeDetector()
    alerts = AlertSystem()
    state = _DetectionState()
    stop_event = threading.Event()

    ocr_thread = threading.Thread(
        target=_ocr_worker,
        args=(camera, detector, alerts, state, stop_event),
        daemon=True,
    )
    ocr_thread.start()

    print("[Detector] Starting. Press 'q' or Ctrl+C to stop.")

    try:
        while True:
            # One copy per display frame — used for drawing
            frame = camera.get_frame()
            label, confidence, is_defect, scanning, regions, text = state.snapshot()

            annotated = draw_result(
                frame,          # draw_result mutates in-place; frame is already a copy
                label, confidence, is_defect,
                scanning=scanning,
                regions=regions,
                ocr_text=text,
            )

            if not headless:
                # Pipeline is RGB; cv2.imshow needs BGR
                bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.imshow("Batch Code Detector", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("[Detector] Stopped.")
    finally:
        stop_event.set()
        ocr_thread.join(timeout=3.0)
        camera.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run(headless="--headless" in sys.argv)
