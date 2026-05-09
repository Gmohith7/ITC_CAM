import sys
import os
import time
import cv2

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from camera.capture import CameraCapture
from preprocessing.preprocess import preprocess_frame, draw_result
from model.inference import DefectInference
from alerts.alert import AlertSystem
from defect_logging.logger import log_detection


def run(headless: bool = False):
    camera = CameraCapture()
    model = DefectInference()
    alerts = AlertSystem()

    print("[Detector] Starting detection loop. Press Ctrl+C (or 'q') to stop.")
    frame_interval = 1.0 / config.FRAME_RATE

    try:
        while True:
            t0 = time.time()

            frame = camera.get_frame()
            tensor = preprocess_frame(frame)
            label, confidence, is_defect = model.predict(tensor)
            annotated = draw_result(frame.copy(), label, confidence, is_defect)

            if is_defect:
                alerts.trigger()
                log_detection(frame, label, confidence)
            else:
                alerts.clear()

            if not headless:
                bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.imshow("Defect Detection", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed = time.time() - t0
            time.sleep(max(0.0, frame_interval - elapsed))

    except KeyboardInterrupt:
        print("[Detector] Stopped by user.")
    finally:
        camera.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    headless = "--headless" in sys.argv
    run(headless=headless)
