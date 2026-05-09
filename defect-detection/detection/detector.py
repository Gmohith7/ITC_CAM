import sys
import os
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from camera.capture import CameraCapture
from preprocessing.preprocess import draw_result
from model.inference import BatchCodeDetector
from alerts.alert import AlertSystem
from defect_logging.logger import log_detection


def run(headless: bool = False):
    camera = CameraCapture()
    detector = BatchCodeDetector()
    alerts = AlertSystem()

    print("[Detector] Starting. Press 'q' or Ctrl+C to stop.")

    # Display every frame; run detection at FRAME_RATE
    detect_interval = 1.0 / config.FRAME_RATE
    last_detect = 0.0
    last_label, last_is_defect, last_regions, last_text = "...", False, [], ""

    try:
        while True:
            frame = camera.get_frame()
            now = time.time()

            # Run OCR detection at the configured rate (not every display frame)
            if now - last_detect >= detect_interval:
                last_label, _, last_is_defect, last_regions, last_text = detector.predict(frame)
                last_detect = now

                if last_is_defect:
                    alerts.trigger()
                    log_detection(frame, last_label, 1.0)
                else:
                    alerts.clear()

            annotated = draw_result(
                frame.copy(),
                last_label, 1.0, last_is_defect,
                regions=last_regions,
                ocr_text=last_text,
            )

            if not headless:
                # Convert RGB→BGR only for cv2.imshow
                bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.imshow("Batch Code Detector", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("[Detector] Stopped.")
    finally:
        camera.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run(headless="--headless" in sys.argv)
