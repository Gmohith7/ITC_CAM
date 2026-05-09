import sys
import cv2
import numpy as np
import config


class CameraCapture:
    """Abstracts picamera2 on Pi and OpenCV webcam in DEV_MODE."""

    def __init__(self):
        if config.DEV_MODE:
            self._init_webcam()
        else:
            self._init_picamera()

    def _init_picamera(self):
        try:
            from picamera2 import Picamera2
            self._mode = "picamera2"
            self.cam = Picamera2()
            cfg = self.cam.create_preview_configuration(
                main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
            )
            self.cam.configure(cfg)
            self.cam.start()
            print("[Camera] picamera2 initialised.")
        except Exception as e:
            print(f"[Camera] picamera2 unavailable ({e}). Falling back to webcam.")
            self._init_webcam()

    def _init_webcam(self):
        self._mode = "webcam"
        self.cam = cv2.VideoCapture(config.DEV_CAMERA_INDEX)
        if not self.cam.isOpened():
            raise RuntimeError(f"Cannot open webcam index {config.DEV_CAMERA_INDEX}")
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
        print(f"[Camera] Webcam {config.DEV_CAMERA_INDEX} initialised (DEV_MODE).")

    def get_frame(self) -> np.ndarray:
        """Return a single frame as (H, W, 3) RGB numpy array."""
        if self._mode == "picamera2":
            return self.cam.capture_array()
        else:
            ret, frame_bgr = self.cam.read()
            if not ret:
                raise RuntimeError("[Camera] Failed to read frame from webcam.")
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def release(self):
        if self._mode == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()
        print("[Camera] Released.")
