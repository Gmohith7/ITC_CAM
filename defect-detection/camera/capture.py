import os
import sys
import cv2
import numpy as np
import config
import threading
import time


class CameraCapture:
    """Abstracts picamera2 on Pi and OpenCV webcam in DEV_MODE."""

    def __init__(self):
        try:
            self._init_picamera()
        except Exception as e:
            if config.DEV_MODE:
                print(f"[Camera] picamera2 unavailable ({e}). Falling back to webcam.")
                self._init_webcam()
            else:
                raise

        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._frame_thread, daemon=True)
        self._capture_thread.start()

        # Wait for first frame before returning
        deadline = time.time() + 3.0
        while time.time() < deadline:
            with self._frame_lock:
                if self._last_frame is not None:
                    break
            time.sleep(0.01)

    def _init_picamera(self):
        self._ensure_system_dist_packages()
        from picamera2 import Picamera2
        self._mode = "picamera2"
        self.cam = Picamera2()
        # Request BGR to align with OpenCV, then convert as needed for the pipeline.
        cfg = self.cam.create_video_configuration(
            main={"size": config.CAMERA_RESOLUTION, "format": "BGR888"},
            controls={"FrameRate": config.FRAME_RATE}
        )
        self.cam.configure(cfg)
        self.cam.start()
        print(f"[Camera] picamera2 initialised at {config.FRAME_RATE} fps.")

    def _ensure_system_dist_packages(self):
        for path in (
            "/usr/lib/python3/dist-packages",
            "/usr/lib/python3.11/dist-packages",
        ):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    def _init_webcam(self):
        self._mode = "webcam"
        self.cam = None
        candidates = [config.DEV_CAMERA_INDEX] + [i for i in range(5) if i != config.DEV_CAMERA_INDEX]
        for idx in candidates:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            cap.set(cv2.CAP_PROP_FPS, config.FRAME_RATE)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise buffer lag
            ret, _ = cap.read()
            if ret:
                self.cam = cap
                print(f"[Camera] Webcam {idx} initialised at {config.FRAME_RATE} fps.")
                return
            cap.release()
        
        print(f"[Camera] Cannot open any webcam. Falling back to dummy video stream.")
        self._mode = "dummy"

    def _frame_thread(self):
        """Capture frames as fast as the hardware allows; keep only the latest."""
        while not self._stop_event.is_set():
            try:
                if self._mode == "picamera2":
                    # capture_array("main") skips an internal copy vs the default "main" path.
                    frame = self.cam.capture_array("main")
                    if config.GRAYSCALE_MODE:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif self._mode == "webcam":
                    ret, bgr = self.cam.read()
                    if not ret:
                        continue
                    if config.GRAYSCALE_MODE:
                        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    else:
                        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif self._mode == "dummy":
                    if config.GRAYSCALE_MODE:
                        frame = np.random.randint(0, 256, (config.CAMERA_RESOLUTION[1], config.CAMERA_RESOLUTION[0]), dtype=np.uint8)
                        cv2.putText(frame, "DUMMY MODE (NO WEBCAM)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 4)
                    else:
                        frame = np.random.randint(0, 256, (config.CAMERA_RESOLUTION[1], config.CAMERA_RESOLUTION[0], 3), dtype=np.uint8)
                        cv2.putText(frame, "DUMMY MODE (NO WEBCAM)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)
                    time.sleep(1.0 / config.FRAME_RATE)

                with self._frame_lock:
                    self._last_frame = frame
            except Exception as e:
                print(f"[Camera] frame thread error: {e}")
                time.sleep(0.05)

    def get_frame(self) -> np.ndarray:
        """Return the most recently captured frame as (H, W, 3) RGB uint8.
        Returns a copy so callers can mutate freely."""
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self._frame_lock:
                if self._last_frame is not None:
                    return self._last_frame.copy()
            time.sleep(0.002)
        raise RuntimeError("[Camera] No frame available (timeout).")

    def get_frame_ref(self) -> np.ndarray:
        """Return a direct reference to the latest frame — zero copy.
        Caller must NOT mutate it. Use only for read-only inference."""
        with self._frame_lock:
            return self._last_frame

    def release(self):
        self._stop_event.set()
        if hasattr(self, "_capture_thread") and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._mode == "picamera2":
            self.cam.stop()
        elif self._mode == "webcam" and self.cam is not None:
            self.cam.release()
        print("[Camera] Released.")
