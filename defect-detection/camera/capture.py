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
        # Background capture thread for higher FPS: keeps the latest frame available.
        self._frame_lock = threading.Lock()
        self._last_frame = None
        self._stop_event = threading.Event()
        self._capture_thread = threading.Thread(target=self._frame_thread, daemon=True)
        self._capture_thread.start()

    def _init_picamera(self):
        self._ensure_system_dist_packages()
        # Prefer rpicam (if installed) as a lightweight Pi camera backend
        try:
            import rpicam
            cam = None
            # Try common constructor patterns
            if hasattr(rpicam, "Camera"):
                cam = rpicam.Camera()
            elif hasattr(rpicam, "RpiCam"):
                cam = rpicam.RpiCam()
            elif hasattr(rpicam, "open"):
                cam = rpicam.open()

            if cam is not None:
                self._mode = "rpicam"
                self.cam = cam
                print("[Camera] rpicam initialised.")
                return
        except Exception:
            # rpicam not available or failed to init; fall through to picamera2
            pass

        # Fallback to picamera2 (default for Raspberry Pi Camera Module)
        from picamera2 import Picamera2

        self._mode = "picamera2"
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
        )
        self.cam.configure(cfg)
        self.cam.start()
        print("[Camera] picamera2 initialised.")

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

        candidate_indices = [config.DEV_CAMERA_INDEX]
        candidate_indices.extend(index for index in range(10) if index != config.DEV_CAMERA_INDEX)

        for index in candidate_indices:
            capture = cv2.VideoCapture(index)
            if not capture.isOpened():
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            ret, _ = capture.read()
            if ret:
                self.cam = capture
                self._camera_index = index
                print(f"[Camera] Webcam {index} initialised (real-time source).")
                return

            capture.release()

        raise RuntimeError(f"Cannot open any webcam index (tried {candidate_indices})")

    def get_frame(self) -> np.ndarray:
        """Return a single frame as (H, W, 3) RGB numpy array."""
        # Return the most recent background-captured frame. Wait briefly if none available.
        wait_start = time.time()
        while True:
            with self._frame_lock:
                if self._last_frame is not None:
                    return self._last_frame.copy()
            if time.time() - wait_start > 1.0:
                raise RuntimeError("[Camera] No frame available (timeout)")
            time.sleep(0.005)

    def _frame_thread(self):
        """Continuously capture frames in a background thread and keep the latest one."""
        while not self._stop_event.is_set():
            try:
                frame = None
                if self._mode == "picamera2":
                    frame = self.cam.capture_array()
                elif self._mode == "rpicam":
                    if hasattr(self.cam, "capture_array"):
                        frame = self.cam.capture_array()
                    elif hasattr(self.cam, "capture"):
                        frame = self.cam.capture()
                    elif hasattr(self.cam, "read"):
                        ret, f = self.cam.read()
                        if ret:
                            frame = f
                else:  # webcam
                    ret, f = self.cam.read()
                    if ret:
                        frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

                # Enforce RGB color scheme for non-picamera2 backends
                if frame is not None and self._mode != "picamera2":
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    except Exception:
                        # If conversion fails, assume frame is already RGB
                        pass

                if frame is not None:
                    with self._frame_lock:
                        self._last_frame = frame
            except Exception as e:
                print(f"[Camera] frame thread error: {e}")
            time.sleep(0.001)

    def release(self):
        # Stop background thread first
        try:
            self._stop_event.set()
            if hasattr(self, "_capture_thread") and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=1.0)
        except Exception:
            pass

        if self._mode == "picamera2":
            self.cam.stop()
        elif self._mode == "rpicam":
            if hasattr(self.cam, "close"):
                try:
                    self.cam.close()
                except Exception:
                    pass
            elif hasattr(self.cam, "stop"):
                try:
                    self.cam.stop()
                except Exception:
                    pass
        elif self._mode == "webcam" and self.cam is not None:
            self.cam.release()
        print("[Camera] Released.")
