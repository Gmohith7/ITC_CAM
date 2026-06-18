import os
import sys
import traceback
import cv2
import numpy as np
import config
import threading
import time


class CameraCapture:
    """Abstracts picamera2 on Pi and OpenCV webcam in DEV_MODE."""

    def __init__(self):
        self._picam_fmt = "XRGB8888"
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

        # Wait for first frame before returning (up to 3 s)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with self._frame_lock:
                if self._last_frame is not None:
                    break
            time.sleep(0.01)

    def _init_picamera(self):
        self._ensure_system_dist_packages()
        from picamera2 import Picamera2
        self._mode = "picamera2"
        self.cam = Picamera2()
        # Pi 5 ISP always outputs 32-bit aligned pixels.
        # XRGB8888 → capture_array returns (H, W, 4) with byte layout [X, R, G, B].
        # We strip channel 0 (padding) and keep [1, 2, 3] to get clean RGB.
        self._picam_fmt = "XRGB8888"
        cfg = self.cam.create_video_configuration(
            main={"size": config.CAMERA_RESOLUTION, "format": self._picam_fmt},
            controls={"FrameRate": config.FRAME_RATE}
        )
        self.cam.configure(cfg)
        self.cam.start()
        print(f"[Camera] picamera2 initialised ({self._picam_fmt}) at {config.FRAME_RATE} fps.")
        self._apply_focus()
        self._apply_exposure()

    def _apply_focus(self):
        """Configure the Camera Module 3 autofocus lens.

        Without this, picamera2 parks the lens at its default (far) position, so a
        close-up product is permanently out of focus. Continuous AF makes the lens
        keep hunting until whatever is in view is sharp. No-op on sensors that have
        no autofocus motor (Camera Module 1/2, HQ cam).
        """
        try:
            from libcamera import controls
        except Exception as e:
            print(f"[Camera] libcamera controls unavailable ({e}); skipping focus setup.")
            return

        af_mode = getattr(config, "AF_MODE", "continuous")
        try:
            if af_mode == "manual":
                self.cam.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": config.LENS_POSITION,
                })
                dist_cm = (100.0 / config.LENS_POSITION) if config.LENS_POSITION > 0 else float("inf")
                print(f"[Camera] Manual focus locked at LensPosition={config.LENS_POSITION} (~{dist_cm:.0f} cm).")
                return

            # Continuous AF — the lens hunts automatically and never stops adjusting.
            self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            print("[Camera] Continuous autofocus enabled.")
        except Exception as e:
            # Camera Module 1/2 / HQ cam have no AF motor — fixed lens, ignore.
            print(f"[Camera] Autofocus not available on this sensor ({e}); using fixed lens.")
            return

        # Best-effort speed/range tuning (older libcamera builds may lack these enums).
        extra = {}
        try:
            extra["AfSpeed"] = {
                "fast": controls.AfSpeedEnum.Fast,
                "normal": controls.AfSpeedEnum.Normal,
            }.get(config.AF_SPEED, controls.AfSpeedEnum.Fast)
        except Exception:
            pass
        try:
            extra["AfRange"] = {
                "normal": controls.AfRangeEnum.Normal,
                "macro": controls.AfRangeEnum.Macro,
                "full": controls.AfRangeEnum.Full,
            }.get(config.AF_RANGE, controls.AfRangeEnum.Normal)
        except Exception:
            pass
        if extra:
            try:
                self.cam.set_controls(extra)
                print(f"[Camera] AF speed={config.AF_SPEED}, range={config.AF_RANGE}.")
            except Exception as e:
                print(f"[Camera] AF speed/range not applied ({e}).")

    def _apply_exposure(self):
        """
        Optional manual exposure / brightness for dim stations (opt-in via .env).

        Dark frames (measured ~37 mean vs ~64 on a readable reference) starve the
        OCR of contrast. Setting EXPOSURE_TIME and/or ANALOGUE_GAIN disables AEC
        and brightens the frame; BRIGHTNESS nudges the ISP output. All empty by
        default → full auto-exposure, behaviour unchanged.
        """
        ctrls = {}
        try:
            if str(getattr(config, "EXPOSURE_TIME", "")).strip():
                ctrls["AeEnable"] = False
                ctrls["ExposureTime"] = int(float(config.EXPOSURE_TIME))
            if str(getattr(config, "ANALOGUE_GAIN", "")).strip():
                ctrls["AeEnable"] = False
                ctrls["AnalogueGain"] = float(config.ANALOGUE_GAIN)
            if str(getattr(config, "BRIGHTNESS", "")).strip():
                ctrls["Brightness"] = float(config.BRIGHTNESS)
        except ValueError as e:
            print(f"[Camera] Bad exposure/brightness value ({e}); using auto-exposure.")
            return

        if not ctrls:
            return
        try:
            self.cam.set_controls(ctrls)
            print(f"[Camera] Manual exposure controls applied: {ctrls}")
        except Exception as e:
            print(f"[Camera] Exposure controls not applied ({e}); using auto-exposure.")

    def _ensure_system_dist_packages(self):
        """
        Make picamera2's system dist-packages importable from inside the venv.

        APPEND (not insert-at-front): picamera2/libcamera live only in the system
        dist-packages, so adding the path at the END is enough to import them —
        while keeping the venv's own packages at higher priority. Inserting at the
        front shadows venv packages with older system copies (e.g. an old
        typing_extensions that breaks PaddleOCR's import).
        """
        import glob
        candidates = [
            "/usr/lib/python3/dist-packages",
            *glob.glob("/usr/lib/python3.*/dist-packages"),
        ]
        for path in candidates:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)

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
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                self.cam = cap
                print(f"[Camera] Webcam {idx} initialised at {config.FRAME_RATE} fps.")
                return
            cap.release()
        print("[Camera] Cannot open any webcam. Falling back to dummy video stream.")
        self._mode = "dummy"

    # ── Frame conversion ──────────────────────────────────────────────────────

    def _picam_to_rgb(self, raw: np.ndarray) -> np.ndarray:
        """
        Convert whatever picamera2 gives us into a clean (H, W, 3) RGB array.

        Pi 5 / libcamera ISP always outputs 32-bit aligned pixels regardless of
        the format name requested. Observed layouts:

          XRGB8888 → (H, W, 4)  byte order per pixel: [X, R, G, B]
                     → take channels [1, 2, 3]
          XBGR8888 → (H, W, 4)  byte order: [X, B, G, R]
                     → take channels [3, 2, 1]
          BGR888   → (H, W, 3)  byte order: [B, G, R]
                     → cvtColor BGR2RGB
          RGB888   → (H, W, 3)  byte order: [R, G, B]
                     → use as-is
        """
        fmt = self._picam_fmt

        if raw.ndim == 3 and raw.shape[2] == 4:
            # Pi 5 ISP actual byte order per pixel (confirmed empirically): B G R X
            # i.e. BGRX — channels [0,1,2] are BGR, channel 3 is padding.
            # To get RGB we reverse the first three channels.
            return np.ascontiguousarray(raw[:, :, 2::-1])   # [B,G,R,X] → [R,G,B]

        if raw.ndim == 3 and raw.shape[2] == 3:
            if "BGR" in fmt:
                return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            return np.ascontiguousarray(raw)   # already RGB

        # Unexpected — best-effort fallback
        if raw.ndim == 2:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(raw[..., :3])

    # ── Capture thread ────────────────────────────────────────────────────────

    def _frame_thread(self):
        """Capture frames as fast as the hardware allows; keep only the latest."""
        while not self._stop_event.is_set():
            try:
                frame = self._capture_one()
                if frame is not None:
                    with self._frame_lock:
                        self._last_frame = frame
            except Exception:
                print(f"[Camera] frame thread error:\n{traceback.format_exc()}")
                time.sleep(0.1)

    def _capture_one(self):
        if self._mode == "picamera2":
            raw = self.cam.capture_array("main")
            rgb = self._picam_to_rgb(raw)
            if config.GRAYSCALE_MODE:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            return rgb

        if self._mode == "webcam":
            ret, bgr = self.cam.read()
            if not ret:
                return None
            if config.GRAYSCALE_MODE:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self._mode == "dummy":
            h, w = config.CAMERA_RESOLUTION[1], config.CAMERA_RESOLUTION[0]
            if config.GRAYSCALE_MODE:
                frame = np.full((h, w), 80, dtype=np.uint8)
                cv2.putText(frame, "DUMMY MODE", (w // 6, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, w / 640, 200, 2)
            else:
                frame = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
                cv2.putText(frame, "DUMMY MODE", (w // 6, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, w / 640, (200, 200, 200), 2)
            time.sleep(1.0 / config.FRAME_RATE)
            return frame

        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray:
        """Return a copy of the most recent frame (safe for mutation)."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._frame_lock:
                if self._last_frame is not None:
                    return self._last_frame.copy()
            time.sleep(0.002)
        raise RuntimeError("[Camera] No frame available (timeout).")

    def get_frame_ref(self) -> np.ndarray:
        """Return a direct reference — zero copy. Do NOT mutate."""
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
