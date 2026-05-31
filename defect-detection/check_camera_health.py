"""
Camera health check (Pi camera only).

Usage:
    python check_camera_health.py

Exit codes:
    0 = healthy
    1 = failed
"""

import os
import sys

import config


def _check_picamera2():
    """Return (ok, message)."""
    try:
        # Ensure system dist-packages are visible (Pi installs picamera2 via apt).
        for path in (
            "/usr/lib/python3/dist-packages",
            "/usr/lib/python3.11/dist-packages",
        ):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        try:
            from picamera2 import Picamera2
        except Exception as exc:
            return False, (
                "picamera2 import failed. Install it with: sudo apt install python3-picamera2"
                f" (error: {exc})"
            )

        cam = Picamera2()
        cfg = cam.create_video_configuration(
            main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"},
            controls={"FrameRate": config.FRAME_RATE},
        )
        cam.configure(cfg)
        cam.start()
        frame = cam.capture_array("main")
        cam.stop()
        if frame is None or getattr(frame, "size", 0) == 0:
            return False, "picamera2 returned an empty frame"
        return True, "picamera2 OK"
    except IndexError as exc:
        return False, f"no camera detected by picamera2: {exc}"
    except Exception as exc:
        return False, f"picamera2 failed: {exc}"


def main():
    print("[CameraCheck] Starting camera health check...")
    print("[CameraCheck] This test validates only the Raspberry Pi Camera.")

    ok, message = _check_picamera2()
    print(f"[CameraCheck] {message}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
