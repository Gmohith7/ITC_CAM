import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- Camera ---
CAMERA_RESOLUTION = (1920, 1080)
FRAME_RATE = int(os.getenv("FRAME_RATE", "30"))
GRAYSCALE_MODE = os.getenv("GRAYSCALE_MODE", "false").lower() == "true"

# --- Autofocus (Camera Module 3 / autofocus lenses only; ignored on fixed-focus sensors) ---
# "continuous" = lens keeps refocusing automatically on whatever is in view (default).
# "manual"     = lock the lens at a fixed distance (best for a fixed-distance station).
AF_MODE = os.getenv("AF_MODE", "continuous").lower()
# Manual focus distance, used ONLY when AF_MODE=manual. LensPosition = 1 / distance_in_metres.
# e.g. 20 cm -> 5.0, 25 cm -> 4.0, 10 cm -> 10.0, 0.0 = infinity.
LENS_POSITION = float(os.getenv("LENS_POSITION", "5.0"))
# Continuous-AF convergence speed: "fast" locks quicker, "normal" is smoother.
AF_SPEED = os.getenv("AF_SPEED", "fast").lower()
# AF search range: "normal" (typical distances), "macro" (close-ups only), "full" (macro->infinity).
AF_RANGE = os.getenv("AF_RANGE", "macro").lower()

# --- Tesseract OCR ---
# On Linux/Pi, tesseract is found via PATH automatically.
# On Windows, set TESSERACT_CMD in .env if not in a standard location.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# --- Detection tuning ---
# Minimum OCR region height (pixels); smaller regions are upscaled before OCR.
OCR_MIN_HEIGHT = int(os.getenv("OCR_MIN_HEIGHT", "140"))
# White-label brightness threshold (0-255); pixels above this are treated as "white sticker".
WHITE_THRESHOLD = int(os.getenv("WHITE_THRESHOLD", "185"))
# Morphological kernel size for sticker region detection (width, height in pixels).
MORPH_KERNEL_W = int(os.getenv("MORPH_KERNEL_W", "28"))
MORPH_KERNEL_H = int(os.getenv("MORPH_KERNEL_H", "14"))
# Padding added around each detected sticker region before OCR (pixels).
REGION_PADDING = int(os.getenv("REGION_PADDING", "14"))
# Confidence threshold to declare a batch code present.
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.55"))
# Minimum average frame brightness before OCR is attempted (0-255).
# Skips frames while the camera is still stabilising / lens cap on.
DARK_FRAME_THRESHOLD = float(os.getenv("DARK_FRAME_THRESHOLD", "8.0"))
# Max seconds between OCR passes in the worker thread (rate-limits CPU use).
OCR_INTERVAL_S = float(os.getenv("OCR_INTERVAL_S", "0.0"))

# --- Logging ---
LOG_DIR = os.getenv(
    "LOG_DIR",
    os.path.join(os.path.dirname(__file__), "data", "results")
)
# Save a JPEG snapshot of each defect frame alongside the CSV row.
LOG_SNAPSHOTS = os.getenv("LOG_SNAPSHOTS", "true").lower() == "true"

# --- GPIO (Raspberry Pi only) ---
GPIO_BUZZER_PIN = int(os.getenv("GPIO_BUZZER_PIN", "17"))
GPIO_LED_PIN = int(os.getenv("GPIO_LED_PIN", "27"))
# Duration of the alert trigger in seconds.
ALERT_DURATION_S = float(os.getenv("ALERT_DURATION_S", "1.0"))

# --- Dashboard ---
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")

# --- Dev mode: uses webcam instead of picamera2 ---
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
DEV_CAMERA_INDEX = int(os.getenv("DEV_CAMERA_INDEX", "0"))

# --- Debug: print raw Tesseract output to console ---
OCR_DEBUG = os.getenv("OCR_DEBUG", "false").lower() == "true"
