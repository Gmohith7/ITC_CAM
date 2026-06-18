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

# --- Camera exposure (optional; only applied when set) ---
# For dim stations where auto-exposure leaves the frame too dark for OCR
# (target frame brightness ~50-120). Leave empty to keep full auto-exposure.
# Manual shutter time in microseconds (e.g. 20000 = 1/50 s). Empty = auto.
EXPOSURE_TIME = os.getenv("EXPOSURE_TIME", "")
# Analogue gain, ISO-like (e.g. 2.0, 4.0). Higher = brighter but noisier. Empty = auto.
ANALOGUE_GAIN = os.getenv("ANALOGUE_GAIN", "")
# ISP brightness shift, -1.0 .. 1.0 (0 = default). Empty = default.
BRIGHTNESS = os.getenv("BRIGHTNESS", "")

# --- OCR engine ---
#   "tesseract" (default; fast, needs clean binary)
#   "rapidocr"  (PP-OCR on onnxruntime; accurate AND reliable on Pi/ARM)
#               install: pip install rapidocr_onnxruntime
#   "paddle"    (PaddleOCR; accurate but paddlepaddle native inference can
#               segfault on Pi 5 / ARM / Python 3.13) install: pip install paddlepaddle paddleocr
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").lower()
# Neural OCR (paddle/rapidocr): downscale the frame so its longest side is at
# most this many px before inference. Fewer/smaller text boxes = much faster
# (lower lag), and the large batch-code digits stay readable. 0 = no downscale.
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "960"))

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

# --- Live performance (bound OCR work per frame so the OK/DEFECT indicator
#     updates continuously instead of freezing for tens of seconds) ---
# Second 1280px full-frame OCR pass. Off by default for low latency — the
# full-res pass already reads a product that fills the frame. Enable only if
# the code sits far/small in the frame.
OCR_MULTISCALE = os.getenv("OCR_MULTISCALE", "false").lower() == "true"
# Stage 2 region-crop fallback (for white-sticker packaging). Direct-print
# products are fully handled by Stage 1, so turning this OFF makes the negative
# (no-code) path much faster. Left on by default for sticker products, but
# hard-capped (see _STAGE2_MAX_REGIONS) so it can never blow up latency.
STAGE2_ENABLED = os.getenv("STAGE2_ENABLED", "true").lower() == "true"

# --- Debug: print raw Tesseract output to console ---
OCR_DEBUG = os.getenv("OCR_DEBUG", "false").lower() == "true"
# Focus quality: frames whose Laplacian variance is below this are flagged
# FOCUS:SOFT in the per-frame debug summary. Informational only — does not
# change detection. A sharp text frame is typically > 100; blurred < 50.
FOCUS_MIN_SHARPNESS = float(os.getenv("FOCUS_MIN_SHARPNESS", "60.0"))
# Debug: also dump the raw frame + binarised OCR variants to DEBUG_DIR so the
# images Tesseract actually sees can be inspected (verify focus/framing/threshold).
OCR_DEBUG_IMAGES = os.getenv("OCR_DEBUG_IMAGES", "false").lower() == "true"
# When OCR_DEBUG_IMAGES is on, dump one image set every N processed frames.
DEBUG_IMAGE_EVERY = int(os.getenv("DEBUG_IMAGE_EVERY", "15"))
DEBUG_DIR = os.getenv(
    "DEBUG_DIR", os.path.join(os.path.dirname(__file__), "data", "debug")
)
