import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- Camera ---
CAMERA_RESOLUTION = (1920, 1080)
FRAME_RATE = int(os.getenv("FRAME_RATE", "30"))

# --- Tesseract OCR ---
# On Windows, point to the Tesseract install location.
# On Linux/Pi it is found automatically via PATH.
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# --- Logging ---
LOG_DIR = os.getenv(
    "LOG_DIR",
    os.path.join(os.path.dirname(__file__), "data", "results")
)

# --- GPIO (Raspberry Pi only) ---
GPIO_BUZZER_PIN = int(os.getenv("GPIO_BUZZER_PIN", "17"))
GPIO_LED_PIN = int(os.getenv("GPIO_LED_PIN", "27"))

# --- Dashboard ---
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")

# --- Dev mode: uses webcam instead of picamera2 ---
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
DEV_CAMERA_INDEX = int(os.getenv("DEV_CAMERA_INDEX", "0"))
