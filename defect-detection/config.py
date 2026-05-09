import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- Camera ---
CAMERA_RESOLUTION = (1920, 1080)
INFERENCE_SIZE = (224, 224)
FRAME_RATE = int(os.getenv("FRAME_RATE", "10"))

# --- Model ---
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model", "model.tflite")
)
LABELS_PATH = os.getenv(
    "LABELS_PATH",
    os.path.join(os.path.dirname(__file__), "model", "labels.txt")
)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

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
