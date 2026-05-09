# Defect Detection System — Raspberry Pi 5 + Camera Module 3

## Project Overview

An end-to-end industrial defect detection system built on a Raspberry Pi 5 using the Raspberry Pi Camera Module 3. The system captures live frames from a conveyor or inspection surface, preprocesses them, runs a lightweight deep learning inference model, classifies defects in real time, and triggers appropriate alerts or logs. The entire pipeline — from capture to decision — runs locally on the Raspberry Pi 5 with no cloud dependency.

---

## Hardware

| Component | Specification |
|---|---|
| SBC | Raspberry Pi 5 (8GB RAM recommended) |
| Camera | Raspberry Pi Camera Module 3 (12MP, Sony IMX708, autofocus) |
| Storage | MicroSD 32GB+ (Class 10 / A2) or NVMe via PCIe HAT |
| Power | Official Pi 5 27W USB-C PSU |
| Alerts (optional) | LED / Buzzer via GPIO |
| Display (optional) | HDMI monitor or VNC remote |

---

## Software Stack

| Layer | Tool |
|---|---|
| OS | Raspberry Pi OS 64-bit (Bookworm) |
| Camera Interface | `picamera2` |
| Image Processing | `opencv-python` |
| ML Inference | `tflite-runtime` or `onnxruntime` |
| Model Training (offline) | TensorFlow / PyTorch (on PC/Colab) |
| Data Handling | `numpy`, `Pillow` |
| Logging | Python `logging`, CSV |
| Dashboard (optional) | `flask` |
| GPIO Alerts | `RPi.GPIO` or `gpiozero` |

---

## Project Structure

```
defect-detection/
├── README.md
├── requirements.txt
├── config.py                  # All tunable parameters
├── setup/
│   └── install.sh             # One-shot dependency installer
├── camera/
│   └── capture.py             # Camera initialisation and frame capture
├── preprocessing/
│   └── preprocess.py          # Resize, normalize, convert frames
├── model/
│   ├── model.tflite           # Exported TFLite model
│   ├── labels.txt             # Class labels (e.g. OK, DEFECT)
│   └── inference.py           # Load model, run inference
├── detection/
│   └── detector.py            # Main detection loop — ties capture + inference
├── alerts/
│   └── alert.py               # GPIO buzzer/LED + log trigger
├── logging/
│   └── logger.py              # CSV + image snapshot logging
├── dashboard/
│   └── app.py                 # Optional Flask web dashboard
├── data/
│   ├── raw/                   # Raw captured images
│   ├── annotated/             # Labelled training data
│   └── results/               # Logged defect snapshots
├── training/
│   ├── train.py               # Model training script (run on PC)
│   ├── export_tflite.py       # Convert trained model to TFLite
│   └── dataset_prep.py        # Dataset organisation and augmentation
└── tests/
    └── test_pipeline.py       # Unit tests for each module
```

---

## Environment Setup

### 1. Flash OS
- Download **Raspberry Pi OS 64-bit (Bookworm)** from raspberrypi.com
- Flash using Raspberry Pi Imager
- Enable SSH and set hostname/credentials in Imager advanced options

### 2. Enable Camera
```bash
sudo raspi-config
# Interface Options → Camera → Enable
sudo reboot
```

Verify camera is detected:
```bash
libcamera-hello --list-cameras
```

### 3. Install Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv libatlas-base-dev \
    python3-picamera2 libjpeg-dev libcamera-dev

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### `requirements.txt`
```
numpy
opencv-python
picamera2
tflite-runtime
Pillow
flask
gpiozero
RPi.GPIO
```

> **Note:** `tflite-runtime` wheel for Pi 5 (aarch64): use the official Google or `piwheels` build.

---

## Module Implementations

### `config.py`
```python
# config.py — Central configuration

CAMERA_RESOLUTION = (1920, 1080)   # capture resolution
INFERENCE_SIZE = (224, 224)         # model input size
MODEL_PATH = "model/model.tflite"
LABELS_PATH = "model/labels.txt"
CONFIDENCE_THRESHOLD = 0.75         # minimum confidence to flag defect
FRAME_RATE = 10                     # frames per second to process
LOG_DIR = "data/results"
GPIO_BUZZER_PIN = 17
GPIO_LED_PIN = 27
FLASK_PORT = 5000
```

---

### `camera/capture.py`
```python
# camera/capture.py — Camera initialisation and frame capture

from picamera2 import Picamera2
import cv2
import config

class CameraCapture:
    def __init__(self):
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
        )
        self.cam.configure(cfg)
        self.cam.start()
        print("[Camera] Initialised.")

    def get_frame(self):
        """Capture a single frame as a numpy array (H, W, 3) RGB."""
        frame = self.cam.capture_array()
        return frame

    def release(self):
        self.cam.stop()
        print("[Camera] Released.")
```

---

### `preprocessing/preprocess.py`
```python
# preprocessing/preprocess.py — Frame preprocessing for inference

import cv2
import numpy as np
import config

def preprocess_frame(frame):
    """
    Resize frame to model input size, normalise pixel values.
    Returns: numpy array of shape (1, H, W, 3) float32
    """
    resized = cv2.resize(frame, config.INFERENCE_SIZE)
    normalised = resized.astype(np.float32) / 255.0
    batched = np.expand_dims(normalised, axis=0)
    return batched

def draw_result(frame, label, confidence, defect=False):
    """Overlay prediction result on the raw frame."""
    color = (0, 0, 255) if defect else (0, 255, 0)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    return frame
```

---

### `model/inference.py`
```python
# model/inference.py — TFLite model loader and inference runner

import numpy as np
import tflite_runtime.interpreter as tflite
import config

class DefectInference:
    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path=config.MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(config.LABELS_PATH, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        print(f"[Model] Loaded. Labels: {self.labels}")

    def predict(self, input_tensor):
        """
        Run inference on preprocessed tensor.
        Returns: (label: str, confidence: float, is_defect: bool)
        """
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        class_idx = int(np.argmax(output))
        confidence = float(output[class_idx])
        label = self.labels[class_idx]
        is_defect = (label.upper() == "DEFECT") and (confidence >= config.CONFIDENCE_THRESHOLD)

        return label, confidence, is_defect
```

---

### `alerts/alert.py`
```python
# alerts/alert.py — GPIO alerts for defect detection

from gpiozero import LED, Buzzer
import config

class AlertSystem:
    def __init__(self):
        self.led = LED(config.GPIO_LED_PIN)
        self.buzzer = Buzzer(config.GPIO_BUZZER_PIN)

    def trigger(self):
        """Activate LED and buzzer for 1 second on defect."""
        self.led.on()
        self.buzzer.on()
        import time; time.sleep(1)
        self.led.off()
        self.buzzer.off()

    def clear(self):
        self.led.off()
        self.buzzer.off()
```

---

### `logging/logger.py`
```python
# logging/logger.py — Defect logging to CSV and image snapshots

import csv
import os
import cv2
from datetime import datetime
import config

os.makedirs(config.LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(config.LOG_DIR, "detections.csv")

def log_detection(frame, label, confidence):
    """Save frame snapshot and append row to CSV log."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = os.path.join(config.LOG_DIR, f"{timestamp}.jpg")
    cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, label, f"{confidence:.4f}", img_path])

    print(f"[Log] Defect logged at {timestamp} — {label} ({confidence:.2f})")
```

---

### `detection/detector.py` — Main Loop
```python
# detection/detector.py — Main detection loop

import time
import config
from camera.capture import CameraCapture
from preprocessing.preprocess import preprocess_frame, draw_result
from model.inference import DefectInference
from alerts.alert import AlertSystem
from logging.logger import log_detection

def run():
    camera = CameraCapture()
    model = DefectInference()
    alerts = AlertSystem()

    print("[Detector] Starting detection loop. Press Ctrl+C to stop.")
    frame_interval = 1.0 / config.FRAME_RATE

    try:
        while True:
            start = time.time()

            # 1. Capture
            frame = camera.get_frame()

            # 2. Preprocess
            tensor = preprocess_frame(frame)

            # 3. Infer
            label, confidence, is_defect = model.predict(tensor)

            # 4. Annotate frame
            annotated = draw_result(frame.copy(), label, confidence, is_defect)

            # 5. React
            if is_defect:
                alerts.trigger()
                log_detection(frame, label, confidence)
            else:
                alerts.clear()

            # 6. Display (optional — remove if headless)
            import cv2
            cv2.imshow("Defect Detection", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.time() - start
            time.sleep(max(0, frame_interval - elapsed))

    except KeyboardInterrupt:
        print("[Detector] Stopped.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
```

---

## Model Training (Done Offline on PC / Google Colab)

### `training/train.py`
```python
# training/train.py — Train MobileNetV2 classifier (run on PC)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "data/annotated"   # subfolders: OK/, DEFECT/

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)
val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

# Transfer learning base
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
model.save("trained_model.h5")
print("[Train] Model saved.")
```

### `training/export_tflite.py`
```python
# training/export_tflite.py — Convert H5 model to TFLite

import tensorflow as tf

model = tf.keras.models.load_model("trained_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # quantize for speed
tflite_model = converter.convert()

with open("model/model.tflite", "wb") as f:
    f.write(tflite_model)

print("[Export] TFLite model saved to model/model.tflite")
```

---

## Optional: Flask Dashboard

### `dashboard/app.py`
```python
# dashboard/app.py — Live web dashboard

from flask import Flask, render_template, Response
from camera.capture import CameraCapture
from preprocessing.preprocess import preprocess_frame, draw_result
from model.inference import DefectInference
import cv2
import config

app = Flask(__name__)
camera = CameraCapture()
model = DefectInference()

def generate_frames():
    while True:
        frame = camera.get_frame()
        tensor = preprocess_frame(frame)
        label, confidence, is_defect = model.predict(tensor)
        annotated = draw_result(frame.copy(), label, confidence, is_defect)
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return "<h2>Defect Detection Live Feed</h2><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False)
```

---

## Running the System

```bash
# Activate environment
source venv/bin/activate

# Run headless detection
python detection/detector.py

# OR run dashboard (access via browser at http://<pi-ip>:5000)
python dashboard/app.py
```

---

## Dataset Collection Guide

1. Mount Pi Camera over the inspection surface
2. Run a capture script to collect ~500+ images each of `OK` and `DEFECT` samples
3. Organise into `data/annotated/OK/` and `data/annotated/DEFECT/`
4. Transfer to PC/Colab for training
5. Export TFLite model back to Pi

```bash
# Quick capture helper
python -c "
from camera.capture import CameraCapture
import cv2, os, time
cam = CameraCapture()
os.makedirs('data/raw', exist_ok=True)
for i in range(200):
    f = cam.get_frame()
    cv2.imwrite(f'data/raw/frame_{i:04d}.jpg', cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    time.sleep(0.5)
cam.release()
print('Done')
"
```

---

## Performance Expectations (Pi 5)

| Model | Inference Time | FPS (approx) |
|---|---|---|
| MobileNetV2 (float32) | ~120ms | ~8 fps |
| MobileNetV2 (quantized int8) | ~60ms | ~15 fps |
| EfficientNet-Lite0 (quantized) | ~45ms | ~20 fps |

Pi 5's improved CPU over Pi 4 gives roughly **1.5–2x** faster inference without any accelerator.

---

## Future Enhancements

- Add **Coral USB Accelerator** for 5–10x inference speedup
- Integrate **MQTT** to push alerts to a central factory dashboard
- Use **YOLO-nano** for multi-defect bounding box detection
- Add **SQLite** backend for structured defect logging
- Implement **OTA model updates** over local network

---

## Notes for Claude Code

- All modules are independently importable and testable
- `config.py` is the single source of truth — adjust resolution, thresholds, GPIO pins here
- The system is designed to run headless (no display) in production; remove `cv2.imshow()` calls in `detector.py`
- Camera Module 3 uses `libcamera` backend — ensure `picamera2` version is ≥ 0.3.12
- Run `libcamera-hello` to verify camera before starting the pipeline
- GPIO alerts are optional — system runs fine without them if pins are not configured
