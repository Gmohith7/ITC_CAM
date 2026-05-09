# Defect Detection System

Real-time industrial defect detection running entirely on a **Raspberry Pi 5** with the **Camera Module 3**. Frames are captured, preprocessed, and classified by a quantized MobileNetV2 TFLite model — no cloud dependency.

---

## Hardware

| Component | Spec |
|---|---|
| SBC | Raspberry Pi 5 (8 GB recommended) |
| Camera | Raspberry Pi Camera Module 3 (12 MP, Sony IMX708, autofocus) |
| Storage | MicroSD 32 GB+ (Class 10 / A2) or NVMe via PCIe HAT |
| Power | Official Pi 5 27 W USB-C PSU |
| Alerts (optional) | LED + buzzer via GPIO |

---

## Project Structure

```
defect-detection/
├── config.py                  # All tunable parameters (reads from .env)
├── run.py                     # Convenience launcher
├── requirements.txt           # Core dependencies
├── requirements-dev.txt       # Dev/test extras
├── requirements-pi.txt        # Pi 5 runtime instructions
├── .env                       # Local config (gitignored)
├── .env.example               # Template for .env
├── setup/
│   └── install.sh             # One-shot Pi 5 setup script
├── camera/
│   └── capture.py             # picamera2 wrapper (webcam fallback in DEV_MODE)
├── preprocessing/
│   └── preprocess.py          # Resize, normalise, annotate frames
├── model/
│   ├── inference.py           # TFLite model loader and runner
│   └── labels.txt             # Class labels: OK, DEFECT
├── detection/
│   └── detector.py            # Main detection loop
├── alerts/
│   └── alert.py               # GPIO LED + buzzer (lgpio / RP1 backend)
├── defect_logging/
│   └── logger.py              # CSV log + JPEG snapshot per defect
├── dashboard/
│   └── app.py                 # Flask live-feed web dashboard
├── data/
│   ├── raw/                   # Raw captured frames
│   ├── annotated/             # Labelled training data (OK/ DEFECT/)
│   └── results/               # Defect snapshots + detections.csv
├── training/
│   ├── train.py               # MobileNetV2 transfer learning (run on PC)
│   ├── export_tflite.py       # Convert .h5 → quantized .tflite
│   └── dataset_prep.py        # Organise raw images into train/val split
└── tests/
    └── test_pipeline.py       # Unit tests for each module
```

---

## Quickstart

### On a dev machine (Windows / Mac / Linux)

```bash
cd defect-detection

# Activate the venv (already created with all deps installed)
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Run detection loop using your webcam
python run.py

# Or run the web dashboard at http://localhost:5000
python run.py --dashboard

# Run tests
python run.py --test
```

`DEV_MODE=true` is set in `.env` by default — the system uses a USB/built-in webcam and a dummy random classifier until you supply a real model.

---

### On Raspberry Pi 5

```bash
# 1. Clone the repo
git clone https://github.com/Gmohith7/ITC-cam.git
cd ITC-cam/defect-detection

# 2. Run the one-shot installer
bash setup/install.sh

# 3. Activate the venv
source venv/bin/activate

# 4. Verify camera
rpicam-hello --list-cameras

# 5. Set DEV_MODE=false in .env, then run
python run.py
```

---

## Configuration

All settings live in [config.py](config.py) and are overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `DEV_MODE` | `false` | `true` = use webcam instead of picamera2 |
| `DEV_CAMERA_INDEX` | `0` | Webcam index when DEV_MODE is on |
| `CONFIDENCE_THRESHOLD` | `0.75` | Minimum confidence to flag a defect |
| `FRAME_RATE` | `10` | Frames per second to process |
| `FLASK_PORT` | `5000` | Dashboard port |
| `GPIO_LED_PIN` | `27` | BCM pin for defect LED |
| `GPIO_BUZZER_PIN` | `17` | BCM pin for defect buzzer |

Copy `.env.example` to `.env` and adjust for your setup.

---

## Model

The system uses a **MobileNetV2** classifier trained with TensorFlow and exported to **TFLite (int8 quantized)** for Pi inference.

| Model | Inference (Pi 5) | FPS |
|---|---|---|
| MobileNetV2 float32 | ~120 ms | ~8 |
| MobileNetV2 int8 quantized | ~60 ms | ~15 |
| EfficientNet-Lite0 int8 | ~45 ms | ~20 |

### Train your own model (on PC / Google Colab)

```bash
# 1. Collect and label images into data/annotated/OK/ and data/annotated/DEFECT/

# 2. (Optional) organise a raw dump into the folder structure
python training/dataset_prep.py --raw data/raw --out data/annotated --label OK

# 3. Train
python training/train.py --data data/annotated --epochs 20 --output trained_model.h5

# 4. Export to TFLite
python training/export_tflite.py --model trained_model.h5 --output model/model.tflite

# 5. Copy model/model.tflite to the Pi
```

Until a real model is present, inference runs in **dummy mode** (random outputs) so the rest of the pipeline can still be developed and tested.

---

## GPIO Wiring (Pi 5)

| Component | BCM Pin | Physical Pin |
|---|---|---|
| LED (anode via 330Ω) | GPIO 27 | Pin 13 |
| Buzzer (+) | GPIO 17 | Pin 11 |
| GND | GND | Pin 6 |

The alert system uses **gpiozero with the lgpio backend** (`chip=4`, RP1). `RPi.GPIO` is not used — it is incompatible with the Pi 5's RP1 I/O chip.

---

## Dependencies

| Package | Used for |
|---|---|
| `numpy` | Array ops throughout the pipeline |
| `opencv-python` | Frame resize, colour convert, display, JPEG encode |
| `Pillow` | Image I/O utilities |
| `flask` | Web dashboard |
| `python-dotenv` | `.env` loading |
| `tflite-runtime` | TFLite inference on Pi (PyPI, not apt) |
| `gpiozero` | GPIO LED + buzzer control |
| `picamera2` | Camera capture on Pi (via apt) |

PyTorch and ONNX are not used. TensorFlow is only needed locally to run the training scripts.

---

## Running Tests

```bash
python run.py --test
# or directly:
pytest tests/ -v
```

All 7 tests run on any machine without a camera, GPIO, or model file.
