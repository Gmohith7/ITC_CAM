# ITC Batch Code Detector

Real-time **batch code detection** on ITC packaged-food products, running entirely on a
**Raspberry Pi 5** with the **Camera Module 3** — no cloud dependency. The camera watches a
product; the system OCRs the printed batch code block and signals **OK** (code present &
readable) or **DEFECT** (missing / unreadable) on screen and via a GPIO LED + buzzer.

OCR uses **RapidOCR** (PP-OCR detection + recognition on **onnxruntime**), which is accurate
on the low-contrast embossed print and reliable on ARM. A strict, block-specific evidence
gate makes the decision — so ingredient/nutrition text on the pack can never trigger a false
OK.

---

## What it detects

The ITC batch code block (printed directly on dark cardboard):

```
Batch No.:   07:04  09A11       ← time + alphanumeric code
PKD.:        31/05/26            ← packed date
Use By:      24/02/27            ← expiry date
MRP Rs. incl. of all taxes/(Rs. per g)
40.00/(0.58)
```

A frame scores **OK** only when the block's signature is present: **two dates (PKD + Use By)
plus a corroborating signal** (the printed time, the alphanumeric code, or a block label).
Threshold `0.55`; a full block scores `1.00`.

---

## Hardware

| Component | Spec |
|---|---|
| SBC | Raspberry Pi 5 |
| Camera | Camera Module 3 (Sony IMX708, autofocus) via CSI |
| Display | HDMI monitor for the live preview |
| Alerts | LED on GPIO 27, buzzer on GPIO 17 (gpiozero + lgpio, RP1 `chip=4`) |
| Lighting | A **side/raking light** on the code — essential for contrast on embossed print |

---

## Project structure

```
defect-detection/
├── run.py                      # Launcher: detection / --dashboard / --headless / --test
├── config.py                   # All tunable settings (reads from .env)
├── camera/capture.py           # picamera2 wrapper (Pi 5 BGRX fix, AF, webcam fallback)
├── model/
│   ├── inference.py            # BatchCodeDetector: RapidOCR + evidence gate
│   └── ocr_engines.py          # RapidOCREngine (PP-OCR on onnxruntime)
├── preprocessing/preprocess.py # HUD overlay (BATCH CODE OK / NO BATCH CODE)
├── detection/detector.py       # Main loop: display thread + OCR worker thread
├── alerts/alert.py             # Non-blocking GPIO LED + buzzer
├── defect_logging/logger.py    # CSV log + JPEG snapshot per defect
├── dashboard/app.py            # Flask MJPEG stream + /status endpoint
├── tools/
│   ├── focus_meter.py          # Live focus/exposure meter (+ --sweep for manual focus)
│   ├── ocr_image.py            # Run the full pipeline on one saved frame (no camera)
│   ├── ocr_report.py           # Batch-test the OCR on test_images/
│   └── diagnose_camera.py      # Camera format diagnostic
└── tests/                      # pytest unit + real-image tests
```

---

## Quickstart (Raspberry Pi 5)

```bash
git clone https://github.com/Gmohith7/ITC_CAM.git
cd ITC_CAM/defect-detection
bash setup/install.sh                 # system deps + venv (system-site-packages for picamera2)
source venv/bin/activate
pip install rapidocr_onnxruntime      # OCR engine (first run downloads PP-OCR models)

# Live detection with HDMI preview:
python run.py

# Headless (SSH, logs only):
python run.py --headless

# Web dashboard at http://<pi-ip>:5000 :
python run.py --dashboard

# Tests:
python run.py --test
```

`picamera2` is installed system-wide via apt (not pip-installable on Pi 5); the camera module
injects `/usr/lib/python3/dist-packages` onto `sys.path` so it imports inside the venv.

---

## Dialing it in (do this first)

Frame quality is everything — RapidOCR (or any OCR) returns noise on a blurry/dark frame.

```bash
# 1. Add a side light on the code, then find the sharpest focus:
python tools/focus_meter.py --sweep      # prints the LENS_POSITION to lock in .env

# 2. Live-check until sharp >= ~80 and brightness ~50-120:
python tools/focus_meter.py
```

Targets: **sharpness ≥ 80**, **brightness 50–120**. With a side light the rig runs sharp
250–330. In `OCR_DEBUG=true` runs, the `[FRAME #n] ... FOCUS:OK/SOFT` line is the fastest
triage — low `sharp` means fix the lens/light, not the code.

---

## Configuration (`.env`, see `.env.example`)

| Variable | Default | Description |
|---|---|---|
| `DETECTION_THRESHOLD` | `0.55` | Min evidence score to declare a batch code present |
| `OCR_MAX_SIDE` | `960` | Downscale longest frame side before OCR (lower = faster/less lag) |
| `AF_MODE` | `continuous` | `continuous` or `manual` (+ `LENS_POSITION`) for a fixed station |
| `AF_RANGE` | `macro` | AF search range for close-up inspection |
| `DARK_FRAME_THRESHOLD` | `8.0` | Skip frames darker than this (camera warming up) |
| `EXPOSURE_TIME` / `ANALOGUE_GAIN` / `BRIGHTNESS` | auto | Optional manual exposure for dim stations |
| `OCR_DEBUG` | `false` | Print per-frame OCR result + focus/evidence diagnostics |
| `FLASK_PORT` | `5000` | Dashboard port |
| `GPIO_LED_PIN` / `GPIO_BUZZER_PIN` | `27` / `17` | Alert pins |

---

## Dependencies

| Package | Used for |
|---|---|
| `rapidocr_onnxruntime` | OCR engine (PP-OCR detection + recognition on onnxruntime) |
| `numpy` | Array ops throughout the pipeline |
| `opencv-python` | Frame resize, colour convert, display, JPEG encode |
| `Pillow` | Image I/O utilities |
| `flask` | Web dashboard |
| `python-dotenv` | `.env` loading |
| `gpiozero` | GPIO LED + buzzer (lgpio / RP1 backend) |
| `picamera2` | Camera capture on Pi (apt, system-wide) |

> Tesseract and PaddleOCR were removed: Tesseract was too weak on the embossed dark-cardboard
> print, and paddlepaddle's native inference segfaults on Pi 5 / ARM / Python 3.13. RapidOCR
> (onnxruntime) replaced both.

---

## Tests

```bash
python run.py --test
# or:  pytest tests/ -v
```

33 unit tests run on any machine without a camera, GPIO, or OCR engine. The real-image tests
skip automatically when RapidOCR is not installed.
