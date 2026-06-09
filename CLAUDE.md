# ITC Batch Code Detector — Project Context for Claude

## What this project does

Real-time batch code detection on ITC packaged food products using a Raspberry Pi 5 + Camera Module 3. The camera watches a product on a conveyor/table; the system OCRs the batch code block and signals OK or DEFECT via GPIO LED + buzzer.

## Hardware

| Component | Detail |
|-----------|--------|
| SBC | Raspberry Pi 5 |
| Camera | Camera Module 3 (connected via CSI) |
| GPIO | LED on pin 27, Buzzer on pin 17 |
| GPIO backend | `gpiozero` + `lgpio` with `LGPIOFactory(chip=4)` — Pi 5 uses the RP1 chip, NOT RPi.GPIO |
| Display | HDMI monitor for `cv2.imshow` window |

## Critical: Pi 5 camera colour fix (hard-won)

`picamera2` on Pi 5 always returns **4-channel (H, W, 4) BGRX arrays** from `capture_array("main")` regardless of the format string requested. The byte layout per pixel is:

```
Index:  0   1   2   3
Byte:  [B,  G,  R,  X]   ← X is unused padding
```

**The correct conversion to RGB is:**
```python
rgb = raw[:, :, 2::-1]   # reverse channels 0-2: [B,G,R] → [R,G,B], drops X
```

**Do NOT use** `COLOR_BGR2RGB`, `COLOR_BGR2GRAY` on the raw 4-channel array, or any format string other than `XRGB8888` (which is what actually gets used internally). Previous attempts that failed:
- `BGR888` format → still gives 4-channel BGRX, wrong colours
- `RGB888` format → still gives 4-channel BGRX, wrong colours  
- `raw[:, :, 1:4]` → picks [G, R, X], completely wrong
- `raw[:, :, 3:0:-1]` → wrong order

This is implemented in `camera/capture.py` → `_picam_to_rgb()`.

## Running the project

```bash
# On the Pi, always activate the venv first:
cd ~/Desktop/ITC_CAM/defect-detection
source .venv/bin/activate   # or: source venv/bin/activate

# Run with display window:
python run.py

# Run headless (no display, useful over SSH):
python run.py --headless

# Run Flask web dashboard (view at http://<pi-ip>:5000):
python run.py --dashboard

# Run test suite:
python run.py --test
```

**Important:** `picamera2` is installed system-wide via `apt`, NOT inside the venv. The camera module adds `/usr/lib/python3/dist-packages` to `sys.path` at runtime so picamera2 is importable from within the venv.

## Architecture

```
run.py                          ← launcher (detection / dashboard / test modes)
config.py                       ← all tunable settings, reads from .env
camera/capture.py               ← CameraCapture: picamera2 + webcam fallback
model/inference.py              ← BatchCodeDetector: OCR + evidence scoring
detection/detector.py           ← main loop: display thread + OCR worker thread
preprocessing/preprocess.py     ← draw_result: HUD overlay on frame
alerts/alert.py                 ← AlertSystem: non-blocking GPIO LED + buzzer
defect_logging/logger.py        ← CSV log + JPEG snapshot per defect
dashboard/app.py                ← Flask MJPEG stream + /status JSON endpoint
tests/test_pipeline.py          ← pytest unit tests (23 tests)
tools/diagnose_camera.py        ← camera format diagnostic (run with python3, not venv)
```

## Detection algorithm

The detector is **strict** — it requires co-occurrence of a label keyword AND a date to produce an OK result. A date alone, a keyword alone, a bright region alone, or no Tesseract = always DEFECT.

### ITC batch code block structure (all products)
```
Batch No.:   HH:MM  XXXXXB11      ← time + alphanumeric code
PKD.:        DD/MM/YY              ← packed date
Use By:      DD/MM/YY              ← expiry date
MRP Rs. incl. of all taxes/(Rs. per g)
NN.NN/(N.NN)
```

### Scoring (AND-gate)
| Evidence | Score |
|----------|-------|
| keyword alone | 0.0 (hard gate) |
| date alone | 0.0 (hard gate) |
| keyword + date (minimum) | 0.60 |
| + second date (PKD + Use By) | 0.80 |
| + time HH:MM | +0.10 |
| + alphanumeric batch code | +0.10 |
| + MRP price line | +0.10 |

Threshold: **0.55** (set in config / `.env` as `DETECTION_THRESHOLD`).

### Two-stage scan
1. **Stage 1** — find white sticker regions (brightness threshold + morphology) + Canny edge rectangles. OCR each crop.
2. **Stage 2** — full-frame OCR on downscaled image (for direct-print on coloured cardboard).

## Key config values (.env / config.py)

```
GRAYSCALE_MODE=false        # must be false — colour is required for correct display
DETECTION_THRESHOLD=0.55    # minimum evidence score for OK
WHITE_THRESHOLD=185         # sticker brightness threshold
OCR_MIN_HEIGHT=140          # upscale OCR crops shorter than this
DARK_FRAME_THRESHOLD=8.0    # skip frames darker than this (camera warming up)
LOG_SNAPSHOTS=true          # save JPEG of each defect frame
ALERT_DURATION_S=1.0        # GPIO pulse duration
```

## GPIO notes

- `gpiozero` with `lgpio` backend is required on Pi 5. `RPi.GPIO` does NOT work on Pi 5.
- `LGPIOFactory(chip=4)` — chip index 4 is the RP1 I/O controller on Pi 5.
- `trigger()` is non-blocking (runs in a daemon thread). A threading lock debounces rapid calls.
- All `led.on()` / `led.off()` calls are wrapped in try/except — `GPIODeviceClosed` can occur if `clear()` races with the alert thread; both simply catch and ignore the error.

## Test images

Located in `test_images/` at the project root. These are real ITC product photos used to validate the OCR pipeline. Products include Right Shift cookies, Sunfeast biscuits — all with the same batch code block layout.

## Known issues / history

- **Colour was wrong for many iterations** — root cause was Pi 5 BGRX pixel layout. Resolved in commit `171631f`.
- `picamera2` cannot be installed in the venv via pip on Pi 5 — must use the system apt package and inject the path.
- `diagnose_camera.py` must be run with `python3` (system), not `python` (venv), for the same reason.
