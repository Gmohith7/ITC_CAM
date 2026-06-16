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

## Camera Module 3 autofocus

The Camera Module 3 (IMX708) has a motorised autofocus lens. picamera2 does **not**
enable it by default — it parks the lens at a far/hyperfocal position, so a close-up
product (~15–25 cm) is permanently blurred regardless of distance. `_apply_focus()` in
`camera/capture.py` fixes this: by default it sets `AfMode=Continuous` so the lens keeps
hunting until whatever is in view is sharp. Set `AF_MODE=manual` + `LENS_POSITION` to lock
focus at a fixed bench distance instead. Minimum focus distance is **~10 cm** (standard
lens) / ~5 cm (wide) — the product physically cannot be focused closer than that.

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

### Two-stage scan (Stage 1 runs first — fastest path)
1. **Stage 1** — full-frame OCR at two resolutions (full-res first, then 1280px). Uses PSM 11 (sparse text) + PSM 3. Early-exits the moment a score ≥ threshold is found. Handles direct-print cardboard where keywords and dates must be read together in one pass.
2. **Stage 2** — region crop OCR (fallback). Finds white-sticker bounding boxes and OCRs each crop with the full PSM set (6, 11, 4, 3).

### OCR preprocessing (hard-won)
- Two grayscale sources are tried per image:
  - **Standard luminance** (`cv2.COLOR_RGB2GRAY`)
  - **min(G, B) channel** — for white text on dark red/maroon background: standard gray ≈ 58, min(G,B) ≈ 20, giving ~12:1 contrast vs ~4:1. Critical for dark cardboard packaging.
- Date separators (`/`) are often misread as `9` or `1` by Tesseract on direct-print packaging. `_PAT_DATE` uses `[/\-\.|l19]?` as the separator so `24902127` still matches `24/02/27`.
- `mrp` is accepted as a keyword fallback — "MRP Rs." always appears in the ITC batch block and survives OCR noise better than multi-word labels.
- Tesseract must be installed system-wide: `sudo apt install tesseract-ocr`. It is NOT pip-installable in the venv.

## Key config values (.env / config.py)

```
GRAYSCALE_MODE=false        # must be false — colour is required for correct display
AF_MODE=continuous          # Camera Module 3 autofocus: "continuous" (hunts automatically) or "manual"
LENS_POSITION=5.0           # manual focus only; LensPosition = 1/distance_m (20cm→5.0, 25cm→4.0, 10cm→10.0)
AF_SPEED=fast               # continuous-AF convergence: "fast" or "normal"
AF_RANGE=macro              # AF search range: "macro" (default, ~10-30cm) | "normal" | "full"
DETECTION_THRESHOLD=0.55    # minimum evidence score for OK
WHITE_THRESHOLD=185         # sticker brightness threshold
OCR_MIN_HEIGHT=140          # upscale OCR crops shorter than this
DARK_FRAME_THRESHOLD=8.0    # skip frames darker than this (camera warming up)
LOG_SNAPSHOTS=true          # save JPEG of each defect frame
ALERT_DURATION_S=1.0        # GPIO pulse duration
OCR_DEBUG=false             # print raw Tesseract output per variant to console
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
- **Camera not focusing** — `AF_RANGE=normal` starts at ~30 cm; changed default to `macro` (~10-30 cm) for inspection distances.
- **"NO BATCH CODE" on clearly visible text** — Tesseract was not installed (`sudo apt install tesseract-ocr`). After install, dates were misread (/ → 9 or 1); fixed in `_PAT_DATE`. Stage 1 was OCRing individual text lines (no keyword+date co-occurrence); fixed by running full-frame first.
- `picamera2` cannot be installed in the venv via pip on Pi 5 — must use the system apt package and inject the path.
- `diagnose_camera.py` must be run with `python3` (system), not `python` (venv), for the same reason.
