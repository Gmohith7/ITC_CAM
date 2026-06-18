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
tests/test_pipeline.py          ← pytest unit tests (28 tests)
tools/diagnose_camera.py        ← camera format diagnostic (run with python3, not venv)
```

## Detection algorithm

The detector is **strict** — it requires the batch block's signature to produce an OK result. A lone date, a lone keyword, or no OCR engine = always DEFECT (see Scoring below).

### ITC batch code block structure (all products)
```
Batch No.:   HH:MM  XXXXXB11      ← time + alphanumeric code
PKD.:        DD/MM/YY              ← packed date
Use By:      DD/MM/YY              ← expiry date
MRP Rs. incl. of all taxes/(Rs. per g)
NN.NN/(N.NN)
```

### Scoring (block-specific gate)
We care ONLY about the batch code block, which ALWAYS prints **two dates**
(PKD + Use By). The gate requires **≥2 dates AND one corroborating block signal**
(printed time HH:MM, the alphanumeric batch code, or a block label). This locks
detection onto the batch block — stray text elsewhere on the pack (ingredients,
a lone date or MRP line) can never trigger a false OK.

| Evidence | Score |
|----------|-------|
| anything with < 2 dates | 0.0 (hard gate) |
| 2 dates with no time/code/label | 0.0 (hard gate — needs a corroborator) |
| 2 dates + (keyword OR code) | 0.80 |
| + time HH:MM | +0.10 |
| + alphanumeric batch code | +0.10 |
| + MRP price line | +0.10 |
| full block (2 dates + time + code) | 1.00 |

Threshold: **0.55** (set in config / `.env` as `DETECTION_THRESHOLD`).

### OCR engine — RapidOCR (PP-OCR on onnxruntime)
- **One pass per frame.** RapidOCR detects + recognises all text on the frame in a single call; the evidence gate (`_evaluate`) decides OK/DEFECT. No binarisation, no PSM sweep, no region finder — those Tesseract-era stages were removed.
- **Install:** `pip install rapidocr_onnxruntime` (in the venv). It bundles the PP-OCR ONNX models; no `apt` package needed. Paddle/Tesseract are gone — paddlepaddle segfaults on Pi 5 / ARM / Python 3.13, Tesseract was too weak on the embossed dark-cardboard print.
- **Do NOT set `OMP_NUM_THREADS=1`** — it throttles onnxruntime to a single core (~10 s/frame instead of ~1–2 s). onnxruntime uses all cores by default.
- **Speed:** `OCR_MAX_SIDE` (default 960) downscales the frame before inference — fewer/smaller text boxes ⇒ much lower lag; the large batch digits stay readable.
- The regex tolerances remain because OCR still mis-renders the direct-print digits: `_PAT_DATE` accepts any digit 1-9 as a `/` separator substitute (`24902127`, `31105126`, `31408126` all match); `_PAT_TIME` accepts `O` for `0` in minutes (`07 O4` → `07:04`).

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
OCR_DEBUG=false             # print per-frame OCR result + focus/evidence diagnostics
OCR_MAX_SIDE=960            # downscale longest frame side before OCR (lower = faster)
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
- **"NO BATCH CODE" on clearly visible text (history)** — root cause was frame quality (out of focus + too dark); a raking side light fixed it (`sharp` 34→300+). The OCR engine evolved Tesseract → (PaddleOCR, segfaults on Pi) → **RapidOCR**, which reads the block reliably. The `_PAT_DATE` / `_PAT_TIME` tolerances for mis-rendered `/` and `O` separators remain useful and were kept.
- **OCR returned pure noise / always DEFECT (dark cardboard)** — root cause was **frame quality, not detection logic**. The captured frame measured `sharp≈34 / bright≈37` (FOCUS:SOFT, too dark); the same pipeline reads a `sharp≈83 / bright≈64` frame perfectly (full-frame inverted-Otsu renders the whole batch block cleanly). Fix was **physical: a raking/side light** on the embossed code, which gave continuous AF enough contrast to lock. Known-good operating point after lighting: **`sharp≈307 / bright≈92` with continuous AF + auto-exposure** — no manual exposure override needed (would only add noise). Diagnose with `tools/focus_meter.py` (live `bright`/`sharp` meter; `--sweep` finds the sharpest manual `LensPosition`). The per-frame `[FRAME #n] ... FOCUS:OK/SOFT` line in `OCR_DEBUG=true` runs is the fastest triage: low `sharp` ⇒ fix the lens/light, not the OCR code.
- `picamera2` cannot be installed in the venv via pip on Pi 5 — must use the system apt package and inject the path.
- `diagnose_camera.py` must be run with `python3` (system), not `python` (venv), for the same reason.
