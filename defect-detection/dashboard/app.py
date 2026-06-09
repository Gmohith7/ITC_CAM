import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from flask import Flask, Response, render_template_string, jsonify
import config
from camera.capture import CameraCapture
from preprocessing.preprocess import draw_result
from model.inference import BatchCodeDetector
from alerts.alert import AlertSystem
from detection.detector import _DetectionState, _ocr_worker
import threading

app = Flask(__name__)
camera = CameraCapture()
detector = BatchCodeDetector()
alerts = AlertSystem()
state = _DetectionState()
_stop = threading.Event()
_start_time = time.time()

threading.Thread(
    target=_ocr_worker,
    args=(camera, detector, alerts, state, _stop),
    daemon=True,
).start()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Batch Code Detector</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d0d0d;
      color: #e0e0e0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
      gap: 16px;
    }
    header {
      width: 100%;
      max-width: 960px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #2a2a2a;
      padding-bottom: 12px;
    }
    header h1 { font-size: 1.2rem; color: #00e676; letter-spacing: .5px; }
    #uptime { font-size: .8rem; color: #666; }

    #feed-wrap {
      width: 100%;
      max-width: 960px;
      border-radius: 8px;
      overflow: hidden;
      border: 2px solid #222;
      background: #111;
    }
    #feed-wrap img { display: block; width: 100%; }

    #status-bar {
      width: 100%;
      max-width: 960px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }
    .stat-card {
      background: #161616;
      border: 1px solid #252525;
      border-radius: 8px;
      padding: 14px 18px;
    }
    .stat-card .label { font-size: .7rem; color: #666; text-transform: uppercase; letter-spacing: .8px; }
    .stat-card .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
    #val-label.ok    { color: #00e676; }
    #val-label.defect { color: #ff1744; }
    #val-label.scanning { color: #ffab40; }
    #val-conf { color: #64b5f6; }
    #val-text { font-size: .95rem; color: #aaa; font-family: monospace; }

    footer { color: #333; font-size: .75rem; }
  </style>
</head>
<body>
  <header>
    <h1>Batch Code Detector &mdash; Live Feed</h1>
    <span id="uptime">uptime: 0s</span>
  </header>

  <div id="feed-wrap">
    <img src="/video_feed" alt="Live camera feed">
  </div>

  <div id="status-bar">
    <div class="stat-card">
      <div class="label">Status</div>
      <div class="value" id="val-label">SCANNING</div>
    </div>
    <div class="stat-card">
      <div class="label">Confidence</div>
      <div class="value" id="val-conf">—</div>
    </div>
    <div class="stat-card">
      <div class="label">OCR Text</div>
      <div class="value" id="val-text">—</div>
    </div>
  </div>

  <footer>Raspberry Pi 5 &mdash; Camera Module 3</footer>

  <script>
    const startTs = Date.now();

    async function poll() {
      try {
        const r = await fetch('/status');
        const d = await r.json();
        const el = document.getElementById('val-label');
        el.textContent = d.label;
        el.className = d.scanning ? 'scanning' : (d.is_defect ? 'defect' : 'ok');
        document.getElementById('val-conf').textContent =
          d.scanning ? '—' : (d.confidence * 100).toFixed(0) + '%';
        document.getElementById('val-text').textContent = d.ocr_text || '—';
      } catch (_) {}
      const s = Math.round((Date.now() - startTs) / 1000);
      document.getElementById('uptime').textContent =
        'uptime: ' + (s < 60 ? s + 's' : Math.floor(s/60) + 'm ' + (s%60) + 's');
    }

    setInterval(poll, 500);
    poll();
  </script>
</body>
</html>
"""


def generate_frames():
    while True:
        frame = camera.get_frame()
        label, confidence, is_defect, scanning, regions, text = state.snapshot()

        annotated = draw_result(
            frame, label, confidence, is_defect,
            scanning=scanning, regions=regions, ocr_text=text,
        )

        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """JSON endpoint for polling current detection state."""
    label, confidence, is_defect, scanning, regions, text = state.snapshot()
    return jsonify({
        "label": label,
        "confidence": confidence,
        "is_defect": is_defect,
        "scanning": scanning,
        "regions": regions,
        "ocr_text": text.split('\n')[0][:80] if text else "",
        "uptime_s": round(time.time() - _start_time),
    })


if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, threaded=True)
