import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from flask import Flask, Response, render_template_string
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

threading.Thread(
    target=_ocr_worker,
    args=(camera, detector, alerts, state, _stop),
    daemon=True,
).start()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Batch Code Detector</title>
  <meta http-equiv="refresh" content="0">
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; text-align:center; padding:20px; }
    h1 { color:#0f0; }
    img { border: 3px solid #333; max-width:100%; }
    .sub { color:#888; font-size:.9em; margin-top:8px; }
  </style>
</head>
<body>
  <h1>Batch Code Detector &mdash; Live Feed</h1>
  <img src="/video_feed" alt="Live feed">
  <p class="sub">Raspberry Pi 5 &mdash; Camera Module 3</p>
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


if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, threaded=True)
