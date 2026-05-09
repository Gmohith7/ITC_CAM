import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from flask import Flask, Response, render_template_string
import config
from camera.capture import CameraCapture
from preprocessing.preprocess import preprocess_frame, draw_result
from model.inference import DefectInference
from defect_logging.logger import log_detection

app = Flask(__name__)
camera = CameraCapture()
model = DefectInference()

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Defect Detection Dashboard</title>
  <meta http-equiv="refresh" content="0">
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; text-align:center; padding:20px; }
    h1 { color:#0f0; }
    img { border: 3px solid #333; max-width:100%; }
    .sub { color:#888; font-size:.9em; margin-top:8px; }
  </style>
</head>
<body>
  <h1>Defect Detection Live Feed</h1>
  <img src="/video_feed" alt="Live feed">
  <p class="sub">Raspberry Pi 5 &mdash; Camera Module 3 &mdash; MobileNetV2</p>
</body>
</html>
"""


def generate_frames():
    while True:
        frame = camera.get_frame()
        tensor = preprocess_frame(frame)
        label, confidence, is_defect = model.predict(tensor)
        annotated = draw_result(frame.copy(), label, confidence, is_defect)

        if is_defect:
            log_detection(frame, label, confidence)

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
