"""
Driver Drowsiness Detection System
===================================
Student : Hrishitha Prasad A S
USN     : 1NT23AD022
Run     : python app.py
Open    : http://localhost:5000
"""

import os, base64, io, json, time
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from collections import deque
prob_buffer = deque(maxlen=10)
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16        import preprocess_input as vgg_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.resnet_v2    import preprocess_input as res_pre

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ── Load class info saved during training ────────────────────────
CLASS_INFO_PATH = os.path.join(MODELS_DIR, 'class_info.json')
if os.path.exists(CLASS_INFO_PATH):
    with open(CLASS_INFO_PATH) as f:
        CLASS_INFO = json.load(f)
    CLASS_INDICES  = CLASS_INFO['class_indices']
    INDEX_TO_LABEL = CLASS_INFO['index_to_label']
else:
    # fallback: alphabetical → DROWSY=0, NATURAL=1
    CLASS_INDICES  = {'DROWSY': 0, 'NATURAL': 1}
    INDEX_TO_LABEL = {'0': 'DROWSY', '1': 'NATURAL'}

# sigmoid output: close to 1 = class with index 1
# alphabetical: DROWSY=0, NATURAL=1
# so prob > 0.5 → NATURAL (alert), prob < 0.5 → DROWSY
DROWSY_IDX = CLASS_INDICES.get('DROWSY', CLASS_INDICES.get('drowsy', 0))
# If DROWSY has index 1 → prob>0.5 means drowsy
# If DROWSY has index 0 → prob<0.5 means drowsy
DROWSY_IS_HIGH = (DROWSY_IDX == 1)

print(f"\n  Class mapping  : {CLASS_INDICES}")
print(f"  DROWSY index   : {DROWSY_IDX}")
print(f"  Drowsy = high prob: {DROWSY_IS_HIGH}")

# ── Model registry ───────────────────────────────────────────────
MODEL_META = {
    "ResNet50V2": {
        "file"    : "resnet_final.keras",
        "params"  : "24,123,393",
        "type"    : "Transfer Learning",
        "input"   : (224, 224),
        "color"   : "rgb",
        "desc"    : "50-layer residual network — Best Model",
        "accuracy": "98.31%",
        "f1"      : "0.9830",
        "auc"     : "0.9995",
    },
    "MobileNetV2": {
        "file"    : "mobilenet_final.keras",
        "params"  : "2,430,785",
        "type"    : "Transfer Learning",
        "input"   : (224, 224),
        "color"   : "rgb",
        "desc"    : "Lightweight depthwise separable CNN",
        "accuracy": "93.46%",
        "f1"      : "0.9370",
        "auc"     : "0.9964",
    },
    "VGG16": {
        "file"    : "vgg16_final.keras",
        "params"  : "14,880,065",
        "type"    : "Transfer Learning",
        "input"   : (224, 224),
        "color"   : "rgb",
        "desc"    : "16-layer CNN pretrained on ImageNet",
        "accuracy": "60.49%",
        "f1"      : "0.7125",
        "auc"     : "0.9959",
    },
    "DrowsyNet": {
        "file"    : "drowsynet_final.keras",
        "params"  : "661,641",
        "type"    : "Custom Novel",
        "input"   : (64, 64),
        "color"   : "grayscale",
        "desc"    : "CNN + SE Attention + BiLSTM",
        "accuracy": "51.31%",
        "f1"      : "0.6679",
        "auc"     : "0.8157",
    },
}

PREPROCESS = {
    "VGG16"      : vgg_pre,
    "MobileNetV2": mob_pre,
    "ResNet50V2" : res_pre,
    "DrowsyNet"  : None,
}

loaded = {}

# ── Webcam state ─────────────────────────────────────────────────
webcam_active = False
webcam_cap    = None
webcam_result = {
    "status"    : "UNKNOWN",
    "is_drowsy" : False,
    "drowsy_pct": 0.0,
    "alert_pct" : 0.0,
}
WEBCAM_MODEL  = "ResNet50V2"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Load models ──────────────────────────────────────────────────
def load_all_models():
    for name, meta in MODEL_META.items():
        path = os.path.join(MODELS_DIR, meta['file'])
        if os.path.exists(path):
            print(f"  Loading {name}...", end=" ", flush=True)
            loaded[name] = load_model(path)
            print("✓")
        else:
            print(f"  ⚠  {name} not found at {path}")

# ── Preprocessing ────────────────────────────────────────────────
def preprocess_image(img_rgb, model_name):
    meta = MODEL_META[model_name]
    size = meta['input']
    if meta['color'] == 'grayscale':
        gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        eq      = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(eq, (3, 3), 0)
        resized = cv2.resize(blurred, size)
        norm    = resized.astype(np.float32) / 255.0
        return norm.reshape(1, size[0], size[1], 1)
    else:
        resized = cv2.resize(img_rgb, size).astype(np.float32)
        return np.expand_dims(PREPROCESS[model_name](resized), 0)

def predict_drowsiness(img_rgb, model_name):
    """
    Returns (is_drowsy, drowsy_pct, alert_pct)
    Handles class index mapping correctly regardless of alphabetical order.
    """
    inp  = preprocess_image(img_rgb, model_name)
    prob = float(loaded[model_name].predict(inp, verbose=0)[0][0])

    # prob = probability of class with index 1
    drowsy_prob = 1.0 - prob

    prob_buffer.append(drowsy_prob)
    avg_prob = sum(prob_buffer) / len(prob_buffer)
    is_drowsy  = avg_prob > 0.35
    drowsy_pct = round(avg_prob * 100, 2)
    alert_pct  = round((1.0 - avg_prob) * 100, 2)
    return is_drowsy, drowsy_pct, alert_pct

# ── DIP pipeline ─────────────────────────────────────────────────
def dip_pipeline_b64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq      = cv2.equalizeHist(gray)
    blur    = cv2.GaussianBlur(eq, (3, 3), 0)

    roi = img_rgb.copy()
    for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)):
        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 220, 120), 2)

    norm_disp = (cv2.resize(blur, (64, 64)).astype(np.float32) / 255.0 * 255).astype(np.uint8)

    steps = [
        ("Original",      cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)),
        ("Grayscale",     cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
        ("Histogram EQ",  cv2.cvtColor(eq,   cv2.COLOR_GRAY2BGR)),
        ("Gaussian Blur", cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)),
        ("Face ROI",      cv2.cvtColor(roi,  cv2.COLOR_RGB2BGR)),
        ("Normalised",    cv2.cvtColor(norm_disp, cv2.COLOR_GRAY2BGR)),
    ]

    result = []
    for label, img in steps:
        _, buf = cv2.imencode('.png', img)
        result.append({"label": label, "data": base64.b64encode(buf).decode()})
    return result

# ── Webcam stream ────────────────────────────────────────────────
def generate_frames():
    global webcam_cap, webcam_active, webcam_result

    webcam_cap    = cv2.VideoCapture(0)
    webcam_active = True

    if not webcam_cap.isOpened():
        webcam_active = False
        return

    frame_count   = 0
    PREDICT_EVERY = 8

    while webcam_active:
        success, frame = webcam_cap.read()
        if not success:
            break

        frame_count += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count % PREDICT_EVERY == 0 and WEBCAM_MODEL in loaded:
            try:
                is_drowsy, drowsy_pct, alert_pct = predict_drowsiness(img_rgb, WEBCAM_MODEL)
                webcam_result = {
                    "status"    : "DROWSY" if is_drowsy else "ALERT",
                    "is_drowsy" : is_drowsy,
                    "drowsy_pct": drowsy_pct,
                    "alert_pct" : alert_pct,
                }
            except Exception:
                pass

        is_drowsy  = webcam_result.get("is_drowsy", False)
        status     = webcam_result.get("status", "UNKNOWN")
        drowsy_pct = webcam_result.get("drowsy_pct", 0.0)
        alert_pct  = webcam_result.get("alert_pct", 0.0)

        color = (0, 0, 255) if is_drowsy else (0, 200, 80)

        # Banner
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), color, -1)
        cv2.putText(frame, f"  {status}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Bottom bar
        cv2.rectangle(frame, (0, frame.shape[0]-45),
                      (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
        bar_w = int((drowsy_pct / 100) * frame.shape[1])
        cv2.rectangle(frame, (0, frame.shape[0]-45),
                      (bar_w, frame.shape[0]), (0, 0, 200), -1)
        cv2.putText(frame,
                    f"Drowsy: {drowsy_pct:.1f}%   Alert: {alert_pct:.1f}%   [{WEBCAM_MODEL}]",
                    (10, frame.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    if webcam_cap:
        webcam_cap.release()
    webcam_active = False

# ── Routes ───────────────────────────────────────────────────────
@app.route('/')
def index():
    model_info   = {n: {k: v for k, v in m.items() if k != 'file'}
                    for n, m in MODEL_META.items()}
    loaded_names = list(loaded.keys())
    return render_template('index.html',
                           model_info=json.dumps(model_info),
                           loaded_models=json.dumps(loaded_names))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    model_name = request.form.get('model', 'ResNet50V2')
    if model_name not in loaded:
        return jsonify({"error": f"{model_name} not loaded"}), 400

    file    = request.files['image']
    img_pil = Image.open(file.stream).convert('RGB')
    img_rgb = np.array(img_pil)

    t0 = time.time()
    is_drowsy, drowsy_pct, alert_pct = predict_drowsiness(img_rgb, model_name)
    ms = int((time.time() - t0) * 1000)

    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "status"    : "DROWSY" if is_drowsy else "ALERT",
        "is_alert"  : not is_drowsy,
        "is_drowsy" : is_drowsy,
        "alert_pct" : alert_pct,
        "drowsy_pct": drowsy_pct,
        "model"     : model_name,
        "latency_ms": ms,
        "dip_steps" : dip_pipeline_b64(img_rgb),
        "image_b64" : img_b64,
        "meta"      : {k: v for k, v in MODEL_META[model_name].items() if k != 'file'},
    })

@app.route('/webcam/feed')
def webcam_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam/status')
def webcam_status_route():
    return jsonify(webcam_result)

@app.route('/webcam/stop', methods=['POST'])
def webcam_stop():
    global webcam_active, webcam_cap
    webcam_active = False
    if webcam_cap:
        webcam_cap.release()
        webcam_cap = None
    return jsonify({"status": "stopped"})

@app.route('/models')
def models_status():
    return jsonify({n: "loaded" if n in loaded else "not found"
                    for n in MODEL_META})

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Driver Drowsiness Detection System")
    print("  Student: Hrishitha Prasad A S | 1NT23AD022")
    print("="*55)
    print("\n📦 Loading models...")
    load_all_models()
    print(f"\n✅ {len(loaded)}/{len(MODEL_META)} models loaded")
    print("\n🌐 Starting server...")
    print("   Open → http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
