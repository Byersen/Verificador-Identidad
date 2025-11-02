import os
import io
import json
import logging

from flask import request, jsonify
from PIL import Image
import torch

log = logging.getLogger(__name__)

MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_MB", 5))
MAX_CONTENT_LENGTH = MAX_CONTENT_MB * 1024 * 1024
VERIFY_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", 0.75))


@identity_app.before_request
def check_content_length():
    if request.content_length is not None and request.content_length > MAX_CONTENT_LENGTH:
        try:
            log.warning(f"Payload exceeds the limit of {MAX_CONTENT_MB} MB.")
        except Exception:
            pass
        return jsonify({"error": f"Payload exceeds {MAX_CONTENT_MB} MB limit"}), 413

@identity_app.route('/verify', methods=['POST'])
def verify_identity():
    if MODELS is None:
        return jsonify({"error": "Models not loaded"}), 503

    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' in form-data"}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')

    device, mtcnn, resnet, clf, scaler = MODELS
    with torch.no_grad():
        face = mtcnn(img)
        if face is None:
            return jsonify({"error": "no face detected"}), 400
        
        emb = resnet(face.unsqueeze(0)).cpu().numpy()
        scaled_emb = scaler.transform(emb)
        prob = clf.predict_proba(scaled_emb)[0, 1]

    return jsonify({"score": float(prob), "is_me": bool(prob >= VERIFY_THRESHOLD)}), 200

@identity_app.route('/healthz', methods=['GET'])
def health_check():
    best_threshold = None
    try:
        if REPORTS_PATH.exists():
            with open(REPORTS_PATH, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                best_threshold = metrics.get('best_threshold')
    except Exception as e:
        try:
            log.warning(f"No se pudo leer el umbral óptimo de metrics.json: {e}")
        except Exception:
            pass

    status = "ok" if MODELS is not None else "error"

    return jsonify({
        "status": status,
        "version": MODEL_VERSION,
        "best_threshold": best_threshold
    }), 200
