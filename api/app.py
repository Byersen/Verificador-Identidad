import os

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
from PIL import Image
import torch
import time

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
        try:
            log.error("Models not loaded.")
        except Exception:
            pass
        return jsonify({"error": "Service unavailable"}), 503

    start = time.monotonic()

    if 'image' not in request.files:
        try:
            log.warning("Request missing 'image' field.")
        except Exception:
            pass
        return jsonify({"error": "Missing 'image' in form-data"}), 400

    file = request.files['image']
    if file.mimetype not in ['image/jpeg', 'image/png']:
        try:
            log.warning(f"Unsupported MIME type: {file.mimetype}")
        except Exception:
            pass
        return jsonify({"error": "Only JPEG and PNG allowed"}), 415

    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        device, mtcnn, resnet, clf, scaler = MODELS

        with torch.no_grad():
            face_tensor = mtcnn(pil_image)
            if face_tensor is None:
                try:
                    log.info("No face detected in image.")
                except Exception:
                    pass
                return jsonify({"error": "no face detected"}), 400

            embedding = resnet(face_tensor.unsqueeze(0))
            scaled = scaler.transform(embedding.cpu().numpy())
            prob = clf.predict_proba(scaled)[0, 1]

        elapsed = (time.monotonic() - start) * 1000
        try:
            log.info(f"Verification done: prob={prob:.4f}, time={elapsed:.2f}ms")
        except Exception:
            pass

        return jsonify({
            "is_me": bool(prob >= VERIFY_THRESHOLD),
            "score": float(prob),
            "threshold": VERIFY_THRESHOLD,
            "timing_ms": round(elapsed, 2)
        }), 200

    except Exception as e:
        try:
            log.error(f"Unexpected error during verification: {e}", exc_info=True)
        except Exception:
            pass
        return jsonify({"error": "Internal server error"}), 500

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
