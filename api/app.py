@identity_app.before_request
def check_content_length():
    if request.content_length is not None and request.content_length > MAX_CONTENT_LENGTH:
        try:
            log.warning(f"Payload exceeds the limit of {MAX_CONTENT_MB} MB.")
        except Exception:
            pass
        return jsonify({"error": f"Payload exceeds {MAX_CONTENT_MB} MB limit"}), 413

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
