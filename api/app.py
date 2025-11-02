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
