@identity_app.route('/healthz', methods=['GET'])
def health_check():
    """Basic health check for service availability."""
    status = "ok" if MODELS is not None else "error"
    return jsonify({
        "status": status,
        "version": MODEL_VERSION
    }), 200
