import torch
import joblib
import json
import os
import io
import time
import logging
from flask import Flask, request, jsonify, render_template
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/"))
API_PORT = int(os.getenv("API_PORT", 5000))
MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_MB", 5))
MAX_CONTENT_LENGTH = MAX_CONTENT_MB * 1024 * 1024
VERIFY_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", 0.75))
MODEL_VERSION = os.getenv("MODEL_VERSION", "me-verifier-v1")
REPORTS_PATH = Path("reports/metrics.json")

identity_app = Flask(__name__)

def load_models(device):
    log.info("Cargando modelos en memoria...")
    try:
        mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20, 
            keep_all=False, 
            device=device,
            select_largest=True
        )
        
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        scaler = joblib.load(MODEL_PATH / "scaler.joblib")
        classifier = joblib.load(MODEL_PATH / "model.joblib")
        
        log.info("Todos los modelos han sido cargados exitosamente.")
        return device, mtcnn, resnet, classifier, scaler
        
    except FileNotFoundError as e:
        log.error(f"Error: No se encontró un archivo de modelo: {e}. Asegúrate de ejecutar train.py.")
        return None
    except Exception as e:
        log.error(f"Error inesperado al cargar modelos: {e}")
        return None

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODELS = load_models(DEVICE)
if MODELS is None:
    log.critical("El servidor no pudo iniciarse. Faltan los artefactos del modelo.")

@identity_app.before_request
def check_content_length():
    if request.content_length is not None and request.content_length > MAX_CONTENT_LENGTH:
        log.warning(f"Payload excede el límite de {MAX_CONTENT_MB} MB.")
        return jsonify({"error": f"Payload excede el límite de {MAX_CONTENT_MB} MB"}), 413

@identity_app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        log.error(f"Error al renderizar index.html: {e}")
        return "Interfaz no disponible", 500

@identity_app.route('/healthz', methods=['GET'])
def health_check():
    best_threshold = None
    try:
        if REPORTS_PATH.exists():
            with open(REPORTS_PATH, 'r') as f:
                metrics = json.load(f)
                best_threshold = metrics.get('best_threshold')
    except Exception as e:
        log.warning(f"No se pudo leer el umbral óptimo de metrics.json: {e}")

    status = "ok" if MODELS is not None else "error"
    
    return jsonify({
        "status": status,
        "version": MODEL_VERSION,
        "best_threshold": best_threshold
    }), 200

@identity_app.route('/verify', methods=['POST'])
def verify_identity():
    
    if MODELS is None:
        log.error("Llamada a /verify fallida porque los modelos no están cargados.")
        return jsonify({"error": "Servicio no disponible, modelos no cargados"}), 503

    start_time = time.monotonic()
    
    if 'image' not in request.files:
        log.warning("Petición a /verify sin archivo 'image'.")
        return jsonify({"error": "Falta el archivo 'image' en la solicitud multipart/form-data"}), 400
        
    file = request.files['image']
    
    if file.mimetype not in ['image/jpeg', 'image/png']:
        log.warning(f"Tipo de archivo no permitido: {file.mimetype}")
        return jsonify({"error": "Tipo de archivo no soportado. Usar solo image/jpeg o image/png"}), 415

    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        device, face_prep, embed_gen, classifier, scaler = MODELS
        
        with torch.no_grad():
            face_tensor = face_prep(pil_image, save_path=None)
            
            if face_tensor is None:
                log.info("No se detectó rostro en la imagen.")
                return jsonify({"error": "no face detected"}), 400
                
            face_tensor = face_tensor.to(device)
            
            embedding = embed_gen(face_tensor.unsqueeze(0))
            
            embedding_np = embedding.cpu().numpy()
            scaled_embedding = scaler.transform(embedding_np)
            
            probability = classifier.predict_proba(scaled_embedding)[0, 1]

        is_me_decision = bool(probability >= VERIFY_THRESHOLD)
        final_score = float(probability)
        
        timing_ms = (time.monotonic() - start_time) * 1000

        response_data = {
            "model_version": MODEL_VERSION,
            "is_me": is_me_decision,
            "score": round(final_score, 4),
            "threshold": VERIFY_THRESHOLD,
            "timing_ms": round(timing_ms, 2)
        }
        
        log.info(f"Verificación completada: is_me={is_me_decision}, score={final_score:.4f}, time_ms={timing_ms:.2f}")
        return jsonify(response_data), 200
        
    except Image.DecompressionBombError:
        log.error("Error: Imagen demasiado grande (Decompression Bomb).")
        return jsonify({"error": "imagen corrupta o demasiado grande para procesar"}), 413
    except Exception as e:
        log.error(f"Error inesperado durante la verificación: {e}", exc_info=True)
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

if __name__ == "__main__":
    if MODELS is not None:
        log.info(f"Iniciando servidor de desarrollo en http://0.0.0.0:{API_PORT}")
        identity_app.run(host='0.0.0.0', port=API_PORT, debug=True)
    else:
        log.critical("El servidor no puede iniciarse. Revisa los logs de carga de modelos.")