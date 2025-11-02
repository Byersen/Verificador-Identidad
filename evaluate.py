import numpy as np
import joblib
import json
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, RocCurveDisplay, 
    precision_recall_curve, PrecisionRecallDisplay,
    f1_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

MODEL_DIR = Path(os.getenv("MODEL_PATH", "models/"))
EMBEDDINGS_FILE = Path(os.getenv("EMBEDDINGS_DATA_PATH", "data/embeddings.npy"))
LABELS_FILE = Path(os.getenv("LABELS_DATA_PATH", "data/labels.csv"))
REPORTS_DIR = Path("reports/")
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

def load_artifacts_and_data():
    """Carga modelo, escalador y datos de prueba."""
    try:
        model = joblib.load(MODEL_DIR / "model.joblib")
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    except FileNotFoundError:
        logging.error("model.joblib o scaler.joblib no encontrados. Ejecuta train.py primero.")
        return None
        
    try:
        X = np.load(EMBEDDINGS_FILE)
        y = np.loadtxt(LABELS_FILE, delimiter=',')
    except FileNotFoundError:
        logging.error("Archivos de embeddings/labels no encontrados. Ejecuta embeddings.py primero.")
        return None
        
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    
    X_test_scaled = scaler.transform(X_test)
    
    logging.info(f"Artefactos cargados y datos de prueba listos ({len(y_test)} muestras).")
    return model, X_test_scaled, y_test

def find_best_threshold(y_true, y_proba):
    """Encuentra el umbral que maximiza el F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Excluir el último valor de precision/recall (1.0, 0.0)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)
    
    if len(f1_scores) == 0:
        return 0.5
        
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    return float(best_threshold)

def generate_reports(model, X_test, y_test):
    """Genera y guarda las visualizaciones de métricas."""
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 1. Encontrar umbral óptimo
    best_threshold = find_best_threshold(y_test, y_proba)
    logging.info(f"Umbral óptimo (max F1) calculado: {best_threshold:.4f}")
    
    # Usar umbral óptimo para predicciones binarias
    y_pred = (y_proba >= best_threshold).astype(int)
    
    # 2. Matriz de Confusión
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Me", "Me"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión (Umbral = {best_threshold:.2f})")
        cm_path = REPORTS_DIR / "confusion_matrix.png"
        plt.savefig(cm_path)
        logging.info(f"Matriz de confusión guardada en: {cm_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Error al generar matriz de confusión: {e}")

    # 3. Curva ROC
    try:
        display = RocCurveDisplay.from_predictions(y_test, y_proba, name="Verificador 'YO'")
        display.plot()
        plt.title("Curva ROC (Receiver Operating Characteristic)")
        roc_path = REPORTS_DIR / "roc_curve.png"
        plt.savefig(roc_path)
        logging.info(f"Curva ROC guardada en: {roc_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Error al generar curva ROC: {e}")

    # 4. Curva Precision-Recall
    try:
        display = PrecisionRecallDisplay.from_predictions(y_test, y_proba, name="Verificador 'YO'")
        display.plot()
        plt.title("Curva Precision-Recall")
        pr_path = REPORTS_DIR / "precision_recall_curve.png"
        plt.savefig(pr_path)
        logging.info(f"Curva PR guardada en: {pr_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Error al generar curva PR: {e}")

    # 5. Guardar/actualizar metrics.json
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics = {}
    try:
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
    except Exception:
        metrics = {}

    metrics['best_threshold'] = best_threshold
    metrics['evaluation_f1_score'] = f1_score(y_test, y_pred)
    metrics['evaluation_accuracy'] = float(np.mean(y_test == y_pred))
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Métricas de evaluación actualizadas en: {metrics_path}")

def main():
    artifacts = load_artifacts_and_data()
    if artifacts:
        model, X_test_scaled, y_test = artifacts
        generate_reports(model, X_test_scaled, y_test)
        logging.info("Evaluación completada. Reportes guardados.")

if __name__ == "__main__":
    main()