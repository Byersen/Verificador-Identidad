import numpy as np
import joblib
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, f1_score
)
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

MODEL_DIR = Path(os.getenv("MODEL_PATH", "models/"))
EMBEDDINGS_FILE = Path(os.getenv("EMBEDDINGS_DATA_PATH", "data/embeddings.npy"))
LABELS_FILE = Path(os.getenv("LABELS_DATA_PATH", "data/labels.csv"))
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

def load_artifacts_and_data():
    """Load trained model, scaler, and test data."""
    try:
        model = joblib.load(MODEL_DIR / "model.joblib")
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    except FileNotFoundError:
        logging.error("Model or scaler not found. Please run train.py first.")
        return None
        
    try:
        X = np.load(EMBEDDINGS_FILE)
        y = np.loadtxt(LABELS_FILE, delimiter=',')
    except FileNotFoundError:
        logging.error("Embeddings or labels not found. Please run embeddings.py first.")
        return None
        
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    X_test_scaled = scaler.transform(X_test)
    logging.info(f"Loaded artifacts and test data ({len(y_test)} samples).")
    return model, X_test_scaled, y_test


def find_best_threshold(y_true, y_proba):
    """Find the probability threshold that maximizes the F1-score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)
    
    if len(f1_scores) == 0:
        return 0.5
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    return float(best_threshold)


REPORTS_DIR = Path("reports/")


def generate_reports(model, X_test, y_test):
    """Generate evaluation reports and visualizations."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    best_threshold = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_threshold).astype(int)
    
    # Confusion Matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Me", "Me"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (Threshold = {best_threshold:.2f})")
        plt.savefig(REPORTS_DIR / "confusion_matrix.png")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to create confusion matrix: {e}")

    # ROC Curve
    try:
        display = RocCurveDisplay.from_predictions(y_test, y_proba, name="Identity Verifier")
        display.plot()
        plt.title("ROC Curve")
        plt.savefig(REPORTS_DIR / "roc_curve.png")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to create ROC curve: {e}")

    # Precision-Recall Curve
    try:
        display = PrecisionRecallDisplay.from_predictions(y_test, y_proba, name="Identity Verifier")
        display.plot()
        plt.title("Precision-Recall Curve")
        plt.savefig(REPORTS_DIR / "precision_recall_curve.png")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to create PR curve: {e}")

    # Metrics
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}
    
    metrics["best_threshold"] = best_threshold
    metrics["evaluation_f1_score"] = f1_score(y_test, y_pred)
    metrics["evaluation_accuracy"] = float(np.mean(y_test == y_pred))
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logging.info("All reports generated and metrics updated.")


def main():
    artifacts = load_artifacts_and_data()
    if artifacts:
        model, X_test_scaled, y_test = artifacts
        generate_reports(model, X_test_scaled, y_test)
        logging.info("Evaluation completed. Reports saved successfully.")


if __name__ == "__main__":
    main()
