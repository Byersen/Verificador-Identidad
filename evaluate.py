import numpy as np
import joblib
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

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
    
    # Exclude the last value where recall=0
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
    f1_scores = np.nan_to_num(f1_scores, nan=0.0)
    
    if len(f1_scores) == 0:
        return 0.5  # fallback threshold
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    return float(best_threshold)
