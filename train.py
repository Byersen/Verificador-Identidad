import numpy as np
import joblib
import json
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

MODEL_DIR = Path(os.getenv("MODEL_PATH", "models/"))
EMBEDDINGS_FILE = Path(os.getenv("EMBEDDINGS_DATA_PATH", "data/embeddings.npy"))
LABELS_FILE = Path(os.getenv("LABELS_DATA_PATH", "data/labels.csv"))
REPORTS_DIR = Path("reports/")
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

def load_data():
    """Load embeddings and labels."""
    if not EMBEDDINGS_FILE.exists() or not LABELS_FILE.exists():
        logging.error("Data files not found. Run 'scripts/embeddings.py' first.")
        return None, None
        
    X = np.load(EMBEDDINGS_FILE)
    y = np.loadtxt(LABELS_FILE, delimiter=',')
    logging.info(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    return X, y

def main():
    X, y = load_data()
    if X is None:
        return

    # 1. Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    logging.info(f"Data split: {len(y_train)} train, {len(y_val)} validation")

    # 2. Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    logging.info("Features scaled (StandardScaler).")

    # 3. Train classifier
    identity_classifier = LogisticRegression(
        max_iter=300,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        solver='liblinear'
    )
    
    logging.info("Training LogisticRegression...")
    identity_classifier.fit(X_train_scaled, y_train)
    logging.info("Training completed.")

    # 4. Validation metrics
    y_pred_val = identity_classifier.predict(X_val_scaled)
    y_proba_val = identity_classifier.predict_proba(X_val_scaled)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred_val),
        "f1_score": f1_score(y_val, y_pred_val),
        "roc_auc": roc_auc_score(y_val, y_proba_val),
        "total_validation_samples": len(y_val),
        "positive_samples_val": int(np.sum(y_val))
    }
    
    logging.info(f"Validation metrics: {json.dumps(metrics, indent=2)}")

    # 5. Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODEL_DIR / "model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    metrics_path = REPORTS_DIR / "metrics.json"
    
    joblib.dump(identity_classifier, model_path)
    joblib.dump(feature_scaler, scaler_path)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logging.info(f"Model saved to: {model_path}")
    logging.info(f"Scaler saved to: {scaler_path}")
    logging.info(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()