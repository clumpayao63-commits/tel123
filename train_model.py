from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "student_data.csv"
MODEL_PATH = ROOT / "student_model.pkl"
METRICS_PATH = ROOT / "model_metrics.json"

TARGET = "sla_breach"
NUMERIC_FEATURES = ["days_open", "current_workload", "submitted_hour"]
CATEGORICAL_FEATURES = ["request_type", "priority", "department"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def main():
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in student_data.csv: {missing}")

    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model_name": "Random Forest classification pipeline",
        "target": TARGET,
        "features": FEATURES,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "data_note": "Demonstration metrics from synthetic sample data; not real institutional performance."
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "precision", "recall", "f1_score"]}, indent=2))

if __name__ == "__main__":
    main()
