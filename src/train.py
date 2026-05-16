"""
Model training entrypoint.

Loads processed data, trains Logistic Regression, Random Forest, and XGBoost
classifiers, logs all experiments to MLflow, and saves the best model (by
AUC-ROC) to models/best_model.joblib.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

EXPERIMENT_NAME = "customer-retention-prediction"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_processed_data(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature matrix and target vector from disk.

    Args:
        processed_dir: Directory containing X.csv and y.csv produced by
                       data_processing.run_pipeline().

    Returns:
        Tuple (X, y) where X is a DataFrame of features and y is a binary
        Series (0 = retained, 1 = churned).

    Raises:
        FileNotFoundError: If X.csv or y.csv are missing. Run the data
                           pipeline first: python -m src.data_processing
    """
    x_path = Path(processed_dir) / "X.csv"
    y_path = Path(processed_dir) / "y.csv"

    for p in (x_path, y_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Processed file not found: '{p}'. "
                "Run `python -m src.data_processing` first."
            )

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()  # single column → Series

    print(f"[load_processed_data] X: {X.shape}, y: {y.shape}  "
          f"(churn rate: {y.mean() * 100:.2f}%)")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Split data into train / validation / test sets (70 / 15 / 15).

    Args:
        X:            Feature DataFrame.
        y:            Target Series.
        val_size:     Fraction of total data to allocate to validation.
        test_size:    Fraction of total data to allocate to test.
        random_state: Seed for reproducibility.

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # First carve out the test set, then split the remainder into train/val
    test_frac = test_size
    val_frac = val_size / (1.0 - test_size)  # relative to the remainder

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
    )

    print(
        f"[split_data] train={len(X_train):,}  val={len(X_val):,}  "
        f"test={len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Train a Logistic Regression classifier with balanced class weights.

    Uses StandardScaler + L2 Logistic Regression in a Pipeline. Scaling is
    required because label-encoded features span different numeric ranges,
    which prevents lbfgs from converging. class_weight='balanced' compensates
    for the 73/27 churn imbalance.

    Args:
        X_train:      Training feature matrix.
        y_train:      Training target vector.
        random_state: Seed for reproducibility.

    Returns:
        Fitted sklearn Pipeline (scaler + LogisticRegression).
    """
    lr_params = {
        "C": 1.0,
        "max_iter": 2000,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "random_state": random_state,
    }
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**lr_params)),
    ])
    model.fit(X_train, y_train)
    print("[train_logistic_regression] Training complete.")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with balanced class weights.

    Uses 200 estimators with a max_depth of 10 to reduce overfitting on the
    relatively small Telco dataset. class_weight='balanced' adjusts for the
    73/27 churn imbalance automatically.

    Args:
        X_train:      Training feature matrix.
        y_train:      Training target vector.
        random_state: Seed for reproducibility.

    Returns:
        Fitted RandomForestClassifier model.
    """
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print("[train_random_forest] Training complete.")
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    """Train an XGBoost classifier with scale_pos_weight for class imbalance.

    scale_pos_weight is set to the negative/positive class ratio (~2.75),
    which is the XGBoost-native way to handle imbalanced binary targets.

    Args:
        X_train:      Training feature matrix.
        y_train:      Training target vector.
        random_state: Seed for reproducibility.

    Returns:
        Fitted XGBClassifier model.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = round(neg / pos, 4)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    print(f"[train_xgboost] Training complete. scale_pos_weight={scale_pos_weight}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """Compute classification metrics for a fitted model on a given split.

    Calculates accuracy, precision, recall, F1-score (all with threshold 0.5),
    and AUC-ROC (threshold-free, using predicted probabilities).

    Args:
        model:      Any fitted sklearn-compatible classifier.
        X:          Feature matrix for the split to evaluate.
        y:          True labels for the split.
        model_name: Human-readable label used in log output.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc_roc.
        All values are floats rounded to 4 decimal places.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y, y_prob), 4),
    }

    print(
        f"[evaluate_model] {model_name:25s} | "
        + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )
    return metrics


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiments(run_mlflow: bool = True) -> dict:
    """Train all three models, log to MLflow, print comparison, save best.

    Orchestrates the full training workflow:
        1. Load processed data and split into train / val / test.
        2. Train Logistic Regression, Random Forest, and XGBoost.
        3. Evaluate each on the validation set.
        4. Log parameters, metrics, and .joblib artifacts to MLflow.
        5. Print a formatted comparison table.
        6. Save the best model (highest val AUC-ROC) to models/best_model.joblib.

    Args:
        run_mlflow: If True (default), log runs to the MLflow tracking server.
                    Set to False to skip MLflow for fast local iteration.

    Returns:
        Dict mapping model names to their validation metrics dicts, e.g.:
        {"Logistic Regression": {"accuracy": 0.80, "auc_roc": 0.84, ...}, ...}
    """
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    X, y = load_processed_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ------------------------------------------------------------------
    # 2. MLflow setup
    # ------------------------------------------------------------------
    if run_mlflow:
        mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
        mlflow.set_experiment(EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    # 3. Model definitions
    # ------------------------------------------------------------------
    model_registry = {
        "Logistic Regression": train_logistic_regression,
        "Random Forest":       train_random_forest,
        "XGBoost":             train_xgboost,
    }

    results: dict[str, dict] = {}

    for model_name, train_fn in model_registry.items():
        print(f"\n{'=' * 55}")
        print(f"  Training: {model_name}")
        print(f"{'=' * 55}")

        model = train_fn(X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val, model_name)
        test_metrics = evaluate_model(model, X_test, y_test, f"{model_name} [test]")

        results[model_name] = {"val": val_metrics, "test": test_metrics, "model": model}

        if run_mlflow:
            with mlflow.start_run(run_name=model_name):
                # Parameters — flatten nested Pipeline params; truncate long values
                raw_params = model.get_params() if hasattr(model, "get_params") else {}
                loggable = {
                    k: str(v)[:250]
                    for k, v in raw_params.items()
                    if v is not None and not hasattr(v, "fit")
                }
                mlflow.log_params(loggable)

                # Metrics — log both val and test with prefixes
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                # Artifact — model file
                artifact_path = f"{model_name.lower().replace(' ', '_')}.joblib"
                tmp_path = PROJECT_ROOT / "models" / artifact_path
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, tmp_path)
                mlflow.log_artifact(str(tmp_path))

    # ------------------------------------------------------------------
    # 4. Comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("  MODEL COMPARISON (Validation Set)")
    print(f"{'=' * 75}")
    header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>9}"
    print(header)
    print("-" * 75)
    for name, res in results.items():
        m = res["val"]
        print(
            f"{name:<25} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['auc_roc']:>9.4f}"
        )
    print(f"{'=' * 75}")

    # ------------------------------------------------------------------
    # 5. Best model
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda n: results[n]["val"]["auc_roc"])
    best_model = results[best_name]["model"]
    best_auc = results[best_name]["val"]["auc_roc"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_path)

    print(f"\n  Best model : {best_name}")
    print(f"  Reason     : Highest validation AUC-ROC = {best_auc:.4f}")
    print(f"  Saved to   : {best_path}")
    print(f"{'=' * 75}\n")

    return {name: res["val"] for name, res in results.items()}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiments(run_mlflow=True)
