"""
Model evaluation and SHAP explainability.

Loads the persisted best model, evaluates it on the held-out test set,
generates standard diagnostic plots (confusion matrix, ROC, PR curve),
and explains predictions globally and locally using SHAP TreeExplainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.train import (
    MODELS_DIR,
    PROJECT_ROOT,
    load_processed_data,
    split_data,
)

PLOT_DIR = PROJECT_ROOT / "data" / "eda_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"


# ---------------------------------------------------------------------------
# Helpers for Pipeline-wrapped estimators
# ---------------------------------------------------------------------------


def _unwrap_pipeline(
    model, X: pd.DataFrame
) -> Tuple[object, pd.DataFrame]:
    """Return (final_estimator, transformed_X) so SHAP sees raw tree model.

    If *model* is a sklearn Pipeline, applies all pre-processing steps to *X*
    and returns the final estimator with the transformed feature matrix.
    For bare estimators it is a no-op.

    Args:
        model: A fitted sklearn estimator or Pipeline.
        X:     Feature DataFrame in the original (pre-pipeline) feature space.

    Returns:
        Tuple (estimator, X_transformed). X_transformed is always a DataFrame
        with the original column names preserved.
    """
    if not hasattr(model, "named_steps"):
        return model, X

    steps = list(model.named_steps.items())
    transformers, (_, final_estimator) = steps[:-1], steps[-1]

    X_t = X.copy()
    for _, step in transformers:
        X_t = step.transform(X_t)

    if not isinstance(X_t, pd.DataFrame):
        X_t = pd.DataFrame(X_t, columns=X.columns, index=X.index)

    return final_estimator, X_t


def _shap_for_positive_class(shap_values) -> np.ndarray:
    """Normalise SHAP output to a 2-D (n_samples, n_features) array for class 1.

    Different SHAP versions return different shapes for binary classification:
        - list of two arrays [class_0, class_1]    (older)
        - ndarray of shape (n_samples, n_features, 2)  (newer)
        - already 2-D                              (regression/binary-XGBoost)

    Args:
        shap_values: Raw output of ``TreeExplainer.shap_values``.

    Returns:
        2-D ndarray with one row per sample and one column per feature,
        carrying the contributions toward the positive (churn=1) class.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def _expected_value_for_positive_class(expected_value) -> float:
    """Pick the scalar base value for the positive class from any SHAP format.

    Args:
        expected_value: ``TreeExplainer.expected_value`` — may be a scalar,
                        a 2-element list, or a numpy array.

    Returns:
        A single float representing the model's baseline probability for
        the positive class.
    """
    if np.isscalar(expected_value):
        return float(expected_value)
    arr = np.atleast_1d(np.asarray(expected_value))
    return float(arr[1] if arr.size > 1 else arr[0])


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_model(path: Path = BEST_MODEL_PATH):
    """Load a serialized model from disk.

    Args:
        path: Filesystem path to a joblib-serialized scikit-learn model.

    Returns:
        The deserialized model object.

    Raises:
        FileNotFoundError: If *path* does not exist. Train the model first
                           with ``python -m src.train``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. "
            "Run `python -m src.train` first."
        )
    model = joblib.load(path)
    print(f"[load_model] Loaded {type(model).__name__} from '{path}'.")
    return model


def evaluate_on_test(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Score the model on the held-out test set and persist diagnostic plots.

    Computes accuracy / precision / recall / F1 / AUC-ROC, prints sklearn's
    classification report, and writes three PNGs to ``data/eda_plots/``:
        - confusion_matrix.png
        - roc_curve.png
        - pr_curve.png

    Args:
        model:  Fitted sklearn-compatible classifier.
        X_test: Test feature matrix.
        y_test: True binary labels for the test set.

    Returns:
        Dict of headline metrics (accuracy, precision, recall, f1, auc_roc).
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    print("\n" + "=" * 55)
    print("  TEST-SET METRICS")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<10s} : {v:.4f}")

    print("\n  Classification report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Retained", "Churned"],
                                digits=4))

    # 1. Confusion matrix ----------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix — Test Set", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. ROC curve -----------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#DD8452", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Precision-Recall curve ---------------------------------------------
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#4C72B0", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set", fontsize=12)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Diagnostic plots saved to: {PLOT_DIR.resolve()}")
    return metrics


def compute_shap_values(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, shap.TreeExplainer, pd.DataFrame]:
    """Compute SHAP values for the test set using TreeExplainer.

    Handles two complications transparently:
        1. If *model* is a sklearn Pipeline, the final tree estimator is
           extracted and *X_test* is passed through the preprocessing steps
           so the explainer sees inputs in the same space as the tree saw
           during training.
        2. SHAP output format varies across versions; this function returns
           a clean 2-D ndarray for the positive class.

    Args:
        model:   Fitted classifier (sklearn estimator or Pipeline).
        X_train: Training feature matrix (kept for API symmetry; not used
                 by ``feature_perturbation='tree_path_dependent'``).
        X_test:  Test feature matrix to explain.

    Returns:
        Tuple (shap_values_pos, explainer, X_test_transformed):
            shap_values_pos:      (n_test, n_features) ndarray, class-1 contributions.
            explainer:            The fitted TreeExplainer for downstream use.
            X_test_transformed:   X_test in the same space the explainer used.
    """
    estimator, X_test_t = _unwrap_pipeline(model, X_test)

    print(f"[compute_shap_values] Using TreeExplainer on "
          f"{type(estimator).__name__}…")
    explainer = shap.TreeExplainer(estimator)
    raw_shap = explainer.shap_values(X_test_t)
    shap_values_pos = _shap_for_positive_class(raw_shap)

    print(f"[compute_shap_values] SHAP matrix shape: {shap_values_pos.shape}")
    return shap_values_pos, explainer, X_test_t


def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Generate and save the SHAP global-importance summary plots.

    Saves two PNGs to ``data/eda_plots/``:
        - shap_summary_bar.png     (mean |SHAP| per feature)
        - shap_summary_beeswarm.png (per-sample SHAP distribution)

    Args:
        shap_values: 2-D array of SHAP values for the positive class.
        X_test:      Matching feature matrix (rows align with *shap_values*).

    Returns:
        DataFrame ranking features by mean absolute SHAP value (descending).
    """
    # Bar plot -- global mean |SHAP|
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
        color="#4C72B0",
    )
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary_bar.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm plot -- per-sample distribution
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary — Per-Sample Feature Impact", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary_beeswarm.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    importance = (
        pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    print(f"[plot_shap_summary] Saved bar + beeswarm to '{PLOT_DIR}'.")
    return importance


def explain_single_prediction(
    model,
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    index: int = 0,
) -> None:
    """Produce a local explanation for one customer in the test set.

    Prints the model's churn probability, lists the top-5 features driving
    that prediction (signed SHAP values), and writes a waterfall plot to
    ``data/eda_plots/shap_waterfall_sample.png``.

    Args:
        model:     The original (possibly Pipeline-wrapped) classifier — used
                   for ``predict_proba``.
        explainer: TreeExplainer returned by ``compute_shap_values``.
        X_test:    Test feature matrix. May be original or transformed; the
                   function unwraps Pipelines internally as needed.
        index:     Row position (0-based) of the customer to explain.
    """
    if not (0 <= index < len(X_test)):
        raise IndexError(
            f"index={index} is out of range for X_test with {len(X_test)} rows."
        )

    # Probability comes from the *full* model (so any pre-processing applies)
    prob_churn = float(model.predict_proba(X_test.iloc[[index]])[:, 1][0])

    # Need a transformed slice that matches the explainer
    _, X_test_t = _unwrap_pipeline(model, X_test)
    sample = X_test_t.iloc[[index]]
    raw_shap = explainer.shap_values(sample)
    shap_pos = _shap_for_positive_class(raw_shap)[0]
    base_value = _expected_value_for_positive_class(explainer.expected_value)

    contribution = pd.DataFrame({
        "feature":    X_test_t.columns,
        "value":      sample.iloc[0].values,
        "shap_value": shap_pos,
        "abs_shap":   np.abs(shap_pos),
    }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 55)
    print(f"  LOCAL EXPLANATION — Test row #{index}")
    print("=" * 55)
    print(f"  Churn probability : {prob_churn:.4f}")
    print(f"  Baseline (E[f(x)]): {base_value:.4f}")
    print("\n  Top-5 features driving this prediction:")
    print("  " + "-" * 51)
    print(f"  {'feature':<22}{'value':>10}{'shap':>12}{'direction':>9}")
    print("  " + "-" * 51)
    for _, row in contribution.head(5).iterrows():
        direction = "→ churn" if row["shap_value"] > 0 else "→ stay"
        print(f"  {row['feature']:<22}{row['value']:>10.3f}"
              f"{row['shap_value']:>12.4f}{direction:>9}")

    # Waterfall plot
    explanation = shap.Explanation(
        values=shap_pos,
        base_values=base_value,
        data=sample.iloc[0].values,
        feature_names=list(X_test_t.columns),
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_waterfall_sample.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Waterfall plot saved to: "
          f"{(PLOT_DIR / 'shap_waterfall_sample.png').resolve()}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_evaluation() -> dict:
    """End-to-end evaluation: metrics → diagnostics → SHAP global + local.

    Loads the saved best model, reconstructs the test split with the same
    random_state used in training, runs all evaluation functions, and prints
    a final summary table with test-set metrics and the top-5 features by
    mean absolute SHAP value.

    Returns:
        Dict of test-set metrics (accuracy, precision, recall, f1, auc_roc).
    """
    # --- Data ----------------------------------------------------------------
    X, y = load_processed_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # --- Model ---------------------------------------------------------------
    model = load_model()

    # --- Test metrics + diagnostic plots -------------------------------------
    metrics = evaluate_on_test(model, X_test, y_test)

    # --- SHAP: global --------------------------------------------------------
    shap_values, explainer, _ = compute_shap_values(model, X_train, X_test)
    importance = plot_shap_summary(shap_values, X_test)

    # --- SHAP: local on first row -------------------------------------------
    explain_single_prediction(model, explainer, X_test, index=0)

    # --- Final summary -------------------------------------------------------
    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Model        : {type(model).__name__}")
    print(f"  Test samples : {len(X_test):,}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1           : {metrics['f1']:.4f}")
    print(f"  AUC-ROC      : {metrics['auc_roc']:.4f}")
    print("\n  Top-5 features by mean |SHAP|:")
    for i, row in importance.head(5).iterrows():
        print(f"    {i + 1}. {row['feature']:<22} {row['mean_abs_shap']:.4f}")
    print("=" * 55 + "\n")

    return metrics


if __name__ == "__main__":
    run_evaluation()
