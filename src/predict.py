"""
Batch and single-record inference utilities.

Loads a trained model artifact from disk (via joblib/MLflow), applies the
same preprocessing pipeline used at training time, and returns churn
probability scores with optional SHAP explanations.
"""
