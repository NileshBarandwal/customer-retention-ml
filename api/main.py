"""
FastAPI REST API for serving churn predictions.

Exposes a /predict endpoint that accepts customer feature data (validated via
Pydantic schemas), runs inference through the loaded model, and returns a
churn probability score along with top SHAP feature contributions.
"""
