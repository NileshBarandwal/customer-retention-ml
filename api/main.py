"""
FastAPI REST API for serving churn predictions.

Exposes:
    GET  /health   — liveness probe + loaded model info
    POST /predict  — accepts a customer profile, returns churn probability,
                     human-readable prediction, confidence band, and the
                     top-5 SHAP factors driving the decision.

Model and SHAP explainer are loaded once at startup via a FastAPI lifespan
context so that requests are not blocked by repeated I/O.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"

# Column order MUST match what the model was trained on.
FEATURE_ORDER: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Categorical → integer mappings. These match sklearn LabelEncoder's
# alphabetical sort behaviour used during training (see data_processing.py).
CATEGORY_MAPPINGS: dict[str, dict[str, int]] = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "MultipleLines":    {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity":   {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup":     {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies":  {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)":   1,
        "Electronic check":          2,
        "Mailed check":              3,
    },
}

# In-memory model store, populated at startup by the lifespan handler.
state: dict = {"model": None, "explainer": None, "model_name": None}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CustomerProfile(BaseModel):
    """Validated customer record submitted to /predict.

    Categorical fields use Literal types to constrain input to known values;
    invalid categories produce a 422 response automatically. Numeric ranges
    were derived from the Telco dataset (tenure 0-72 months, monthly charges
    \\$18-\\$120, total charges \\$0-\\$9000).
    """

    # Demographics
    gender:        Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner:       Literal["No", "Yes"]
    Dependents:    Literal["No", "Yes"]

    # Account
    tenure:           int = Field(..., ge=0, le=72, description="Months as customer")
    Contract:         Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod:    Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    # Services
    PhoneService:     Literal["No", "Yes"]
    MultipleLines:    Literal["No", "No phone service", "Yes"]
    InternetService:  Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:   Literal["No", "No internet service", "Yes"]
    OnlineBackup:     Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport:      Literal["No", "No internet service", "Yes"]
    StreamingTV:      Literal["No", "No internet service", "Yes"]
    StreamingMovies:  Literal["No", "No internet service", "Yes"]

    # Billing
    MonthlyCharges: float = Field(..., ge=0.0, le=200.0)
    TotalCharges:   float = Field(..., ge=0.0, le=10_000.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.50,
                "TotalCharges": 906.00,
            }
        }
    }


class FactorImpact(BaseModel):
    """One row of the top-factors response: a feature and its direction."""

    feature: str = Field(..., description="Feature name")
    value:   str = Field(..., description="Value of this feature for the customer")
    impact:  Literal["increases risk", "decreases risk"]
    shap_value: float = Field(..., description="Signed SHAP contribution")


class PredictionResponse(BaseModel):
    """Schema returned by POST /predict."""

    churn_probability: float = Field(..., ge=0.0, le=1.0)
    prediction:        Literal["Will Leave", "Will Stay"]
    confidence:        Literal["High", "Medium", "Low"]
    top_factors:       list[FactorImpact]


class HealthResponse(BaseModel):
    """Schema returned by GET /health."""

    status:    Literal["ok", "degraded"]
    model:     str | None
    explainer: bool
    features:  int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_profile(profile: CustomerProfile) -> pd.DataFrame:
    """Convert a validated CustomerProfile into a one-row encoded DataFrame.

    Categorical strings are mapped to the integer codes used during training;
    numerical fields pass through unchanged. Column order matches FEATURE_ORDER
    exactly so the model receives features in the expected positions.

    Args:
        profile: A validated CustomerProfile instance.

    Returns:
        A single-row DataFrame with 19 columns ready for model.predict().
    """
    raw = profile.model_dump()
    encoded: dict = {}
    for feat in FEATURE_ORDER:
        val = raw[feat]
        if feat in CATEGORY_MAPPINGS:
            encoded[feat] = CATEGORY_MAPPINGS[feat][val]
        else:
            encoded[feat] = val
    return pd.DataFrame([encoded], columns=FEATURE_ORDER)


def _confidence_band(prob: float) -> Literal["High", "Medium", "Low"]:
    """Bucket a churn probability into a qualitative confidence band.

    Distance from 0.5 is used as a proxy for model conviction:
        ≥ 0.30 → High   (prob ≤ 0.20 or ≥ 0.80)
        ≥ 0.15 → Medium (0.20 < prob ≤ 0.35 or 0.65 ≤ prob < 0.80)
        otherwise → Low (model is near its decision boundary)

    Args:
        prob: Predicted churn probability in [0, 1].

    Returns:
        One of "High", "Medium", "Low".
    """
    distance = abs(prob - 0.5)
    if distance >= 0.30:
        return "High"
    if distance >= 0.15:
        return "Medium"
    return "Low"


def _shap_values_class1(raw_shap) -> np.ndarray:
    """Reduce SHAP output to a 1-D array of class-1 contributions per feature."""
    if isinstance(raw_shap, list):
        arr = np.asarray(raw_shap[1])
    else:
        arr = np.asarray(raw_shap)
    # Single-row input → squeeze sample dimension
    if arr.ndim == 3:           # (1, n_features, n_classes)
        return arr[0, :, 1]
    if arr.ndim == 2:           # (1, n_features)
        return arr[0]
    return arr                   # already 1-D


def _format_value(feature: str, raw_value) -> str:
    """Render a feature value for the API response in human-readable form.

    Profile values arrive already human-readable (validated by Pydantic),
    so categorical fields are stringified as-is and only floats get a fixed
    two-decimal format for tidy display.
    """
    if isinstance(raw_value, float):
        return f"{raw_value:.2f}"
    return str(raw_value)


def compute_top_factors(
    explainer: shap.TreeExplainer,
    X_encoded: pd.DataFrame,
    profile_raw: dict,
    k: int = 5,
) -> list[FactorImpact]:
    """Compute the top-*k* SHAP-ranked drivers of a single prediction.

    Args:
        explainer:   Pre-loaded shap.TreeExplainer.
        X_encoded:   1-row encoded feature DataFrame passed to the model.
        profile_raw: Original (un-encoded) profile dict, used for display.
        k:           Number of factors to return (default 5).

    Returns:
        List of FactorImpact entries sorted by |SHAP value| descending.
    """
    raw_shap = explainer.shap_values(X_encoded)
    shap_pos = _shap_values_class1(raw_shap)

    pairs = sorted(
        zip(X_encoded.columns, shap_pos),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:k]

    out: list[FactorImpact] = []
    for feature, sv in pairs:
        out.append(FactorImpact(
            feature=feature,
            value=_format_value(feature, profile_raw[feature]),
            impact="increases risk" if sv > 0 else "decreases risk",
            shap_value=round(float(sv), 4),
        ))
    return out


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and SHAP explainer once when the app boots.

    Both artifacts are required; a missing file is a hard startup failure
    so the operator notices immediately rather than at first request.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at '{MODEL_PATH}'. "
            "Run `python -m src.train` to generate it."
        )
    if not EXPLAINER_PATH.exists():
        raise RuntimeError(
            f"SHAP explainer not found at '{EXPLAINER_PATH}'. "
            "Re-run `python -m src.train` after pulling the latest code."
        )

    state["model"] = joblib.load(MODEL_PATH)
    state["explainer"] = joblib.load(EXPLAINER_PATH)
    state["model_name"] = type(state["model"]).__name__
    print(f"[startup] Loaded {state['model_name']} and SHAP explainer.")
    yield
    state.clear()
    print("[shutdown] Cleared model state.")


app = FastAPI(
    title="Customer Retention Prediction API",
    description=(
        "Predicts churn probability for telco customers and explains the "
        "decision using SHAP values."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Streamlit, local dev — relax for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe — returns API status and model info."""
    return HealthResponse(
        status="ok" if state.get("model") is not None else "degraded",
        model=state.get("model_name"),
        explainer=state.get("explainer") is not None,
        features=len(FEATURE_ORDER),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(profile: CustomerProfile) -> PredictionResponse:
    """Score a single customer profile.

    Returns the churn probability, a human-readable verdict ("Will Leave" /
    "Will Stay"), a confidence band derived from the probability's distance
    from 0.5, and the top-5 SHAP factors that drove the decision (each
    annotated as "increases risk" or "decreases risk").

    Args:
        profile: Validated customer feature record.

    Returns:
        A PredictionResponse object.
    """
    model = state.get("model")
    explainer = state.get("explainer")
    if model is None or explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /health.",
        )

    try:
        X = encode_profile(profile)
        prob = float(model.predict_proba(X)[0, 1])
        top_factors = compute_top_factors(explainer, X, profile.model_dump())
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown category value: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    return PredictionResponse(
        churn_probability=round(prob, 4),
        prediction="Will Leave" if prob >= 0.5 else "Will Stay",
        confidence=_confidence_band(prob),
        top_factors=top_factors,
    )


@app.get("/", tags=["meta"])
def root() -> dict:
    """Tiny landing page — points to the OpenAPI docs."""
    return {
        "service": "Customer Retention Prediction API",
        "docs":    "/docs",
        "health":  "/health",
    }
