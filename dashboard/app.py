"""
Streamlit interactive dashboard for Customer Retention Prediction.

Collects a customer profile through grouped sidebar controls, calls the
FastAPI /predict endpoint, and renders the result with a probability metric,
a colour-coded verdict, a horizontal bar chart of SHAP factors, and a
plain-English explanation tailored to whichever direction the model leans.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = os.environ.get("RETENTION_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Customer Retention Predictor",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------


def collect_inputs() -> dict:
    """Render the sidebar form and return the entered customer profile.

    All 19 features required by the API are gathered, grouped into four
    sections: Account Info, Services, Billing, Demographics. Defaults
    match a typical month-to-month fibre customer for quick experimentation.

    Returns:
        Dict matching the API's CustomerProfile schema.
    """
    st.sidebar.title("Customer Profile")
    st.sidebar.caption("Adjust values and click **Predict**.")

    # -- Account Info --
    st.sidebar.subheader("Account Info")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    # -- Services --
    st.sidebar.subheader("Services")
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.sidebar.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet = st.sidebar.selectbox(
        "Internet Service", ["Fiber optic", "DSL", "No"]
    )
    online_sec = st.sidebar.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_bk = st.sidebar.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )
    device_prot = st.sidebar.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )
    tech_sup = st.sidebar.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )
    stream_tv = st.sidebar.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    stream_mv = st.sidebar.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )

    # -- Billing --
    st.sidebar.subheader("Billing")
    monthly = st.sidebar.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=75.50, step=0.50
    )
    total = st.sidebar.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10_000.0, value=906.0, step=10.0
    )

    # -- Demographics --
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.radio("Gender", ["Female", "Male"], horizontal=True)
    senior = st.sidebar.radio(
        "Senior Citizen", [0, 1], horizontal=True,
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    partner = st.sidebar.radio("Partner", ["Yes", "No"], horizontal=True)
    dependents = st.sidebar.radio("Dependents", ["Yes", "No"], horizontal=True)

    return {
        "gender":           gender,
        "SeniorCitizen":    int(senior),
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           int(tenure),
        "PhoneService":     phone,
        "MultipleLines":    multi_lines,
        "InternetService":  internet,
        "OnlineSecurity":   online_sec,
        "OnlineBackup":     online_bk,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_sup,
        "StreamingTV":      stream_tv,
        "StreamingMovies":  stream_mv,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "MonthlyCharges":   float(monthly),
        "TotalCharges":     float(total),
    }


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def call_predict_api(payload: dict) -> dict | None:
    """POST the profile to the FastAPI /predict endpoint.

    Args:
        payload: Customer profile dict matching CustomerProfile.

    Returns:
        Parsed JSON response on success, None on failure (errors are
        reported in-page using st.error).
    """
    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=15
        )
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the API at `{API_URL}`. "
            "Start it with `uvicorn api.main:app --reload` and try again."
        )
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Try again or check the server logs.")
        return None

    if response.status_code == 200:
        return response.json()

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"API returned {response.status_code}: {detail}")
    return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_top_factors_chart(top_factors: list[dict]) -> plt.Figure:
    """Render the top-factors horizontal bar chart.

    Args:
        top_factors: List of dicts from the API response, each with keys
                     feature, value, impact, shap_value.

    Returns:
        A matplotlib Figure ready for st.pyplot.
    """
    items = list(reversed(top_factors))  # largest at top of horizontal bar
    labels = [f"{f['feature']} = {f['value']}" for f in items]
    values = [f["shap_value"] for f in items]
    colors = ["#DD8452" if v > 0 else "#4C72B0" for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels, values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("SHAP value (impact on churn probability)")
    ax.set_title("Top 5 Factors Driving This Prediction")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def render_english_explanation(result: dict) -> str:
    """Build a plain-English summary of the prediction.

    Args:
        result: Full API response dict.

    Returns:
        A markdown-formatted sentence summarising the call-out.
    """
    verdict = result["prediction"]
    prob = result["churn_probability"]
    confidence = result["confidence"].lower()
    factors = result["top_factors"]

    sign = "stay" if verdict == "Will Stay" else "leave"
    drivers = [
        f for f in factors
        if (f["impact"] == "decreases risk") == (sign == "stay")
    ][:3]
    driver_strs = [f"`{f['feature']}` is `{f['value']}`" for f in drivers]

    if not driver_strs:
        # Edge case: no factors aligned with the verdict direction
        return (
            f"This customer is predicted to **{sign}** "
            f"(probability {prob:.0%}, {confidence} confidence)."
        )

    if len(driver_strs) == 1:
        because = driver_strs[0]
    else:
        because = ", ".join(driver_strs[:-1]) + f", and {driver_strs[-1]}"

    return (
        f"This customer is likely to **{sign}** because {because}. "
        f"Predicted churn probability: **{prob:.1%}** ({confidence} confidence)."
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def main() -> None:
    """Top-level entry point: render the dashboard and handle interactions."""
    st.title("📊 Customer Retention Predictor")
    st.markdown(
        "Predicts whether a Telco customer is likely to churn, "
        "and explains the model's reasoning using SHAP values."
    )

    payload = collect_inputs()

    if st.sidebar.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Calling prediction API..."):
            st.session_state["result"] = call_predict_api(payload)
            st.session_state["last_payload"] = payload

    result = st.session_state.get("result")

    if result is None:
        st.info(
            "Configure the customer profile in the sidebar, then click "
            "**Predict** to see the result."
        )
        return

    # ---- Headline metric + verdict
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{result['churn_probability']:.1%}",
            delta=f"{result['confidence']} confidence",
            delta_color="off",
        )
    with col2:
        if result["prediction"] == "Will Leave":
            st.error(f"🚨 Prediction: **{result['prediction']}**")
        else:
            st.success(f"✅ Prediction: **{result['prediction']}**")

    # ---- Top factors chart
    st.divider()
    st.subheader("Why the model decided this")
    fig = render_top_factors_chart(result["top_factors"])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ---- Plain-English explanation
    st.markdown(render_english_explanation(result))

    # ---- Raw payload + response (collapsible)
    with st.expander("Show raw request / response"):
        st.json({"request": st.session_state.get("last_payload"),
                 "response": result})


if __name__ == "__main__":
    main()
