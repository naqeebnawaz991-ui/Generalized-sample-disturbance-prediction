import streamlit as st
import joblib
import pandas as pd

# ---------- Page config ----------
st.set_page_config(
    page_title="Sample Disturbance Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------- Load trained model (cached) ----------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

try:
    model = load_model("xgb_trained_model.joblib")
except Exception as e:
    st.error(
        f"Could not load model file 'xgb_trained_model.joblib'. Place it in the same folder as app.py.\nDetails: {e}"
    )
    st.stop()

# Feature names
FEATURES = ["AR", "normalstress", "PI", "OCR"]

# ---------- Header ----------
st.title("📊 Generalized Sample Disturbance (Sampling Quality) Assessment")
st.markdown(
    "Enter Area Ratio (AR) and other soil parameters to predict **Δe/e₀**."
)

# ---------- Sidebar Instructions ----------
st.sidebar.header("ℹ️ Instructions")
st.sidebar.write(
    """
    1. Enter Area Ratio (AR) and soil parameters.
    2. Click **Predict** to compute **Δe/e₀**.
    3. The app also shows quality categories based on OCR.
    """
)

# ---------- Inputs for Prediction ----------
st.subheader("⚙️ Input Parameters for Prediction")
col1, col2 = st.columns(2)

with col1:
    AR = st.number_input("Area Ratio (AR) [%]", min_value=0.0, value=15.0, step=0.1, format="%0.2f")
    PI = st.number_input("Plasticity Index (PI) [%]", min_value=0, value=20, step=1, format="%d")
with col2:
    normalstress = st.number_input("Effective Normal Stress (σ'ₙ) [kPa]", min_value=0.0, value=200.0, step=1.0, format="%0.2f")
    OCR = st.number_input("Overconsolidation Ratio (OCR)", min_value=0.0, value=2.0, step=0.1, format="%0.2f")

# ---------- Predict ----------
if st.button("🔮 Predict"):
    X_df = pd.DataFrame([[AR, normalstress, PI, OCR]], columns=FEATURES)
    try:
        y_pred = model.predict(X_df)[0]
    except Exception as e:
        st.error(
            f"Prediction failed. Check that your saved model was trained with features {FEATURES}.\nDetails: {e}"
        )
        st.stop()

    st.success(f"Prediction complete! Δe/e₀ value: **{y_pred:.3f}**")
    st.metric(label="Predicted Δe/e₀", value=f"{y_pred:.3f}")

    # ---------- Sample Quality Categories ----------
    st.subheader("📌 Sample Quality Category")
    if OCR < 2:
        if y_pred < 0.04:
            category = "Excellent"
        elif 0.04 <= y_pred <= 0.07:
            category = "Good to Fair"
        elif 0.07 < y_pred <= 0.14:
            category = "Poor"
        else:
            category = "Very Poor"
    else:
        if y_pred < 0.03:
            category = "Very Good to Excellent"
        elif 0.03 <= y_pred <= 0.05:
            category = "Good to Fair"
        elif 0.05 < y_pred <= 0.10:
            category = "Poor"
        else:
            category = "Very Poor"

    st.info(f"Sample Quality Category based on OCR: **{category}**")

    # Keep history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "AR [%]": AR,
        "Normal Stress σ'_n [kPa]": normalstress,
        "PI [%]": PI,
        "OCR": OCR,
        "Predicted Δe/e₀": float(y_pred),
        "Category": category
    })

# ---------- Show history & download ----------
if "history" in st.session_state and len(st.session_state.history) > 0:
    st.subheader("🧾 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download history (CSV)", data=csv, file_name="sample_quality_history.csv", mime="text/csv")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit · Model: XGBoost Regressor · Inputs: AR, σ'_n, PI, OCR")
