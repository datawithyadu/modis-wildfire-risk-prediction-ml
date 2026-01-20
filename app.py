import streamlit as st
import geemap.foliumap as geemap

from gee_utils import (
    init_ee,
    get_vegetation_image,
    get_burn_label,
    get_fire_risk_prediction,
)
from ml_results import get_model_metrics
from config import AREAS


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(layout="wide")
st.title("🔥 Forest Wildfire Decision Support System")
st.caption("MODIS • Google Earth Engine • Machine Learning")


# ----------------------------
# Earth Engine safe initialization
# ----------------------------
@st.cache_resource
def start_ee():
    """
    Initialize Earth Engine.
    Returns True if EE is available, False otherwise.
    """
    return init_ee()


ee_ready = start_ee()


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

area = st.sidebar.selectbox("Study Area", list(AREAS.keys()))
year = st.sidebar.selectbox("Year", [2021, 2022, 2023])
model = st.sidebar.radio("ML Model", ["Random Forest", "XGBoost"])

run = st.sidebar.button("Run Analysis")


# ----------------------------
# SAFE MODE (Cloud protection)
# ----------------------------
if run and not ee_ready:
    st.warning(
        "⚠️ **Google Earth Engine authentication is not available on "
        "Streamlit Community Cloud.**\n\n"
        "This app runs in **demo mode** on the cloud.\n\n"
        "👉 To view satellite maps and predictions, please run the app **locally**."
    )
    st.stop()


# ----------------------------
# Main analysis
# ----------------------------
if run:
    col1, col2, col3 = st.columns(3)

    # ---- NDVI ----
    with col1:
        st.subheader("🌿 NDVI")
        m1 = geemap.Map()
        m1.addLayer(
            get_vegetation_image(area, year).select("NDVI"),
            {"min": 0, "max": 1, "palette": ["brown", "yellow", "green"]},
            "NDVI",
        )
        m1.to_streamlit(height=350)

    # ---- Fire Risk Prediction ----
    with col2:
        st.subheader("🔥 Predicted Fire Risk")
        m2 = geemap.Map()
        m2.addLayer(
            get_fire_risk_prediction(area, year),
            {"min": 0, "max": 1, "palette": ["green", "red"]},
            "Fire Risk",
        )
        m2.to_streamlit(height=350)

    # ---- Burned Area ----
    with col3:
        st.subheader("✅ Burned Area (MODIS)")
        m3 = geemap.Map()
        m3.addLayer(
            get_burn_label(area, year),
            {"min": 0, "max": 1, "palette": ["black", "orange"]},
            "Burn Label",
        )
        m3.to_streamlit(height=350)

    # ----------------------------
    # Model performance
    # ----------------------------
    st.divider()
    st.subheader("📊 Model Performance")

    metrics = get_model_metrics(model)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", metrics["Accuracy"])
    c2.metric("AUC", metrics["AUC"])
    c3.metric("Precision", metrics["Precision"])
    c4.metric("Recall", metrics["Recall"])
    c5.metric("F1", metrics["F1"])

