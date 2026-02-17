import streamlit as st
import folium
from streamlit_folium import st_folium
import ee

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
        "⚠️ Google Earth Engine authentication is not available on "
        "Streamlit Community Cloud.\n\n"
        "Run locally to view satellite layers."
    )
    st.stop()


# ----------------------------
# Helper function to add EE layer
# ----------------------------
def add_ee_layer(map_object, ee_image, vis_params, layer_name):
    map_id = ee.Image(ee_image).getMapId(vis_params)
    folium.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=layer_name,
        overlay=True,
        control=True,
    ).add_to(map_object)


# ----------------------------
# Main analysis
# ----------------------------
if run:

    col1, col2, col3 = st.columns(3)

    # ---- NDVI ----
    with col1:
        st.subheader("🌿 NDVI")
        m1 = folium.Map(location=[55, -125], zoom_start=5)

        add_ee_layer(
            m1,
            get_vegetation_image(area, year).select("NDVI"),
            {"min": 0, "max": 1, "palette": ["brown", "yellow", "green"]},
            "NDVI",
        )

        st_folium(m1, width=350, height=350)

    # ---- Fire Risk Prediction ----
    with col2:
        st.subheader("🔥 Predicted Fire Risk")
        m2 = folium.Map(location=[55, -125], zoom_start=5)

        add_ee_layer(
            m2,
            get_fire_risk_prediction(area, year),
            {"min": 0, "max": 1, "palette": ["green", "red"]},
            "Fire Risk",
        )

        st_folium(m2, width=350, height=350)

    # ---- Burned Area ----
    with col3:
        st.subheader("✅ Burned Area (MODIS)")
        m3 = folium.Map(location=[55, -125], zoom_start=5)

        add_ee_layer(
            m3,
            get_burn_label(area, year),
            {"min": 0, "max": 1, "palette": ["black", "orange"]},
            "Burn Label",
        )

        st_folium(m3, width=350, height=350)

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
