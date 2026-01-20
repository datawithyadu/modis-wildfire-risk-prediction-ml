import streamlit as st
import geemap.foliumap as geemap

from gee_utils import init_ee, get_vegetation_image, get_burn_label, get_fire_risk_prediction
from ml_results import get_model_metrics
from config import AREAS

st.set_page_config(layout="wide")
st.title("Forest Wildfire Decision Support System")
st.caption("MODIS + Google Earth Engine + Machine Learning")

@st.cache_resource
def start_ee():
    init_ee()
    return True

start_ee()

# Sidebar
st.sidebar.header("Controls")
area = st.sidebar.selectbox("Study Area", AREAS.keys())
year = st.sidebar.selectbox("Year", [2021, 2022, 2023])
model = st.sidebar.radio("ML Model", ["Random Forest", "XGBoost"])

run = st.sidebar.button("Run Analysis")

if run:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("NDVI")
        m1 = geemap.Map()
        m1.addLayer(
            get_vegetation_image(area, year).select("NDVI"),
            {"min": 0, "max": 1, "palette": ["brown", "yellow", "green"]},
            "NDVI"
        )
        m1.to_streamlit(height=350)

    with col2:
        st.subheader("Predicted Fire Risk")
        m2 = geemap.Map()
        m2.addLayer(
            get_fire_risk_prediction(area, year),
            {"min": 0, "max": 1, "palette": ["green", "red"]},
            "Fire Risk"
        )
        m2.to_streamlit(height=350)

    with col3:
        st.subheader("Burned Area")
        m3 = geemap.Map()
        m3.addLayer(
            get_burn_label(area, year),
            {"min": 0, "max": 1, "palette": ["black", "orange"]},
            "Burn Label"
        )
        m3.to_streamlit(height=350)


    st.divider()

    st.subheader("Model Performance")
    metrics = get_model_metrics(model)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", metrics["Accuracy"])
    c2.metric("AUC", metrics["AUC"])
    c3.metric("Precision", metrics["Precision"])
    c4.metric("Recall", metrics["Recall"])
    c5.metric("F1", metrics["F1"])


