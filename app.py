"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Application : Streamlit Dashboard
Author : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
Interactive dashboard for

✔ Sales Analytics

✔ Demand Forecasting

✔ Anomaly Detection

✔ Product Segmentation

✔ Executive Insights

==============================================================
"""
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from src.dashboard_utils import *
from src.model_evaluation import ForecastEvaluation

from src.anomaly_detection import AnomalyDetection

from src.clustering import ProductClustering
from src.theme import apply_theme
from src.components import (
    enterprise_header,
    section_title,
    footer,
    kpi_card,
    recommendation_card,
)
from pathlib import Path

LOGO = Path("assets/logo.png")

logo = Image.open(LOGO)
apply_theme()
st.set_page_config(
    page_title="Sales Intelligence Workspace",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.image(
    logo,
    width=220
)

st.markdown(
    """
    <h1 style="text-align:center;">
    Sales Intelligence Platform
    </h1>

    <p style="text-align:center;">
    AI Powered Forecasting & Demand Intelligence
    </p>
    """,
    unsafe_allow_html=True
)
with open("assets/custom.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )
st.markdown("""
<div style='display:flex;
align-items:center;
justify-content:space-between;
padding:15px;
background:#1E293B;
border-radius:18px;'>

<div>

<h1 style='margin:0'>
Sales Forecasting &
Demand Intelligence
</h1>

<p>
Enterprise AI Analytics Platform
</p>

</div>

<div>



</div>

</div>
""",unsafe_allow_html=True)
st.caption(
    "AI-powered Business Intelligence Dashboard using SARIMA, Prophet, XGBoost, Isolation Forest and KMeans."
)
st.markdown(
"""
Professional Business Intelligence Dashboard

Forecasting Models

• SARIMA

• Prophet

• XGBoost

Machine Learning

• Isolation Forest

• KMeans Clustering

Business Analytics

• Forecasting

• Anomaly Detection

• Demand Segmentation
"""
)
sales = load_sales_data()
st.sidebar.title("Navigation")

page = option_menu(
    menu_title="Navigation",
    options=[
        "Dashboard",
        "Forecasting",
        "Anomaly Detection",
        "Demand Segmentation",
        "Reports",
        "About",
    ],
    icons=[
        "speedometer2",
        "graph-up-arrow",
        "exclamation-triangle",
        "diagram-3",
        "download",
        "info-circle",
    ],
    menu_icon="cast",
    default_index=0,
)
with st.sidebar:

    st.image(
        logo,
        width=140
    )

    st.markdown(
        """
        <h2 style="text-align:center;margin-bottom:0;">
        Sales Intelligence
        </h2>

        <p style="text-align:center;color:#94A3B8;">
        Enterprise Dashboard v1.0
        </p>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("# Sales Intelligence")

st.sidebar.caption(
    "Enterprise Dashboard v1.0"
)

st.sidebar.markdown("---")
sales = sidebar_filters(sales)
col1, col2 = st.columns([1,6])

with col1:
    st.image(logo, width=90)

with col2:
    st.markdown("""
    <h1 style='margin-bottom:0;'>
    Sales Intelligence Workspace
    </h1>

    <p style='color:#94A3B8;'>
    End-to-End Sales Forecasting & Demand Intelligence System
    </p>
    """, unsafe_allow_html=True)
if page == "Dashboard":

    section_title("📊 Executive Dashboard")

    show_kpis(sales)

    st.markdown("")

    ai_business_summary(sales)

    col1, col2 = st.columns(2)

    with col1:
        monthly_sales_chart(sales)

    with col2:
        category_sales_chart(sales)

    col3, col4 = st.columns(2)

    with col3:
        region_sales_chart(sales)

    with col4:
        top_products(sales)
elif page == "Forecasting":

    section_title("📈 Forecast Center")
    forecast_accuracy()
    st.divider()
    show_forecast_metrics()
    st.divider()
    interactive_xgboost()
    st.divider()
    future_forecast_table()
    st.divider()
    forecast_insights()
    if st.button(
        "🚀 Run Forecast Models"
    ):

        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
        evaluator = ForecastEvaluation()
        evaluator.run()
        progress.empty()

        st.toast(
            "Forecast completed successfully!",
            icon="✅"
     )
# ==========================================================
# ANOMALY DETECTION
# ==========================================================

elif page == "Anomaly Detection":

    enterprise_header()

    st.write(
        """
This module identifies unusual sales patterns
using Isolation Forest and Z-Score Detection.
"""
    )
    show_anomalies()
    st.divider()
    interactive_anomaly_chart()

    st.divider()

    anomaly_insights()

    if st.button("🚀 Run Anomaly Detection"):

        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
        evaluator = AnomalyDetection()
        evaluator.run()
        progress.empty()
        st.toast(
            "Anomaly Detection Completed",
            icon="🚨"
        )
# ==========================================================
# PRODUCT SEGMENTATION
# ==========================================================

elif page == "Demand Segmentation":

    section_title("📦 Demand Intelligence")

    show_clusters()

    st.divider()

    cluster_distribution()

    st.divider()

    cluster_scatter()

    st.divider()

    interactive_cluster_chart()

    st.divider()

    cluster_insights()

    if st.button(

        "🚀 Run Product Segmentation"

    ):

        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
        ProductClustering().run()
        progress.empty()

        st.toast(
            "Demand Segmentation Completed",
            icon="📦"
        )
elif page == "Reports":

    section_title("📑 Reports Center")

    executive_summary()

    st.divider()

    best_model_summary()

    st.divider()

    show_reports()
# ==========================================================
# DOWNLOAD CENTER
# ==========================================================

elif page == "Downloads":

    enterprise_header()

    metrics = load_csv("evaluation_results.csv")
    anomalies = load_csv("anomalies.csv")
    clusters = load_csv("clusters.csv")

    if not metrics.empty:

        download_csv(
            metrics,
            "evaluation_results.csv"
     )

    if not anomalies.empty:

        download_csv(
            anomalies,
            "anomalies.csv"
        )

    if not clusters.empty:

        download_csv(
            clusters,
            "clusters.csv"
        )

    st.success("Downloads Ready")
# ==========================================================
# ABOUT
# ==========================================================
elif page == "About":

    section_title("ℹ About")

    st.markdown("""
## 📊 End-to-End Sales Forecasting & Demand Intelligence System

An enterprise Business Intelligence platform developed for internship submission.

---

### 🚀 Technologies

- Python
- Streamlit
- Plotly
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Prophet
- SARIMA
- KMeans
- Isolation Forest

---

### 🤖 Machine Learning Modules

✅ SARIMA Forecasting

✅ Prophet Forecasting

✅ XGBoost Forecasting

✅ Model Evaluation

✅ Anomaly Detection

✅ Product Demand Segmentation

---

### 📈 Dashboard Features

- Executive Dashboard
- Forecast Center
- Business Analytics
- Anomaly Center
- Demand Intelligence
- Reports Center

---

### 👨‍💻 Developed By

**Ravindra Nathtagoor**

B.Tech Artificial Intelligence & Machine Learning

St. Martin's Engineering College

Hyderabad

---

### 📌 Version

Enterprise Dashboard v2.0

Internship Submission
""")
footer()
st.markdown("---")

col1,col2=st.columns([1,8])

with col1:
    st.image(logo,width=45)

with col2:
    st.markdown("""
    **Sales Intelligence Platform**

    Developed by **Ravindra Nathtagoor**

    Powered by **XYLOFY.AI**
    """)