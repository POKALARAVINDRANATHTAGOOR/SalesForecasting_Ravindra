"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Dashboard Utilities
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
Reusable utility functions for Streamlit Dashboard.

Provides

✔ Data Loading
✔ KPI Cards
✔ Sidebar Filters
✔ Download Buttons
✔ Plotly Charts
✔ Cached Functions
✔ Metric Formatting

==============================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from src.config import (
    TRAIN_DATA,
    CHARTS_DIR,
    ROOT_DIR
)
from src.theme import (
    PRIMARY,
    SUCCESS,
    WARNING,
    DANGER,
    CARD,
    BACKGROUND,
    TEXT,
    PLOTLY_LAYOUT,
)

from src.components import (
    kpi_card,
    recommendation_card,
)
OUTPUT_DIR = ROOT_DIR / "outputs"
# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

@st.cache_data
def load_sales_data():

    df = pd.read_csv(TRAIN_DATA)

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    return df
@st.cache_data
def load_csv(filename):
    """
    Load any CSV file from outputs folder.
    """

    file = OUTPUT_DIR / filename

    if file.exists():
        return pd.read_csv(file)
    st.info(
        f"📂  {filename} Report not available yet. Run the corresponding module to generate it."
    )
    return pd.DataFrame()
# ==========================================================
# FORECAST DATA
# ==========================================================

@st.cache_data(show_spinner=False)
def load_xgboost_forecast():

    return load_csv("xgboost_forecast.csv")


@st.cache_data(show_spinner=False)
def load_predictions():

    return load_csv("xgboost_predictions.csv")


@st.cache_data(show_spinner=False)
def load_evaluation():

    return load_csv("evaluation_results.csv")
# ==========================================================
# INTERACTIVE XGBOOST FORECAST
# ==========================================================

def interactive_xgboost():

    prediction = load_predictions()

    future = load_xgboost_forecast()

    if prediction.empty:

        st.warning("Prediction file not found.")

        return

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=list(range(len(prediction))),

            y=prediction["Actual"],

            mode="lines+markers",

            name="Actual"

        )

    )

    fig.add_trace(

        go.Scatter(

            x=list(range(len(prediction))),

            y=prediction["Predicted"],

            mode="lines+markers",

            name="Predicted"

        )

    )

    if not future.empty:

        start = len(prediction)

        fig.add_trace(

            go.Scatter(

                x=list(
                    range(
                        start,
                        start + len(future)
                    )
                ),

                y=future["Forecast"],

                mode="lines+markers",

                name="Future Forecast",

                line=dict(
                    dash="dash"
                )

            )

        )

    fig.update_layout(

        **PLOTLY_LAYOUT,

        title="XGBoost Sales Forecast",

        xaxis_title="Months",

        yaxis_title="Sales",

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
# ==========================================================
# FORECAST ACCURACY
# ==========================================================

def forecast_accuracy():

    evaluation = load_evaluation()

    if evaluation.empty:

        return

    best = evaluation.loc[
        evaluation["RMSE"].idxmin()
    ]

    c1, c2, c3 = st.columns(3)

    with c1:

        kpi_card(

            "Best Model",

            best["Model"],

            "",

            "🏆"

        )

    with c2:

        kpi_card(

            "RMSE",

            f"{best['RMSE']:.2f}",

            "",

            "📉"

        )

    with c3:

        accuracy = 100 - best["MAPE"]

        kpi_card(

            "Accuracy",

            f"{accuracy:.2f}%",

            "",

            "🎯"

        )
# ==========================================================
# EXECUTIVE KPI DASHBOARD
# ==========================================================

def show_kpis(df):
    """
    Enterprise KPI Dashboard
    """

    total_sales = df["Sales"].sum()

    avg_sales = df["Sales"].mean()

    total_orders = df["Order ID"].nunique()

    total_customers = df["Customer ID"].nunique()

    forecast = load_csv("evaluation_results.csv")

    accuracy = "--"

    if not forecast.empty and "MAPE" in forecast.columns:

        best_mape = forecast["MAPE"].min()

        accuracy = f"{100-best_mape:.1f}%"

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        kpi_card(
            title="Revenue",
            value=f"${total_sales:,.0f}",
            change="▲ Sales",
            icon="💰"
        )

    with col2:

        kpi_card(
            title="Orders",
            value=f"{total_orders:,}",
            change="▲ Orders",
            icon="📦"
        )

    with col3:

        kpi_card(
            title="Customers",
            value=f"{total_customers:,}",
            change="▲ Customers",
            icon="👥"
        )

    with col4:

        kpi_card(
            title="Forecast Accuracy",
            value=accuracy,
            change="Best Model",
            icon="🎯"
        )
# ==========================================================
# AI BUSINESS INSIGHTS
# ==========================================================

def ai_business_summary(df):
    """
    AI Business Summary
    """

    total_sales = df["Sales"].sum()

    avg_sales = df["Sales"].mean()

    best_category = (
        df.groupby("Category")["Sales"]
        .sum()
        .idxmax()
    )

    recommendation = f"""
Total Revenue : ${total_sales:,.0f}

Average Order Value : ${avg_sales:,.2f}

Best Performing Category : {best_category}

Recommendation

Increase inventory for the highest-selling category
and continue monitoring monthly sales trends using
the forecasting models.
"""

    recommendation_card(
        "AI Business Recommendation",
        recommendation
    )
# ==========================================================
# SIDEBAR FILTERS
# ==========================================================

def sidebar_filters(df):

    st.sidebar.markdown("## 🔍 Filters")

    region = st.sidebar.multiselect(
        "Region",
        sorted(df["Region"].dropna().unique()),
        default=sorted(df["Region"].dropna().unique())
    )

    category = st.sidebar.multiselect(
        "Category",
        sorted(df["Category"].dropna().unique()),
        default=sorted(df["Category"].dropna().unique())
    )

    segment = st.sidebar.multiselect(
        "Segment",
        sorted(df["Segment"].dropna().unique()),
        default=sorted(df["Segment"].dropna().unique())
    )

    filtered = df[
        df["Region"].isin(region)
        &
        df["Category"].isin(category)
        &
        df["Segment"].isin(segment)
    ]

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Showing **{len(filtered):,}** records"
    )

    return filtered
# ==========================================================
# MONTHLY SALES TREND
# ==========================================================

def monthly_sales_chart(df):

    st.markdown("### 📈 Monthly Sales Trend")

    data = df.copy()

    data["Order Date"] = pd.to_datetime(data["Order Date"])

    monthly = (
        data
        .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        template="plotly_dark"
    )

    fig.update_traces(
        line=dict(
            color=PRIMARY,
            width=4
        ),
        marker=dict(
            size=8
        ),
        hovertemplate=
        "<b>%{x|%b %Y}</b><br>"
        "Sales: $%{y:,.0f}<extra></extra>"
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        title=None,
        xaxis_title="",
        yaxis_title="Sales ($)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# ==========================================================
# CATEGORY SALES
# ==========================================================

def category_sales_chart(df):

    st.markdown("### 📦 Category Performance")

    category = (
        df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        category,
        x="Category",
        y="Sales",
        color="Sales",
        color_continuous_scale="Blues",
        template="plotly_dark"
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        title=None,
        coloraxis_showscale=False
    )

    fig.update_traces(
        hovertemplate=
        "<b>%{x}</b><br>"
        "Sales : $%{y:,.0f}<extra></extra>"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# ==========================================================
# REGION SALES
# ==========================================================

def region_sales_chart(df):

    st.markdown("### 🌍 Regional Analysis")

    region = (
        df.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        region,
        names="Region",
        values="Sales",
        hole=.55,
        template="plotly_dark"
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        showlegend=True
    )

    fig.update_traces(
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# ==========================================================
# TOP PRODUCTS
# ==========================================================

def top_products(df):

    st.markdown("### 🏆 Top Products")

    products = (
        df.groupby("Product Name")["Sales"]
        .sum()
        .nlargest(10)
        .reset_index()
    )

    fig = px.bar(
        products,
        x="Sales",
        y="Product Name",
        orientation="h",
        color="Sales",
        color_continuous_scale="Viridis",
        template="plotly_dark"
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        title=None,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed")
    )

    fig.update_traces(
        hovertemplate=
        "<b>%{y}</b><br>"
        "$%{x:,.0f}<extra></extra>"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.divider()
    executive_insights()
# ==========================================================
# FORECAST MODEL COMPARISON
# ==========================================================

def show_forecast_metrics():

    st.markdown("### 🎯 Forecast Model Performance")

    metrics = load_csv("evaluation_results.csv")

    if metrics.empty:
        st.warning("No evaluation results found.")
        return

    # Highlight best RMSE
    best_rmse = metrics["RMSE"].min()

    def highlight_best(row):
        color = "#22C55E" if row["RMSE"] == best_rmse else ""
        return ["background-color: {}".format(color)] * len(row)

    styled = (
        metrics.style
        .format({
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "MAPE": "{:.2f}%"
        })
        .apply(highlight_best, axis=1)
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True
    )

    winner = metrics.loc[
        metrics["RMSE"].idxmin(),
        "Model"
    ]

    st.success(f"🏆 Best Forecast Model : {winner}")
# ==========================================================
# FORECAST CHARTS
# ==========================================================

def forecast_charts():

    st.markdown("## 📈 Forecast Visualizations")

    charts = [

        (
            "SARIMA Forecast",
            "sarima_forecast.png"
        ),

        (
            "Prophet Forecast",
            "prophet_forecast.png"
        ),

        (
            "XGBoost Forecast",
            "xgboost_forecast.png"
        ),

        (
            "Model Comparison",
            "forecast_model_comparison.png"
        )

    ]

    for title, image in charts:

        chart = CHARTS_DIR / image

        with st.container():

            st.markdown(
                f"### {title}"
            )

            if chart.exists():

                st.image(
                    chart,
                    use_column_width=True
                )

            else:

                st.info(
                    f"{image} not generated."
                )
# ==========================================================
# FUTURE FORECAST TABLE
# ==========================================================

def future_forecast_table():

    forecast = load_csv(
        "xgboost_forecast.csv"
    )

    if forecast.empty:
        return

    st.markdown(
        "### 📅 Next Forecast"
    )

    st.dataframe(
        forecast,
        hide_index=True,
        use_container_width=True
    )
    # ==========================================================
# FORECAST INSIGHTS
# ==========================================================

def forecast_insights():

    recommendation_card(

        "Forecast Recommendation",

        """
Use the forecast to prepare inventory for the
coming months.

Recommended Actions

• Increase stock for fast-moving products

• Monitor seasonal demand

• Review forecast monthly

• Use anomaly reports before purchasing

Forecast Confidence

High
"""
    )
# ==========================================================
# ANOMALY DASHBOARD
# ==========================================================

def show_anomalies():

    st.markdown("## 🚨 Sales Anomaly Center")

    anomalies = load_csv("anomalies.csv")

    if anomalies.empty:

        st.warning("No anomaly report found.")

        return

    total = len(anomalies)

    highest = anomalies["Sales"].max()

    average = anomalies["Sales"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:

        kpi_card(
            "Anomalies",
            total,
            "Detected Weeks",
            "🚨"
        )

    with col2:

        kpi_card(
            "Highest Sales",
            f"${highest:,.0f}",
            "Peak Week",
            "📈"
        )

    with col3:

        kpi_card(
            "Average",
            f"${average:,.0f}",
            "Average Anomaly",
            "📊"
        )

    st.markdown("---")

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomalies,
        use_container_width=True,
        hide_index=True
    )
# ==========================================================
# LOAD ANOMALY DATA
# ==========================================================

@st.cache_data(show_spinner=False)
def load_anomaly_data():

    return load_csv("anomalies.csv")
# ==========================================================
# INTERACTIVE ANOMALY CHART
# ==========================================================

def interactive_anomaly_chart():

    anomalies = load_anomaly_data()

    if anomalies.empty:
        return

    if "Date" not in anomalies.columns:
        st.info("Date column not available.")
        return

    anomalies["Date"] = pd.to_datetime(anomalies["Date"])

    fig = px.scatter(

        anomalies,

        x="Date",

        y="Sales",

        color="Sales",

        size="Sales",

        hover_data=anomalies.columns,

        template="plotly_dark",

        color_continuous_scale="Reds"

    )

    fig.update_layout(

        **PLOTLY_LAYOUT,

        title="Sales Anomaly Timeline",

        height=550

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# ==========================================================
# ANOMALY TIMELINE
# ==========================================================

def anomaly_timeline():

    anomalies = load_csv("anomalies.csv")

    if anomalies.empty:
        return

    if "Date" not in anomalies.columns:
        return

    anomalies["Date"] = pd.to_datetime(anomalies["Date"])

    fig = px.scatter(

        anomalies,

        x="Date",

        y="Sales",

        color="Sales",

        size="Sales",

        template="plotly_dark",

        color_continuous_scale="Reds"

    )

    fig.update_layout(

        **PLOTLY_LAYOUT,

        title="Weekly Sales Anomalies",

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
    # ==========================================================
# ANOMALY INSIGHTS
# ==========================================================

def anomaly_insights():

    recommendation_card(

        "Business Risk Assessment",

        """
Potential causes

• Festival demand

• Flash sales

• Inventory shortage

• Supply chain delays

Recommended Actions

✔ Review abnormal weeks

✔ Increase inventory

✔ Validate promotions

✔ Compare against forecasts

Overall Risk

MEDIUM
"""
    )
# ==========================================================
# LOAD CLUSTERS
# ==========================================================

@st.cache_data(show_spinner=False)
def load_cluster_data():

    return load_csv("clusters.csv")
# ==========================================================
# DEMAND SEGMENTATION DASHBOARD
# ==========================================================

def show_clusters():

    st.markdown("## 📦 Product Demand Intelligence")

    clusters = load_csv("clusters.csv")

    if clusters.empty:

        st.warning("No clustering results found.")

        return

    total_products = len(clusters)

    total_clusters = clusters["Cluster"].nunique()

    top_cluster = (
        clusters["Cluster"]
        .value_counts()
        .idxmax()
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        kpi_card(
            "Products",
            total_products,
            "Clustered",
            "📦"
        )

    with c2:

        kpi_card(
            "Clusters",
            total_clusters,
            "Demand Groups",
            "🎯"
        )

    with c3:

        kpi_card(
            "Largest Cluster",
            top_cluster,
            "Most Products",
            "🏆"
        )

    st.markdown("---")

    st.subheader("Cluster Summary")

    st.dataframe(

        clusters,

        hide_index=True,

        use_container_width=True

    )
# ==========================================================
# INTERACTIVE PCA
# ==========================================================

def interactive_cluster_chart():

    clusters = load_cluster_data()

    if clusters.empty:
        return

    required = [

        "PCA1",

        "PCA2",

        "Cluster"

    ]

    if not all(col in clusters.columns for col in required):

        st.info("PCA columns unavailable.")

        return

    fig = px.scatter(

        clusters,

        x="PCA1",

        y="PCA2",

        color="Cluster",

        size_max=18,

        hover_data=clusters.columns,

        template="plotly_dark"

    )

    fig.update_layout(

        **PLOTLY_LAYOUT,

        title="Demand Intelligence",

        height=550

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
# ==========================================================
# CLUSTER DISTRIBUTION
# ==========================================================

def cluster_distribution():

    clusters = load_csv("clusters.csv")

    if clusters.empty:

        return

    counts = (

        clusters["Cluster"]

        .value_counts()

        .sort_index()

        .reset_index()

    )

    counts.columns = [

        "Cluster",

        "Products"

    ]

    fig = px.bar(

        counts,

        x="Cluster",

        y="Products",

        color="Products",

        template="plotly_dark",

        color_continuous_scale="Blues"

    )

    fig.update_layout(

        **PLOTLY_LAYOUT,

        title="Cluster Distribution",

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
# ==========================================================
# PCA CLUSTER PLOT
# ==========================================================

def cluster_scatter():

    clusters = load_csv("clusters.csv")

    if clusters.empty:

        return

    needed = [

        "PCA1",

        "PCA2",

        "Cluster"

    ]

    if not all(col in clusters.columns for col in needed):

        st.info(

            "PCA columns not available."

        )

        return

    fig = px.scatter(

        clusters,

        x="PCA1",

        y="PCA2",

        color="Cluster",

        hover_data=clusters.columns,

        template="plotly_dark"

    )

    fig.update_layout(

        **PLOTLY_LAYOUT,

        height=500,

        title="Demand Segmentation"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
# ==========================================================
# CLUSTER INSIGHTS
# ==========================================================

def cluster_insights():

    recommendation_card(

        "Demand Intelligence",

        """
High Demand Products

Increase stock

Medium Demand

Monitor inventory

Low Demand

Avoid overstocking

Growing Products

Increase marketing

Overall Recommendation

Use demand clusters together with
forecasting before purchasing inventory.
"""
    )
# ==========================================================
# REPORT CENTER
# ==========================================================

def show_reports():

    st.markdown("## 📑 Executive Reports")

    reports = {
        "Forecast Evaluation": "evaluation_results.csv",
        "Forecast Results": "xgboost_forecast.csv",
        "Anomaly Report": "anomalies.csv",
        "Demand Segments": "clusters.csv"
    }

    for title, file in reports.items():

        data = load_csv(file)

        with st.container():

            st.markdown(f"### 📄 {title}")

            if data.empty:

                st.warning(f"{file} not found.")

            else:

                st.success(f"{len(data)} records available")

                st.dataframe(
                    data.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                csv = data.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label=f"⬇ Download {file}",
                    data=csv,
                    file_name=file,
                    mime="text/csv",
                    key=file
                )

        st.markdown("---")
# ==========================================================
# BEST MODEL SUMMARY
# ==========================================================

def best_model_summary():

    results = load_csv("evaluation_results.csv")

    if results.empty:

        return

    best = results.loc[
        results["RMSE"].idxmin()
    ]

    recommendation_card(

        "🏆 Best Forecast Model",

        f"""
Model : {best['Model']}

RMSE : {best['RMSE']:.2f}

MAE : {best['MAE']:.2f}

MAPE : {best['MAPE']:.2f}%

Recommendation

Deploy this model for production forecasting.
"""
    )
# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

def executive_summary():

    metrics = load_csv("evaluation_results.csv")

    anomalies = load_csv("anomalies.csv")

    clusters = load_csv("clusters.csv")

    col1, col2, col3 = st.columns(3)

    with col1:

        kpi_card(
            "Forecast Models",
            len(metrics),
            "Evaluated",
            "📈"
        )

    with col2:

        kpi_card(
            "Anomalies",
            len(anomalies),
            "Detected",
            "🚨"
        )

    with col3:

        kpi_card(
            "Demand Segments",
            clusters["Cluster"].nunique()
            if not clusters.empty else 0,
            "Clusters",
            "📦"
        )
# ---------------------------------------------------------
# Download CSV
# ---------------------------------------------------------
def download_csv(df, filename):
    """
    Create download button.
    """

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )
# ==========================================================
# EXECUTIVE INSIGHTS
# ==========================================================

def executive_insights():

    recommendation_card(

        "Executive AI Insights",

        """
📈 Sales Forecast

Demand expected to remain stable.

📦 Inventory

Increase stock for high-demand products.

🚨 Risk

Monitor anomaly weeks before replenishment.

🤖 Best Model

Deploy the model with the lowest RMSE.

📊 Overall Status

Healthy
"""
    )
# ---------------------------------------------------------
# Section Header
# ---------------------------------------------------------

def section_title(title):
    """
    Dashboard section title.
    """

    st.markdown("---")

    st.subheader(title)
# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

def footer():

    st.markdown("---")

    st.markdown(
        """
<div style="text-align:center;color:#94A3B8">

Sales Forecasting & Demand Intelligence Platform

<br>

Powered by

Python • Streamlit • Plotly • XGBoost • Prophet • SARIMA

<br><br>

Developed by <b>Ravindra Nathtagoor</b>

<br>

Enterprise Dashboard Version 1.0

<br>

© 2026 All Rights Reserved

</div>
""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
"""
Enterprise Dashboard

Version 1.0

Developed by

Ravindra Nathtagoor
"""
    )
# ---------------------------------------------------------
# Dashboard Theme
# ---------------------------------------------------------

def page_configuration():
    """
    Configure Streamlit page.
    """

    st.set_page_config(

        page_title="Sales Forecasting Dashboard",

        page_icon="📈",

        layout="wide",

        initial_sidebar_state="expanded"

    )
if __name__ == "__main__":
    print("=" * 60)
    print("Dashboard Utilities Loaded Successfully")
    print("=" * 60)

    print(load_csv("evaluation_results.csv").head())