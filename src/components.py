"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence
Module  : Enterprise UI Components
Author  : Ravindra Nathtagoor
Version : 2.0
==============================================================
"""

from datetime import datetime
import streamlit as st

from src.theme import (
    PRIMARY,
    SUCCESS,
    WARNING,
    DANGER,
    TEXT,
)

# ==========================================================
# Enterprise Header
# ==========================================================

def enterprise_header():

    today = datetime.now().strftime("%d %B %Y")

    st.markdown(
        f"""
<div class="enterprise-header">

<div class="enterprise-title">
📊 Sales Intelligence Workspace
</div>

<div class="enterprise-subtitle">

AI Powered Forecasting • Demand Intelligence • Machine Learning

<br>

Last Updated : {today}

</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# Section Title
# ==========================================================

def section_title(title):

    st.markdown(
        f"""
<div class="section-title">
{title}
</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# KPI Card
# ==========================================================

def kpi_card(title, value, change="", icon="📈"):

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-title">
{icon} {title}
</div>

<div class="metric-value">
{value}
</div>

<div class="metric-change">
{change}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# Status Badge
# ==========================================================

def status_badge(text, status="success"):

    colors = {
        "success": SUCCESS,
        "warning": WARNING,
        "danger": DANGER,
        "primary": PRIMARY,
    }

    color = colors.get(status, PRIMARY)

    st.markdown(
        f"""
<span style="
background:{color};
padding:6px 12px;
border-radius:20px;
font-size:13px;
font-weight:600;
color:white;">
{text}
</span>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# Glass Card
# ==========================================================

def glass_card(title, body):

    st.markdown(
        f"""
<div class="glass-card">

<h4>{title}</h4>

<p>{body}</p>

</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# Recommendation Card
# ==========================================================

def recommendation_card(title, recommendation):

    st.markdown(
        f"""
<div class="glass-card">

<h3>🤖 {title}</h3>

<p style="font-size:16px;">
{recommendation}
</p>

</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================================
# Loading
# ==========================================================

def loading(message="Processing..."):

    with st.spinner(message):
        pass


# ==========================================================
# Success Message
# ==========================================================

def success(message):

    st.success(f"✅ {message}")


# ==========================================================
# Warning Message
# ==========================================================

def warning(message):

    st.warning(f"⚠ {message}")


# ==========================================================
# Error Message
# ==========================================================

def error(message):

    st.error(f"❌ {message}")


# ==========================================================
# Divider
# ==========================================================

def divider():

    st.markdown("---")


# ==========================================================
# Footer
# ==========================================================

def footer():

    st.markdown(
        """
<div class="footer">

Sales Forecasting & Demand Intelligence System

<br>

Powered by

Python • Streamlit • Plotly • SARIMA • Prophet • XGBoost

<br><br>

Developed by <b>Ravindra Nathtagoor</b>

<br>

Version 1.0 Enterprise Edition

</div>
""",
        unsafe_allow_html=True,
    )