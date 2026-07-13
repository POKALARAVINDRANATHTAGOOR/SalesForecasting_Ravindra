"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence
Module  : Enterprise Theme
Author  : Ravindra Nathtagoor
Version : 2.0
==============================================================
"""

from pathlib import Path
import streamlit as st

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = ROOT_DIR / "assets"

CSS_FILE = ASSETS_DIR / "custom.css"

# ---------------------------------------------------------
# Theme Colors
# ---------------------------------------------------------

BACKGROUND = "#0B1220"

SURFACE = "#111827"

CARD = "#1F2937"

PRIMARY = "#2563EB"

SECONDARY = "#3B82F6"

SUCCESS = "#22C55E"

WARNING = "#F59E0B"

DANGER = "#EF4444"

TEXT = "#F9FAFB"

MUTED = "#94A3B8"

# ---------------------------------------------------------
# Typography
# ---------------------------------------------------------

FONT = "Inter"

TITLE_SIZE = 34

SUBTITLE_SIZE = 15

HEADER_SIZE = 26

BODY_SIZE = 15

# ---------------------------------------------------------
# Icons
# ---------------------------------------------------------

ICONS = {
    "overview": "🏠",
    "forecast": "📈",
    "analytics": "📊",
    "ai": "🤖",
    "anomaly": "🚨",
    "cluster": "📦",
    "reports": "📁",
    "download": "📥",
    "settings": "⚙",
    "about": "ℹ",
    "success": "✅",
    "warning": "⚠",
    "error": "❌",
}

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

def configure_page():
    """
    Configure Streamlit page.
    Call this ONCE at the top of app.py.
    """

    st.set_page_config(
        page_title="Sales Intelligence Workspace",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ---------------------------------------------------------
# Load CSS
# ---------------------------------------------------------

def load_css():
    """
    Load enterprise CSS.
    """

    if CSS_FILE.exists():
        with open(CSS_FILE, "r", encoding="utf-8") as css:
            st.markdown(
                f"<style>{css.read()}</style>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------

PLOTLY_LAYOUT = {
    "paper_bgcolor": BACKGROUND,
    "plot_bgcolor": CARD,
    "font": {
        "family": FONT,
        "color": TEXT,
    },
    "margin": dict(l=30, r=30, t=50, b=30),
    "hovermode": "x unified",
}

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def apply_theme():
    """
    Configure page + load css.
    """

    configure_page()

    load_css()