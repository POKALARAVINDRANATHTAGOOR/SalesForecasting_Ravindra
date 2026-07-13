"""
==========================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Configuration
Author  : Ravindra Nathtagoor
==========================================================
"""

from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Project folders
DATA_DIR = ROOT_DIR / "data"
CHARTS_DIR = ROOT_DIR / "charts"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Dataset paths
TRAIN_DATA = DATA_DIR / "train.csv"
VGSALES_DATA = DATA_DIR / "vgsales.csv"
# ==========================================================
# Forecast Configuration
# ==========================================================
FORECAST_HORIZON = 3
TEST_SIZE = 3
RANDOM_STATE = 42
# Figure settings
FIG_WIDTH = 15
FIG_HEIGHT = 7
DPI = 120