"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Sales Anomaly Detection
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
This module detects abnormal sales behaviour using two methods:

1. Isolation Forest
2. Z-Score Detection

Features
--------
✔ Weekly Sales Aggregation
✔ Isolation Forest Detection
✔ Z-Score Detection
✔ Comparison of Methods
✔ Automatic Chart Generation
✔ CSV Export
✔ Business Insights
✔ Streamlit Ready

==============================================================
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

from src.feature_engineering import FeatureEngineering
from src.config import (
    CHARTS_DIR,
    ROOT_DIR,
    RANDOM_STATE
)

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")


class AnomalyDetection:
    """
    Professional Sales Anomaly Detection

    Detects unusual sales patterns using:

    • Isolation Forest
    • Z-Score

    Saves

    • Charts
    • CSV Reports
    • Business Summary
    """

    def __init__(self):

        feature = FeatureEngineering()

        self.weekly = feature.weekly_sales()

        self.weekly = self.weekly.rename(
            columns={
                "Order Date": "Date"
            }
        )

        self.output_dir = ROOT_DIR / "outputs"

        self.output_dir.mkdir(
            exist_ok=True
        )

        self.weekly.set_index(
            "Date",
            inplace=True
        )

        self.iforest = None

        self.results = self.weekly.copy()

        print("=" * 70)
        print("ANOMALY DETECTION MODULE")
        print("=" * 70)

        print(f"Weekly Records : {len(self.results)}")
    # ---------------------------------------------------------
    # Dataset Information
    # ---------------------------------------------------------

    def dataset_summary(self):
        """
        Display dataset information.
        """

        print("\n")
        print("=" * 70)
        print("WEEKLY SALES SUMMARY")
        print("=" * 70)

        print(f"Rows : {self.results.shape[0]}")

        print(f"Columns : {self.results.shape[1]}")

        print("\n")

        print(self.results.head())

        print("\n")

        print(self.results.describe())
    # ---------------------------------------------------------
    # Prepare Features
    # ---------------------------------------------------------

    def prepare_features(self):
        """
        Prepare numerical features
        for anomaly detection.
        """

        self.results["Sales"] = (
            self.results["Sales"]
            .astype(float)
        )

        self.features = self.results[
            ["Sales"]
        ]

        print("\nFeatures Prepared Successfully")
    # ---------------------------------------------------------
    # Isolation Forest
    # ---------------------------------------------------------
    
    def run_isolation_forest(self):
        """
        Detect anomalies using Isolation Forest.
        """
    
        print("\n" + "=" * 70)
        print("ISOLATION FOREST")
        print("=" * 70)
    
        self.iforest = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=RANDOM_STATE
        )
    
        self.results["IF_Prediction"] = self.iforest.fit_predict(
            self.features
        )
    
        self.results["IF_Anomaly"] = (
            self.results["IF_Prediction"] == -1
        )
    
        anomaly_count = self.results["IF_Anomaly"].sum()
    
        print(f"Anomalies Detected : {anomaly_count}")
    
        return self.results
    # ---------------------------------------------------------
    # Isolation Scores
    # ---------------------------------------------------------
    
    def isolation_scores(self):
        """
        Calculate anomaly scores.
        """
    
        self.results["IF_Score"] = self.iforest.decision_function(
            self.features
        )
    
        print("\nIsolation Scores Calculated")
    
        return self.results
    # ---------------------------------------------------------
    # Plot Isolation Forest
    # ---------------------------------------------------------
    
    def plot_isolation_forest(self):
        """
        Plot anomalies detected using
        Isolation Forest.
        """
    
        plt.figure(figsize=(16,6))
    
        plt.plot(
            self.results.index,
            self.results["Sales"],
            linewidth=2,
            label="Weekly Sales"
        )
    
        anomalies = self.results[
            self.results["IF_Anomaly"]
        ]
    
        plt.scatter(
            anomalies.index,
            anomalies["Sales"],
            color="red",
            s=80,
            marker="o",
            label="Anomaly"
        )
    
        plt.title("Isolation Forest Anomaly Detection")
    
        plt.xlabel("Date")
    
        plt.ylabel("Sales")
    
        plt.legend()
    
        plt.grid(True)
    
        plt.tight_layout()
    
        plt.savefig(
            CHARTS_DIR /
            "isolation_forest_anomalies.png",
            dpi=300
        )
    
        plt.close()
    
        print("✓ Isolation Forest Chart Saved")
    # ---------------------------------------------------------
    # Isolation Forest Report
    # ---------------------------------------------------------
    
    def isolation_report(self):
        """
        Print anomaly records.
        """
    
        anomalies = self.results[
            self.results["IF_Anomaly"]
        ]
    
        print("\n")
        print("=" * 70)
        print("ISOLATION FOREST REPORT")
        print("=" * 70)
    
        print(anomalies[["Sales"]])
    
        return anomalies
    # ---------------------------------------------------------
    # Z-Score Detection
    # ---------------------------------------------------------
    
    def run_zscore_detection(self):
        """
        Detect anomalies using Z-Score.
        """
    
        print("\n" + "=" * 70)
        print("Z-SCORE ANOMALY DETECTION")
        print("=" * 70)
    
        self.results["Z_Score"] = np.abs(
            zscore(self.results["Sales"])
        )
    
        self.results["Z_Anomaly"] = (
            self.results["Z_Score"] > 2
        )
    
        anomaly_count = self.results["Z_Anomaly"].sum()
    
        print(f"Anomalies Detected : {anomaly_count}")
    
        return self.results
    # ---------------------------------------------------------
    # Plot Z-Score
    # ---------------------------------------------------------
    
    def plot_zscore(self):
        """
        Plot anomalies detected using
        Z-Score.
        """
    
        plt.figure(figsize=(16,6))
    
        plt.plot(
            self.results.index,
            self.results["Sales"],
            linewidth=2,
            label="Weekly Sales"
        )
    
        anomalies = self.results[
            self.results["Z_Anomaly"]
        ]
    
        plt.scatter(
            anomalies.index,
            anomalies["Sales"],
            color="orange",
            marker="D",
            s=90,
            label="Z-Score Anomaly"
        )
    
        plt.title("Z-Score Anomaly Detection")
    
        plt.xlabel("Date")
    
        plt.ylabel("Sales")
    
        plt.grid(True)
    
        plt.legend()
    
        plt.tight_layout()
    
        plt.savefig(
            CHARTS_DIR /
            "zscore_anomalies.png",
            dpi=300
        )
    
        plt.close()
    
        print("✓ Z-Score Chart Saved")
    # ---------------------------------------------------------
    # Z-Score Report
    # ---------------------------------------------------------
    
    def zscore_report(self):
        """
        Display Z-Score anomalies.
        """
    
        anomalies = self.results[
            self.results["Z_Anomaly"]
        ]
    
        print("\n")
        print("=" * 70)
        print("Z-SCORE REPORT")
        print("=" * 70)
    
        print(anomalies[["Sales","Z_Score"]])
    
        return anomalies
    # ---------------------------------------------------------
    # Compare Detection Methods
    # ---------------------------------------------------------
    
    def compare_methods(self):
        """
        Compare Isolation Forest
        and Z-Score results.
        """
    
        print("\n")
        print("=" * 70)
        print("METHOD COMPARISON")
        print("=" * 70)
    
        isolation = self.results["IF_Anomaly"].sum()
    
        zscore = self.results["Z_Anomaly"].sum()
    
        print(f"Isolation Forest : {isolation}")
    
        print(f"Z-Score          : {zscore}")
    
        common = (
    
            self.results["IF_Anomaly"]
    
            &
    
            self.results["Z_Anomaly"]
    
        ).sum()
    
        print(f"Common Anomalies : {common}")
    # ---------------------------------------------------------
    # Combined Anomaly Plot
    # ---------------------------------------------------------
    
    def plot_combined_anomalies(self):
        """
        Plot anomalies detected by both
        Isolation Forest and Z-Score.
        """
    
        plt.figure(figsize=(16,6))
    
        plt.plot(
            self.results.index,
            self.results["Sales"],
            linewidth=2,
            color="blue",
            label="Weekly Sales"
        )
    
        # Isolation Forest
        iso = self.results[
            self.results["IF_Anomaly"]
        ]
    
        plt.scatter(
            iso.index,
            iso["Sales"],
            color="red",
            s=90,
            label="Isolation Forest"
        )
    
        # Z Score
        z = self.results[
            self.results["Z_Anomaly"]
        ]
    
        plt.scatter(
            z.index,
            z["Sales"],
            color="orange",
            marker="D",
            s=90,
            label="Z-Score"
        )
    
        plt.title("Sales Anomaly Detection Comparison")
    
        plt.xlabel("Week")
    
        plt.ylabel("Sales")
    
        plt.grid(True)
    
        plt.legend()
    
        plt.tight_layout()
    
        plt.savefig(
            CHARTS_DIR /
            "combined_anomalies.png",
            dpi=300
        )
    
        plt.close()
    
        print("✓ Combined Anomaly Chart Saved")
    # ---------------------------------------------------------
    # Export CSV
    # ---------------------------------------------------------
    
    def export_results(self):
        """
        Save anomaly detection results.
        """
    
        output = self.output_dir / "anomalies.csv"
    
        self.results.to_csv(
            output
        )
    
        print(f"✓ Results exported to {output}")
    # ---------------------------------------------------------
    # Business Insights
    # ---------------------------------------------------------
    
    def business_insights(self):
        """
        Generate business interpretation.
        """
    
        print("\n")
        print("="*70)
        print("BUSINESS INSIGHTS")
        print("="*70)
    
        total = len(self.results)
    
        isolation = self.results["IF_Anomaly"].sum()
    
        zscore = self.results["Z_Anomaly"].sum()
    
        print(f"Total Weeks : {total}")
    
        print(f"Isolation Forest Anomalies : {isolation}")
    
        print(f"Z-Score Anomalies : {zscore}")
    
        print("\nPossible Reasons")
    
        print("• Festival Season")
    
        print("• Discount Campaign")
    
        print("• Product Launch")
    
        print("• Stock Shortage")
    
        print("• Supply Chain Delay")
    
        print("• Marketing Campaign")
    
        print("\nRecommendation")
    
        print(
            """
    Monitor these periods carefully.
    
    Increase inventory before high demand periods.
    
    Investigate low sales weeks for operational issues.
    
    Use anomaly reports together with forecasting
    results before making inventory decisions.
    """
        )
    # ---------------------------------------------------------
    # Run Pipeline
    # ---------------------------------------------------------
    
    def run(self):
        """
        Execute anomaly detection workflow.
        """
    
        self.dataset_summary()
    
        self.prepare_features()
    
        self.run_isolation_forest()
    
        self.isolation_scores()
    
        self.plot_isolation_forest()
    
        self.isolation_report()
    
        self.run_zscore_detection()
    
        self.plot_zscore()
    
        self.zscore_report()
    
        self.compare_methods()
    
        self.plot_combined_anomalies()
    
        self.export_results()
    
        self.business_insights()
    
        print("\n✓ Anomaly Detection Completed Successfully")
# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    detector = AnomalyDetection()

    detector.run()