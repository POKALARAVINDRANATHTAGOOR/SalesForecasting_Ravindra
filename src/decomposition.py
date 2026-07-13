"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Time Series Analysis & Decomposition
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================
"""

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.feature_engineering import FeatureEngineering
from src.config import CHARTS_DIR
from pathlib import Path
import pandas as pd
class TimeSeriesAnalysis:

    def __init__(self):

        feature = FeatureEngineering()

        self.monthly = feature.monthly_sales()

        self.monthly = self.monthly.rename(
            columns={
                "Order Date": "Date"
            }
        )

        self.monthly.set_index("Date", inplace=True)
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    # ---------------------------------------------------------

    def plot_monthly_sales(self):

        plt.figure(figsize=(15, 6))

        plt.plot(
            self.monthly.index,
            self.monthly["Sales"],
            linewidth=2
        )

        plt.title("Monthly Sales Trend")

        plt.xlabel("Year")

        plt.ylabel("Sales")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "monthly_sales_trend.png",
            dpi=300
        )

        plt.close()

    # ---------------------------------------------------------

    def decomposition(self):

        result = seasonal_decompose(
            self.monthly["Sales"],
            model="additive",
            period=12
        )

        fig = result.plot()

        fig.set_size_inches(15, 10)

        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "time_series_decomposition.png",
            dpi=300
        )

        plt.close()

        return result

    # ---------------------------------------------------------

    def adf_test(self):
        print("=" * 70)
        print("ADF TEST")
        print("=" * 70)

        result = adfuller(self.monthly["Sales"])

        print(f"ADF Statistic : {result[0]:.4f}")

        print(f"P-Value       : {result[1]:.4f}")

        print("\nCritical Values")

        for key, value in result[4].items():

            print(f"{key} : {value:.4f}")

        if result[1] < 0.05:
            print("\nBusiness Insight:")
            print("Monthly sales are stationary and suitable for ARIMA/SARIMA forecasting.")
            print("\nTime Series is Stationary")

        else:
            print("\nBusiness Insight:")
            print("Differencing is required before forecasting.")
            print("\nTime Series is NOT Stationary")
        adf_results = pd.DataFrame({
            "Metric": ["ADF Statistic", "P-Value"],
            "Value": [result[0], result[1]]
        })

        adf_results.to_csv(
            self.output_dir / "adf_results.csv",
            index=False
       )
    # ---------------------------------------------------------
    def differencing(self):

        diff = self.monthly["Sales"].diff().dropna()

        plt.figure(figsize=(15, 6))

        plt.plot(diff)

        plt.title("First Order Differencing")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / "first_difference.png",
            dpi=300
        )
        diff.to_csv(
            self.output_dir / "first_difference.csv",
            header=["Sales"]
       )
        plt.close()

        return diff

    # ---------------------------------------------------------

    def run(self):

        print("=" * 70)
        print("TIME SERIES ANALYSIS")
        print("Generating monthly sales trend...")
        print("Performing seasonal decomposition...")
        print("Running ADF stationarity test...")
        print("Generating first-order differencing plot...")
        print("=" * 70)

        self.plot_monthly_sales()

        self.decomposition()

        self.adf_test()

        self.differencing()

        print("\n✓ Time Series Analysis Completed")
        print(f"\nCharts saved to : {CHARTS_DIR}")
        print(f"Outputs saved to: {self.output_dir}")


if __name__ == "__main__":

    analysis = TimeSeriesAnalysis()

    analysis.run()