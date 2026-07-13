"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Visualization
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Creates all charts used in:

• analysis.ipynb
• Streamlit Dashboard
• Executive Report

Charts are automatically saved inside charts/
==============================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CHARTS_DIR
from src.eda import EDA

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")


class SalesVisualization:

    def __init__(self):

        self.eda = EDA()

        self.df = self.eda.df

        Path(CHARTS_DIR).mkdir(exist_ok=True)

    # ---------------------------------------------------------

    def save_plot(self, filename):

        plt.tight_layout()

        plt.savefig(
            CHARTS_DIR / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ---------------------------------------------------------

    def yearly_sales_chart(self):

        yearly = self.eda.yearly_sales()

        plt.figure(figsize=(12,6))

        sns.barplot(
            data=yearly,
            x="Year",
            y="Sales"
        )

        plt.title("Year-wise Sales")

        self.save_plot("yearly_sales.png")

    # ---------------------------------------------------------

    def category_sales_chart(self):

        category = self.eda.highest_revenue_category()

        plt.figure(figsize=(10,6))

        sns.barplot(
            data=category,
            x="Category",
            y="Sales"
        )

        plt.title("Category-wise Sales")

        self.save_plot("category_sales.png")

    # ---------------------------------------------------------

    def region_sales_chart(self):

        region = self.eda.regional_sales()

        plt.figure(figsize=(10,6))

        sns.barplot(
            data=region,
            x="Region",
            y="Sales"
        )

        plt.title("Region-wise Sales")

        self.save_plot("region_sales.png")

    # ---------------------------------------------------------

    def monthly_sales_line(self):

        monthly = self.eda.monthly_sales()

        monthly["Date"] = (
            monthly["Year"].astype(str)
            + "-"
            + monthly["Month"].astype(str)
        )

        plt.figure(figsize=(15,6))

        plt.plot(
            monthly["Date"],
            monthly["Sales"],
            linewidth=2
        )

        plt.xticks(rotation=90)

        plt.title("Monthly Sales Trend")

        self.save_plot("monthly_sales.png")

    # ---------------------------------------------------------

    def shipping_days_chart(self):

        shipping = self.eda.shipping_analysis()

        plt.figure(figsize=(10,6))

        sns.barplot(
            data=shipping,
            x="Region",
            y="Shipping Days"
        )

        plt.title("Average Shipping Days")

        self.save_plot("shipping_days.png")

    # ---------------------------------------------------------

    def top_products_chart(self):

        products = self.eda.top_products()

        plt.figure(figsize=(12,8))

        sns.barplot(
            data=products,
            y="Product Name",
            x="Sales"
        )

        plt.title("Top 10 Products")

        self.save_plot("top_products.png")

    # ---------------------------------------------------------

    def sales_distribution(self):

        plt.figure(figsize=(10,6))

        sns.histplot(
            self.df["Sales"],
            kde=True,
            bins=30
        )

        plt.title("Sales Distribution")

        self.save_plot("sales_distribution.png")

    # ---------------------------------------------------------

    def correlation_heatmap(self):

        numeric = self.df.select_dtypes(include="number")

        plt.figure(figsize=(10,8))

        sns.heatmap(
            numeric.corr(),
            annot=True,
            cmap="coolwarm"
        )

        plt.title("Correlation Heatmap")

        self.save_plot("correlation_heatmap.png")

    # ---------------------------------------------------------

    def generate_all(self):

        self.yearly_sales_chart()

        self.category_sales_chart()

        self.region_sales_chart()

        self.monthly_sales_line()

        self.shipping_days_chart()

        self.top_products_chart()

        self.sales_distribution()

        self.correlation_heatmap()

        print("✓ All charts generated successfully.")


if __name__ == "__main__":

    chart = SalesVisualization()

    chart.generate_all()