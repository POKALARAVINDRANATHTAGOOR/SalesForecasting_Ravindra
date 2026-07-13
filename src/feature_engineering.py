"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Feature Engineering
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
Creates analytical datasets required for:

1. Daily Sales
2. Weekly Sales
3. Monthly Sales
4. Region-wise Sales
5. Category-wise Sales
6. Sub-Category Analysis
7. Lag Features
8. Rolling Statistics
9. Growth Rate
==============================================================
"""

import pandas as pd

from src.preprocessing import DataPreprocessor


class FeatureEngineering:

    def __init__(self):

        processor = DataPreprocessor()

        self.df = processor.preprocess()

    # ---------------------------------------------------------

    def daily_sales(self):

        daily = (
            self.df
            .groupby("Order Date")["Sales"]
            .sum()
            .reset_index()
        )

        return daily

    # ---------------------------------------------------------

    def weekly_sales(self):

        weekly = (
            self.df
            .groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
            .sum()
            .reset_index()
        )

        return weekly

    # ---------------------------------------------------------

    def monthly_sales(self):

        monthly = (
            self.df
            .groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
            .sum()
            .reset_index()
        )

        return monthly

    # ---------------------------------------------------------

    def yearly_sales(self):

        yearly = (
            self.df
            .groupby("Year")["Sales"]
            .sum()
            .reset_index()
        )

        return yearly

    # ---------------------------------------------------------

    def region_sales(self):

        region = (
            self.df
            .groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        return region

    # ---------------------------------------------------------

    def category_sales(self):

        category = (
            self.df
            .groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        return category

    # ---------------------------------------------------------

    def subcategory_sales(self):

        subcategory = (
            self.df
            .groupby("Sub-Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        return subcategory

    # ---------------------------------------------------------

    def create_lag_features(self):

        monthly = self.monthly_sales()

        monthly["Lag_1"] = monthly["Sales"].shift(1)

        monthly["Lag_2"] = monthly["Sales"].shift(2)

        monthly["Lag_3"] = monthly["Sales"].shift(3)

        monthly["Rolling_Mean_3"] = (
            monthly["Sales"]
            .rolling(3)
            .mean()
        )

        monthly["Rolling_STD_3"] = (
            monthly["Sales"]
            .rolling(3)
            .std()
        )

        monthly.dropna(inplace=True)

        return monthly

    # ---------------------------------------------------------

    def sales_growth(self):

        monthly = self.monthly_sales()

        monthly["Growth Rate (%)"] = (
            monthly["Sales"]
            .pct_change()
            * 100
        )

        return monthly

    # ---------------------------------------------------------

    def summary(self):

        print("="*70)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*70)

        print("Daily Sales Shape      :", self.daily_sales().shape)

        print("Weekly Sales Shape     :", self.weekly_sales().shape)

        print("Monthly Sales Shape    :", self.monthly_sales().shape)

        print("Yearly Sales Shape     :", self.yearly_sales().shape)

        print("Region Sales Shape     :", self.region_sales().shape)

        print("Category Sales Shape   :", self.category_sales().shape)

        print("SubCategory Shape      :", self.subcategory_sales().shape)

        print("Lag Dataset Shape      :", self.create_lag_features().shape)
if __name__ == "__main__":

    feature = FeatureEngineering()

    feature.summary()

    print("\nMonthly Sales")
    monthly_sales = feature.monthly_sales()
    print(monthly_sales.head())

    print("\nLag Features")
    lag_df = feature.create_lag_features()
    print(lag_df.head())

    # -----------------------------
    # Save Outputs
    # -----------------------------
    from pathlib import Path

    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    daily_sales = feature.daily_sales()
    weekly_sales = feature.weekly_sales()
    yearly_sales = feature.yearly_sales()
    region_sales = feature.region_sales()
    category_sales = feature.category_sales()
    subcategory_sales = feature.subcategory_sales()

    daily_sales.to_csv(OUTPUT_DIR / "daily_sales.csv", index=False)
    weekly_sales.to_csv(OUTPUT_DIR / "weekly_sales.csv", index=False)
    monthly_sales.to_csv(OUTPUT_DIR / "monthly_sales.csv", index=False)
    yearly_sales.to_csv(OUTPUT_DIR / "yearly_sales.csv", index=False)
    region_sales.to_csv(OUTPUT_DIR / "region_sales.csv", index=False)
    category_sales.to_csv(OUTPUT_DIR / "category_sales.csv", index=False)
    subcategory_sales.to_csv(OUTPUT_DIR / "subcategory_sales.csv", index=False)
    lag_df.to_csv(OUTPUT_DIR / "lag_features.csv", index=False)

    print("\n✓ Feature Engineering outputs saved successfully.")