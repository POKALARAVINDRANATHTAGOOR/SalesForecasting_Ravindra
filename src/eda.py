"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Exploratory Data Analysis (EDA)
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
Performs Exploratory Data Analysis including:

1. Dataset Overview
2. Sales Analysis
3. Category Analysis
4. Region Analysis
5. Shipping Analysis
6. Monthly Trend Analysis
7. Business Insights

==============================================================
"""

import pandas as pd

from src.preprocessing import DataPreprocessor


class EDA:

    def __init__(self):

        processor = DataPreprocessor()

        self.df = processor.preprocess()

    # ---------------------------------------------------------

    def dataset_overview(self):

        print("=" * 70)
        print("DATASET OVERVIEW")
        print("=" * 70)

        print(f"Rows : {self.df.shape[0]}")
        print(f"Columns : {self.df.shape[1]}")

        print("\nColumns\n")

        print(self.df.columns.tolist())

    # ---------------------------------------------------------

    def descriptive_statistics(self):

        print("=" * 70)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 70)

        print(self.df.describe())

    # ---------------------------------------------------------

    def highest_revenue_category(self):

        category = (

            self.df

            .groupby("Category")["Sales"]

            .sum()

            .sort_values(ascending=False)

            .reset_index()

        )

        print("\nHighest Revenue Category\n")

        print(category)

        return category

    # ---------------------------------------------------------

    def regional_sales(self):

        region = (

            self.df

            .groupby("Region")["Sales"]

            .sum()

            .sort_values(ascending=False)

            .reset_index()

        )

        print("\nRegional Sales\n")

        print(region)

        return region

    # ---------------------------------------------------------

    def shipping_analysis(self):

        shipping = (

            self.df

            .groupby("Region")["Shipping Days"]

            .mean()

            .round(2)

            .reset_index()

        )

        print("\nAverage Shipping Time\n")

        print(shipping)

        return shipping

    # ---------------------------------------------------------

    def monthly_sales(self):

        monthly = (

            self.df

            .groupby(["Year", "Month"])["Sales"]

            .sum()

            .reset_index()

        )

        print("\nMonthly Sales\n")

        print(monthly.head())

        return monthly

    # ---------------------------------------------------------

    def yearly_sales(self):

        yearly = (

            self.df

            .groupby("Year")["Sales"]

            .sum()

            .reset_index()

        )

        print("\nYearly Sales\n")

        print(yearly)

        return yearly

    # ---------------------------------------------------------

    def top_products(self):

        products = (

            self.df

            .groupby("Product Name")["Sales"]

            .sum()

            .sort_values(ascending=False)

            .head(10)

            .reset_index()

        )

        print("\nTop 10 Products\n")

        print(products)

        return products

    # ---------------------------------------------------------

    def top_customers(self):

        customers = (

            self.df

            .groupby("Customer Name")["Sales"]

            .sum()

            .sort_values(ascending=False)

            .head(10)

            .reset_index()

        )

        print("\nTop Customers\n")

        print(customers)

        return customers

    # ---------------------------------------------------------

    def business_summary(self):

        print("=" * 70)

        print("BUSINESS SUMMARY")

        print("=" * 70)

        print(f"Total Sales : {self.df['Sales'].sum():,.2f}")

        print(f"Average Sales : {self.df['Sales'].mean():,.2f}")

        print(f"Maximum Sale : {self.df['Sales'].max():,.2f}")

        print(f"Minimum Sale : {self.df['Sales'].min():,.2f}")

        print(f"Total Orders : {self.df['Order ID'].nunique()}")

        print(f"Customers : {self.df['Customer ID'].nunique()}")

        print(f"Products : {self.df['Product ID'].nunique()}")

    # ---------------------------------------------------------

    def run_complete_eda(self):

        self.dataset_overview()

        self.descriptive_statistics()

        self.business_summary()

        self.highest_revenue_category()

        self.regional_sales()

        self.shipping_analysis()

        self.monthly_sales()

        self.yearly_sales()

        self.top_products()

        self.top_customers()


# ===========================================================

if __name__ == "__main__":

    eda = EDA()

    eda.run_complete_eda()