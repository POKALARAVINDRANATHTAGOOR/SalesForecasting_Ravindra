"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Data Preprocessing
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
This module performs:

1. Data Cleaning
2. Missing Value Handling
3. Duplicate Removal
4. Time Feature Engineering
5. Shipping Time Calculation
6. Data Validation

==============================================================
"""

import pandas as pd

from src.data_loader import DataLoader


class DataPreprocessor:

    def __init__(self):

        loader = DataLoader()

        self.df = loader.load_sales_dataset()

    # ---------------------------------------------------------

    def remove_duplicates(self):
        """
        Remove duplicate records
        """

        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        print(f"Removed {before-after} duplicate rows")

    # ---------------------------------------------------------

    def handle_missing_values(self):
        """
         Handle missing values
        """
        print("\nMissing Values")

        print(self.df.isnull().sum())
    # Remove rows with missing dates
        self.df.dropna(
            subset=["Order Date", "Ship Date"],
            inplace=True
        )

    # Fill Postal Code
        if "Postal Code" in self.df.columns:
            self.df["Postal Code"] = (
                self.df["Postal Code"]
                .fillna(0)
                .astype(int)
            )

    # ---------------------------------------------------------

    def create_date_features(self):
        """Extract Time Features"""

    # Ensure datetime format
        self.df["Order Date"] = pd.to_datetime(
            self.df["Order Date"],
            errors="coerce"
       )

        self.df["Ship Date"] = pd.to_datetime(
             self.df["Ship Date"],
             errors="coerce")

    # Remove rows with invalid dates
        self.df.dropna(
            subset=["Order Date", "Ship Date"],
            inplace=True
        )

        self.df["Year"] = self.df["Order Date"].dt.year
        self.df["Month"] = self.df["Order Date"].dt.month
        self.df["Month Name"] = self.df["Order Date"].dt.month_name()
        self.df["Quarter"] = self.df["Order Date"].dt.quarter
        self.df["Week"] = (
        self.df["Order Date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )
        self.df["Day"] = self.df["Order Date"].dt.day
        self.df["Day Name"] = self.df["Order Date"].dt.day_name()
        self.df["Day Of Week"] = self.df["Order Date"].dt.dayofweek

    # ---------------------------------------------------------

    def create_shipping_days(self):
        """
        Shipping Duration
        """

        self.df["Shipping Days"] = (
            self.df["Ship Date"] -
            self.df["Order Date"]
        ).dt.days

    # ---------------------------------------------------------

    @staticmethod
    def get_season(month):

        if month in [12,1,2]:
            return "Winter"

        elif month in [3,4,5]:
            return "Spring"

        elif month in [6,7,8]:
            return "Summer"

        return "Autumn"

    # ---------------------------------------------------------

    def create_season_feature(self):

        self.df["Season"] = self.df["Month"].apply(
            self.get_season
        )

    # ---------------------------------------------------------

    def validate_dataset(self):
        print("\nDataset Shape")
        print(self.df.shape)
        print("\nColumns")
        print(self.df.columns.tolist())
        print("\nData Types")
        print(self.df.dtypes)

        print("\nMissing Values")
        print(self.df.isnull().sum())

    # ---------------------------------------------------------

    def preprocess(self):

        print("="*70)
        print("DATA PREPROCESSING")
        print("="*70)

        self.remove_duplicates()

        self.handle_missing_values()

        self.create_date_features()

        self.create_shipping_days()

        self.create_season_feature()

        self.validate_dataset()

        return self.df


# ===========================================================

if __name__=="__main__":

    processor = DataPreprocessor()

    df = processor.preprocess()

    print(df.head())