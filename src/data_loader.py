"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Data Loader
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description:
------------
This module is responsible for:

1. Loading both datasets
2. Parsing date columns
3. Dataset validation
4. Dataset information
5. Missing value summary
6. Duplicate checking

This module is used by:
    • analysis.ipynb
    • app.py
    • preprocessing.py
==============================================================
"""

import pandas as pd
from pathlib import Path

from src.config import TRAIN_DATA, VGSALES_DATA


class DataLoader:
    """
    DataLoader class for loading project datasets.
    """

    def __init__(self):

        self.sales_df = None
        self.vg_df = None

    # ---------------------------------------------------------

    def load_sales_dataset(self):
        """
    Load Superstore Sales Dataset
    """
        if not Path(TRAIN_DATA).exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{TRAIN_DATA}"
           )

        self.sales_df = pd.read_csv(TRAIN_DATA)

        self.sales_df.columns = (
            self.sales_df.columns.str.strip()
        )

        self.sales_df["Order Date"] = pd.to_datetime(
            self.sales_df["Order Date"],
            format="%d/%m/%Y",
            errors="raise"
        )

        self.sales_df["Ship Date"] = pd.to_datetime(
            self.sales_df["Ship Date"],
            format="%d/%m/%Y",
            errors="raise"
        )

        print("✓ Superstore Sales Dataset Loaded")

        return self.sales_df
    # ---------------------------------------------------------

    def load_vgsales_dataset(self):
        """
        Load Video Game Sales Dataset
        """

        if not Path(VGSALES_DATA).exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{VGSALES_DATA}"
            )

        self.vg_df = pd.read_csv(VGSALES_DATA)

       # Remove rows with missing Year
        self.vg_df = self.vg_df.dropna(subset=["Year"])
        # Replace missing Publisher
        self.vg_df["Publisher"] = (
            self.vg_df["Publisher"]
            .fillna("Unknown")
        )

       # Convert Year to integer
        self.vg_df["Year"] = self.vg_df["Year"].astype(int)
        print("✓ Video Game Sales Dataset Loaded")

        return self.vg_df

    # ---------------------------------------------------------

    @staticmethod
    def dataset_info(df, name):
        """
        Display dataset information
        """

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print(f"Rows      : {df.shape[0]}")
        print(f"Columns   : {df.shape[1]}")

        print("\nColumn Names")
        print(df.columns.tolist())

        print("\nData Types")
        print(df.dtypes)

    # ---------------------------------------------------------

    @staticmethod
    def missing_values(df):
        """
        Missing Value Report
        """

        print("\nMissing Values")

        missing = df.isnull().sum()

        print(missing[missing > 0])

    # ---------------------------------------------------------

    @staticmethod
    def duplicate_values(df):
        """
        Duplicate Report
        """

        duplicates = df.duplicated().sum()

        print(f"\nDuplicate Records : {duplicates}")

    # ---------------------------------------------------------

    def summary(self):
        """
        Complete Dataset Summary
        """

        self.dataset_info(
            self.sales_df,
            "Superstore Sales Dataset"
        )

        self.missing_values(self.sales_df)

        self.duplicate_values(self.sales_df)

        print("\n")

        self.dataset_info(
            self.vg_df,
            "Video Game Dataset"
        )

        self.missing_values(self.vg_df)

        self.duplicate_values(self.vg_df)


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    loader = DataLoader()

    loader.load_sales_dataset()

    loader.load_vgsales_dataset()

    loader.summary()