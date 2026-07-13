"""
==========================================================
Project : Sales Forecasting & Demand Intelligence System
Module  : Utility Functions
Author  : Ravindra Nathtagoor
==========================================================
"""

from datetime import datetime


def print_title(title: str):
    """
    Print formatted section title.
    """

    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)


def current_time():
    """
    Returns current timestamp.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def separator():
    """
    Print separator line.
    """

    print("-" * 70)