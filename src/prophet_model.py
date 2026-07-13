"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Prophet Forecasting Model
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from src.feature_engineering import FeatureEngineering
from src.config import (
    MODELS_DIR,
    CHARTS_DIR,
    TEST_SIZE,
    FORECAST_HORIZON
)


class ProphetForecasting:

    def __init__(self):

        feature = FeatureEngineering()

        monthly = feature.monthly_sales()

        monthly = monthly.rename(
            columns={
                "Order Date": "ds",
                "Sales": "y"
            }
        )

        self.df = monthly

        self.model = None

        self.train = None

        self.test = None

    # ---------------------------------------------------------

    def train_test_split(self):
       """
       Split data into training and testing datasets.
       """
       self.train = self.df.iloc[:-TEST_SIZE].copy()
       self.test = self.df.iloc[-TEST_SIZE:].copy()
       print(f"Training Samples : {len(self.train)}")
       print(f"Testing Samples  : {len(self.test)}")

    # ---------------------------------------------------------

    def build_model(self):

        self.model = Prophet(

            yearly_seasonality=True,

            weekly_seasonality=False,

            daily_seasonality=False

        )

        self.model.fit(self.train)

        print("✓ Prophet Model Trained")

    # ---------------------------------------------------------

    def evaluate(self):

        future = self.model.make_future_dataframe(

            periods=FORECAST_HORIZON,
            freq="ME"

        )

        forecast = self.model.predict(future)

        prediction = forecast["yhat"].tail(len(self.test)).values

        actual = self.test["y"].values
        mae = mean_absolute_error(actual, prediction)

        rmse = np.sqrt(

            mean_squared_error(actual, prediction)

        )

        mape = np.mean(

            np.abs(

                (actual - prediction)

                / actual

            )

        ) * 100

        print("=" * 60)

        print("PROPHET PERFORMANCE")

        print("=" * 60)

        print(f"MAE  : {mae:.2f}")

        print(f"RMSE : {rmse:.2f}")

        print(f"MAPE : {mape:.2f}%")

        return forecast

    # ---------------------------------------------------------

    def future_forecast(self):

        future = self.model.make_future_dataframe(

            periods=FORECAST_HORIZON,
            freq="ME"

        )

        forecast = self.model.predict(future)

        print("\n3 Month Forecast")

        print(

            forecast[

                ["ds", "yhat"]

            ].tail(3)

        )

        return forecast

    # ---------------------------------------------------------

    def forecast_plot(self, forecast):

        fig = self.model.plot(forecast)

        plt.title("Prophet Forecast")

        plt.tight_layout()

        plt.savefig(

            CHARTS_DIR /

            "prophet_forecast.png",

            dpi=300

        )

        plt.close()

    # ---------------------------------------------------------

    def components_plot(self, forecast):

        fig = self.model.plot_components(

            forecast

        )

        plt.tight_layout()

        plt.savefig(

            CHARTS_DIR /

            "prophet_components.png",

            dpi=300

        )

        plt.close()

    # ---------------------------------------------------------

    def save_model(self):

        joblib.dump(

            self.model,

            MODELS_DIR /

            "prophet_model.pkl"

        )

        print("✓ Prophet Model Saved")

    # ---------------------------------------------------------

    def run(self):
        print("=" * 60)
        print("PROPHET FORECASTING")
        print("=" * 60)
        self.train_test_split()

        self.build_model()

        self.evaluate()

        forecast = self.future_forecast()

        self.forecast_plot(forecast)

        self.components_plot(forecast)

        self.save_model()
        print("\n✓ Prophet Forecasting Completed")

if __name__ == "__main__":

    model = ProphetForecasting()

    model.run()