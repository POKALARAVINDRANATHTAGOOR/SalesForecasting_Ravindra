"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : SARIMA Forecasting Model
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
Builds SARIMA model for monthly sales forecasting.

Functions:
1. Train SARIMA Model
2. Forecast Next 3 Months
3. Evaluate Model
4. Save Model
5. Save Forecast Chart
==============================================================
"""
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import matplotlib.pyplot as plt
from src.config import TEST_SIZE, CHARTS_DIR, MODELS_DIR
from src.config import FORECAST_HORIZON
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.feature_engineering import FeatureEngineering
from src.config import FORECAST_HORIZON, MODELS_DIR, CHARTS_DIR


class SARIMAForecasting:

    def __init__(self):

        feature = FeatureEngineering()
        self.monthly = feature.monthly_sales()
        self.monthly = self.monthly.rename(
            columns={"Order Date": "Date"}
        )

        self.monthly.set_index("Date", inplace=True)
        self.monthly = self.monthly.asfreq("ME")
        self.train = None
        self.test = None
        self.model = None
        print(self.monthly.index.freq)

    # ---------------------------------------------------------

    def train_test_split(self):

        self.train = self.monthly.iloc[:-TEST_SIZE]
        self.test = self.monthly.iloc[-TEST_SIZE:]
        print(f"Training Samples : {len(self.train)}")

        print(f"Testing Samples  : {len(self.test)}")

    # ---------------------------------------------------------

    def build_model(self):

        print("\nTraining SARIMA Model...\n")

        self.model = SARIMAX(

            self.train["Sales"],

            order=(1,1,1),

            seasonal_order=(1,1,1,12),

            enforce_stationarity=False,

            enforce_invertibility=False

        ).fit()

        print("✓ Model Training Completed")

    # ---------------------------------------------------------

    def forecast_test(self):

        forecast = self.model.forecast(
            steps=len(self.test)
        )

        return forecast

    # ---------------------------------------------------------

    def evaluate(self):

        forecast = self.forecast_test()

        mae = mean_absolute_error(
            self.test["Sales"],
            forecast
        )

        rmse = np.sqrt(

            mean_squared_error(

                self.test["Sales"],

                forecast

            )

        )

        mape = np.mean(

            np.abs(

                (self.test["Sales"] - forecast)

                / self.test["Sales"]

            )

        ) * 100

        print("="*60)

        print("SARIMA PERFORMANCE")

        print("="*60)

        print(f"MAE  : {mae:.2f}")

        print(f"RMSE : {rmse:.2f}")

        print(f"MAPE : {mape:.2f}%")

        return mae, rmse, mape

    # ---------------------------------------------------------

    def future_forecast(self):

        future = self.model.get_forecast(steps=FORECAST_HORIZON)

        prediction = future.predicted_mean

        confidence = future.conf_int()

        print("\n3-Month Forecast")

        print(prediction)

        return prediction, confidence

    # ---------------------------------------------------------

    def forecast_plot(self):

        forecast = self.forecast_test()

        plt.figure(figsize=(15,6))

        plt.plot(

            self.train.index,

            self.train["Sales"],

            label="Training"

        )

        plt.plot(

            self.test.index,

            self.test["Sales"],

            label="Actual"

        )

        plt.plot(

            self.test.index,

            forecast,

            label="Forecast"

        )

        plt.title("SARIMA Forecast")

        plt.xlabel("Date")

        plt.ylabel("Sales")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            CHARTS_DIR /

            "sarima_forecast.png",

            dpi=300

        )

        plt.close()

        print("✓ Forecast Chart Saved")

    # ---------------------------------------------------------

    def save_model(self):

        joblib.dump(

            self.model,

            MODELS_DIR / "sarima_model.pkl"

        )

        print("✓ Model Saved")

    # ---------------------------------------------------------

    def run(self):

        self.train_test_split()

        self.build_model()

        self.evaluate()

        self.future_forecast()

        self.forecast_plot()

        self.save_model()


if __name__ == "__main__":

    model = SARIMAForecasting()

    model.run()