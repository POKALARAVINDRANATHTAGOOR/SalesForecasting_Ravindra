import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from src.feature_engineering import FeatureEngineering

from src.config import (
    TEST_SIZE,
    FORECAST_HORIZON,
    RANDOM_STATE,
    MODELS_DIR,
    CHARTS_DIR
)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
class XGBoostForecast:
    def __init__(self):

        feature = FeatureEngineering()

        self.df = feature.create_lag_features()

        self.model = None

        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

    def train_test_split(self):
        X = self.df[
            [
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Rolling_Mean_3",
            "Rolling_STD_3"] ]
        y = self.df["Sales"]
        self.X_train = X.iloc[:-TEST_SIZE]
        self.X_test = X.iloc[-TEST_SIZE:]
        self.y_train = y.iloc[:-TEST_SIZE]
        self.y_test = y.iloc[-TEST_SIZE:]
        print(f"Training Samples : {len(self.X_train)}")
        print(f"Testing Samples : {len(self.X_test)}")
    def build_model(self):
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE
        )

        self.model.fit(self.X_train, self.y_train)
        print("✓ XGBoost Model Trained")
    
    def evaluate(self):

        prediction = self.model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, prediction)
        rmse = np.sqrt(mean_squared_error(self.y_test, prediction))
        mape = np.mean(np.abs((self.y_test - prediction) / self.y_test)) * 100

        results = pd.DataFrame({
            "Actual": self.y_test.values,
            "Predicted": prediction
        })

        results.to_csv(
            OUTPUT_DIR / "xgboost_predictions.csv",
            index=False
        )

        return {
            "Model": "XGBoost",
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "Prediction": prediction,
            "Actual": self.y_test.values
        }
    
    def future_forecast(self):
        """
        Generate recursive forecast for the next FORECAST_HORIZON months.
        """
        current = self.df.tail(1)[
            ["Lag_1", "Lag_2", "Lag_3",
            "Rolling_Mean_3", "Rolling_STD_3"]
        ].copy()
        forecast = []
        for _ in range(FORECAST_HORIZON):
        # Predict next month
            pred = self.model.predict(current)[0]
            forecast.append(pred)
        # Previous lag values
            lag1_old = current.iloc[0]["Lag_1"]
            lag2_old = current.iloc[0]["Lag_2"]
        # Update lag features
            current.loc[:, "Lag_3"] = lag2_old
            current.loc[:, "Lag_2"] = lag1_old
            current.loc[:, "Lag_1"] = pred

        # Update rolling statistics
            values = [
                current.iloc[0]["Lag_1"],
                current.iloc[0]["Lag_2"],
                current.iloc[0]["Lag_3"]
            ]
            current.loc[:, "Rolling_Mean_3"] = np.mean(values)
            current.loc[:, "Rolling_STD_3"] = np.std(values)
        return forecast
    
    def plot_forecast(self, prediction):
        plt.figure(figsize=(15,6))
        plt.plot(self.y_test.values,marker="o",label="Actual")
        plt.plot(prediction,marker="o",label="Predicted"    )

        plt.title("XGBoost Forecast")

        plt.xlabel("Test Months")

        plt.ylabel("Sales")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(CHARTS_DIR /"xgboost_forecast.png",dpi=300)
        plt.close()
    def save_model(self):
        joblib.dump(self.model,MODELS_DIR /"xgboost_model.pkl")
        print("✓ XGBoost Model Saved")
    def run(self):
        self.train_test_split()
        self.build_model()
        result = self.evaluate()
        future = self.future_forecast()
         # ---------------------------------------
         # Save future forecast
         # ---------------------------------------
        forecast_df = pd.DataFrame({
            "Month": [
                "Month 1",
                "Month 2",
                "Month 3"
            ],
            "Forecast": future
        })
        forecast_df.to_csv(
            OUTPUT_DIR / "xgboost_forecast.csv",
            index=False
        )

        self.plot_forecast(
            result["Prediction"]
        )
        self.save_model()
        print("\n============================================================")
        print("XGBOOST PERFORMANCE")
        print("============================================================")
        print(f"MAE  : {result['MAE']:.2f}")
        print(f"RMSE : {result['RMSE']:.2f}")
        print(f"MAPE : {result['MAPE']:.2f}%")

        print("\nNext 3 Month Forecast")
        for i, value in enumerate(future, start=1):
            print(f"Month {i} : {value:.2f}")
        return result


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    model = XGBoostForecast()

    model.run()