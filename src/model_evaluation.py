"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Model Evaluation & Comparison
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
This module compares all forecasting models:

1. SARIMA
2. Prophet
3. XGBoost

It automatically

✔ Calculates MAE
✔ Calculates RMSE
✔ Calculates MAPE
✔ Calculates R² Score
✔ Creates Model Comparison Table
✔ Saves Results
✔ Selects Best Model

==============================================================
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.config import ROOT_DIR
from src.config import CHARTS_DIR

from src.sarima_model import SARIMAForecasting
from src.prophet_model import ProphetForecasting
from src.xgboost_model import XGBoostForecast

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")
class ForecastEvaluation:
    """
    Professional Forecast Model Comparison

    Compares

    • SARIMA

    • Prophet

    • XGBoost

    Produces

    • Metrics

    • CSV Report

    • Comparison Graph

    • Best Model Recommendation
    """

    def __init__(self):

        self.output_dir = ROOT_DIR / "outputs"

        self.output_dir.mkdir(
            exist_ok=True
        )

        self.results = []

        self.best_model = None

        self.best_metrics = None
    # ---------------------------------------------------------

    @staticmethod
    def calculate_metrics(actual, prediction):
        """
        Calculate Evaluation Metrics
        """

        mae = mean_absolute_error(
            actual,
            prediction
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                prediction
            )
        )

        mape = np.mean(
            np.abs(
                (actual - prediction) / actual
            )
        ) * 100

        r2 = r2_score(
            actual,
            prediction
        )

        return {

            "MAE": round(mae, 2),

            "RMSE": round(rmse, 2),

            "MAPE": round(mape, 2),

            "R2 Score": round(r2, 4)

        }

    # ---------------------------------------------------------

    def add_result(
        self,
        model_name,
        metrics,
        forecast
    ):
        """
        Store model result
        """

        row = {

            "Model": model_name,

            "MAE": metrics["MAE"],

            "RMSE": metrics["RMSE"],

            "MAPE": metrics["MAPE"],

            "R2 Score": metrics["R2 Score"],

            "Forecast Month 1": forecast[0],

            "Forecast Month 2": forecast[1],

            "Forecast Month 3": forecast[2]

        }

        self.results.append(row)  
    # ---------------------------------------------------------
    # SARIMA Evaluation
    # ---------------------------------------------------------
    
    def evaluate_sarima(self):
        """
        Evaluate SARIMA Forecasting Model
        """
    
        print("=" * 70)
        print("Evaluating SARIMA Model")
        print("=" * 70)
    
        model = SARIMAForecasting()
    
        model.train_test_split()
    
        model.build_model()
    
        result = model.evaluate()
    
        forecast, _ = model.future_forecast()
    
        prediction = model.forecast_test()
    
        metrics = self.calculate_metrics(
            model.test["Sales"],
            prediction
        )
    
        self.add_result(
            model_name="SARIMA",
            metrics=metrics,
            forecast=forecast.values
        )
    
        print("✓ SARIMA Evaluation Completed")
    # ---------------------------------------------------------
    # Prophet Evaluation
    # ---------------------------------------------------------
    
    def evaluate_prophet(self):
        """
        Evaluate Prophet Forecasting Model
        """
    
        print("=" * 70)
        print("Evaluating Prophet Model")
        print("=" * 70)
    
        model = ProphetForecasting()
    
        model.train_test_split()
    
        model.build_model()
    
        forecast = model.evaluate()
    
        future = model.future_forecast()
    
        prediction = forecast["yhat"].tail(
            len(model.test)
        ).values
    
        metrics = self.calculate_metrics(
            model.test["y"],
            prediction
        )
    
        self.add_result(
            model_name="Prophet",
            metrics=metrics,
            forecast=future["yhat"].tail(3).values
        )
    
        print("✓ Prophet Evaluation Completed")
    # ---------------------------------------------------------
    # XGBoost Evaluation
    # ---------------------------------------------------------
    
    def evaluate_xgboost(self):
        """
        Evaluate XGBoost Forecasting Model
        """
    
        print("=" * 70)
        print("Evaluating XGBoost Model")
        print("=" * 70)
    
        model = XGBoostForecast()
    
        model.train_test_split()
    
        model.build_model()
    
        result = model.evaluate()
    
        future = model.future_forecast()
    
        metrics = self.calculate_metrics(
            model.y_test,
            result["Prediction"]
        )
    
        self.add_result(
            model_name="XGBoost",
            metrics=metrics,
            forecast=future
        )
    
        print("✓ XGBoost Evaluation Completed")
    # ---------------------------------------------------------
    # Run All Models
    # ---------------------------------------------------------
    
    def evaluate_all_models(self):
        """
        Execute all forecasting models.
        """
    
        self.evaluate_sarima()
    
        self.evaluate_prophet()
    
        self.evaluate_xgboost()
    
        print("\n✓ All Models Evaluated Successfully")
    # ---------------------------------------------------------
    # Comparison Table
    # ---------------------------------------------------------
    
    def create_comparison_table(self):
        """
        Create comparison dataframe.
        """
    
        self.results_df = pd.DataFrame(self.results)
    
        print("\n")
        print("=" * 90)
        print("FORECAST MODEL COMPARISON")
        print("=" * 90)
    
        print(self.results_df)
    
        return self.results_df
    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------    
    
    def save_results(self):
        """
        Save model comparison.
        """
    
        output_file = self.output_dir / "evaluation_results.csv"
    
        self.results_df.to_csv(
            output_file,
            index=False
        )
    
        print(f"\n✓ Results saved to {output_file}")
    # ---------------------------------------------------------
    # Best Model
    # ---------------------------------------------------------
    
    def select_best_model(self):
        """
        Select best model using RMSE.
        """
    
        best = self.results_df.loc[
            self.results_df["RMSE"].idxmin()
        ]
    
        self.best_model = best["Model"]
    
        self.best_metrics = best
    
        print("\n")
        print("=" * 70)
        print("BEST MODEL")
        print("=" * 70)
    
        print(f"Model : {self.best_model}")
    
        print(f"MAE   : {best['MAE']:.2f}")
    
        print(f"RMSE  : {best['RMSE']:.2f}")
    
        print(f"MAPE  : {best['MAPE']:.2f}%")
    
        print(f"R²    : {best['R2 Score']:.4f}")
        pd.DataFrame([best]).to_csv(
            self.output_dir / "best_model.csv",
            index=False
       )
    # ---------------------------------------------------------
    # Comparison Plot
    # ---------------------------------------------------------
    
    def plot_model_comparison(self):
        """
        Plot comparison of RMSE.
        """
    
        plt.figure(figsize=(10,6))
    
        sns.barplot(
            data=self.results_df,
            x="Model",
            y="RMSE"
        )
    
        plt.title("Forecast Model Comparison")
    
        plt.ylabel("RMSE")
    
        plt.tight_layout()
    
        plt.savefig(
            CHARTS_DIR / "forecast_model_comparison.png",
            dpi=300
        )
    
        plt.close()
    
        print("✓ Comparison Chart Saved")
    # ---------------------------------------------------------
    # Business Summary
    # ---------------------------------------------------------
    def create_comparison_table(self):
        """
        Create comparison dataframe.
        """
        self.results_df = pd.DataFrame(self.results) 
    # Sort models by RMSE (lowest first)
        self.results_df = self.results_df.sort_values(
            by="RMSE"
        ).reset_index(drop=True)
        print("\n")
        print("=" * 90)
        print("FORECAST MODEL COMPARISON")
        print("=" * 90)
        print(self.results_df)
        return self.results_df
    def business_summary(self):
        """
        Business recommendation.
        """
    
        print("\n")
        print("=" * 70)
        print("BUSINESS RECOMMENDATION")
        print("=" * 70)
    
        print(
            f"""
    Based on evaluation metrics,
    {self.best_model} achieved
    the lowest forecasting error.
    
    Recommended Production Model
    
    {self.best_model}
    
    Reason
    
    • Lowest RMSE
    
    • Lowest MAE
    
    • Reliable Forecast Accuracy
    
    This model is recommended
    for deployment inside the
    Streamlit Dashboard.
    """
        )
        with open(
            self.output_dir / "business_recommendation.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(f"Recommended Model : {self.best_model}\n")
            f.write(f"RMSE : {self.best_metrics['RMSE']:.2f}\n")
            f.write(f"MAE : {self.best_metrics['MAE']:.2f}\n")
            f.write(f"MAPE : {self.best_metrics['MAPE']:.2f}%\n")
            f.write(f"R2 Score : {self.best_metrics['R2 Score']:.4f}\n")
    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------
    
    def run(self):
        """
        Execute evaluation pipeline.
        """
    
        self.evaluate_all_models()
        self.create_comparison_table()
        self.select_best_model()
        self.save_results()
        self.plot_model_comparison()
        self.business_summary()
    
        print("\n✓ Model Evaluation Completed Successfully")
# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    evaluation = ForecastEvaluation()

    evaluation.run()