"""
==============================================================
Project : End-to-End Sales Forecasting & Demand Intelligence System
Module  : Product Demand Segmentation
Author  : Ravindra Nathtagoor
Version : 1.0
==============================================================

Description
-----------
This module performs Product Demand Segmentation using
Machine Learning.

Techniques Used

✔ Feature Engineering
✔ Standard Scaling
✔ Elbow Method
✔ K-Means Clustering
✔ PCA Visualization
✔ Business Recommendations

Outputs

✔ clusters.csv
✔ elbow_method.png
✔ pca_clusters.png

==============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.preprocessing import DataPreprocessor
from src.config import (
    ROOT_DIR,
    CHARTS_DIR,
    RANDOM_STATE
)

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")
class ProductClustering:
    """
    Product Demand Segmentation
    using K-Means Clustering.
    """

    def __init__(self):
        processor = DataPreprocessor()
        self.df = processor.preprocess()
        self.output_dir = ROOT_DIR / "outputs"

        self.output_dir.mkdir(
            exist_ok=True
        )

        self.cluster_data = None

        self.scaled_data = None

        self.model = None

        self.pca = None

        print("="*70)
        print("PRODUCT DEMAND SEGMENTATION")
        print("="*70)
    # ---------------------------------------------------------
    # Product Aggregation
    # ---------------------------------------------------------
    
    def prepare_dataset(self):
        """
        Create Sub-Category level dataset for clustering.
        """
        print("\nPreparing Product Dataset...")
    # Total Sales
        sales = (
            self.df
            .groupby("Sub-Category")["Sales"]
            .sum()
        )

    # Average Order Value
        avg_order = (
            self.df
            .groupby("Sub-Category")["Sales"]
            .mean()
        )
    # Monthly Sales
        monthly = (
            self.df
            .groupby([
                "Sub-Category",
                pd.Grouper(key="Order Date", freq="ME")
            ])["Sales"]
            .sum()
            .reset_index()
         )

    # Sales Volatility
        volatility = (
            monthly
            .groupby("Sub-Category")["Sales"]
            .std()
        )

    # Yearly Sales
        yearly = (
            self.df
            .groupby([
                "Sub-Category",
                "Year"
            ])["Sales"]
            .sum()
            .reset_index()
        )

    # Growth Rate
        growth = (
            yearly
            .groupby("Sub-Category")["Sales"]
            .pct_change()
            .groupby(yearly["Sub-Category"])
            .mean()
            * 100
        )

        self.cluster_data = pd.DataFrame({
            "Sub-Category": sales.index,
            "Total Sales": sales.values,
            "Average Order Value": avg_order.values,
            "Sales Volatility": volatility.values,
            "Growth Rate": growth.values
       })
        self.cluster_data.fillna(0, inplace=True)
        print("\nDataset Shape :", self.cluster_data.shape)
        print(self.cluster_data.head())
        print("\nNumber of Products :", len(self.cluster_data))
        print("Number of Features :", self.cluster_data.shape[1] - 1)
        return self.cluster_data
    # ---------------------------------------------------------
    # Feature Selection
    # ---------------------------------------------------------
    def select_features(self):
        self.features = self.cluster_data[
            [
                "Total Sales",
                "Average Order Value",
                "Sales Volatility",
                "Growth Rate"
            ]
        ].copy()
        print("\nSelected Features")
        print(self.features.head())
        print("Number of Features :", len(self.features.columns))
    # ---------------------------------------------------------
    # Cluster Summary
    # ---------------------------------------------------------
    def cluster_summary(self):
        summary = (
            self.cluster_data
            .groupby("Cluster")
            .agg({
                "Total Sales": "mean",
                "Average Order Value": "mean",
                "Sales Volatility": "mean",
                "Growth Rate": "mean"
            })
           .round(2)
        )
        print("\n")
        print("=" * 70)
        print("CLUSTER SUMMARY")
        print("=" * 70)
        print(summary)
        summary.to_csv(
            self.output_dir / "cluster_summary.csv"
        )
        return summary
    # ---------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------
    
    def scale_features(self):
        """
        Scale numerical features.
        """
    
        scaler = StandardScaler()
    
        self.scaled_data = scaler.fit_transform(
            self.features
        )
    
        print("\n✓ Feature Scaling Completed")
    # ---------------------------------------------------------
    # Elbow Method
    # ---------------------------------------------------------
    
    def elbow_method(self):
        """
        Find optimal number of clusters
        using the Elbow Method.
        """
    
        print("\n" + "=" * 70)
        print("ELBOW METHOD")
        print("=" * 70)
    
        inertia = []
    
        K = range(2, 11)
    
        for k in K:
    
            model = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=10
            )
    
            model.fit(self.scaled_data)
    
            inertia.append(model.inertia_)
    
        plt.figure(figsize=(10,6))
    
        plt.plot(
            K,
            inertia,
            marker="o",
            linewidth=2
        )
    
        plt.xlabel("Number of Clusters")
    
        plt.ylabel("Inertia")
    
        plt.title("Elbow Method")
    
        plt.grid(True)
    
        plt.tight_layout()
    
        plt.savefig(
            CHARTS_DIR / "elbow_method.png",
            dpi=300
        )
    
        plt.close()
    
        print("✓ Elbow Method Chart Saved")
    # ---------------------------------------------------------
    # Train KMeans
    # ---------------------------------------------------------
    
    def train_model(self, n_clusters=4):
        """
        Train KMeans clustering model.
        """
    
        print("\nTraining KMeans...")
    
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init= 20
        )
    
        self.cluster_data["Cluster"] = self.model.fit_predict(
            self.scaled_data
        )
    
        print("✓ KMeans Model Trained")
        print("\nCluster Centers")
        centers = pd.DataFrame(
            self.model.cluster_centers_,
            columns=self.features.columns
        )
        print(centers)
        centers.to_csv(
            self.output_dir / "cluster_centers.csv",
            index=False
        )
    # ---------------------------------------------------------
    # Silhouette Score
    # ---------------------------------------------------------
    
    def silhouette(self):
        """
        Evaluate clustering quality.
        """
    
        score = silhouette_score(
            self.scaled_data,
            self.cluster_data["Cluster"]
        )
        pd.DataFrame({
            "Silhouette Score": [score]
        }).to_csv(
            self.output_dir / "silhouette_score.csv",
            index=False
        )
        print("\nSilhouette Score")
    
        print(f"{score:.4f}")
    
        return score
    # ---------------------------------------------------------
    # Cluster Sizes
    # ---------------------------------------------------------
    
    def cluster_size(self):
        """
        Number of products in each cluster.
        """
    
        counts = self.cluster_data[
            "Cluster"
        ].value_counts()
    
        print("\nCluster Sizes\n")
    
        print(counts)
        counts.to_csv(
            self.output_dir / "cluster_sizes.csv",
            header=["Count"]
        )
    
        return counts
    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------
    
    def apply_pca(self):
        """
        Reduce features to 2 dimensions
        for visualization.
        """
    
        print("\nApplying PCA...")
    
        self.pca = PCA(
            n_components=2,
            random_state=RANDOM_STATE
        )
    
        components = self.pca.fit_transform(
            self.scaled_data
        )
        pca_df = pd.DataFrame({
            "Component": ["PCA1", "PCA2"],
            "Explained Variance": self.pca.explained_variance_ratio_
        })
        pca_df.to_csv(
            self.output_dir / "pca_variance.csv",
            index=False
        )
        self.cluster_data["PCA1"] = components[:, 0]
        self.cluster_data["PCA2"] = components[:, 1]
        print("\nExplained Variance")
        for i, value in enumerate(
            self.pca.explained_variance_ratio_,
            start=1
      ):
            print(f"PCA{i} : {value:.2%}")

        print("✓ PCA Completed")
    # ---------------------------------------------------------
    # PCA Visualization
    # ---------------------------------------------------------
    
    def plot_clusters(self):
        """
        Visualize product clusters.
        """
    
        plt.figure(figsize=(12,8))
    
        sns.scatterplot(
            data=self.cluster_data,
            x="PCA1",
            y="PCA2",
            hue="Cluster",
            palette="Set2",
            s=150
        )
    
        for _, row in self.cluster_data.iterrows():
    
            plt.text(
                row["PCA1"],
                row["PCA2"],
                row["Sub-Category"],
                fontsize=8
            )
    
        plt.title("Product Demand Segments")
    
        plt.xlabel("Principal Component 1")
    
        plt.ylabel("Principal Component 2")
    
        plt.grid(True)
    
        plt.tight_layout()
    
        plt.savefig(
            CHARTS_DIR / "product_clusters.png",
            dpi=300
        )
    
        plt.close()
    
        print("✓ Cluster Plot Saved")
    # ---------------------------------------------------------
    # Business Labels
    # ---------------------------------------------------------
    
    def assign_cluster_labels(self):
        """
        Assign meaningful business names
        to each cluster.
        """
    
        labels = {
    
            0: "High Volume, Stable Demand",
    
            1: "Growing Demand",
    
            2: "Low Volume, High Volatility",
    
            3: "Declining Demand"
    
        }
    
        self.cluster_data["Demand Segment"] = (
            self.cluster_data["Cluster"]
            .map(labels)
            .fillna("Other")
        )
    
        print("\nDemand Segments Assigned")
    # ---------------------------------------------------------
    # Export CSV
    # ---------------------------------------------------------
    
    def export_results(self):
        """
        Export clustering results.
        """
    
        output_file = (
            self.output_dir /
            "clusters.csv"
        )
    
        self.cluster_data.to_csv(
            output_file,
            index=False
        )
    
        print(f"✓ Results exported to {output_file}")
    # ---------------------------------------------------------
    # Business Recommendation
    # ---------------------------------------------------------
    
    def business_recommendations(self):
        """
        Recommend stocking strategies
        based on demand segments.
        """
    
        print("\n")
        print("=" * 70)
        print("BUSINESS RECOMMENDATIONS")
        print("=" * 70)
    
        recommendations = {
            "High Volume, Stable Demand":
                "Maintain high inventory and prioritize replenishment.",
            "Growing Demand":
                "Increase stock gradually and monitor future trends.",
            "Low Volume, High Volatility":
                "Keep limited stock and review demand frequently.",
            "Declining Demand":
                "Reduce inventory and avoid overstocking."
        }
    
        for segment in self.cluster_data["Demand Segment"].unique():
    
            print(f"\n{segment}")
    
            print(
                recommendations.get(
                    segment,
                    "Monitor performance."
                )
            )
            recommendation_df = pd.DataFrame({
                "Demand Segment": list(recommendations.keys()),
                "Recommendation": list(recommendations.values())
          })

            recommendation_df.to_csv(
                self.output_dir / "business_recommendations.csv",
                index=False
            )
    # ---------------------------------------------------------
    # Run Pipeline
    # ---------------------------------------------------------
    
    def run(self):
        """
        Execute product segmentation workflow.
        """
    
        self.prepare_dataset()
    
        self.select_features()
    
        self.scale_features()
    
        self.elbow_method()
    
        self.train_model()
    
        self.silhouette()
    
        self.cluster_summary()
    
        self.cluster_size()
    
        self.apply_pca()
    
        self.assign_cluster_labels()
    
        self.plot_clusters()
    
        self.export_results()
    
        self.business_recommendations()

        print("\nOutputs Saved")

        print(f"Charts : {CHARTS_DIR}")

        print(f"CSV Files : {self.output_dir}")
    
        print("\n✓ Product Demand Segmentation Completed Successfully")
# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    clustering = ProductClustering()

    clustering.run()
    