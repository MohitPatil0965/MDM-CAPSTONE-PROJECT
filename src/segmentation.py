"""Audience segmentation module using K-Means clustering.

Fits StandardScaler and KMeans strictly on the training set, evaluates K via
elbow method and silhouette scores, generates descriptive segment personas,
and profiles clusters.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_persona_labels(centroids_df: pd.DataFrame) -> Dict[int, str]:
    """Generate meaningful human-readable marketing persona labels based on cluster centroids.

    Args:
        centroids_df: DataFrame of unscaled cluster centroids with features like
                      Income, Total_Spend, Recency, NumWebVisitsMonth, Age, Total_Purchases.

    Returns:
        Dictionary mapping cluster ID to persona name.
    """
    labels = {}
    k = len(centroids_df)
    
    # Ranks across key dimensions
    spend_ranks = centroids_df["Total_Spend"].rank(ascending=False).astype(int)
    income_ranks = centroids_df["Income"].rank(ascending=False).astype(int)
    recency_ranks = centroids_df["Recency"].rank(ascending=True).astype(int) # lower recency = more recent
    web_visits_ranks = centroids_df["NumWebVisitsMonth"].rank(ascending=False).astype(int)

    assigned_names = set()

    for idx, row in centroids_df.iterrows():
        spend = row["Total_Spend"]
        inc = row["Income"]
        rec = row["Recency"]
        web = row["NumWebVisitsMonth"]

        if spend_ranks[idx] == 1:
            name = "High-Value Loyalists"
        elif spend_ranks[idx] == k:
            if web_visits_ranks[idx] <= 2:
                name = "Price-Sensitive Deal Seekers"
            else:
                name = "Low-Engagement Budget Shoppers"
        elif recency_ranks[idx] == k:
            name = "Lapsed Moderate Spenders"
        elif web_visits_ranks[idx] == 1:
            name = "Active Digital Window Shoppers"
        else:
            name = f"Mid-Tier Core Segment {idx+1}"

        # Ensure uniqueness
        original_name = name
        counter = 2
        while name in assigned_names:
            name = f"{original_name} (Group {counter})"
            counter += 1
            
        assigned_names.add(name)
        labels[idx] = name

    return labels


def perform_segmentation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    config_path: str = "config.yaml"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Fit K-Means segmentation pipeline on training data and transform splits.

    Returns:
        (train_with_segments, test_with_segments, full_with_segments, segment_profiles, artifacts)
    """
    cfg = load_config(config_path)
    features = cfg["segmentation"]["features"]
    k_min = cfg["segmentation"]["k_min"]
    k_max = cfg["segmentation"]["k_max"]
    k_override = cfg["segmentation"]["k_override"]
    seed = cfg["seed"]

    Path(cfg["reports"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["reports"]["tables_dir"]).mkdir(parents=True, exist_ok=True)

    # Scale features strictly on train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[features])
    X_test_scaled = scaler.transform(test_df[features])
    X_full_scaled = scaler.transform(full_df[features])

    # Evaluate K candidates
    k_candidates = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels_k = km.fit_predict(X_train_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_train_scaled, labels_k)
        silhouettes.append(sil)
        logger.info("K=%d | Inertia: %.2f | Silhouette: %.4f", k, km.inertia_, sil)

    # Plot Elbow Curve
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(k_candidates, inertias, marker="o", color="#1f77b4", lw=2)
    plt.title("Elbow Method for Optimal K", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    elbow_fig_path = Path(cfg["reports"]["figures_dir"]) / "elbow_plot.png"
    plt.savefig(elbow_fig_path)
    plt.close()

    # Plot Silhouette Scores
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(k_candidates, silhouettes, marker="s", color="#2ca02c", lw=2)
    plt.title("Silhouette Score vs Number of Clusters", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    sil_fig_path = Path(cfg["reports"]["figures_dir"]) / "silhouette_plot.png"
    plt.savefig(sil_fig_path)
    plt.close()

    # Select optimal K
    if k_override is not None:
        best_k = int(k_override)
        logger.info("Using configured K override: %d", best_k)
    else:
        best_idx = int(np.argmax(silhouettes))
        best_k = k_candidates[best_idx]
        logger.info("Automatically selected optimal K=%d (Best Silhouette: %.4f)", best_k, silhouettes[best_idx])

    # Fit final model with best K
    final_kmeans = KMeans(n_clusters=best_k, random_state=seed, n_init=20)
    train_clusters = final_kmeans.fit_predict(X_train_scaled)
    test_clusters = final_kmeans.predict(X_test_scaled)
    full_clusters = final_kmeans.predict(X_full_scaled)

    train_out = train_df.copy()
    test_out = test_df.copy()
    full_out = full_df.copy()

    train_out["Segment"] = train_clusters
    test_out["Segment"] = test_clusters
    full_out["Segment"] = full_clusters

    # Calculate unscaled cluster centroids
    centroids_unscaled = pd.DataFrame(
        scaler.inverse_transform(final_kmeans.cluster_centers_),
        columns=features
    )
    persona_dict = assign_persona_labels(centroids_unscaled)

    train_out["Segment_Label"] = train_out["Segment"].map(persona_dict)
    test_out["Segment_Label"] = test_out["Segment"].map(persona_dict)
    full_out["Segment_Label"] = full_out["Segment"].map(persona_dict)

    # Compute Segment Profiles Table across the full dataset
    target_col = cfg["model"]["target"]
    profiles = []
    for seg_id in range(best_k):
        seg_data = full_out[full_out["Segment"] == seg_id]
        profiles.append({
            "Segment_ID": seg_id,
            "Persona": persona_dict[seg_id],
            "Customer_Count": len(seg_data),
            "Share_Pct": round(len(seg_data) / len(full_out) * 100, 2),
            "Mean_Income": round(seg_data["Income"].mean(), 2),
            "Mean_Total_Spend": round(seg_data["Total_Spend"].mean(), 2),
            "Mean_Purchases": round(seg_data["Total_Purchases"].mean(), 2),
            "Mean_Recency": round(seg_data["Recency"].mean(), 2),
            "Mean_Web_Visits": round(seg_data["NumWebVisitsMonth"].mean(), 2),
            "Mean_Age": round(seg_data["Age"].mean(), 2),
            "Observed_Response_Rate_Pct": round(seg_data[target_col].mean() * 100, 2),
        })

    profiles_df = pd.DataFrame(profiles)

    # Save Profile Tables
    tables_dir = Path(cfg["reports"]["tables_dir"])
    profiles_df.to_csv(tables_dir / "segment_profiles.csv", index=False)
    with open(tables_dir / "segment_profiles.tex", "w", encoding="utf-8") as f:
        f.write(profiles_df.to_latex(index=False, caption="Customer Segment Profiles and Behavioral Attributes", label="tab:segment_profiles"))

    logger.info("Segment profiles generated and saved to %s", tables_dir / "segment_profiles.csv")

    # Save models
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "segmentation_scaler.joblib")
    joblib.dump(final_kmeans, models_dir / "segmentation_kmeans.joblib")
    joblib.dump(persona_dict, models_dir / "segment_personas.joblib")

    artifacts = {
        "best_k": best_k,
        "silhouettes": dict(zip(k_candidates, silhouettes)),
        "inertias": dict(zip(k_candidates, inertias)),
        "persona_dict": persona_dict,
        "scaler": scaler,
        "kmeans": final_kmeans,
    }

    return train_out, test_out, full_out, profiles_df, artifacts


if __name__ == "__main__":
    from src.data_prep import prepare_data
    tr, te, fu = prepare_data()
    perform_segmentation(tr, te, fu)
