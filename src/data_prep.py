"""Data preparation and feature engineering module.

Implements outlier filtering, categorical cleaning, feature engineering,
and stratified train/test splitting with strict no-data-leakage median imputation.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load system configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize messy categorical values for Education and Marital_Status.

    Mappings:
        Marital_Status:
            Partner: Married, Together
            Single: Single, Divorced, Widow, Widowed, Alone, Absurd, YOLO
        Education:
            Basic: Basic
            Graduate: Graduation
            Postgraduate: 2n Cycle, Master, PhD
    """
    df = df.copy()
    
    marital_map = {
        "Married": "Partner",
        "Together": "Partner",
        "Single": "Single",
        "Divorced": "Single",
        "Widow": "Single",
        "Widowed": "Single",
        "Alone": "Single",
        "Absurd": "Single",
        "YOLO": "Single",
    }
    df["Marital_Status"] = df["Marital_Status"].map(marital_map).fillna("Single")

    edu_map = {
        "Basic": "Basic",
        "Graduation": "Graduate",
        "2n Cycle": "Postgraduate",
        "Master": "Postgraduate",
        "PhD": "Postgraduate",
    }
    df["Education"] = df["Education"].map(edu_map).fillna("Graduate")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived marketing features using the latest Dt_Customer as reference."""
    df = df.copy()
    
    # Parse dates flexibly
    dt_series = pd.to_datetime(df["Dt_Customer"], format="mixed", dayfirst=True)
    ref_date = dt_series.max()
    logger.info("Reference date for tenure and age: %s", ref_date.strftime("%Y-%m-%d"))

    # Demographic & tenure features
    df["Age"] = ref_date.year - df["Year_Birth"]
    df["Tenure_Days"] = (ref_date - dt_series).dt.days

    # Spending & Purchases aggregations
    spend_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df["Total_Spend"] = df[spend_cols].sum(axis=1)
    
    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    df["Total_Purchases"] = df[purchase_cols].sum(axis=1)

    # Family & Campaign history
    df["Kids"] = df["Kidhome"] + df["Teenhome"]
    
    cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    existing_cmp_cols = [c for c in cmp_cols if c in df.columns]
    df["Prev_Accepted"] = df[existing_cmp_cols].sum(axis=1)

    # Interaction & Ratio features (safe from division by zero)
    safe_purchases = np.maximum(df["Total_Purchases"].values, 1.0)
    df["Spend_per_Purchase"] = df["Total_Spend"] / safe_purchases
    df["Web_Purchase_Share"] = df["NumWebPurchases"] / safe_purchases
    
    total_deals_purchases = df["Total_Purchases"] + df["NumDealsPurchases"]
    df["Deal_Share"] = df["NumDealsPurchases"] / np.maximum(total_deals_purchases.values, 1.0)

    # Drop raw and redundant columns
    drop_cols = [
        "ID", "Dt_Customer", "Z_CostContact", "Z_Revenue",
        "Year_Birth", "Kidhome", "Teenhome"
    ]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def prepare_data(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete data preparation pipeline with strict train-only imputation.

    Returns:
        Tuple of (train_df, test_df, full_cleaned_df)
    """
    cfg = load_config(config_path)
    raw_path = cfg["data"]["raw_path"]
    min_year = cfg["data"]["min_year_birth"]
    max_inc = cfg["data"]["max_income"]
    test_size = cfg["data"]["test_size"]
    seed = cfg["seed"]

    logger.info("Reading raw dataset from: %s", raw_path)
    if raw_path.endswith(".xlsx") or raw_path.endswith(".xls"):
        df_raw = pd.read_excel(raw_path, sheet_name="Sheet1")
    else:
        df_raw = pd.read_csv(raw_path)

    initial_len = len(df_raw)
    logger.info("Initial raw record count: %d", initial_len)

    # Filter outliers
    valid_mask = (df_raw["Year_Birth"] > min_year) & (
        (df_raw["Income"] < max_inc) | df_raw["Income"].isna()
    )
    df_cleaned = df_raw[valid_mask].copy().reset_index(drop=True)
    removed_count = initial_len - len(df_cleaned)
    logger.info("Outlier filtering removed %d rows (Remaining: %d)", removed_count, len(df_cleaned))

    # Standardize categories and engineer features
    df_cat = clean_categories(df_cleaned)
    df_feat = engineer_features(df_cat)

    # Stratified Train/Test split on target 'Response'
    target_col = cfg["model"]["target"]
    train_df, test_df = train_test_split(
        df_feat,
        test_size=test_size,
        stratify=df_feat[target_col],
        random_state=seed,
    )
    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    # No Data Leakage: compute median income ONLY on train split
    train_income_median = train_df["Income"].median()
    logger.info("Train-set Income median for imputation: %.2f", train_income_median)

    train_df["Income"] = train_df["Income"].fillna(train_income_median)
    test_df["Income"] = test_df["Income"].fillna(train_income_median)

    # Construct unified cleaned dataset for full scoring
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

    # Log class balance
    train_pos = train_df[target_col].sum()
    train_rate = train_df[target_col].mean()
    test_pos = test_df[target_col].sum()
    test_rate = test_df[target_col].mean()

    logger.info("Train Split: %d rows | Positives: %d (%.2f%%)", len(train_df), train_pos, train_rate * 100)
    logger.info("Test Split:  %d rows | Positives: %d (%.2f%%)", len(test_df), test_pos, test_rate * 100)

    # Persist processed parquet files
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(cfg["data"]["processed_train_path"], index=False)
    test_df.to_parquet(cfg["data"]["processed_test_path"], index=False)
    full_df.to_parquet(cfg["data"]["processed_full_path"], index=False)
    logger.info("Processed datasets saved to parquet format in data/processed/")

    return train_df, test_df, full_df


if __name__ == "__main__":
    prepare_data()
