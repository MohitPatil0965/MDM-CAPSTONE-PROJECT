"""Unit tests for data preparation module."""

import pandas as pd
import numpy as np
from src.data_prep import clean_categories, engineer_features, prepare_data


def test_clean_categories():
    raw_sample = pd.DataFrame({
        "Education": ["Graduation", "PhD", "Master", "2n Cycle", "Basic", "Unknown"],
        "Marital_Status": ["Married", "Together", "Single", "Divorced", "Widow", "YOLO"]
    })
    cleaned = clean_categories(raw_sample)
    
    assert cleaned["Education"].tolist() == ["Graduate", "Postgraduate", "Postgraduate", "Postgraduate", "Basic", "Graduate"]
    assert cleaned["Marital_Status"].tolist() == ["Partner", "Partner", "Single", "Single", "Single", "Single"]


def test_engineer_features():
    raw_sample = pd.DataFrame({
        "ID": [1],
        "Year_Birth": [1980],
        "Dt_Customer": ["2014-06-15"],
        "Kidhome": [1],
        "Teenhome": [1],
        "MntWines": [100],
        "MntFruits": [20],
        "MntMeatProducts": [50],
        "MntFishProducts": [10],
        "MntSweetProducts": [5],
        "MntGoldProds": [15],
        "NumWebPurchases": [4],
        "NumCatalogPurchases": [2],
        "NumStorePurchases": [4],
        "NumDealsPurchases": [2],
        "AcceptedCmp1": [0],
        "AcceptedCmp2": [1],
        "AcceptedCmp3": [0],
        "AcceptedCmp4": [0],
        "AcceptedCmp5": [1],
        "Z_CostContact": [3],
        "Z_Revenue": [11],
    })
    feat = engineer_features(raw_sample)
    
    assert "ID" not in feat.columns
    assert "Z_CostContact" not in feat.columns
    assert "Total_Spend" in feat.columns
    assert feat["Total_Spend"].iloc[0] == 200  # 100+20+50+10+5+15
    assert feat["Total_Purchases"].iloc[0] == 10  # 4+2+4
    assert feat["Kids"].iloc[0] == 2
    assert feat["Prev_Accepted"].iloc[0] == 2
    assert feat["Spend_per_Purchase"].iloc[0] == 20.0
    assert feat["Web_Purchase_Share"].iloc[0] == 0.4


def test_prepare_data_e2e():
    train_df, test_df, full_df = prepare_data()
    
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert train_df["Income"].isna().sum() == 0
    assert test_df["Income"].isna().sum() == 0
    assert "Response" in train_df.columns
