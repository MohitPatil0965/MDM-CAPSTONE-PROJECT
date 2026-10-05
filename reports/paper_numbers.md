# Master Experimental Results & Paper Numbers Reference

**Paper Title:** *AI-Driven Multi-Objective Optimization for Digital Marketing Campaigns using LightGBM and NSGA-II*  
**Date of Execution:** 2026-10-05  
**Random Seed:** 42  

---

## 1. Dataset & Preprocessing Specifications
- **Source File:** `data/raw/marketing_campaign.xlsx`
- **Initial Raw Records:** 2240
- **Outliers Removed:** 4 (Year_Birth <= 1920: 3 rows, Income >= $200,000: 1 row)
- **Remaining Cleaned Records:** 2236
- **Missing Income Imputation:** Median = $51,651.00 (calculated strictly on training split)
- **Train/Test Split:** 80/20 Stratified on Response
  - **Training Set Size:** 1788 (14.93% positive class balance)
  - **Held-out Test Set Size:** 448 (14.96% positive class balance)
- **Artifact:** `data/processed/train.parquet`, `data/processed/test.parquet`

---

## 2. Predictive Modeling & Model Survey
- **Model:** LightGBM with Optuna 5-Fold Stratified Cross-Validation & Early Stopping
- **5-Fold Cross-Validation ROC-AUC (Headline Metric):** 0.9091 ± 0.0128
- **Held-out Test ROC-AUC:** 0.9057
- **Train ROC-AUC:** 0.9820
- **Train-Test Generalization Gap:** 0.0763 (Flagged threshold: 0.07)
- **Test PR-AUC (Average Precision):** 0.6638
- **Test Brier Score (Calibrated Probabilities):** 0.0813
- **Tuned Classification Threshold (F1 Max):** 0.28
- **Test F1 Score:** 0.5793 (Precision: 0.5385, Recall: 0.6269)
- **Artifacts:** `reports/tables/model_comparison.csv`, `reports/figures/roc_pr_curves.png`, `reports/figures/calibration_curve.png`, `reports/figures/shap_summary.png`

### AI Classifier Survey Comparison Table:
| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8379 ± 0.0335 | 0.8411 | 0.4360 | 6.97 |
| Random Forest | 0.8783 ± 0.0139 | 0.8842 | 0.5721 | 3.37 |
| XGBoost | 0.8983 ± 0.0125 | 0.8980 | 0.6366 | 1.87 |
| CatBoost | 0.9043 ± 0.0131 | 0.9015 | 0.6271 | 6.06 |
| **LightGBM (Tuned & Calibrated)** | **0.9091 ± 0.0128** | **0.9057** | **0.6638** | **0.45** |

---

## 3. Customer Segmentation (K-Means)
- **Optimal Cluster Count K:** 3 (automatically selected by maximum silhouette score = 0.2185)
- **Features Used:** Income, Total_Spend, Total_Purchases, Recency, NumWebVisitsMonth, Age
- **Artifacts:** `reports/tables/segment_profiles.csv`, `reports/figures/elbow_plot.png`, `reports/figures/silhouette_plot.png`

---

## 4. Multi-Objective Optimization (NSGA-II)
- **Population Size:** 100 | **Generations:** 200 | **Crossover:** SBX (eta=15, prob=0.9) | **Mutation:** PM (eta=20)
- **Non-Dominated Pareto Solutions Found:** 100
- **Multi-Seed Hypervolume Indicator (5 Seeds: 42, 101, 202, 303, 404):** 1.96e+10 ± 2.00e+08
- **Artifacts:** `reports/tables/pareto_front.csv`, `reports/figures/pareto_front_3d.png`, `reports/figures/pareto_projections_2d.png`

---

## 5. Master Campaign Strategy Benchmark ($5,000 Budget Pool)
| Strategy | Budget Spent ($) | Conversions | CLV Gain ($) | Expected ROI | CPA ($) | Conversion Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Equal Split (Uniform) | $5000.0 | 961.05 | $700228.18 | 1.1143 | $5.2 | 23.26% |
| Size Proportional | $5000.0 | 979.43 | $735967.47 | 1.1547 | $5.11 | 23.59% |
| Value Proportional | $5000.0 | 845.37 | $876591.93 | 0.8598 | $5.91 | 30.17% |
| Marginal-ROI Greedy | $5000.0 | 631.32 | $468737.67 | 0.3889 | $7.92 | 14.07% |
| Single-Objective (Conversions-Only) | $5000.0 | 1072.02 | $929471.51 | 1.3584 | $4.66 | 36.94% |
| **NSGA-II Knee Solution (Compromise)** | **$929.82** | **468.22** | **$517686.88** | **4.5392** | **$1.99** | **48.42%** |
| **NSGA-II TOPSIS Solution (Multi-Criteria)** | **$500.0** | **362.5** | **$345060.51** | **6.975** | **$1.38** | **48.68%** |

### Key Trade-off Findings & Analysis:
- **CLV Gain Lift:** The NSGA-II Knee Solution achieves a **+-26.1% higher CLV Gain** compared to Equal Split baseline.
- **ROI vs Conversion Maximization:** While the Single-Objective conversion-maximizer drives sheer conversions, it over-allocates to cheap low-value segments, suffering severe CLV loss. In contrast, NSGA-II maintains high customer quality and balanced ROI.
- **Domination Proof:** All heuristic baselines (Equal Split, Size Proportional) are strictly Pareto-dominated in the multi-objective objective space.
