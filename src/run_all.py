"""End-to-End Pipeline Execution Script.

Runs the complete campaign optimization workflow from raw XLSX data to
segmentation, modeling, multi-objective optimization, baseline evaluation,
and academic paper artifacts in a single command:
    python -m src.run_all
"""

import logging
import time
from pathlib import Path
import yaml
import pandas as pd

from src.data_prep import prepare_data
from src.segmentation import perform_segmentation
from src.model import train_pipeline
from src.optimizer import run_nsga2_optimization
from src.evaluate import run_full_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("CampaignOptimizer")


def run_pipeline(config_path: str = "config.yaml"):
    """Execute all pipeline steps end to end."""
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("STARTING AI-DRIVEN MULTI-OBJECTIVE CAMPAIGN OPTIMIZATION PIPELINE")
    logger.info("================================================================================")

    # 1. Data Preparation
    logger.info("\n--- STEP 1: DATA PREPARATION & STRATIFIED SPLIT ---")
    train_df, test_df, full_df = prepare_data(config_path)

    # 2. Customer Segmentation
    logger.info("\n--- STEP 2: AUDIENCE SEGMENTATION (K-MEANS) ---")
    train_seg, test_seg, full_seg, profiles_df, seg_artifacts = perform_segmentation(
        train_df, test_df, full_df, config_path
    )
    logger.info("Optimal K: %d | Segments: %s", seg_artifacts["best_k"], list(seg_artifacts["persona_dict"].values()))

    # 3. Predictive Modeling & Probability Calibration
    logger.info("\n--- STEP 3: PREDICTIVE MODELING & OOF CALIBRATION (LIGHTGBM) ---")
    model, scored_df, model_metrics = train_pipeline(train_seg, test_seg, full_seg, config_path)
    
    logger.info(">>> HEADLINE METRIC - 5-Fold Stratified CV ROC-AUC: %.4f ± %.4f", model_metrics["cv_auc_mean"], model_metrics["cv_auc_std"])
    logger.info(">>> Held-out Test ROC-AUC: %.4f | PR-AUC: %.4f", model_metrics["test_auc"], model_metrics["test_pr_auc"])
    logger.info(">>> Train AUC: %.4f | Train-Test Gap: %.4f", model_metrics["train_auc"], model_metrics["train_test_gap"])
    logger.info(">>> Brier Score: %.4f | F1 Score: %.4f", model_metrics["test_brier_score"], model_metrics["test_f1"])

    # 4. Multi-Objective Optimization
    logger.info("\n--- STEP 4: MULTI-OBJECTIVE OPTIMIZATION (NSGA-II) ---")
    pareto_df, alloc_df, opt_stats, helper = run_nsga2_optimization(scored_df, config_path)
    logger.info("Generated %d non-dominated Pareto solutions across 5 seeds.", len(pareto_df))
    logger.info("Multi-seed Hypervolume: %.4e ± %.4e", opt_stats["mean_hypervolume"], opt_stats["std_hypervolume"])

    # 5. Baseline Evaluation & Paper Reporting
    logger.info("\n--- STEP 5: BASELINE BENCHMARKING & PUBLICATION REPORTING ---")
    run_full_evaluation(config_path)

    elapsed = time.time() - start_time
    logger.info("\n================================================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS", elapsed)
    logger.info("================================================================================")


if __name__ == "__main__":
    run_pipeline()
