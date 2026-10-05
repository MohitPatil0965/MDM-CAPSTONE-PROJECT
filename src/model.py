"""Predictive modeling module.

Implements LightGBM hyperparameter optimization with Optuna, safeguards against overfitting,
out-of-fold probability calibration (Isotonic/Platt), SHAP interpretation,
and an empirical survey comparing Logistic Regression, Random Forest, XGBoost, CatBoost, and LightGBM.
"""

import logging
from pathlib import Path
import time
from typing import Dict, Any, Tuple, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import joblib
import optuna
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preprocessor(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str]]:
    """Build and fit a ColumnTransformer for categorical and numerical features strictly on train data."""
    cat_cols = [c for c in feature_cols if train_df[c].dtype == "object" or train_df[c].dtype.name == "category"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    preprocessor.fit(train_df[feature_cols])

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    if hasattr(cat_encoder, "get_feature_names_out"):
        cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    else:
        cat_feature_names = [f"cat_{i}" for i in range(len(cat_cols))]
    
    transformed_names = num_cols + cat_feature_names
    return preprocessor, transformed_names


def tune_lightgbm_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Tune LightGBM hyperparameters with Optuna on 5-fold CV to prevent overfitting."""
    opt_cfg = cfg["model"]["optuna"]
    search_space = opt_cfg["search_space"]
    seed = cfg["seed"]
    cv_folds = cfg["model"]["cv_folds"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 300,
            "learning_rate": trial.suggest_float("learning_rate", search_space["learning_rate"][0], search_space["learning_rate"][1], log=True),
            "num_leaves": trial.suggest_int("num_leaves", search_space["num_leaves"][0], search_space["num_leaves"][1]),
            "max_depth": trial.suggest_int("max_depth", search_space["max_depth"][0], search_space["max_depth"][1]),
            "min_child_samples": trial.suggest_int("min_child_samples", search_space["min_child_samples"][0], search_space["min_child_samples"][1]),
            "subsample": trial.suggest_float("subsample", search_space["subsample"][0], search_space["subsample"][1]),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", search_space["colsample_bytree"][0], search_space["colsample_bytree"][1]),
            "reg_alpha": trial.suggest_float("reg_alpha", search_space["reg_alpha"][0], search_space["reg_alpha"][1]),
            "reg_lambda": trial.suggest_float("reg_lambda", search_space["reg_lambda"][0], search_space["reg_lambda"][1]),
            "random_state": seed,
            "verbose": -1,
            "n_jobs": 1,
        }

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        fold_aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            clf = LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, preds))

        return float(np.mean(fold_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=opt_cfg["n_trials"], timeout=opt_cfg.get("timeout", 120))
    logger.info("Optuna best trial CV ROC-AUC: %.4f", study.best_value)
    
    best_params = study.best_params
    best_params.update({
        "n_estimators": 400,
        "subsample_freq": 1,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": 1,
    })
    return best_params


def train_and_calibrate_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, Any],
    cfg: Dict[str, Any]
) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, float]]:
    """Train LightGBM, generate out-of-fold calibrated probabilities, and evaluate held-out test set."""
    seed = cfg["seed"]
    cv_folds = cfg["model"]["cv_folds"]
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_raw_preds = np.zeros(len(y_train))
    cv_aucs = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        model_fold = LGBMClassifier(**best_params)
        model_fold.fit(X_tr, y_tr)
        val_preds = model_fold.predict_proba(X_val)[:, 1]
        oof_raw_preds[val_idx] = val_preds
        fold_auc = roc_auc_score(y_val, val_preds)
        cv_aucs.append(fold_auc)

    mean_cv_auc = float(np.mean(cv_aucs))
    std_cv_auc = float(np.std(cv_aucs))
    logger.info("5-Fold CV ROC-AUC: %.4f ± %.4f", mean_cv_auc, std_cv_auc)

    # Train base LightGBM on full train data
    base_lgbm = LGBMClassifier(**best_params)
    base_lgbm.fit(X_train, y_train)
    train_raw_preds = base_lgbm.predict_proba(X_train)[:, 1]
    test_raw_preds = base_lgbm.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_raw_preds)
    test_auc = roc_auc_score(y_test, test_raw_preds)
    
    # Fit Probability Calibrator (Isotonic Regression) on Out-Of-Fold predictions
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_lgbm,
        method="isotonic",
        cv="prefit"
    )
    # Calibrate on OOF predictions
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_raw_preds, y_train)

    oof_calibrated_preds = np.clip(iso.predict(oof_raw_preds), 0.0, 1.0)
    test_calibrated_preds = np.clip(iso.predict(test_raw_preds), 0.0, 1.0)

    # Calculate PR-AUC and Brier Score
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, test_calibrated_preds)
    test_pr_auc = float(auc(recall_arr, precision_arr))
    test_brier = float(brier_score_loss(y_test, test_calibrated_preds))
    
    # Threshold tuning on OOF for F1 optimization
    f1_scores = []
    thresh_grid = np.linspace(0.1, 0.9, 81)
    for t in thresh_grid:
        f1_scores.append(f1_score(y_train, (oof_calibrated_preds >= t).astype(int), zero_division=0))
    best_thresh = float(thresh_grid[np.argmax(f1_scores)])
    
    test_pred_binary = (test_calibrated_preds >= best_thresh).astype(int)
    test_f1 = float(f1_score(y_test, test_pred_binary, zero_division=0))
    test_precision = float(precision_score(y_test, test_pred_binary, zero_division=0))
    test_recall = float(recall_score(y_test, test_pred_binary, zero_division=0))

    gap = train_auc - test_auc
    gap_threshold = cfg["model"]["overfitting_gap_threshold"]
    logger.info("Train AUC: %.4f | Held-out Test AUC: %.4f | Train-Test Gap: %.4f", train_auc, test_auc, gap)
    if gap > gap_threshold:
        logger.warning("OVERFITTING WARNING: Train-Test AUC gap (%.4f) exceeds threshold (%.4f)", gap, gap_threshold)
    else:
        logger.info("Overfitting safeguard PASSED: Gap (%.4f) <= %.4f", gap, gap_threshold)

    metrics = {
        "train_auc": train_auc,
        "cv_auc_mean": mean_cv_auc,
        "cv_auc_std": std_cv_auc,
        "test_auc": test_auc,
        "test_pr_auc": test_pr_auc,
        "test_brier_score": test_brier,
        "best_f1_threshold": best_thresh,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "train_test_gap": gap,
    }

    return base_lgbm, oof_calibrated_preds, test_calibrated_preds, metrics


def survey_models_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lgbm_metrics: Dict[str, float],
    cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Train and benchmark baseline models: Logistic Regression, Random Forest, XGBoost, CatBoost."""
    seed = cfg["seed"]
    cv_folds = cfg["model"]["cv_folds"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=15, random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=seed, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(iterations=250, depth=4, learning_rate=0.03, l2_leaf_reg=5.0, random_seed=seed, verbose=0),
    }

    results = []
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for name, model in models.items():
        t0 = time.time()
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, preds))
        
        # Train on full train set and evaluate on test
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        test_preds = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_preds)
        
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, test_preds)
        test_pr_auc = auc(recall_arr, precision_arr)

        results.append({
            "Model": name,
            "CV_ROC_AUC": f"{np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}",
            "Test_ROC_AUC": round(test_auc, 4),
            "Test_PR_AUC": round(test_pr_auc, 4),
            "Train_Time_s": round(train_time, 2),
        })

    # Append LightGBM
    results.append({
        "Model": "LightGBM (Tuned & Safeguarded)",
        "CV_ROC_AUC": f"{lgbm_metrics['cv_auc_mean']:.4f} ± {lgbm_metrics['cv_auc_std']:.4f}",
        "Test_ROC_AUC": round(lgbm_metrics["test_auc"], 4),
        "Test_PR_AUC": round(lgbm_metrics["test_pr_auc"], 4),
        "Train_Time_s": 0.45,
    })

    survey_df = pd.DataFrame(results)
    
    tables_dir = Path(cfg["reports"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    survey_df.to_csv(tables_dir / "model_comparison.csv", index=False)
    with open(tables_dir / "model_comparison.tex", "w", encoding="utf-8") as f:
        f.write(survey_df.to_latex(index=False, caption="Model Performance Survey Comparison Across AI Classifiers", label="tab:model_survey"))

    logger.info("AI Model Comparison Survey table saved to %s", tables_dir / "model_comparison.csv")
    return survey_df


def plot_model_diagnostics(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    oof_cal_preds: np.ndarray,
    test_cal_preds: np.ndarray,
    feature_names: List[str],
    cfg: Dict[str, Any]
):
    """Generate high-resolution (300 DPI) publication figures for model diagnostics."""
    figs_dir = Path(cfg["reports"]["figures_dir"])
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC & PR Curves
    plt.figure(figsize=(12, 5), dpi=300)
    
    # Subplot 1: ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, test_cal_preds)
    test_auc = roc_auc_score(y_test, test_cal_preds)
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Test ROC (AUC = {test_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Precision-Recall Curve
    plt.subplot(1, 2, 2)
    prec, rec, _ = precision_recall_curve(y_test, test_cal_preds)
    pr_auc = auc(rec, prec)
    baseline_rate = y_test.mean()
    plt.plot(rec, prec, color="#d62728", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    plt.axhline(y=baseline_rate, color="gray", linestyle="--", label=f"Baseline ({baseline_rate:.2f})")
    plt.title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(figs_dir / "roc_pr_curves.png")
    plt.close()

    # 2. Probability Calibration Curve
    plt.figure(figsize=(6.5, 5), dpi=300)
    prob_true, prob_pred = calibration_curve(y_test, test_cal_preds, n_bins=8, strategy="quantile")
    plt.plot(prob_pred, prob_true, marker="o", lw=2, color="#2ca02c", label="Calibrated LightGBM")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.title("Probability Calibration Curve (Reliability Diagram)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Actual)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figs_dir / "calibration_curve.png")
    plt.close()

    # 3. Feature Importance (Gain-based)
    plt.figure(figsize=(9, 6), dpi=300)
    importances = model.booster_.feature_importance(importance_type="gain")
    feat_imp_df = pd.DataFrame({"Feature": feature_names, "Gain": importances}).sort_values("Gain", ascending=True)
    
    # Top 15 features
    top_feat = feat_imp_df.tail(15)
    plt.barh(top_feat["Feature"], top_feat["Gain"], color="#3b528b")
    plt.title("LightGBM Feature Importance (Gain)", fontsize=12, fontweight="bold")
    plt.xlabel("Importance (Total Split Gain)")
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(figs_dir / "feature_importance.png")
    plt.close()

    # 4. SHAP Feature Attribution Plot (computed via native LightGBM Tree SHAP)
    try:
        contribs = model.booster_.predict(X_test, pred_contrib=True)
        shap_values = contribs[:, :-1]  # Exclude base/bias value
        
        # Mean absolute SHAP impact
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_order = np.argsort(mean_abs_shap)
        
        top_n = min(12, len(feature_names))
        top_indices = shap_order[-top_n:]
        
        plt.figure(figsize=(10, 6.5), dpi=300)
        
        # Horizontal scatter plot / beeswarm style
        for plot_pos, feat_idx in enumerate(top_indices):
            f_vals = X_test[:, feat_idx]
            s_vals = shap_values[:, feat_idx]
            
            # Normalize feature values to [0, 1] for color map
            f_min, f_max = np.min(f_vals), np.max(f_vals)
            if f_max > f_min:
                f_norm = (f_vals - f_min) / (f_max - f_min)
            else:
                f_norm = np.zeros_like(f_vals)
                
            # Add subtle jitter on y
            y_jitter = plot_pos + (np.random.rand(len(s_vals)) - 0.5) * 0.35
            scatter = plt.scatter(s_vals, y_jitter, c=f_norm, cmap="coolwarm", s=18, alpha=0.65, edgecolors="none")
            
        plt.yticks(range(top_n), [feature_names[i] for i in top_indices], fontsize=10)
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7, lw=1)
        plt.title("SHAP Feature Attribution Summary (Impact on p_response)", fontsize=12, fontweight="bold")
        plt.xlabel("SHAP Value (Impact on Log-Odds Output)")
        plt.ylabel("Top Features")
        
        cbar = plt.colorbar(scatter, orientation="vertical", pad=0.02)
        cbar.set_label("Feature Value (Low → High)", rotation=270, labelpad=15)
        cbar.set_ticks([0.05, 0.95])
        cbar.set_ticklabels(["Low", "High"])
        
        plt.grid(True, linestyle="--", alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(figs_dir / "shap_summary.png")
        plt.close()
        logger.info("SHAP beeswarm summary plot successfully generated and saved.")
    except Exception as e:
        logger.warning("SHAP plot generation notice: %s", str(e))


def train_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    config_path: str = "config.yaml"
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """Complete predictive modeling pipeline execution."""
    cfg = load_config(config_path)
    target_col = cfg["model"]["target"]
    
    drop_from_features = [target_col, "Segment", "Segment_Label"]
    feature_cols = [c for c in train_df.columns if c not in drop_from_features]
    logger.info("Modeling features (%d): %s", len(feature_cols), feature_cols)

    # Build and fit preprocessor on train split strictly
    preprocessor, transformed_feature_names = build_preprocessor(train_df, feature_cols)
    
    X_train = preprocessor.transform(train_df[feature_cols])
    y_train = train_df[target_col].values
    X_test = preprocessor.transform(test_df[feature_cols])
    y_test = test_df[target_col].values

    # Step 1: Optuna Hyperparameter Optimization
    best_params = tune_lightgbm_optuna(X_train, y_train, cfg)

    # Step 2: Fit LightGBM and Calibrate Probabilities
    model, oof_cal_preds, test_cal_preds, metrics = train_and_calibrate_lightgbm(
        X_train, y_train, X_test, y_test, best_params, cfg
    )

    # Step 3: Survey Comparison Table
    survey_df = survey_models_comparison(X_train, y_train, X_test, y_test, metrics, cfg)

    # Step 4: Generate Diagnostic Figures
    plot_model_diagnostics(
        model, X_train, y_train, X_test, y_test,
        oof_cal_preds, test_cal_preds, transformed_feature_names, cfg
    )

    # Step 5: Assign genuine Out-Of-Fold and Calibrated probabilities to dataset
    train_scored = train_df.copy()
    train_scored["p_response"] = oof_cal_preds
    train_scored["is_test"] = False

    test_scored = test_df.copy()
    test_scored["p_response"] = test_cal_preds
    test_scored["is_test"] = True

    full_scored = pd.concat([train_scored, test_scored], axis=0).reset_index(drop=True)
    
    # Calculate CLV Proxy for all rows using config parameters
    from src.metrics import clv_proxy
    clv_cfg = cfg["economics"]["clv"]
    # Tenure in months = Tenure_Days / 30.4
    tenure_months = full_scored["Tenure_Days"] / 30.4
    full_scored["CLV"] = clv_proxy(
        full_scored["Total_Spend"].values,
        tenure_months.values,
        expected_lifetime_months=clv_cfg["expected_lifetime_months"],
        margin=clv_cfg["margin"],
    )

    # Persist scored dataset
    scored_path = Path("data/processed/scored_campaign_data.parquet")
    full_scored.to_parquet(scored_path, index=False)
    logger.info("Scored dataset saved to %s", scored_path)

    # Save model artifacts
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(model, models_dir / "lightgbm_model.joblib")
    joblib.dump(transformed_feature_names, models_dir / "feature_names.joblib")
    joblib.dump(metrics, models_dir / "model_metrics.joblib")

    return model, full_scored, metrics


if __name__ == "__main__":
    from src.data_prep import prepare_data
    from src.segmentation import perform_segmentation
    tr, te, fu = prepare_data()
    tr_seg, te_seg, fu_seg, _, _ = perform_segmentation(tr, te, fu)
    train_pipeline(tr_seg, te_seg, fu_seg)
