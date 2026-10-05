"""Non-optimized and single-objective baseline campaign allocation strategies.

Implements standard industry heuristics and single-objective benchmarks:
1. Equal Split (Uniform across segments and channels)
2. Proportional to Segment Size
3. Proportional to Predicted Segment Value (Sum of CLV * p_response)
4. Marginal-ROI Greedy Allocation
5. Single-Objective Optimization (Maximize Conversions / CTR-only surrogate)

All strategies are evaluated using the EXACT same evaluation function as the multi-objective optimizer.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize

from src.optimizer import SegmentDataHelper, evaluate_allocation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def strategy_equal_split(helper: SegmentDataHelper, budget_fraction: float = 1.0) -> np.ndarray:
    """Equal budget allocation across all segments and channels."""
    K = helper.num_segments
    C = helper.num_channels
    weights = np.ones((K, C)) / (K * C)
    return weights


def strategy_size_proportional(helper: SegmentDataHelper, budget_fraction: float = 1.0) -> np.ndarray:
    """Allocation proportional to customer count per segment, split equally across channels."""
    K = helper.num_segments
    C = helper.num_channels
    seg_shares = helper.segment_sizes / np.sum(helper.segment_sizes)
    weights = np.zeros((K, C))
    for s in range(K):
        weights[s, :] = seg_shares[s] / C
    return weights


def strategy_value_proportional(
    helper: SegmentDataHelper,
    scored_df: pd.DataFrame,
    budget_fraction: float = 1.0
) -> np.ndarray:
    """Allocation proportional to predicted segment monetary value (Sum of CLV * p_response)."""
    K = helper.num_segments
    C = helper.num_channels
    
    seg_values = []
    for s_id in helper.segments:
        seg_data = scored_df[scored_df["Segment"] == s_id]
        val = np.sum(seg_data["p_response"] * seg_data["CLV"])
        seg_values.append(val)
        
    seg_values = np.array(seg_values, dtype=float)
    total_val = np.sum(seg_values)
    if total_val > 0:
        val_shares = seg_values / total_val
    else:
        val_shares = np.ones(K) / K

    weights = np.zeros((K, C))
    for s in range(K):
        weights[s, :] = val_shares[s] / C
    return weights


def strategy_marginal_roi_greedy(helper: SegmentDataHelper, budget_fraction: float = 1.0) -> np.ndarray:
    """Greedy allocation prioritizing cells with highest initial marginal ROI."""
    K = helper.num_segments
    C = helper.num_channels
    weights = np.zeros((K, C))

    # Compute initial marginal ROI for top customer in each (s, c) cell
    rois = np.zeros((K, C))
    for s in range(K):
        top_p = helper.cum_responses[s][1] if len(helper.cum_responses[s]) > 1 else 0.1
        for c in range(C):
            cost_c = helper.costs_per_contact[c]
            mult_c = helper.response_multipliers[c]
            exp_rev = min(top_p * mult_c, 1.0) * helper.revenue_per_conversion
            rois[s, c] = (exp_rev - cost_c) / cost_c

    # Softmax / exponential weighting on positive ROIs
    exp_rois = np.exp(np.clip(rois, -5, 5))
    weights = exp_rois / np.sum(exp_rois)
    return weights


def strategy_single_objective_conversions(
    helper: SegmentDataHelper,
    max_budget: float,
    budget_fraction: float = 1.0
) -> np.ndarray:
    """Single-objective optimization maximizing conversions only (CTR-only/Conversion-only heuristic)."""
    K = helper.num_segments
    C = helper.num_channels
    n_vars = K * C

    def obj_func(w_flat: np.ndarray) -> float:
        w_2d = w_flat.reshape((K, C))
        res = evaluate_allocation(
            f_budget=budget_fraction,
            raw_weights_2d=w_2d,
            helper=helper,
            max_budget=max_budget,
            include_fatigue=False
        )
        # Minimize negative conversions
        return -res["conversions"]

    # Initial guess: equal weights
    x0 = np.ones(n_vars) / n_vars
    bounds = [(0.0, 1.0) for _ in range(n_vars)]

    opt_res = scipy_minimize(obj_func, x0, method="SLSQP", bounds=bounds, options={"maxiter": 150})
    best_w = opt_res.x.reshape((K, C))
    best_w = best_w / (np.sum(best_w) + 1e-9)
    return best_w


def evaluate_all_baselines(
    helper: SegmentDataHelper,
    scored_df: pd.DataFrame,
    max_budget: float,
    budget_fraction: float = 1.0
) -> pd.DataFrame:
    """Evaluate all baseline strategies at the exact same budget level as optimizer solutions."""
    baselines = {}

    # 1. Equal Split
    w_eq = strategy_equal_split(helper, budget_fraction)
    baselines["Equal Split"] = evaluate_allocation(budget_fraction, w_eq, helper, max_budget)

    # 2. Size Proportional
    w_size = strategy_size_proportional(helper, budget_fraction)
    baselines["Size Proportional"] = evaluate_allocation(budget_fraction, w_size, helper, max_budget)

    # 3. Value Proportional
    w_val = strategy_value_proportional(helper, scored_df, budget_fraction)
    baselines["Value Proportional"] = evaluate_allocation(budget_fraction, w_val, helper, max_budget)

    # 4. Marginal-ROI Greedy
    w_roi = strategy_marginal_roi_greedy(helper, budget_fraction)
    baselines["Marginal-ROI Greedy"] = evaluate_allocation(budget_fraction, w_roi, helper, max_budget)

    # 5. Single-Objective Conversions Only
    w_conv = strategy_single_objective_conversions(helper, max_budget, budget_fraction)
    baselines["Single-Objective (Conversions-Only)"] = evaluate_allocation(budget_fraction, w_conv, helper, max_budget)

    records = []
    for name, res in baselines.items():
        records.append({
            "Strategy": name,
            "Total_Cost": round(res["total_cost"], 2),
            "Conversions": round(res["conversions"], 2),
            "CLV_Gain": round(res["clv_gain"], 2),
            "ROI": round(res["roi"], 4),
            "CPA": round(res["cpa"], 2),
            "Conversion_Rate": round(res["conversion_rate"] * 100, 2),
            "Total_Contacts": round(res["total_contacts"], 1),
            "Fatigue_Score": round(res["fatigue"], 3),
        })

    baseline_df = pd.DataFrame(records)
    return baseline_df
