"""Multi-Objective Campaign Optimizer using NSGA-II via pymoo.

Formulates and solves the non-linear multi-objective resource allocation problem across
customer segments and marketing channels under diminishing returns and budget constraints.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import yaml

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

from src.metrics import return_on_investment, cost_per_acquisition, conversion_rate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SegmentDataHelper:
    """Precomputes sorted cumulative response and CLV curves per segment for fast interpolation."""

    def __init__(self, scored_df: pd.DataFrame, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.segments = sorted(scored_df["Segment"].unique())
        self.num_segments = len(self.segments)
        
        # Channel configurations
        channels_cfg = cfg["economics"]["channels"]
        self.channel_keys = list(channels_cfg.keys())
        self.channel_names = [channels_cfg[k]["name"] for k in self.channel_keys]
        self.costs_per_contact = np.array([channels_cfg[k]["cost_per_contact"] for k in self.channel_keys], dtype=float)
        self.response_multipliers = np.array([channels_cfg[k]["response_multiplier"] for k in self.channel_keys], dtype=float)
        self.num_channels = len(self.channel_keys)
        
        self.revenue_per_conversion = float(cfg["economics"]["revenue_per_conversion"])
        self.fatigue_penalty_weight = float(cfg["optimizer"].get("fatigue_penalty_weight", 0.05))

        self.segment_sizes = []
        self.cum_responses = []
        self.cum_clvs = []

        for seg_id in self.segments:
            seg_rows = scored_df[scored_df["Segment"] == seg_id].copy()
            # Sort by calibrated p_response descending (greedy diminishing returns within segment)
            seg_rows = seg_rows.sort_values("p_response", ascending=False).reset_index(drop=True)
            
            n_cust = len(seg_rows)
            self.segment_sizes.append(n_cust)
            
            # Cumulative response curve: C[0]=0, C[k] = sum_{j=1}^k p_j
            p_vals = seg_rows["p_response"].values
            cum_resp = np.concatenate([[0.0], np.cumsum(p_vals)])
            self.cum_responses.append(cum_resp)
            
            # Cumulative CLV gain curve: V[0]=0, V[k] = sum_{j=1}^k (p_j * CLV_j)
            clv_vals = seg_rows["CLV"].values
            expected_clv_per_cust = p_vals * clv_vals
            cum_clv = np.concatenate([[0.0], np.cumsum(expected_clv_per_cust)])
            self.cum_clvs.append(cum_clv)

        self.segment_sizes = np.array(self.segment_sizes, dtype=int)
        self.total_customers = int(np.sum(self.segment_sizes))


def evaluate_allocation(
    f_budget: float,
    raw_weights_2d: np.ndarray,
    helper: SegmentDataHelper,
    max_budget: float,
    include_fatigue: bool = False
) -> Dict[str, Any]:
    """Evaluate a single campaign allocation strategy.

    Args:
        f_budget: Fraction of max budget in [0.1, 1.0].
        raw_weights_2d: Array of shape (num_segments, num_channels) with allocation weights.
        helper: SegmentDataHelper instance.
        max_budget: Total maximum budget pool in $.
        include_fatigue: Whether to compute contact fatigue.

    Returns:
        Dictionary containing CLV gain, ROI, Cost, Conversions, Revenue, CPA, Fatigue, Allocations.
    """
    K = helper.num_segments
    C = helper.num_channels

    # Repair step: Normalize weights so they sum strictly to 1.0
    weight_sum = np.sum(raw_weights_2d)
    if weight_sum > 1e-9:
        w_norm = raw_weights_2d / weight_sum
    else:
        w_norm = np.ones((K, C)) / (K * C)

    total_cost = float(f_budget * max_budget)
    spend_grid = total_cost * w_norm  # Shape (K, C)

    total_conversions = 0.0
    total_clv_gain = 0.0
    total_contacts = 0.0
    contacts_grid = np.zeros((K, C))
    conversions_grid = np.zeros((K, C))

    for s in range(K):
        N_s = helper.segment_sizes[s]
        grid_indices = np.arange(N_s + 1)
        cum_r = helper.cum_responses[s]
        cum_v = helper.cum_clvs[s]

        seg_contacts_total = 0.0

        for c in range(C):
            cost_c = helper.costs_per_contact[c]
            mult_c = helper.response_multipliers[c]
            spend_sc = spend_grid[s, c]

            # Contact count for channel c
            n_sc = spend_sc / cost_c
            # Cap contacts at segment size
            n_sc_capped = min(n_sc, float(N_s))
            contacts_grid[s, c] = n_sc_capped
            seg_contacts_total += n_sc_capped

            # Interpolate cumulative curves
            base_conv = float(np.interp(n_sc_capped, grid_indices, cum_r))
            channel_conv = min(mult_c * base_conv, n_sc_capped)
            conversions_grid[s, c] = channel_conv
            total_conversions += channel_conv

            base_clv = float(np.interp(n_sc_capped, grid_indices, cum_v))
            channel_clv = mult_c * base_clv
            total_clv_gain += channel_clv

        # Fatigue penalty if multiple channels saturate the segment
        if seg_contacts_total > N_s:
            excess_ratio = (seg_contacts_total - N_s) / N_s
            fatigue_discount = 1.0 / (1.0 + helper.fatigue_penalty_weight * excess_ratio)
            total_clv_gain *= fatigue_discount

        total_contacts += seg_contacts_total

    total_revenue = total_conversions * helper.revenue_per_conversion
    roi = return_on_investment(total_revenue, total_cost)
    cpa = cost_per_acquisition(total_cost, total_conversions)
    cr = conversion_rate(total_conversions, total_contacts)
    fatigue_score = total_contacts / max(helper.total_customers, 1)

    return {
        "clv_gain": float(total_clv_gain),
        "roi": float(roi),
        "total_cost": float(total_cost),
        "conversions": float(total_conversions),
        "revenue": float(total_revenue),
        "cpa": float(cpa),
        "conversion_rate": float(cr),
        "total_contacts": float(total_contacts),
        "fatigue": float(fatigue_score),
        "spend_grid": spend_grid,
        "contacts_grid": contacts_grid,
        "conversions_grid": conversions_grid,
        "weights": w_norm,
        "f_budget": f_budget,
    }


class CampaignOptimizationProblem(Problem):
    """Pymoo Problem definition for Multi-Objective Campaign Budget & Channel Allocation."""

    def __init__(
        self,
        helper: SegmentDataHelper,
        max_budget: float = 5000.0,
        include_fatigue: bool = False
    ):
        self.helper = helper
        self.max_budget = max_budget
        self.include_fatigue = include_fatigue

        K = helper.num_segments
        C = helper.num_channels
        self.n_alloc_vars = K * C
        # Decision variables: [f_budget, w_0_0, w_0_1, ..., w_K_C]
        n_var = 1 + self.n_alloc_vars

        # Objectives: f1 = -CLV, f2 = -ROI, f3 = Cost, (optional f4 = Fatigue)
        n_obj = 4 if include_fatigue else 3

        xl = np.zeros(n_var)
        xl[0] = 0.10  # Min budget fraction: 10%
        xu = np.ones(n_var)  # Max budget fraction: 100%, max raw weight: 1.0

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=0,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, X: np.ndarray, out: Dict[str, np.ndarray], *args, **kwargs):
        pop_size = len(X)
        K = self.helper.num_segments
        C = self.helper.num_channels

        F = np.zeros((pop_size, self.n_obj))

        for i in range(pop_size):
            xi = X[i]
            f_b = xi[0]
            raw_w = xi[1:].reshape((K, C))
            
            res = evaluate_allocation(
                f_budget=f_b,
                raw_weights_2d=raw_w,
                helper=self.helper,
                max_budget=self.max_budget,
                include_fatigue=self.include_fatigue
            )

            # Pymoo minimizes: negate maximized objectives
            F[i, 0] = -res["clv_gain"]   # Objective 1: Maximize CLV Gain
            F[i, 1] = -res["roi"]        # Objective 2: Maximize ROI
            F[i, 2] = res["total_cost"]  # Objective 3: Minimize Cost
            
            if self.include_fatigue:
                F[i, 3] = res["fatigue"] # Objective 4: Minimize Fatigue

        out["F"] = F


def run_nsga2_optimization(
    scored_df: pd.DataFrame,
    config_path: str = "config.yaml",
    custom_max_budget: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], SegmentDataHelper]:
    """Execute NSGA-II multi-objective optimization across multiple random seeds."""
    cfg = load_config(config_path)
    opt_cfg = cfg["optimizer"]
    seeds = opt_cfg["seeds"]
    max_budget = custom_max_budget if custom_max_budget is not None else float(opt_cfg["default_max_budget"])
    include_fatigue = bool(opt_cfg.get("include_fatigue_objective", False))

    helper = SegmentDataHelper(scored_df, cfg)
    problem = CampaignOptimizationProblem(helper, max_budget=max_budget, include_fatigue=include_fatigue)

    algorithm = NSGA2(
        pop_size=opt_cfg["population_size"],
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=opt_cfg["crossover_prob"], eta=opt_cfg["crossover_eta"]),
        mutation=PM(eta=opt_cfg["mutation_eta"]),
        eliminate_duplicates=True
    )

    seed_hypervolumes = []
    # Reference point for Hypervolume [Worst CLV (negated = 0), Worst ROI (negated = 0), Max Cost]
    ref_point = np.array([0.0, 0.0, max_budget * 1.05])
    hv_calc = HV(ref_point=ref_point)

    best_res = None
    best_hv = -1.0

    for seed in seeds:
        res = minimize(
            problem,
            algorithm,
            ("n_gen", opt_cfg["n_generations"]),
            seed=seed,
            verbose=False
        )

        # Calculate hypervolume on 3 objectives
        F_3obj = res.F[:, :3]
        # HV needs non-dominated front within reference point
        hv_val = hv_calc(F_3obj)
        seed_hypervolumes.append(hv_val)
        logger.info("NSGA-II Seed %d | Solutions: %d | Hypervolume: %.4f", seed, len(res.F), hv_val)

        if hv_val > best_hv:
            best_hv = hv_val
            best_res = res

    mean_hv = float(np.mean(seed_hypervolumes))
    std_hv = float(np.std(seed_hypervolumes))
    logger.info("Multi-Seed Hypervolume: %.4f ± %.4f", mean_hv, std_hv)

    # Process Best Pareto Front Solutions
    pareto_X = best_res.X
    pareto_F = best_res.F
    K = helper.num_segments
    C = helper.num_channels

    pareto_records = []
    allocation_records = []

    for idx in range(len(pareto_X)):
        xi = pareto_X[idx]
        f_b = xi[0]
        raw_w = xi[1:].reshape((K, C))
        eval_res = evaluate_allocation(f_b, raw_w, helper, max_budget, include_fatigue)

        pareto_records.append({
            "Solution_ID": idx,
            "CLV_Gain": eval_res["clv_gain"],
            "ROI": eval_res["roi"],
            "Total_Cost": eval_res["total_cost"],
            "Conversions": eval_res["conversions"],
            "Revenue": eval_res["revenue"],
            "CPA": eval_res["cpa"],
            "Conversion_Rate": eval_res["conversion_rate"],
            "Total_Contacts": eval_res["total_contacts"],
            "Fatigue_Score": eval_res["fatigue"],
            "Budget_Fraction": eval_res["f_budget"],
        })

        # Record spend per segment and channel
        alloc_row = {"Solution_ID": idx}
        for s_i, s_id in enumerate(helper.segments):
            for c_i, c_key in enumerate(helper.channel_keys):
                alloc_row[f"Spend_Seg{s_id}_{c_key}"] = eval_res["spend_grid"][s_i, c_i]
                alloc_row[f"Contacts_Seg{s_id}_{c_key}"] = eval_res["contacts_grid"][s_i, c_i]
        allocation_records.append(alloc_row)

    pareto_df = pd.DataFrame(pareto_records)
    alloc_df = pd.DataFrame(allocation_records)

    # Save to reports/tables
    tables_dir = Path(cfg["reports"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    pareto_df.to_csv(tables_dir / "pareto_front.csv", index=False)
    alloc_df.to_csv(tables_dir / "pareto_allocations.csv", index=False)

    opt_stats = {
        "mean_hypervolume": mean_hv,
        "std_hypervolume": std_hv,
        "seed_hypervolumes": seed_hypervolumes,
        "num_pareto_solutions": len(pareto_df),
        "best_seed_hypervolume": best_hv,
    }

    return pareto_df, alloc_df, opt_stats, helper


def select_knee_point(pareto_df: pd.DataFrame) -> int:
    """Select the knee point solution on the Pareto front minimizing distance to ideal point."""
    clv = pareto_df["CLV_Gain"].values
    roi = pareto_df["ROI"].values
    cost = pareto_df["Total_Cost"].values

    # Normalize objectives to [0, 1] where 1 is ideal
    norm_clv = (clv - np.min(clv)) / (np.max(clv) - np.min(clv) + 1e-9)
    norm_roi = (roi - np.min(roi)) / (np.max(roi) - np.min(roi) + 1e-9)
    norm_cost = (np.max(cost) - cost) / (np.max(cost) - np.min(cost) + 1e-9)  # cost is minimized

    # Ideal point is [1.0, 1.0, 1.0] in normalized space
    dist_to_ideal = np.sqrt((1.0 - norm_clv)**2 + (1.0 - norm_roi)**2 + (1.0 - norm_cost)**2)
    knee_idx = int(np.argmin(dist_to_ideal))
    return knee_idx


def select_topsis_solution(
    pareto_df: pd.DataFrame,
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
) -> int:
    """Select solution via Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS).

    Args:
        pareto_df: DataFrame with CLV_Gain, ROI, Total_Cost.
        weights: Preference weights (w_clv, w_roi, w_cost).
    """
    matrix = pareto_df[["CLV_Gain", "ROI", "Total_Cost"]].values
    n_samples = len(matrix)

    # Vector normalization
    denom = np.sqrt(np.sum(matrix**2, axis=0))
    norm_matrix = matrix / denom

    # Apply weights
    w = np.array(weights) / np.sum(weights)
    weighted_matrix = norm_matrix * w

    # Ideal and Anti-Ideal solutions
    ideal_best = np.array([np.max(weighted_matrix[:, 0]), np.max(weighted_matrix[:, 1]), np.min(weighted_matrix[:, 2])])
    ideal_worst = np.array([np.min(weighted_matrix[:, 0]), np.min(weighted_matrix[:, 1]), np.max(weighted_matrix[:, 2])])

    # Euclidean distances
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # Relative closeness
    closeness = dist_worst / (dist_best + dist_worst + 1e-9)
    best_topsis_idx = int(np.argmax(closeness))
    return best_topsis_idx


if __name__ == "__main__":
    scored_data = pd.read_parquet("data/processed/scored_campaign_data.parquet")
    p_df, a_df, stats, _ = run_nsga2_optimization(scored_data)
    print(f"Generated {len(p_df)} non-dominated Pareto solutions.")
    knee = select_knee_point(p_df)
    topsis = select_topsis_solution(p_df)
    print(f"Knee Solution ID: {knee} | TOPSIS Solution ID: {topsis}")
