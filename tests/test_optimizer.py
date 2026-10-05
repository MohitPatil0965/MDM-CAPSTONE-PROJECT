"""Unit tests for multi-objective optimizer and baseline strategies."""

import numpy as np
import pandas as pd
import pytest
from src.optimizer import (
    SegmentDataHelper,
    evaluate_allocation,
    select_knee_point,
    select_topsis_solution,
)
from src.baselines import (
    strategy_equal_split,
    strategy_size_proportional,
    strategy_value_proportional,
    strategy_marginal_roi_greedy,
)


@pytest.fixture
def mock_scored_data():
    """Create a minimal synthetic scored dataset with 2 segments."""
    return pd.DataFrame({
        "ID": range(10),
        "Segment": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "p_response": [0.8, 0.6, 0.4, 0.2, 0.1, 0.9, 0.7, 0.5, 0.3, 0.1],
        "CLV": [1000.0, 800.0, 600.0, 400.0, 200.0, 1200.0, 900.0, 700.0, 500.0, 300.0],
        "Response": [1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    })


@pytest.fixture
def mock_config():
    """Minimal test configuration."""
    return {
        "seed": 42,
        "economics": {
            "revenue_per_conversion": 11.0,
            "channels": {
                "email": {"name": "Email", "cost_per_contact": 0.5, "response_multiplier": 0.8},
                "sms": {"name": "SMS", "cost_per_contact": 1.0, "response_multiplier": 0.9},
            }
        },
        "optimizer": {
            "fatigue_penalty_weight": 0.05,
            "default_topsis_weights": [0.4, 0.4, 0.2]
        }
    }


def test_segment_data_helper(mock_scored_data, mock_config):
    helper = SegmentDataHelper(mock_scored_data, mock_config)
    assert helper.num_segments == 2
    assert helper.num_channels == 2
    assert len(helper.cum_responses[0]) == 6  # 5 customers + index 0
    assert helper.cum_responses[0][0] == 0.0
    assert helper.cum_responses[0][1] == 0.8  # First customer p_response


def test_evaluate_allocation_repair_and_bounds(mock_scored_data, mock_config):
    helper = SegmentDataHelper(mock_scored_data, mock_config)
    
    # Raw unnormalized weights
    raw_weights = np.array([
        [10.0, 20.0],
        [30.0, 40.0],
    ])
    
    res = evaluate_allocation(
        f_budget=0.5,
        raw_weights_2d=raw_weights,
        helper=helper,
        max_budget=1000.0
    )
    
    assert res["total_cost"] == pytest.approx(500.0)
    assert res["conversions"] > 0.0
    assert res["clv_gain"] > 0.0
    assert np.sum(res["weights"]) == pytest.approx(1.0)
    assert res["conversions"] * 11.0 == pytest.approx(res["revenue"])


def test_baselines_generation(mock_scored_data, mock_config):
    helper = SegmentDataHelper(mock_scored_data, mock_config)
    
    w_eq = strategy_equal_split(helper)
    assert np.allclose(w_eq, 0.25)
    
    w_size = strategy_size_proportional(helper)
    assert np.allclose(w_size, 0.25)  # Equal segment sizes in mock data
    
    w_val = strategy_value_proportional(helper, mock_scored_data)
    assert np.sum(w_val) == pytest.approx(1.0)


def test_mcdm_selection():
    pareto_df = pd.DataFrame({
        "CLV_Gain": [100.0, 500.0, 1000.0],
        "ROI": [3.0, 2.0, 0.5],
        "Total_Cost": [500.0, 1500.0, 5000.0],
    })
    
    knee = select_knee_point(pareto_df)
    assert 0 <= knee < len(pareto_df)
    
    topsis = select_topsis_solution(pareto_df, (0.4, 0.4, 0.2))
    assert 0 <= topsis < len(pareto_df)
