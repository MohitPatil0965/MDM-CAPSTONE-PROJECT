"""Unit tests for marketing metrics module."""

import numpy as np
import pytest
from src.metrics import (
    conversion_rate,
    click_through_rate,
    cost_per_click,
    cost_per_acquisition,
    return_on_investment,
    clv_proxy,
)


def test_conversion_rate():
    # Normal scalar case
    assert conversion_rate(15, 100) == pytest.approx(0.15)
    # Zero contacts edge case (zero division safe)
    assert conversion_rate(0, 0) == 0.0
    assert conversion_rate(5, 0) == 0.0
    # Vectorized case
    convs = np.array([10, 0, 25])
    contacts = np.array([100, 50, 0])
    expected = np.array([0.10, 0.0, 0.0])
    np.testing.assert_allclose(conversion_rate(convs, contacts), expected)


def test_click_through_rate():
    # Normal case
    assert click_through_rate(25, 1000) == pytest.approx(0.025)
    # Zero impressions
    assert click_through_rate(10, 0) == 0.0
    assert click_through_rate(0, 0) == 0.0
    # Vectorized
    clicks = np.array([5, 0])
    impr = np.array([100, 0])
    np.testing.assert_allclose(click_through_rate(clicks, impr), np.array([0.05, 0.0]))


def test_cost_per_click():
    # Normal case
    assert cost_per_click(100.0, 50) == pytest.approx(2.0)
    # Zero clicks
    assert cost_per_click(100.0, 0) == 0.0
    assert cost_per_click(0.0, 0) == 0.0


def test_cost_per_acquisition():
    # Normal case
    assert cost_per_acquisition(300.0, 15) == pytest.approx(20.0)
    # Zero conversions
    assert cost_per_acquisition(300.0, 0) == 0.0
    assert cost_per_acquisition(0.0, 0) == 0.0
    # Vectorized
    costs = np.array([300.0, 100.0, 0.0])
    convs = np.array([15, 0, 0])
    np.testing.assert_allclose(cost_per_acquisition(costs, convs), np.array([20.0, 0.0, 0.0]))


def test_return_on_investment():
    # Normal profit case: revenue 250, cost 100 -> ROI = (250-100)/100 = 1.5
    assert return_on_investment(250.0, 100.0) == pytest.approx(1.5)
    # Loss case: revenue 50, cost 100 -> ROI = (50-100)/100 = -0.5
    assert return_on_investment(50.0, 100.0) == pytest.approx(-0.5)
    # Zero cost edge case
    assert return_on_investment(0.0, 0.0) == 0.0
    assert return_on_investment(100.0, 0.0) == 0.0
    # Vectorized
    rev = np.array([200.0, 50.0, 0.0])
    cost = np.array([100.0, 100.0, 0.0])
    np.testing.assert_allclose(return_on_investment(rev, cost), np.array([1.0, -0.5, 0.0]))


def test_clv_proxy():
    # Total spend $1200 over 12 months -> $100/mo * 24 mo * 0.3 margin = $720.0
    assert clv_proxy(1200.0, 12.0, expected_lifetime_months=24.0, margin=0.30) == pytest.approx(720.0)
    # Tenure < 1 month guard (safe tenure max(tenure, 1))
    # Spend $50 over 0 months -> 50 / 1 * 24 * 0.3 = 360.0
    assert clv_proxy(50.0, 0.0, expected_lifetime_months=24.0, margin=0.30) == pytest.approx(360.0)
    # Zero spend
    assert clv_proxy(0.0, 10.0) == pytest.approx(0.0)
    # Vectorized
    spends = np.array([1200.0, 600.0])
    tenures = np.array([12.0, 6.0])
    np.testing.assert_allclose(
        clv_proxy(spends, tenures, expected_lifetime_months=24.0, margin=0.30),
        np.array([720.0, 720.0])
    )
