"""Marketing metrics module.

This module provides pure, vectorized, zero-division-safe functions for standard
digital marketing performance metrics and customer lifetime value (CLV) proxies.
"""

from typing import Union
import numpy as np
import pandas as pd

NumericType = Union[float, int, np.ndarray, pd.Series]


def conversion_rate(
    conversions: NumericType,
    contacts: NumericType
) -> NumericType:
    """Calculate conversion rate (conversions / contacts).

    Formula:
        CR = conversions / contacts

    Zero-division handling:
        Returns 0.0 when contacts == 0.

    Example:
        >>> conversion_rate(15, 100)
        0.15
        >>> conversion_rate(0, 0)
        0.0
    """
    conversions_arr = np.asarray(conversions, dtype=float)
    contacts_arr = np.asarray(contacts, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(contacts_arr > 0, conversions_arr / contacts_arr, 0.0)
    
    return float(result) if np.ndim(result) == 0 else result


def click_through_rate(
    clicks: NumericType,
    impressions: NumericType
) -> NumericType:
    """Calculate click-through rate (clicks / impressions).

    Note:
        Implemented for completeness and benchmark comparisons (e.g. Criteo/Avazu).
        Not directly observed in the direct-marketing catalog/store dataset.

    Formula:
        CTR = clicks / impressions

    Zero-division handling:
        Returns 0.0 when impressions == 0.

    Example:
        >>> click_through_rate(25, 1000)
        0.025
        >>> click_through_rate(10, 0)
        0.0
    """
    clicks_arr = np.asarray(clicks, dtype=float)
    impressions_arr = np.asarray(impressions, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(impressions_arr > 0, clicks_arr / impressions_arr, 0.0)
        
    return float(result) if np.ndim(result) == 0 else result


def cost_per_click(
    cost: NumericType,
    clicks: NumericType
) -> NumericType:
    """Calculate cost per click (total cost / clicks).

    Formula:
        CPC = cost / clicks

    Zero-division handling:
        Returns 0.0 when clicks == 0.

    Example:
        >>> cost_per_click(100.0, 50)
        2.0
        >>> cost_per_click(100.0, 0)
        0.0
    """
    cost_arr = np.asarray(cost, dtype=float)
    clicks_arr = np.asarray(clicks, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(clicks_arr > 0, cost_arr / clicks_arr, 0.0)
        
    return float(result) if np.ndim(result) == 0 else result


def cost_per_acquisition(
    total_cost: NumericType,
    conversions: NumericType
) -> NumericType:
    """Calculate cost per acquisition / conversion (total cost / conversions).

    Formula:
        CPA = total_cost / conversions

    Zero-division handling:
        Returns 0.0 when conversions == 0.

    Example:
        >>> cost_per_acquisition(300.0, 15)
        20.0
        >>> cost_per_acquisition(300.0, 0)
        0.0
    """
    cost_arr = np.asarray(total_cost, dtype=float)
    conv_arr = np.asarray(conversions, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(conv_arr > 0, cost_arr / conv_arr, 0.0)
        
    return float(result) if np.ndim(result) == 0 else result


def return_on_investment(
    revenue: NumericType,
    cost: NumericType
) -> NumericType:
    """Calculate return on marketing investment (ROI).

    Formula:
        ROI = (revenue - cost) / cost

    Zero-division handling:
        Returns 0.0 when cost == 0.

    Example:
        >>> return_on_investment(250.0, 100.0)
        1.5
        >>> return_on_investment(0.0, 0.0)
        0.0
    """
    rev_arr = np.asarray(revenue, dtype=float)
    cost_arr = np.asarray(cost, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(cost_arr > 0, (rev_arr - cost_arr) / cost_arr, 0.0)
        
    return float(result) if np.ndim(result) == 0 else result


def clv_proxy(
    total_spend: NumericType,
    tenure_months: NumericType,
    expected_lifetime_months: float = 24.0,
    margin: float = 0.30
) -> NumericType:
    """Calculate Customer Lifetime Value (CLV) proxy.

    Assumptions & Formula:
        Assumes historical monthly spending rate is indicative of future spend
        over an expected customer lifetime horizon, adjusted for profit margin:
        
        monthly_spend_i = Total_Spend_i / max(tenure_months_i, 1.0)
        CLV_i = monthly_spend_i * expected_lifetime_months * margin

    Args:
        total_spend: Total cumulative historical spend ($ across all product categories).
        tenure_months: Customer relationship tenure in months (must be >= 1.0).
        expected_lifetime_months: [Assumption] Customer lifetime horizon (default 24 months).
        margin: [Assumption] Operating margin percentage (default 30% or 0.30).

    Example:
        >>> clv_proxy(1200.0, 12.0, expected_lifetime_months=24.0, margin=0.30)
        720.0
    """
    spend_arr = np.asarray(total_spend, dtype=float)
    tenure_arr = np.asarray(tenure_months, dtype=float)
    safe_tenure = np.maximum(tenure_arr, 1.0)
    
    monthly_spend = spend_arr / safe_tenure
    clv = monthly_spend * expected_lifetime_months * margin
    
    return float(clv) if np.ndim(clv) == 0 else clv
