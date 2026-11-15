"""
Leak-Free Supply Chain Feature Engineering Module.
Engineers geospatial Haversine distance, out-of-fold historical seller delay rates,
warehouse dispatch lag, temporal seasonality flags, and packaging density metrics.
Strictly ensures zero data leakage by excluding post-delivery timestamps and survey data.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. GEOSPATIAL HAVERSINE DISTANCE
# ---------------------------------------------------------------------------
def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    radius_km: float = 6371.0
) -> np.ndarray:
    """
    Computes great-circle distance between two points on Earth using Haversine formula.
    Vectorized with NumPy for high throughput.
    
    Args:
        lat1, lon1: Coordinates of origin (Seller) in degrees
        lat2, lon2: Coordinates of destination (Customer) in degrees
        radius_km: Earth radius in kilometers (default 6371.0 km)
        
    Returns:
        np.ndarray: Distances in kilometers
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # Clip to prevent numerical errors exceeding 1.0 before arcsin
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_km * c


# ---------------------------------------------------------------------------
# 2. BRAZILIAN HOLIDAYS & PEAK SHOPPING SEASONS
# ---------------------------------------------------------------------------
BRAZILIAN_FIXED_HOLIDAYS = [
    (1, 1),    # New Year's Day (Ano Novo)
    (4, 21),   # Tiradentes
    (5, 1),    # Labor Day (Dia do Trabalho)
    (9, 7),    # Independence Day (Independência)
    (10, 12),  # Our Lady of Aparecida / Children's Day
    (11, 2),   # All Souls' Day (Finados)
    (11, 15),  # Proclamation of the Republic
    (12, 25),  # Christmas (Natal)
]


def is_brazilian_holiday(dt_series: pd.Series) -> pd.Series:
    """Returns boolean series indicating if date falls on a major Brazilian holiday."""
    month = dt_series.dt.month
    day = dt_series.dt.day
    
    is_holiday = pd.Series(False, index=dt_series.index)
    for m, d in BRAZILIAN_FIXED_HOLIDAYS:
        is_holiday |= (month == m) & (day == d)
    return is_holiday.astype(int)


# ---------------------------------------------------------------------------
# 3. LEAK-FREE HISTORICAL SELLER DELAY RATE (EXPANDING WINDOW / BAYESIAN SMOOTHING)
# ---------------------------------------------------------------------------
def compute_leak_free_seller_delay_rate(
    df: pd.DataFrame,
    seller_col: str = "primary_seller_id",
    target_col: str = "is_delayed",
    timestamp_col: str = "order_approved_at",
    prior_weight: float = 10.0,
    global_prior: Optional[float] = None
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Computes an expanding-window historical delay rate per seller strictly
    using orders that occurred BEFORE the current order's timestamp.
    
    Uses Empirical Bayes smoothing:
        smoothed_rate = (prior_weight * global_prior + past_delays) / (prior_weight + past_orders)
        
    Returns:
        (historical_rates_series, final_seller_stats_lookup_df)
    """
    sorted_df = df[[seller_col, target_col, timestamp_col]].copy()
    sorted_df["orig_idx"] = sorted_df.index
    sorted_df[timestamp_col] = pd.to_datetime(sorted_df[timestamp_col])
    sorted_df = sorted_df.sort_values(by=timestamp_col)

    if global_prior is None:
        global_prior = sorted_df[target_col].mean()

    # Expanding cumulative sums by seller (shifted by 1 to exclude current order)
    seller_grouped = sorted_df.groupby(seller_col)[target_col]
    
    # Cumulative count and sum of delays strictly before current order
    sorted_df["past_orders"] = seller_grouped.cumcount()
    sorted_df["past_delays"] = seller_grouped.cumsum() - sorted_df[target_col]

    # Empirical Bayes smoothed delay rate
    smoothed_rate = (
        (prior_weight * global_prior + sorted_df["past_delays"]) /
        (prior_weight + sorted_df["past_orders"])
    )
    sorted_df["seller_historical_delay_rate"] = smoothed_rate

    # Restore original ordering
    result_series = sorted_df.sort_values(by="orig_idx")["seller_historical_delay_rate"]
    result_series.index = df.index

    # Summary table for test-time/production inference lookup
    seller_summary = (
        df.groupby(seller_col)
        .agg(
            total_orders=(target_col, "count"),
            total_delays=(target_col, "sum")
        )
        .reset_index()
    )
    seller_summary["seller_historical_delay_rate"] = (
        (prior_weight * global_prior + seller_summary["total_delays"]) /
        (prior_weight + seller_summary["total_orders"])
    )

    return result_series, seller_summary


# ---------------------------------------------------------------------------
# 4. MASTER LEAK-FREE FEATURE PIPELINE
# ---------------------------------------------------------------------------
# Whitelist of features allowed for machine learning models (zero data leakage)
FEATURE_COLUMNS: List[str] = [
    # Geospatial
    "distance_km",
    "is_cross_state_shipment",
    # Order dimensions & financials
    "total_items",
    "distinct_products_count",
    "total_order_value",
    "total_freight_value",
    "freight_to_price_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "weight_density",
    "freight_per_km",
    "freight_per_weight",
    # Operational duration (available at carrier handoff)
    "dispatch_time_hours",
    "estimated_delivery_window_days",
    "dispatch_to_estimated_ratio",
    # Seller historical reputation
    "seller_historical_delay_rate",
    # Temporal & Seasonality
    "purchase_hour",
    "purchase_dayofweek",
    "is_weekend",
    "purchase_month",
    "is_peak_season",
    "is_holiday",
    # Categorical codes
    "customer_state",
    "seller_state",
    "primary_category_name",
    "dominant_payment_type",
]

# Explicit list of columns forbidden from training to prevent target leakage
LEAKY_COLUMNS: List[str] = [
    "order_delivered_customer_date",
    "delay_days",
    "transit_time_hours",
    "total_delivery_time_hours",
    "review_score",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
]


def engineer_features(
    df: pd.DataFrame,
    seller_summary_lookup: Optional[pd.DataFrame] = None,
    global_delay_prior: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies the full leak-free feature engineering pipeline to the Analytical Base Table.
    
    Args:
        df: Input DataFrame from ORDER_ANALYTICAL_BASE_TABLE_QUERY
        seller_summary_lookup: Optional lookup table for test-set inference
        global_delay_prior: Prior delay probability
        
    Returns:
        (featured_df, seller_summary_df)
    """
    feat_df = df.copy()

    # 1. Parse Datetime Timestamps
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_estimated_delivery_date"
    ]:
        if col in feat_df.columns:
            feat_df[col] = pd.to_datetime(feat_df[col])

    # 2. Geospatial Haversine Distance
    # Impute missing GPS coordinates with Brazil centroid / regional defaults if missing
    valid_coords = (
        feat_df["customer_lat"].notnull() &
        feat_df["customer_lng"].notnull() &
        feat_df["seller_lat"].notnull() &
        feat_df["seller_lng"].notnull()
    )
    
    distances = np.zeros(len(feat_df))
    if valid_coords.any():
        distances[valid_coords] = haversine_distance(
            feat_df.loc[valid_coords, "seller_lat"].values,
            feat_df.loc[valid_coords, "seller_lng"].values,
            feat_df.loc[valid_coords, "customer_lat"].values,
            feat_df.loc[valid_coords, "customer_lng"].values
        )
    
    # Fill invalid/missing distances with median distance
    median_dist = np.median(distances[valid_coords]) if valid_coords.any() else 500.0
    distances[~valid_coords] = median_dist
    feat_df["distance_km"] = np.round(distances, 2)

    # 3. Supply Chain Physical & Economic Ratios
    feat_df["total_order_value"] = feat_df["total_order_value"].fillna(0.0)
    feat_df["total_freight_value"] = feat_df["total_freight_value"].fillna(0.0)
    feat_df["total_weight_g"] = feat_df["total_weight_g"].fillna(0.0)
    feat_df["total_volume_cm3"] = feat_df["total_volume_cm3"].fillna(0.0)

    # Freight to price ratio
    feat_df["freight_to_price_ratio"] = (
        feat_df["total_freight_value"] / (feat_df["total_order_value"] + 1e-4)
    ).clip(upper=10.0)

    # Volumetric weight density (g / cm³)
    feat_df["weight_density"] = (
        feat_df["total_weight_g"] / (feat_df["total_volume_cm3"] + 1.0)
    ).clip(upper=5.0)

    # Freight economic efficiencies
    feat_df["freight_per_km"] = (
        feat_df["total_freight_value"] / (feat_df["distance_km"] + 1.0)
    ).clip(upper=50.0)
    feat_df["freight_per_weight"] = (
        feat_df["total_freight_value"] / (feat_df["total_weight_g"] + 1.0)
    ).clip(upper=10.0)

    # 4. Operational Lag Times (Known at or before Carrier Handoff)
    if "dispatch_time_hours" not in feat_df.columns:
        feat_df["dispatch_time_hours"] = (
            (feat_df["order_delivered_carrier_date"] - feat_df["order_approved_at"])
            .dt.total_seconds() / 3600.0
        )
    feat_df["dispatch_time_hours"] = feat_df["dispatch_time_hours"].clip(lower=0.0, upper=720.0)

    # Promised SLA Window (Estimated Delivery Date - Order Approved Date) in days
    feat_df["estimated_delivery_window_days"] = (
        (feat_df["order_estimated_delivery_date"] - feat_df["order_approved_at"])
        .dt.total_seconds() / 86400.0
    ).clip(lower=1.0, upper=90.0)

    # Ratio of warehouse dispatch time consumed against total promised delivery window
    feat_df["dispatch_to_estimated_ratio"] = (
        (feat_df["dispatch_time_hours"] / 24.0) / 
        feat_df["estimated_delivery_window_days"]
    ).clip(upper=5.0)

    # 5. Temporal & Peak Season Calendar Features
    purchase_dt = feat_df["order_purchase_timestamp"]
    feat_df["purchase_hour"] = purchase_dt.dt.hour
    feat_df["purchase_dayofweek"] = purchase_dt.dt.dayofweek
    feat_df["is_weekend"] = feat_df["purchase_dayofweek"].isin([5, 6]).astype(int)
    feat_df["purchase_month"] = purchase_dt.dt.month

    # Peak e-commerce surge: Black Friday (late Nov) and Christmas shopping (Dec)
    feat_df["is_peak_season"] = (
        (feat_df["purchase_month"] == 11) | (feat_df["purchase_month"] == 12)
    ).astype(int)

    # Brazilian National Holidays
    feat_df["is_holiday"] = is_brazilian_holiday(purchase_dt)

    # 6. Seller Historical Delay Rate (Leak-Free Computation)
    if seller_summary_lookup is None:
        delay_series, seller_summary = compute_leak_free_seller_delay_rate(
            feat_df,
            seller_col="primary_seller_id",
            target_col="is_delayed",
            timestamp_col="order_approved_at",
            prior_weight=10.0,
            global_prior=global_delay_prior
        )
        feat_df["seller_historical_delay_rate"] = delay_series
    else:
        # Test-set lookup from training distribution
        merged = feat_df.merge(
            seller_summary_lookup[["primary_seller_id", "seller_historical_delay_rate"]],
            on="primary_seller_id",
            how="left"
        )
        fallback_prior = global_delay_prior if global_delay_prior is not None else 0.08
        feat_df["seller_historical_delay_rate"] = (
            merged["seller_historical_delay_rate"].fillna(fallback_prior).values
        )
        seller_summary = seller_summary_lookup

    # Categorical string cleanup
    for cat_col in ["customer_state", "seller_state", "primary_category_name", "dominant_payment_type"]:
        if cat_col in feat_df.columns:
            feat_df[cat_col] = feat_df[cat_col].fillna("unknown").astype(str)

    return feat_df, seller_summary

