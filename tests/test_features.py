"""
Unit and Leakage Tests for Phase 2: Feature Engineering & Geospatial Math.
Validates Haversine distance accuracy, Bayesian seller delay rate expanding window,
and ensures zero target leakage from post-delivery artifacts.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.features import (
    haversine_distance,
    is_brazilian_holiday,
    compute_leak_free_seller_delay_rate,
    engineer_features,
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
)


# ---------------------------------------------------------------------------
# 1. HAVERSINE DISTANCE ACCURACY TESTS
# ---------------------------------------------------------------------------
def test_haversine_zero_distance():
    """Identical origin and destination coordinates must yield 0.0 km."""
    lats = np.array([-23.5505])
    lons = np.array([-46.6333])
    dist = haversine_distance(lats, lons, lats, lons)
    assert dist[0] == pytest.approx(0.0, abs=1e-5)


def test_haversine_known_cities():
    """
    Great-circle distance between São Paulo (-23.5505, -46.6333)
    and Rio de Janeiro (-22.9068, -43.1729) is ~358 km.
    """
    sp_lat, sp_lon = np.array([-23.5505]), np.array([-46.6333])
    rj_lat, rj_lon = np.array([-22.9068]), np.array([-43.1729])
    
    dist = haversine_distance(sp_lat, sp_lon, rj_lat, rj_lon)
    assert 350.0 < dist[0] < 365.0, f"Expected ~358 km, got {dist[0]:.2f} km"


def test_haversine_vectorized_shapes():
    """Verify vectorized handling of multiple coordinates and numerical stability."""
    lats1 = np.array([-23.55, -19.92, 0.0])
    lons1 = np.array([-46.63, -43.94, 0.0])
    lats2 = np.array([-22.91, -25.43, 0.0])
    lons2 = np.array([-43.17, -49.27, 0.0])

    dist = haversine_distance(lats1, lons1, lats2, lons2)
    assert len(dist) == 3
    assert np.all(dist >= 0.0)
    assert dist[2] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. LEAKAGE PREVENTION TESTS
# ---------------------------------------------------------------------------
def test_feature_columns_contain_zero_leaky_features():
    """
    STRICT LEAKAGE AUDIT:
    None of the post-delivery timestamps, survey ratings, or transit times
    may appear in FEATURE_COLUMNS.
    """
    leaky_set = set(LEAKY_COLUMNS)
    feature_set = set(FEATURE_COLUMNS)
    overlap = leaky_set.intersection(feature_set)
    assert len(overlap) == 0, f"Critical Leakage Detected! Leaky columns in feature set: {overlap}"


def test_expanding_window_seller_delay_rate_prevents_lookahead():
    """
    Verifies that seller historical delay rate strictly uses past events:
    Order 1 (delayed) at T1 -> Prior only (0 past orders)
    Order 2 (delayed) at T2 -> Sees Order 1
    Order 3 (on-time) at T3 -> Sees Order 1 and Order 2
    Order 1's score must NOT change based on Order 2 or Order 3!
    """
    base_time = datetime(2018, 1, 1, 10, 0, 0)
    mock_df = pd.DataFrame({
        "primary_seller_id": ["SELLER_X", "SELLER_X", "SELLER_X"],
        "is_delayed": [1, 1, 0],
        "order_approved_at": [
            base_time,
            base_time + timedelta(days=5),
            base_time + timedelta(days=10),
        ]
    })

    prior_weight = 10.0
    global_prior = 0.10

    rates, summary = compute_leak_free_seller_delay_rate(
        mock_df,
        seller_col="primary_seller_id",
        target_col="is_delayed",
        timestamp_col="order_approved_at",
        prior_weight=prior_weight,
        global_prior=global_prior
    )

    # Order 1 has 0 past orders -> (10 * 0.10 + 0) / (10 + 0) = 0.10
    assert rates.iloc[0] == pytest.approx(global_prior, abs=1e-4)

    # Order 2 has 1 past order (Order 1, which was delayed) -> (10 * 0.10 + 1) / (10 + 1) = 2.0 / 11 = ~0.1818
    assert rates.iloc[1] == pytest.approx(2.0 / 11.0, abs=1e-4)

    # Order 3 has 2 past orders (both delayed) -> (10 * 0.10 + 2) / (10 + 2) = 3.0 / 12 = 0.25
    assert rates.iloc[2] == pytest.approx(3.0 / 12.0, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. FEATURE PIPELINE INTEGRATION TEST
# ---------------------------------------------------------------------------
def test_engineer_features_end_to_end():
    """Tests the complete feature pipeline on synthetic order records."""
    sample_df = pd.DataFrame({
        "order_id": ["O1", "O2"],
        "customer_id": ["C1", "C2"],
        "primary_seller_id": ["S1", "S2"],
        "order_purchase_timestamp": ["2018-01-01 10:00:00", "2018-11-23 15:00:00"],  # O2 is Black Friday peak
        "order_approved_at": ["2018-01-01 12:00:00", "2018-11-23 16:00:00"],
        "order_delivered_carrier_date": ["2018-01-02 12:00:00", "2018-11-25 16:00:00"],
        "order_delivered_customer_date": ["2018-01-08 12:00:00", "2018-12-05 16:00:00"],
        "order_estimated_delivery_date": ["2018-01-15 00:00:00", "2018-11-30 00:00:00"],
        "is_delayed": [0, 1],
        "total_items": [1, 3],
        "distinct_products_count": [1, 2],
        "distinct_sellers_count": [1, 1],
        "total_order_value": [100.0, 350.0],
        "total_freight_value": [20.0, 45.0],
        "total_weight_g": [1500.0, 8000.0],
        "total_volume_cm3": [5000.0, 24000.0],
        "customer_lat": [-23.55, -22.90],
        "customer_lng": [-46.63, -43.17],
        "seller_lat": [-23.55, -19.92],
        "seller_lng": [-46.63, -43.94],
        "customer_state": ["SP", "RJ"],
        "seller_state": ["SP", "MG"],
        "is_cross_state_shipment": [0, 1],
        "primary_category_name": ["telephony", "bed_bath_table"],
        "dominant_payment_type": ["credit_card", "boleto"],
    })

    feat_df, summary = engineer_features(sample_df, global_delay_prior=0.08)

    # Verify key columns exist and have correct types
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Missing feature column: {col}"
        assert feat_df[col].isnull().sum() == 0, f"Found nulls in feature column: {col}"

    # Verify peak season flag: O1 is Jan (0), O2 is Nov (1)
    assert feat_df.iloc[0]["is_peak_season"] == 0
    assert feat_df.iloc[1]["is_peak_season"] == 1

    # Verify dispatch time hours: O1 is 24h, O2 is 48h
    assert feat_df.iloc[0]["dispatch_time_hours"] == pytest.approx(24.0, abs=0.1)
    assert feat_df.iloc[1]["dispatch_time_hours"] == pytest.approx(48.0, abs=0.1)

