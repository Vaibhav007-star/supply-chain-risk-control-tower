"""
Unit and Integration Tests for Phase 3: Model Inference, Calibration & Threshold Logic.
Verifies calibrated pipeline predictions, probability bounds, and decision consistency.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.features import FEATURE_COLUMNS


@pytest.fixture
def pipeline_artifacts():
    """Loads trained pipeline and metadata if available."""
    pipeline_path = Path("models/calibrated_pipeline.joblib")
    if not pipeline_path.exists():
        pytest.skip("Pipeline artifact not yet created. Run 'python -m src.train' first.")
    return joblib.load(pipeline_path)


def test_pipeline_prediction_bounds(pipeline_artifacts):
    """Calibrated probabilities must strictly lie within [0.0, 1.0]."""
    preprocessor = pipeline_artifacts["preprocessor"]
    calibrator = pipeline_artifacts["calibrator"]

    # Construct synthetic feature row
    sample_row = pd.DataFrame([{
        "distance_km": 350.0,
        "is_cross_state_shipment": 1,
        "total_items": 1,
        "distinct_products_count": 1,
        "total_order_value": 150.0,
        "total_freight_value": 30.0,
        "freight_to_price_ratio": 0.20,
        "total_weight_g": 2000.0,
        "total_volume_cm3": 6000.0,
        "weight_density": 0.33,
        "freight_per_km": 0.08,
        "freight_per_weight": 0.015,
        "dispatch_time_hours": 36.0,
        "estimated_delivery_window_days": 15.0,
        "dispatch_to_estimated_ratio": 0.10,
        "seller_historical_delay_rate": 0.12,
        "purchase_hour": 14,
        "purchase_dayofweek": 2,
        "is_weekend": 0,
        "purchase_month": 5,
        "is_peak_season": 0,
        "is_holiday": 0,
        "customer_state": "RJ",
        "seller_state": "SP",
        "primary_category_name": "health_beauty",
        "dominant_payment_type": "credit_card",
    }])[FEATURE_COLUMNS]

    proc_vec = preprocessor.transform(sample_row)
    probs = calibrator.predict_proba(proc_vec)

    assert probs.shape == (1, 2), f"Expected shape (1, 2), got {probs.shape}"
    p_delay = probs[0, 1]
    assert 0.0 <= p_delay <= 1.0, f"Probability out of bounds: {p_delay}"
    assert probs[0, 0] + probs[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_monotonic_dispatch_sensitivity(pipeline_artifacts):
    """
    Supply chain domain test:
    All else equal, increasing warehouse dispatch lag from 12h to 120h
    should not decrease delay risk for the same order.
    """
    preprocessor = pipeline_artifacts["preprocessor"]
    calibrator = pipeline_artifacts["calibrator"]

    base_dict = {
        "distance_km": 600.0,
        "is_cross_state_shipment": 1,
        "total_items": 2,
        "distinct_products_count": 1,
        "total_order_value": 120.0,
        "total_freight_value": 35.0,
        "freight_to_price_ratio": 0.29,
        "total_weight_g": 3000.0,
        "total_volume_cm3": 8000.0,
        "weight_density": 0.375,
        "freight_per_km": 0.058,
        "freight_per_weight": 0.011,
        "estimated_delivery_window_days": 12.0,
        "seller_historical_delay_rate": 0.25,
        "purchase_hour": 10,
        "purchase_dayofweek": 3,
        "is_weekend": 0,
        "purchase_month": 6,
        "is_peak_season": 0,
        "is_holiday": 0,
        "customer_state": "BA",
        "seller_state": "SP",
        "primary_category_name": "furniture_decor",
        "dominant_payment_type": "credit_card",
    }

    # Low dispatch (12 hours)
    row_fast = base_dict.copy()
    row_fast["dispatch_time_hours"] = 12.0
    row_fast["dispatch_to_estimated_ratio"] = (12.0 / 24.0) / 12.0

    # High dispatch (120 hours)
    row_slow = base_dict.copy()
    row_slow["dispatch_time_hours"] = 120.0
    row_slow["dispatch_to_estimated_ratio"] = (120.0 / 24.0) / 12.0

    df_test = pd.DataFrame([row_fast, row_slow])[FEATURE_COLUMNS]
    proc = preprocessor.transform(df_test)
    probs = calibrator.predict_proba(proc)[:, 1]

    # Delayed warehouse dispatch should produce higher or equal risk
    assert probs[1] >= probs[0], f"Expected higher risk for slow dispatch, got fast={probs[0]:.4f}, slow={probs[1]:.4f}"

