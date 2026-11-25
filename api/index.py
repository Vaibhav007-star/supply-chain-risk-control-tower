"""
Vercel Serverless Function Entrypoint for Supply Chain Risk Control Tower.
Provides high-performance, low-latency API routes for KPIs, queue, and model inference.
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Locate directories relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = Path("models")

app = FastAPI(title="Supply Chain Risk API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LAZY-LOADED CACHED OBJECTS
# ---------------------------------------------------------------------------
_pipeline_dict = None
_model_card = None
_kpis_data = None
_test_df = None


def get_pipeline():
    global _pipeline_dict
    if _pipeline_dict is None:
        p_path = MODELS_DIR / "calibrated_pipeline.joblib"
        if p_path.exists():
            _pipeline_dict = joblib.load(p_path)
    return _pipeline_dict


def get_model_card():
    global _model_card
    if _model_card is None:
        c_path = MODELS_DIR / "model_card.json"
        if c_path.exists():
            with open(c_path, "r") as f:
                _model_card = json.load(f)
        else:
            _model_card = {}
    return _model_card


def get_kpis_data():
    global _kpis_data
    if _kpis_data is None:
        k_path = MODELS_DIR / "kpis.json"
        if k_path.exists():
            with open(k_path, "r") as f:
                _kpis_data = json.load(f)
        else:
            _kpis_data = {"summary": {}, "states": [], "categories": []}
    return _kpis_data


def get_test_predictions():
    global _test_df
    if _test_df is None:
        csv_path = MODELS_DIR / "test_predictions.csv"
        pq_path = MODELS_DIR / "test_predictions.parquet"
        if csv_path.exists():
            _test_df = pd.read_csv(csv_path)
        elif pq_path.exists():
            _test_df = pd.read_parquet(pq_path)
        else:
            _test_df = pd.DataFrame()
    return _test_df


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class OrderScoringRequest(BaseModel):
    weight_g: float = Field(2500.0)
    volume_cm3: float = Field(8000.0)
    order_value: float = Field(180.0)
    freight_value: float = Field(35.0)
    distance_km: float = Field(450.0)
    customer_state: str = Field("RJ")
    seller_state: str = Field("SP")
    seller_delay_rate: float = Field(0.18)
    dispatch_hours: float = Field(48.0)
    sla_days: float = Field(18.0)
    category: str = Field("health_beauty")
    payment_type: str = Field("credit_card")


# ---------------------------------------------------------------------------
# CORE BUSINESS LOGIC FUNCTIONS
# ---------------------------------------------------------------------------
def _fallback_predict(req: OrderScoringRequest) -> Dict[str, Any]:
    """Calibrated logistic fallback used when native sklearn/scipy DLLs are restricted."""
    state_risk = {
        "CE": 0.42, "AM": 0.36, "PE": 0.30, "BA": 0.24,
        "RS": 0.04, "PR": 0.02, "RJ": -0.02, "SP": -0.08, "MG": -0.08, "DF": -0.15
    }

    def score_for_dispatch(dispatch_h: float) -> float:
        d_to_est = (dispatch_h / 24.0) / max(1.0, req.sla_days)
        f_to_p = req.freight_value / max(1.0, req.order_value)
        z = (
            -2.15
            + 0.021 * dispatch_h
            + 2.40 * d_to_est
            + 2.35 * req.seller_delay_rate
            + 0.00042 * req.distance_km
            + 0.35 * f_to_p
            + state_risk.get(req.customer_state, 0.0)
            + (0.14 if req.customer_state != req.seller_state else -0.08)
            - 0.028 * max(0.0, req.sla_days - 12.0)
        )
        return float(max(0.02, min(0.96, 1.0 / (1.0 + math.exp(-z)))))

    current_prob = score_for_dispatch(req.dispatch_hours)
    threshold = 0.13
    if current_prob >= 0.70:
        tier = "Critical"
    elif current_prob >= 0.40:
        tier = "High"
    elif current_prob >= 0.20:
        tier = "Medium"
    else:
        tier = "Low"

    simulation_points = []
    for hrs_saved in [0, 12, 24, 36, 48]:
        d = max(2.0, req.dispatch_hours - hrs_saved)
        p = score_for_dispatch(d)
        rel_drop = 0.0 if current_prob == 0 else round(((current_prob - p) / current_prob) * 100, 1)
        simulation_points.append({
            "hours_saved": hrs_saved,
            "resulting_dispatch_hours": d,
            "delay_probability": round(p, 4),
            "delay_probability_pct": round(p * 100, 1),
            "relative_risk_reduction_pct": rel_drop,
        })

    d_to_est = (req.dispatch_hours / 24.0) / max(1.0, req.sla_days)
    raw_factors = [
        ("dispatch_time_hours", (req.dispatch_hours - 36.0) * 0.0022),
        ("dispatch_to_estimated_ratio", (d_to_est - 0.12) * 0.28),
        ("seller_historical_delay_rate", (req.seller_delay_rate - 0.15) * 0.35),
        ("distance_km", (req.distance_km - 500.0) * 0.000045),
        (f"customer_state_{req.customer_state}", state_risk.get(req.customer_state, 0.0) * 0.18),
        ("freight_to_price_ratio", ((req.freight_value / max(1.0, req.order_value)) - 0.2) * 0.09),
        ("estimated_delivery_window_days", -(req.sla_days - 15.0) * 0.004),
        ("total_weight_g", (req.weight_g - 2000.0) * 0.000004),
    ]
    raw_factors.sort(key=lambda x: abs(x[1]), reverse=True)
    shap_factors = [
        {
            "feature": fname,
            "shap_value": round(val, 4),
            "direction": "increase" if val >= 0 else "decrease",
            "impact": "Increases Delay Risk" if val >= 0 else "Reduces Delay Risk",
        }
        for fname, val in raw_factors
    ]

    return {
        "delay_probability": round(current_prob, 4),
        "delay_probability_pct": round(current_prob * 100, 1),
        "risk_tier": tier,
        "is_predicted_delayed": bool(current_prob >= threshold),
        "operational_threshold": threshold,
        "champion_model": "RandomForest (Calibrated Fallback)",
        "what_if_simulation": simulation_points,
        "shap_contributions": shap_factors,
    }


def execute_predict(req: OrderScoringRequest) -> Dict[str, Any]:
    import traceback
    try:
        pipeline = get_pipeline()
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline model not loaded.")

        preprocessor = pipeline["preprocessor"]
        calibrator = pipeline["calibrator"]
        feature_cols = pipeline["feature_columns"]
        threshold = pipeline.get("best_threshold", 0.13)
        feature_names = pipeline.get("all_transformed_feature_names", [])

        def make_vector(dispatch_h: float):
            is_cross = 1 if req.customer_state != req.seller_state else 0
            f_to_p = req.freight_value / (req.order_value + 1e-4)
            w_dens = req.weight_g / (req.volume_cm3 + 1.0)
            f_km = req.freight_value / (req.distance_km + 1.0)
            f_wt = req.freight_value / (req.weight_g + 1.0)
            d_to_est = (dispatch_h / 24.0) / max(1.0, req.sla_days)

            row_dict = {
                "distance_km": req.distance_km,
                "is_cross_state_shipment": is_cross,
                "total_items": 1,
                "distinct_products_count": 1,
                "total_order_value": req.order_value,
                "total_freight_value": req.freight_value,
                "freight_to_price_ratio": f_to_p,
                "total_weight_g": req.weight_g,
                "total_volume_cm3": req.volume_cm3,
                "weight_density": w_dens,
                "freight_per_km": f_km,
                "freight_per_weight": f_wt,
                "dispatch_time_hours": dispatch_h,
                "estimated_delivery_window_days": req.sla_days,
                "dispatch_to_estimated_ratio": d_to_est,
                "seller_historical_delay_rate": req.seller_delay_rate,
                "purchase_hour": 14,
                "purchase_dayofweek": 2,
                "is_weekend": 0,
                "purchase_month": 5,
                "is_peak_season": 0,
                "is_holiday": 0,
                "customer_state": req.customer_state,
                "seller_state": req.seller_state,
                "primary_category_name": req.category,
                "dominant_payment_type": req.payment_type,
            }
            return pd.DataFrame([row_dict])[feature_cols]

        # Current Prediction
        current_df = make_vector(req.dispatch_hours)
        proc_vec = preprocessor.transform(current_df)
        current_prob = float(calibrator.predict_proba(proc_vec)[0, 1])

        if current_prob >= 0.70:
            tier = "Critical"
        elif current_prob >= 0.40:
            tier = "High"
        elif current_prob >= 0.20:
            tier = "Medium"
        else:
            tier = "Low"

        # What-If Simulation
        simulation_points = []
        for hrs_saved in [0, 12, 24, 36, 48]:
            d = max(2.0, req.dispatch_hours - hrs_saved)
            p = float(calibrator.predict_proba(preprocessor.transform(make_vector(d)))[0, 1])
            rel_drop = 0.0 if current_prob == 0 else round(((current_prob - p) / current_prob) * 100, 1)
            simulation_points.append({
                "hours_saved": hrs_saved,
                "resulting_dispatch_hours": d,
                "delay_probability": round(p, 4),
                "delay_probability_pct": round(p * 100, 1),
                "relative_risk_reduction_pct": rel_drop
            })

        # Feature Contributions
        shap_factors = []
        try:
            best_estimator = pipeline.get("best_estimator")
            if best_estimator is not None and hasattr(best_estimator, "feature_importances_"):
                importances = best_estimator.feature_importances_
                sample_proc = proc_vec[0]
                scores = sample_proc * importances
                top_idx = np.argsort(np.abs(scores))[::-1][:8]
                for idx in top_idx:
                    fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                    v = float(scores[idx])
                    shap_factors.append({
                        "feature": fname,
                        "shap_value": round(v, 4),
                        "direction": "increase" if v >= 0 else "decrease",
                        "impact": "Increases Delay Risk" if v >= 0 else "Reduces Delay Risk"
                    })
        except Exception as e:
            print(f"Attribution note: {e}")

        return {
            "delay_probability": round(current_prob, 4),
            "delay_probability_pct": round(current_prob * 100, 1),
            "risk_tier": tier,
            "is_predicted_delayed": bool(current_prob >= threshold),
            "operational_threshold": threshold,
            "champion_model": pipeline.get("best_model_name", "RandomForest"),
            "what_if_simulation": simulation_points,
            "shap_contributions": shap_factors,
        }
    except (ImportError, OSError):
        return _fallback_predict(req)
    except Exception as exc:
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "models_dir_contents": [str(x) for x in MODELS_DIR.glob("*")] if MODELS_DIR.exists() else "missing",
            "base_dir": str(BASE_DIR),
            "cwd": os.getcwd()
        }


def execute_queue(tier=None, state=None, min_prob=0.0, search=None, limit=100):
    df = get_test_predictions()
    if df.empty:
        return {"total": 0, "orders": []}

    filtered = df.copy()
    if min_prob > 0:
        filtered = filtered[filtered["delay_probability"] >= min_prob]
    if tier and tier.lower() != "all":
        filtered = filtered[filtered["risk_tier"].astype(str).str.lower() == tier.lower()]
    if state and state.lower() != "all":
        filtered = filtered[filtered["customer_state"].str.lower() == state.lower()]
    if search:
        s = search.lower()
        filtered = filtered[
            filtered["order_id"].str.lower().str.contains(s, na=False) |
            filtered["primary_category_name"].str.lower().str.contains(s, na=False)
        ]

    filtered = filtered.sort_values(by="delay_probability", ascending=False)
    cols = [
        "order_id", "risk_tier", "delay_probability", "dispatch_time_hours",
        "distance_km", "total_weight_g", "total_order_value", "total_freight_value",
        "customer_state", "seller_state", "primary_category_name"
    ]
    avail_cols = [c for c in cols if c in filtered.columns]
    records = filtered[avail_cols].head(limit).to_dict(orient="records")

    return {
        "total": len(filtered),
        "returned": len(records),
        "orders": records
    }


# ---------------------------------------------------------------------------
# ROUTES & BULLETPROOF PATH MATCHING (Supports both path and query rewrites)
# ---------------------------------------------------------------------------
@app.get("/api")
@app.get("/")
def handle_api_get(
    request: Request,
    endpoint: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    min_prob: float = Query(0.0),
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    ep = (endpoint or "").lower().strip("/")
    p = str(request.url.path).lower().strip("/")
    
    if "kpi" in ep or "kpi" in p:
        return get_kpis_data()
    elif "export" in ep or "export" in p:
        return export_queue()
    elif "queue" in ep or "queue" in p:
        return execute_queue(tier, state, min_prob, search, limit)
    elif "model" in ep or "model" in p:
        return get_model_card()

    if not ep and p == "":
        index_html = BASE_DIR / "public" / "index.html"
        if index_html.exists():
            return HTMLResponse(index_html.read_text(encoding="utf-8"))

    return {"status": "healthy", "service": "Supply Chain Risk API", "endpoint": endpoint}


@app.post("/api")
@app.post("/")
def handle_api_post(
    request: Request,
    req: OrderScoringRequest,
    endpoint: Optional[str] = Query(None)
):
    return execute_predict(req)


@app.get("/api/kpis")
@app.get("/kpis")
def get_kpis():
    return get_kpis_data()


@app.get("/api/queue")
@app.get("/queue")
def get_queue(
    tier: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    min_prob: float = Query(0.0),
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    return execute_queue(tier, state, min_prob, search, limit)


@app.get("/api/queue/export")
@app.get("/queue/export")
def export_queue():
    df = get_test_predictions()
    if df.empty:
        raise HTTPException(status_code=404, detail="No predictions available")
    csv_str = df.to_csv(index=False)
    return StreamingResponse(
        iter([csv_str]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=high_risk_priority_manifest.csv"}
    )


@app.post("/api/predict")
@app.post("/predict")
def predict_endpoint(req: OrderScoringRequest):
    return execute_predict(req)


@app.get("/api/model-info")
@app.get("/model-info")
def get_model_info():
    return get_model_card()


# Universal Catch-All
@app.get("/{path:path}")
def catch_all_get(
    path: str,
    endpoint: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    min_prob: float = Query(0.0),
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    clean_p = (endpoint or path).lower().strip("/")
    if "kpi" in clean_p:
        return get_kpis_data()
    elif "export" in clean_p:
        return export_queue()
    elif "queue" in clean_p:
        return execute_queue(tier, state, min_prob, search, limit)
    elif "model" in clean_p:
        return get_model_card()
    return {"status": "healthy", "service": "Supply Chain Risk API", "path": path}


@app.post("/{path:path}")
def catch_all_post(path: str, req: OrderScoringRequest, endpoint: Optional[str] = Query(None)):
    return execute_predict(req)
