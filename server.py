"""
FastAPI Enterprise Web Application & REST API Server.
Powers the Supply Chain Risk Control Tower website with real-time inference,
KPI analytics, queue filtering, what-if fulfillment simulation, and SHAP attribution.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
import shap

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db_ingest import get_db_connection, DEFAULT_DB_PATH
from src.queries import load_state_kpis, load_category_kpis

# Initialize FastAPI App
app = FastAPI(
    title="Supply Chain Risk Control Tower API",
    description="Enterprise API for Delivery Delay Prediction & Logistics Simulation",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory paths
STATIC_DIR = PROJECT_ROOT / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------------------------
pipeline_dict = None
model_card = {}
test_predictions_df = pd.DataFrame()
tree_explainer = None


def load_all_artifacts():
    global pipeline_dict, model_card, test_predictions_df, tree_explainer

    pipeline_path = MODELS_DIR / "calibrated_pipeline.joblib"
    card_path = MODELS_DIR / "model_card.json"
    pq_path = MODELS_DIR / "test_predictions.parquet"
    csv_path = MODELS_DIR / "test_predictions.csv"

    if pipeline_path.exists():
        pipeline_dict = joblib.load(pipeline_path)
        best_estimator = pipeline_dict.get("best_estimator")
        if best_estimator is not None and hasattr(best_estimator, "estimators_"):
            try:
                tree_explainer = shap.TreeExplainer(best_estimator)
            except Exception as e:
                print(f"TreeExplainer load note: {e}")

    if card_path.exists():
        with open(card_path, "r") as f:
            model_card = json.load(f)

    if pq_path.exists():
        test_predictions_df = pd.read_parquet(pq_path)
    elif csv_path.exists():
        test_predictions_df = pd.read_csv(csv_path)


load_all_artifacts()


# ---------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# ---------------------------------------------------------------------------
class OrderScoringRequest(BaseModel):
    weight_g: float = Field(2500.0, description="Total order weight in grams")
    volume_cm3: float = Field(8000.0, description="Package volume in cm³")
    order_value: float = Field(180.0, description="Total order value in R$")
    freight_value: float = Field(35.0, description="Freight fee in R$")
    distance_km: float = Field(450.0, description="Transit distance in kilometers")
    customer_state: str = Field("RJ", description="Destination state code")
    seller_state: str = Field("SP", description="Origin seller state code")
    seller_delay_rate: float = Field(0.18, description="Historical seller delay probability")
    dispatch_hours: float = Field(48.0, description="Warehouse dispatch lag in hours")
    sla_days: float = Field(18.0, description="Promised delivery window in days")
    category: str = Field("health_beauty", description="Product category")
    payment_type: str = Field("credit_card", description="Payment method")


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/api/kpis")
def get_kpis():
    """Returns state-level and category-level supply chain KPIs from SQLite DW."""
    conn = get_db_connection(DEFAULT_DB_PATH)
    try:
        state_df = load_state_kpis(conn)
        cat_df = load_category_kpis(conn)
    finally:
        conn.close()

    if state_df.empty:
        return {"summary": {}, "states": [], "categories": []}

    total_orders = int(state_df["total_orders"].sum())
    total_delayed = int(state_df["delayed_orders"].sum())
    otd_pct = round(100.0 * (1.0 - (total_delayed / total_orders)), 2)
    avg_dispatch = round(float(state_df["avg_dispatch_time_hours"].mean()), 1)
    avg_transit = round(float(state_df["avg_transit_time_hours"].mean()), 1)

    return {
        "summary": {
            "total_orders": total_orders,
            "total_delayed": total_delayed,
            "on_time_delivery_pct": otd_pct,
            "delay_rate_pct": round(100.0 - otd_pct, 2),
            "avg_dispatch_hours": avg_dispatch,
            "avg_transit_hours": avg_transit,
            "avg_transit_days": round(avg_transit / 24.0, 1),
        },
        "states": state_df.to_dict(orient="records"),
        "categories": cat_df.head(15).to_dict(orient="records"),
    }


@app.get("/api/queue")
def get_high_risk_queue(
    tier: Optional[str] = Query(None, description="Risk tier: Critical, High, Medium, Low"),
    state: Optional[str] = Query(None, description="Customer state code"),
    min_prob: float = Query(0.0, description="Minimum delay probability"),
    search: Optional[str] = Query(None, description="Search order ID or category"),
    limit: int = Query(100, ge=1, le=500),
):
    """Returns filterable queue of at-risk shipments."""
    if test_predictions_df.empty:
        return {"total": 0, "orders": []}

    df = test_predictions_df.copy()
    if min_prob > 0:
        df = df[df["delay_probability"] >= min_prob]
    if tier and tier.lower() != "all":
        df = df[df["risk_tier"].astype(str).str.lower() == tier.lower()]
    if state and state.lower() != "all":
        df = df[df["customer_state"].str.lower() == state.lower()]
    if search:
        s = search.lower()
        df = df[
            df["order_id"].str.lower().str.contains(s, na=False) |
            df["primary_category_name"].str.lower().str.contains(s, na=False)
        ]

    df = df.sort_values(by="delay_probability", ascending=False)
    total_count = len(df)

    cols = [
        "order_id", "risk_tier", "delay_probability", "dispatch_time_hours",
        "distance_km", "total_weight_g", "total_order_value", "total_freight_value",
        "customer_state", "seller_state", "primary_category_name"
    ]
    avail_cols = [c for c in cols if c in df.columns]
    records = df[avail_cols].head(limit).to_dict(orient="records")

    return {
        "total": total_count,
        "returned": len(records),
        "orders": records
    }


@app.get("/api/queue/export")
def export_queue_csv():
    """Exports filtered high-risk orders as a downloadable CSV."""
    if test_predictions_df.empty:
        raise HTTPException(status_code=404, detail="No predictions available")
    csv_str = test_predictions_df.to_csv(index=False)
    return StreamingResponse(
        iter([csv_str]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=high_risk_priority_manifest.csv"}
    )


@app.post("/api/predict")
def predict_and_simulate(req: OrderScoringRequest):
    """Scores single order, runs SHAP attribution, and simulates dispatch turnaround sensitivity."""
    if pipeline_dict is None:
        raise HTTPException(status_code=500, detail="Model pipeline not loaded.")

    preprocessor = pipeline_dict["preprocessor"]
    calibrator = pipeline_dict["calibrator"]
    best_estimator = pipeline_dict["best_estimator"]
    feature_cols = pipeline_dict["feature_columns"]
    threshold = pipeline_dict.get("best_threshold", 0.13)
    feature_names = pipeline_dict.get("all_transformed_feature_names", [])

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

    # Risk Tier
    if current_prob >= 0.70:
        tier = "Critical"
    elif current_prob >= 0.40:
        tier = "High"
    elif current_prob >= 0.20:
        tier = "Medium"
    else:
        tier = "Low"

    # What-If Simulation Curve (Accelerating warehouse handoff)
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

    # Local SHAP Attribution
    shap_factors = []
    if tree_explainer is not None:
        try:
            shap_raw = tree_explainer.shap_values(proc_vec)
            if isinstance(shap_raw, list):
                sample_shap = shap_raw[1][0]
            elif hasattr(shap_raw, "shape") and len(shap_raw.shape) == 3:
                sample_shap = shap_raw[0, :, 1]
            else:
                sample_shap = shap_raw[0]

            if len(feature_names) == len(sample_shap):
                ranked = sorted(
                    zip(feature_names, sample_shap),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:8]
                shap_factors = [
                    {
                        "feature": f,
                        "shap_value": round(float(v), 4),
                        "direction": "increase" if v >= 0 else "decrease",
                        "impact": "Increases Delay Risk" if v >= 0 else "Reduces Delay Risk"
                    }
                    for f, v in ranked
                ]
        except Exception as e:
            print(f"SHAP local error: {e}")

    return {
        "delay_probability": round(current_prob, 4),
        "delay_probability_pct": round(current_prob * 100, 1),
        "risk_tier": tier,
        "is_predicted_delayed": bool(current_prob >= threshold),
        "operational_threshold": threshold,
        "champion_model": pipeline_dict.get("best_model_name", "RandomForest"),
        "what_if_simulation": simulation_points,
        "shap_contributions": shap_factors,
    }


@app.get("/api/model-info")
def get_model_info():
    """Returns model metadata, leaderboard, and threshold configuration."""
    return model_card


# ---------------------------------------------------------------------------
# SERVE WEBSITE FRONTEND
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serves the main website dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Control Tower website building...</h1>", status_code=200)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
