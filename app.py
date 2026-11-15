"""
Supply Chain Operations Control Tower (Streamlit App).
Module 1: Executive Supply Chain KPIs & Regional Hotspots
Module 2: High-Risk Order Priority Queue
Module 3: Order Risk Scorer & What-If Dispatch Optimization Simulator
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db_ingest import get_db_connection, DEFAULT_DB_PATH
from src.queries import load_state_kpis, load_category_kpis
from src.features import haversine_distance

# Set Streamlit page layout
st.set_page_config(
    page_title="Supply Chain Risk Control Tower",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-high {
        background-color: #ffa500;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #f0e68c;
        color: black;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .badge-low {
        background-color: #28a745;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA & MODEL LOADERS (CACHED)
# ---------------------------------------------------------------------------
@st.cache_data
def load_db_kpis():
    """Loads state and category supply chain KPIs from SQLite DW."""
    conn = get_db_connection(DEFAULT_DB_PATH)
    try:
        state_df = load_state_kpis(conn)
        cat_df = load_category_kpis(conn)
    finally:
        conn.close()
    return state_df, cat_df


@st.cache_data
def load_test_predictions():
    """Loads precomputed high-risk test set queue."""
    parquet_path = Path("models/test_predictions.parquet")
    csv_path = Path("models/test_predictions.csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_resource
def load_production_pipeline():
    """Loads serialized calibrated machine learning pipeline and metadata."""
    pipeline_path = Path("models/calibrated_pipeline.joblib")
    card_path = Path("models/model_card.json")

    pipeline_dict = joblib.load(pipeline_path) if pipeline_path.exists() else None
    model_card = {}
    if card_path.exists():
        with open(card_path, "r") as f:
            model_card = json.load(f)

    return pipeline_dict, model_card


# Load resources
state_kpis, cat_kpis = load_db_kpis()
test_predictions_df = load_test_predictions()
pipeline_dict, model_card = load_production_pipeline()


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM METRICS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/delivery-conveyer.png", width=64)
    st.title("Operations Tower")
    st.caption("E-Commerce Delivery Delay & Supply Chain Risk Predictor")
    st.divider()

    best_model_name = model_card.get("best_model_name", "RandomForest")
    cal_threshold = model_card.get("best_threshold", 0.13)
    st.markdown(f"**Champion Model:** `{best_model_name}`")
    st.markdown(f"**Calibrated Threshold:** `{cal_threshold:.3f}`")

    test_metrics = model_card.get("test_metrics_tuned", {})
    if test_metrics:
        st.metric("Test Delay Recall", f"{test_metrics.get('recall', 0.0):.1%}")
        st.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0.0):.3f}")
        st.metric("Test PR-AUC", f"{test_metrics.get('pr_auc', 0.0):.3f}")

    st.divider()
    st.markdown("🏢 **Target Roles:** Amazon, Flipkart, Myntra")
    st.caption("Fulfillment & Logistics Intelligence")


# ---------------------------------------------------------------------------
# HEADER & APP TITLE
# ---------------------------------------------------------------------------
st.title("🚚 E-Commerce Delivery Delay & Supply Chain Risk Control Tower")
st.markdown("Real-time fulfillment visibility, carrier bottleneck diagnostics, and predictive delay intervention.")

tab1, tab2, tab3 = st.tabs([
    "📊 Executive Supply Chain KPIs",
    "🚨 High-Risk Order Queue",
    "⚡ Risk Scorer & What-If Simulator"
])


# ===========================================================================
# TAB 1: EXECUTIVE SUPPLY CHAIN KPIS
# ===========================================================================
with tab1:
    st.subheader("Fulfillment Velocity & Regional Performance Dashboard")

    if not state_kpis.empty:
        total_orders = int(state_kpis["total_orders"].sum())
        total_delayed = int(state_kpis["delayed_orders"].sum())
        overall_otd_pct = 100.0 * (1.0 - (total_delayed / total_orders))
        avg_dispatch_h = state_kpis["avg_dispatch_time_hours"].mean()
        avg_transit_h = state_kpis["avg_transit_time_hours"].mean()

        # Top Executive KPI Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric(
                label="On-Time Delivery (OTD %)",
                value=f"{overall_otd_pct:.1f}%",
                delta=f"-{100-overall_otd_pct:.1f}% Delay Rate",
                delta_color="inverse"
            )
        with kpi_col2:
            st.metric(
                label="Total Orders Monitored",
                value=f"{total_orders:,}",
                delta=f"{total_delayed:,} Delayed Orders",
                delta_color="off"
            )
        with kpi_col3:
            st.metric(
                label="Avg. Warehouse Dispatch Lag",
                value=f"{avg_dispatch_h:.1f} hrs",
                delta="Approval to Carrier",
                delta_color="off"
            )
        with kpi_col4:
            st.metric(
                label="Avg. Carrier Transit Duration",
                value=f"{avg_transit_h/24.0:.1f} days",
                delta=f"{avg_transit_h:.0f} hrs in-transit",
                delta_color="off"
            )

        st.markdown("---")

        # Visualizations: Row 1
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("##### Regional Delay Hotspots by Destination State")
            sorted_state = state_kpis.sort_values(by="delay_rate_pct", ascending=True)
            fig_state = px.bar(
                sorted_state,
                x="delay_rate_pct",
                y="customer_state",
                orientation="h",
                color="delay_rate_pct",
                color_continuous_scale="Reds",
                labels={"delay_rate_pct": "Delay Rate (%)", "customer_state": "Destination State"},
                title="Delay Rate (%) by Brazilian State"
            )
            fig_state.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_state, use_container_width=True)

        with viz_col2:
            st.markdown("##### Fulfillment Cycle Breakdown: Dispatch vs. Transit")
            fig_cycle = px.scatter(
                state_kpis,
                x="avg_dispatch_time_hours",
                y="avg_transit_time_hours",
                size="total_orders",
                color="delay_rate_pct",
                hover_name="customer_state",
                labels={
                    "avg_dispatch_time_hours": "Warehouse Dispatch Lag (Hours)",
                    "avg_transit_time_hours": "Carrier Transit Time (Hours)",
                    "delay_rate_pct": "Delay %"
                },
                title="Warehouse Dispatch Lag vs Carrier Transit Duration",
                color_continuous_scale="Viridis"
            )
            fig_cycle.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cycle, use_container_width=True)

        # Visualizations: Row 2 (Category Risk)
        if not cat_kpis.empty:
            st.markdown("##### Product Category Delay Vulnerability & Freight Costs")
            fig_cat = px.scatter(
                cat_kpis.head(15),
                x="avg_freight_value",
                y="delay_rate_pct",
                size="total_orders",
                color="avg_dispatch_time_hours",
                hover_name="category",
                labels={
                    "avg_freight_value": "Avg Freight Value (R$)",
                    "delay_rate_pct": "Delay Rate (%)",
                    "avg_dispatch_time_hours": "Dispatch Lag (h)"
                },
                title="Category Freight Cost vs Delay Probability (Top 15 Categories)",
                color_continuous_scale="Plasma"
            )
            fig_cat.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_cat, use_container_width=True)

    else:
        st.warning("No warehouse KPI records found. Ingest SQLite database first.")


# ===========================================================================
# TAB 2: HIGH-RISK ORDER PRIORITY QUEUE
# ===========================================================================
with tab2:
    st.subheader("🚨 Priority Dispatch & Expedited Carrier Intervention Queue")
    st.markdown(
        "Fulfillment center operations queue ranked by **calibrated delay probability**. "
        "Allows warehouse team leads to identify and reroute at-risk shipments prior to promised SLA breaches."
    )

    if not test_predictions_df.empty:
        # Filter controls
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            tier_options = ["All"] + list(test_predictions_df["risk_tier"].dropna().unique())
            selected_tier = st.selectbox("Filter Risk Tier:", tier_options)
        with f_col2:
            state_options = ["All"] + sorted(list(test_predictions_df["customer_state"].dropna().unique()))
            selected_state = st.selectbox("Customer State:", state_options)
        with f_col3:
            min_prob = st.slider("Minimum Calibrated Delay Probability:", 0.0, 1.0, 0.40, 0.05)

        # Apply filtering
        filtered_queue = test_predictions_df[test_predictions_df["delay_probability"] >= min_prob]
        if selected_tier != "All":
            filtered_queue = filtered_queue[filtered_queue["risk_tier"] == selected_tier]
        if selected_state != "All":
            filtered_queue = filtered_queue[filtered_queue["customer_state"] == selected_state]

        filtered_queue = filtered_queue.sort_values(by="delay_probability", ascending=False)

        st.markdown(f"**Identified At-Risk Shipments:** `{len(filtered_queue):,}` orders pending priority routing.")

        display_cols = [
            "order_id",
            "risk_tier",
            "delay_probability",
            "dispatch_time_hours",
            "distance_km",
            "total_weight_g",
            "total_order_value",
            "total_freight_value",
            "customer_state",
            "seller_state",
            "primary_category_name",
        ]
        available_display_cols = [c for c in display_cols if c in filtered_queue.columns]

        # Styled Table Display
        st.dataframe(
            filtered_queue[available_display_cols].head(100),
            column_config={
                "delay_probability": st.column_config.ProgressColumn(
                    "Delay Risk",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "dispatch_time_hours": st.column_config.NumberColumn("Dispatch Lag (hrs)", format="%.1f h"),
                "distance_km": st.column_config.NumberColumn("Distance", format="%.0f km"),
                "total_order_value": st.column_config.NumberColumn("Value (R$)", format="R$ %.2f"),
                "total_freight_value": st.column_config.NumberColumn("Freight (R$)", format="R$ %.2f"),
            },
            use_container_width=True,
            height=450
        )

        # Export CSV Button
        csv_data = filtered_queue[available_display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export High-Risk Priority Manifest (CSV)",
            data=csv_data,
            file_name="high_risk_priority_manifest.csv",
            mime="text/csv",
        )
    else:
        st.info("No test set predictions available. Please run model training pipeline first.")


# ===========================================================================
# TAB 3: ORDER RISK SCORER & WHAT-IF SIMULATOR
# ===========================================================================
with tab3:
    st.subheader("⚡ Real-Time Order Delay Risk Scorer & Fulfillment What-If Simulator")
    st.markdown(
        "Simulate order fulfillment parameters to predict delay probability in real time, "
        "and measure how expediting warehouse dispatch turnaround mitigates delivery SLA risk."
    )

    if pipeline_dict is None:
        st.error("Production pipeline artifact not found in `models/`. Please train the model first.")
    else:
        preprocessor = pipeline_dict["preprocessor"]
        best_estimator = pipeline_dict["best_estimator"]
        calibrator = pipeline_dict["calibrator"]
        feature_cols = pipeline_dict["feature_columns"]
        threshold = pipeline_dict["best_threshold"]
        seller_summary = pipeline_dict["seller_summary"]
        global_prior = pipeline_dict["global_prior"]

        sim_col1, sim_col2 = st.columns([1, 1])

        with sim_col1:
            st.markdown("##### 📦 Order & Logistics Inputs")
            in_weight = st.number_input("Total Order Weight (grams):", min_value=100.0, max_value=30000.0, value=2500.0, step=100.0)
            in_vol = st.number_input("Packaging Volume (cm³):", min_value=500.0, max_value=80000.0, value=8000.0, step=500.0)
            in_order_val = st.number_input("Total Order Value (R$):", min_value=10.0, max_value=5000.0, value=180.0, step=10.0)
            in_freight_val = st.number_input("Freight Cost (R$):", min_value=5.0, max_value=500.0, value=35.0, step=5.0)

            c_state_col, s_state_col = st.columns(2)
            with c_state_col:
                in_cust_state = st.selectbox("Customer State:", ["SP", "RJ", "MG", "BA", "RS", "PE", "CE", "AM", "PR", "DF"], index=1)
            with s_state_col:
                in_seller_state = st.selectbox("Seller State:", ["SP", "RJ", "MG", "BA", "RS", "PE", "CE", "AM", "PR", "DF"], index=0)

            in_dist = st.slider("Estimated Transit Distance (km):", min_value=10.0, max_value=3500.0, value=450.0, step=25.0)
            in_seller_delay_rate = st.slider("Seller Historical Delay Rate:", min_value=0.0, max_value=1.0, value=0.18, step=0.01)
            in_sla_days = st.slider("Promised SLA Window (Days):", min_value=5.0, max_value=45.0, value=18.0, step=1.0)
            in_dispatch_hours = st.slider("Current Warehouse Dispatch Lag (Hours):", min_value=4.0, max_value=168.0, value=48.0, step=4.0)

            in_cat = st.selectbox("Product Category:", [
                "health_beauty", "watches_gifts", "bed_bath_table", "sports_leisure",
                "computers_accessories", "furniture_decor", "housewares", "telephony", "auto", "toys"
            ], index=0)
            in_pay = st.selectbox("Payment Type:", ["credit_card", "boleto", "voucher", "debit_card"], index=0)

        with sim_col2:
            st.markdown("##### 🎯 Real-Time Delay Risk Diagnostics")

            # Helper function to construct feature row
            def build_feature_vector(dispatch_h: float):
                is_cross = 1 if in_cust_state != in_seller_state else 0
                f_to_p = in_freight_val / (in_order_val + 1e-4)
                w_dens = in_weight / (in_vol + 1.0)
                f_km = in_freight_val / (in_dist + 1.0)
                f_wt = in_freight_val / (in_weight + 1.0)
                d_to_est = (dispatch_h / 24.0) / max(1.0, in_sla_days)

                row_dict = {
                    "distance_km": in_dist,
                    "is_cross_state_shipment": is_cross,
                    "total_items": 1,
                    "distinct_products_count": 1,
                    "total_order_value": in_order_val,
                    "total_freight_value": in_freight_val,
                    "freight_to_price_ratio": f_to_p,
                    "total_weight_g": in_weight,
                    "total_volume_cm3": in_vol,
                    "weight_density": w_dens,
                    "freight_per_km": f_km,
                    "freight_per_weight": f_wt,
                    "dispatch_time_hours": dispatch_h,
                    "estimated_delivery_window_days": in_sla_days,
                    "dispatch_to_estimated_ratio": d_to_est,
                    "seller_historical_delay_rate": in_seller_delay_rate,
                    "purchase_hour": 14,
                    "purchase_dayofweek": 2,
                    "is_weekend": 0,
                    "purchase_month": 5,
                    "is_peak_season": 0,
                    "is_holiday": 0,
                    "customer_state": in_cust_state,
                    "seller_state": in_seller_state,
                    "primary_category_name": in_cat,
                    "dominant_payment_type": in_pay,
                }
                return pd.DataFrame([row_dict])[feature_cols]

            # Current prediction
            current_feat_df = build_feature_vector(in_dispatch_hours)
            proc_vec = preprocessor.transform(current_feat_df)
            current_prob = float(calibrator.predict_proba(proc_vec)[0, 1])

            # Badge styling based on risk
            if current_prob >= 0.70:
                badge_html = f"<span class='badge-critical'>CRITICAL DELAY RISK: {current_prob:.1%}</span>"
            elif current_prob >= 0.40:
                badge_html = f"<span class='badge-high'>HIGH DELAY RISK: {current_prob:.1%}</span>"
            elif current_prob >= 0.20:
                badge_html = f"<span class='badge-medium'>MODERATE RISK: {current_prob:.1%}</span>"
            else:
                badge_html = f"<span class='badge-low'>ON-TIME DELIVERY LIKELY: {current_prob:.1%}</span>"

            st.markdown(f"### Status: {badge_html}", unsafe_allow_html=True)
            st.caption(f"Operational Decision Threshold: `{threshold:.3f}` | Model: `{pipeline_dict['best_model_name']}`")

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Calibrated Delay Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 20], 'color': "#d4edda"},
                        {'range': [20, 50], 'color': "#fff3cd"},
                        {'range': [50, 75], 'color': "#ffeeba"},
                        {'range': [75, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # -------------------------------------------------------------------
            # WHAT-IF DISPATCH ACCELERATION SIMULATOR
            # -------------------------------------------------------------------
            st.markdown("---")
            st.markdown("##### 🛠️ What-If Fulfillment Acceleration Simulator")
            st.caption("How much does expediting warehouse handoff reduce the likelihood of late delivery?")

            expedite_hours = st.select_slider(
                "Expedite Dispatch Turnaround By (Hours Saved):",
                options=[0, 12, 24, 36, 48],
                value=24
            )

            new_dispatch = max(2.0, in_dispatch_hours - expedite_hours)
            new_feat_df = build_feature_vector(new_dispatch)
            new_proc_vec = preprocessor.transform(new_feat_df)
            new_prob = float(calibrator.predict_proba(new_proc_vec)[0, 1])
            prob_delta = new_prob - current_prob

            sim_res1, sim_res2 = st.columns(2)
            with sim_res1:
                st.metric(
                    label=f"Delay Risk after -{expedite_hours}h Dispatch",
                    value=f"{new_prob:.1%}",
                    delta=f"{prob_delta:.1%}",
                    delta_color="inverse"
                )
            with sim_res2:
                risk_reduction_pct = 0.0 if current_prob == 0 else (prob_delta / current_prob) * 100
                st.metric(
                    label="Relative Risk Reduction",
                    value=f"{abs(risk_reduction_pct):.1f}%",
                    delta="Fulfillment Optimization Impact",
                    delta_color="normal"
                )

            # Sensitivity Curve
            curve_points = []
            for h in [0, 12, 24, 36, 48, 72]:
                d = max(2.0, in_dispatch_hours - h)
                p = float(calibrator.predict_proba(preprocessor.transform(build_feature_vector(d)))[0, 1])
                curve_points.append({"Hours Saved": h, "Warehouse Dispatch (hrs)": d, "Delay Probability (%)": p * 100})

            curve_df = pd.DataFrame(curve_points)
            fig_sens = px.line(
                curve_df,
                x="Hours Saved",
                y="Delay Probability (%)",
                markers=True,
                title="Sensitivity of Delay Probability to Dispatch Acceleration",
                labels={"Hours Saved": "Warehouse Hours Saved", "Delay Probability (%)": "Delay Probability (%)"}
            )
            fig_sens.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sens, use_container_width=True)

            # -------------------------------------------------------------------
            # LOCAL SHAP WATERFALL / FEATURE ATTRIBUTION
            # -------------------------------------------------------------------
            st.markdown("---")
            st.markdown("##### 🔍 Local SHAP Feature Attribution (Root Cause Diagnosis)")
            st.caption("How specific order features push delay probability higher (+) or pull it lower (-)")

            try:
                import shap
                explainer = shap.TreeExplainer(best_estimator)
                shap_raw = explainer.shap_values(proc_vec)
                
                if isinstance(shap_raw, list):
                    sample_shap = shap_raw[1][0]
                elif hasattr(shap_raw, "shape") and len(shap_raw.shape) == 3:
                    sample_shap = shap_raw[0, :, 1]
                else:
                    sample_shap = shap_raw[0]

                feature_names = pipeline_dict.get("all_transformed_feature_names", [])
                if len(feature_names) == len(sample_shap):
                    shap_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP Value": sample_shap
                    })
                    # Top influential drivers
                    shap_df["Abs SHAP"] = shap_df["SHAP Value"].abs()
                    top_shap = shap_df.sort_values(by="Abs SHAP", ascending=False).head(8)
                    top_shap["Direction"] = np.where(top_shap["SHAP Value"] >= 0, "Increases Delay Risk (+)", "Reduces Delay Risk (-)")

                    fig_shap = px.bar(
                        top_shap.sort_values(by="SHAP Value", ascending=True),
                        x="SHAP Value",
                        y="Feature",
                        orientation="h",
                        color="Direction",
                        color_discrete_map={
                            "Increases Delay Risk (+)": "#d9534f",
                            "Reduces Delay Risk (-)": "#5cb85c"
                        },
                        title="Top Order-Specific SHAP Impact Drivers"
                    )
                    fig_shap.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.caption(f"SHAP local explanation: {e}")

