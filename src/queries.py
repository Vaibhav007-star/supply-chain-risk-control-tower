"""
Analytical Base Table (ABT) & Supply Chain SQL Queries.
Constructs order-level analytical base table using advanced SQL (CTEs, Window Functions),
calculating warehouse dispatch time, carrier transit time, and On-Time Delivery (OTD) rates.
"""

from typing import Optional
import sqlite3
import pandas as pd


# ---------------------------------------------------------------------------
# 1. ORDER-LEVEL ANALYTICAL BASE TABLE (ABT) QUERY
# ---------------------------------------------------------------------------
ORDER_ANALYTICAL_BASE_TABLE_QUERY = """
WITH order_items_aggregated AS (
    -- Aggregate order items to order level and identify dominant seller/category
    SELECT
        oi.order_id,
        COUNT(oi.order_item_id) AS total_items,
        COUNT(DISTINCT oi.product_id) AS distinct_products_count,
        COUNT(DISTINCT oi.seller_id) AS distinct_sellers_count,
        ROUND(SUM(oi.price), 2) AS total_order_value,
        ROUND(SUM(oi.freight_value), 2) AS total_freight_value,
        ROUND(SUM(COALESCE(p.product_weight_g, 0)), 2) AS total_weight_g,
        ROUND(SUM(
            COALESCE(p.product_length_cm, 0) * 
            COALESCE(p.product_height_cm, 0) * 
            COALESCE(p.product_width_cm, 0)
        ), 2) AS total_volume_cm3,
        -- Primary seller ID (seller with highest line-item value in this order)
        (
            SELECT sub_oi.seller_id 
            FROM order_items sub_oi 
            WHERE sub_oi.order_id = oi.order_id 
            ORDER BY sub_oi.price DESC 
            LIMIT 1
        ) AS primary_seller_id,
        -- Primary product category (in English)
        (
            SELECT COALESCE(t.product_category_name_english, sub_p.product_category_name, 'unknown')
            FROM order_items sub_oi
            JOIN products sub_p ON sub_oi.product_id = sub_p.product_id
            LEFT JOIN product_category_translation t 
                ON sub_p.product_category_name = t.product_category_name
            WHERE sub_oi.order_id = oi.order_id
            ORDER BY sub_oi.price DESC
            LIMIT 1
        ) AS primary_category_name
    FROM order_items oi
    LEFT JOIN products p ON oi.product_id = p.product_id
    GROUP BY oi.order_id
),

order_payments_aggregated AS (
    -- Aggregate multiple payment installments/methods per order
    SELECT
        order_id,
        COUNT(payment_sequential) AS payment_installments_count,
        ROUND(SUM(payment_value), 2) AS total_payment_value,
        -- Dominant payment type
        (
            SELECT sub_op.payment_type 
            FROM order_payments sub_op 
            WHERE sub_op.order_id = op.order_id 
            ORDER BY sub_op.payment_value DESC 
            LIMIT 1
        ) AS dominant_payment_type
    FROM order_payments op
    GROUP BY order_id
),

cleaned_delivered_orders AS (
    -- Filter strictly for completed deliveries with valid lifecycle timestamps
    SELECT
        o.order_id,
        o.customer_id,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_carrier_date,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,
        
        -- Target Definition: 1 if actual delivery occurred strictly after estimated delivery date
        CASE 
            WHEN datetime(o.order_delivered_customer_date) > datetime(o.order_estimated_delivery_date) 
            THEN 1 
            ELSE 0 
        END AS is_delayed,

        -- Target duration in days (negative = delivered early, positive = days late)
        ROUND(
            julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date), 
            2
        ) AS delay_days,

        -- Warehouse Dispatch Duration (approval to carrier handoff) in hours
        ROUND(
            (julianday(o.order_delivered_carrier_date) - julianday(o.order_approved_at)) * 24.0, 
            2
        ) AS dispatch_time_hours,

        -- Carrier Transit Duration (carrier handoff to customer door) in hours
        ROUND(
            (julianday(o.order_delivered_customer_date) - julianday(o.order_delivered_carrier_date)) * 24.0, 
            2
        ) AS transit_time_hours,

        -- Total Order Delivery Time in hours
        ROUND(
            (julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)) * 24.0, 
            2
        ) AS total_delivery_time_hours,

        -- Promised SLA Window (purchase to promised delivery date) in hours
        ROUND(
            (julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp)) * 24.0, 
            2
        ) AS estimated_delivery_window_hours

    FROM orders o
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp IS NOT NULL
      AND o.order_approved_at IS NOT NULL
      AND o.order_delivered_carrier_date IS NOT NULL
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
      -- Data hygiene: carrier handoff must be after order approval
      AND julianday(o.order_delivered_carrier_date) >= julianday(o.order_approved_at)
      -- Data hygiene: delivery must be after carrier handoff
      AND julianday(o.order_delivered_customer_date) >= julianday(o.order_delivered_carrier_date)
)

SELECT
    cdo.order_id,
    cdo.customer_id,
    cdo.order_purchase_timestamp,
    cdo.order_approved_at,
    cdo.order_delivered_carrier_date,
    cdo.order_delivered_customer_date,
    cdo.order_estimated_delivery_date,
    
    -- Target & Durations
    cdo.is_delayed,
    cdo.delay_days,
    cdo.dispatch_time_hours,
    cdo.transit_time_hours,
    cdo.total_delivery_time_hours,
    cdo.estimated_delivery_window_hours,

    -- Item & Order Level Features
    oia.total_items,
    oia.distinct_products_count,
    oia.distinct_sellers_count,
    oia.total_order_value,
    oia.total_freight_value,
    ROUND(
        oia.total_freight_value / NULLIF(oia.total_order_value, 0), 
        4
    ) AS freight_to_price_ratio,
    oia.total_weight_g,
    oia.total_volume_cm3,
    oia.primary_seller_id,
    oia.primary_category_name,

    -- Payment Details
    opa.payment_installments_count,
    opa.total_payment_value,
    opa.dominant_payment_type,

    -- Customer Geolocation
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,
    cg.geolocation_lat AS customer_lat,
    cg.geolocation_lng AS customer_lng,

    -- Seller Geolocation
    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,
    sg.geolocation_lat AS seller_lat,
    sg.geolocation_lng AS seller_lng,

    -- Cross-state shipping flag
    CASE 
        WHEN c.customer_state = s.seller_state THEN 0 
        ELSE 1 
    END AS is_cross_state_shipment

FROM cleaned_delivered_orders cdo
JOIN order_items_aggregated oia ON cdo.order_id = oia.order_id
LEFT JOIN order_payments_aggregated opa ON cdo.order_id = opa.order_id
LEFT JOIN customers c ON cdo.customer_id = c.customer_id
LEFT JOIN sellers s ON oia.primary_seller_id = s.seller_id
LEFT JOIN geolocation cg ON c.customer_zip_code_prefix = cg.zip_code_prefix
LEFT JOIN geolocation sg ON s.seller_zip_code_prefix = sg.zip_code_prefix
ORDER BY cdo.order_purchase_timestamp ASC;
"""


# ---------------------------------------------------------------------------
# 2. SUPPLY CHAIN KPI: STATE-LEVEL ON-TIME DELIVERY (OTD) & TRANSIT METRICS
# ---------------------------------------------------------------------------
STATE_OTD_METRICS_QUERY = """
WITH delivered_orders AS (
    SELECT
        o.order_id,
        c.customer_state,
        CASE WHEN datetime(o.order_delivered_customer_date) > datetime(o.order_estimated_delivery_date) THEN 1 ELSE 0 END AS is_delayed,
        ROUND((julianday(o.order_delivered_carrier_date) - julianday(o.order_approved_at)) * 24.0, 2) AS dispatch_time_hours,
        ROUND((julianday(o.order_delivered_customer_date) - julianday(o.order_delivered_carrier_date)) * 24.0, 2) AS transit_time_hours
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
      AND o.order_delivered_carrier_date IS NOT NULL
      AND o.order_approved_at IS NOT NULL
      AND julianday(o.order_delivered_customer_date) >= julianday(o.order_delivered_carrier_date)
)
SELECT
    customer_state,
    COUNT(order_id) AS total_orders,
    SUM(is_delayed) AS delayed_orders,
    ROUND(100.0 * (1.0 - (1.0 * SUM(is_delayed) / COUNT(order_id))), 2) AS on_time_delivery_pct,
    ROUND(100.0 * (1.0 * SUM(is_delayed) / COUNT(order_id)), 2) AS delay_rate_pct,
    ROUND(AVG(dispatch_time_hours), 2) AS avg_dispatch_time_hours,
    ROUND(AVG(transit_time_hours), 2) AS avg_transit_time_hours,
    -- Window Function: Rank states by delay rate (worst to best)
    DENSE_RANK() OVER (ORDER BY (1.0 * SUM(is_delayed) / COUNT(order_id)) DESC) AS risk_rank
FROM delivered_orders
GROUP BY customer_state
HAVING COUNT(order_id) >= 50
ORDER BY delay_rate_pct DESC;
"""


# ---------------------------------------------------------------------------
# 3. SUPPLY CHAIN KPI: PRODUCT CATEGORY ON-TIME DELIVERY (OTD) ANALYSIS
# ---------------------------------------------------------------------------
CATEGORY_OTD_METRICS_QUERY = """
WITH category_orders AS (
    SELECT
        o.order_id,
        COALESCE(t.product_category_name_english, p.product_category_name, 'Other') AS category,
        CASE WHEN datetime(o.order_delivered_customer_date) > datetime(o.order_estimated_delivery_date) THEN 1 ELSE 0 END AS is_delayed,
        ROUND((julianday(o.order_delivered_carrier_date) - julianday(o.order_approved_at)) * 24.0, 2) AS dispatch_time_hours,
        ROUND((julianday(o.order_delivered_customer_date) - julianday(o.order_delivered_carrier_date)) * 24.0, 2) AS transit_time_hours,
        oi.price,
        oi.freight_value
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_category_translation t ON p.product_category_name = t.product_category_name
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
      AND o.order_delivered_carrier_date IS NOT NULL
      AND o.order_approved_at IS NOT NULL
      AND julianday(o.order_delivered_customer_date) >= julianday(o.order_delivered_carrier_date)
)
SELECT
    category,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(is_delayed) AS delayed_orders,
    ROUND(100.0 * (1.0 - (1.0 * SUM(is_delayed) / COUNT(order_id))), 2) AS on_time_delivery_pct,
    ROUND(100.0 * (1.0 * SUM(is_delayed) / COUNT(order_id)), 2) AS delay_rate_pct,
    ROUND(AVG(dispatch_time_hours), 1) AS avg_dispatch_time_hours,
    ROUND(AVG(transit_time_hours), 1) AS avg_transit_time_hours,
    ROUND(AVG(price), 2) AS avg_item_price,
    ROUND(AVG(freight_value), 2) AS avg_freight_value,
    -- Window Function: Percentile ranking of category by risk
    ROUND(PERCENT_RANK() OVER (ORDER BY (1.0 * SUM(is_delayed) / COUNT(order_id)) ASC), 3) AS risk_percentile
FROM category_orders
GROUP BY category
HAVING COUNT(DISTINCT order_id) >= 100
ORDER BY delay_rate_pct DESC;
"""


# ---------------------------------------------------------------------------
# PYTHON WRAPPERS TO FETCH DATA FROM SQLITE
# ---------------------------------------------------------------------------
def load_analytical_base_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Executes the ABT query and returns a pandas DataFrame."""
    return pd.read_sql_query(ORDER_ANALYTICAL_BASE_TABLE_QUERY, conn)


def load_state_kpis(conn: sqlite3.Connection) -> pd.DataFrame:
    """Returns state-level On-Time Delivery KPIs and transit durations."""
    return pd.read_sql_query(STATE_OTD_METRICS_QUERY, conn)


def load_category_kpis(conn: sqlite3.Connection) -> pd.DataFrame:
    """Returns category-level On-Time Delivery KPIs and freight metrics."""
    return pd.read_sql_query(CATEGORY_OTD_METRICS_QUERY, conn)

