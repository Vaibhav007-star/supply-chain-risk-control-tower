"""
Unit and Integration Tests for Phase 1: Database Ingestion & SQL Query Logic.
Validates target calculation, dispatch/transit durations, and order-level aggregation.
"""

import sqlite3
import pytest
import pandas as pd
from src.queries import (
    ORDER_ANALYTICAL_BASE_TABLE_QUERY,
    STATE_OTD_METRICS_QUERY,
    CATEGORY_OTD_METRICS_QUERY,
)


@pytest.fixture
def mock_db():
    """Creates an in-memory SQLite database populated with minimal test data."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # 1. Orders table
    cursor.execute("""
    CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT,
        order_status TEXT,
        order_purchase_timestamp TEXT,
        order_approved_at TEXT,
        order_delivered_carrier_date TEXT,
        order_delivered_customer_date TEXT,
        order_estimated_delivery_date TEXT
    );
    """)

    # 2. Order items table
    cursor.execute("""
    CREATE TABLE order_items (
        order_id TEXT,
        order_item_id INTEGER,
        product_id TEXT,
        seller_id TEXT,
        shipping_limit_date TEXT,
        price REAL,
        freight_value REAL
    );
    """)

    # 3. Products table
    cursor.execute("""
    CREATE TABLE products (
        product_id TEXT PRIMARY KEY,
        product_category_name TEXT,
        product_name_lenght INTEGER,
        product_description_lenght INTEGER,
        product_photos_qty INTEGER,
        product_weight_g REAL,
        product_length_cm REAL,
        product_height_cm REAL,
        product_width_cm REAL
    );
    """)

    # 4. Customers table
    cursor.execute("""
    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        customer_unique_id TEXT,
        customer_zip_code_prefix INTEGER,
        customer_city TEXT,
        customer_state TEXT
    );
    """)

    # 5. Sellers table
    cursor.execute("""
    CREATE TABLE sellers (
        seller_id TEXT PRIMARY KEY,
        seller_zip_code_prefix INTEGER,
        seller_city TEXT,
        seller_state TEXT
    );
    """)

    # 6. Geolocation table
    cursor.execute("""
    CREATE TABLE geolocation (
        zip_code_prefix INTEGER PRIMARY KEY,
        geolocation_lat REAL,
        geolocation_lng REAL,
        geolocation_city TEXT,
        geolocation_state TEXT
    );
    """)

    # 7. Order payments table
    cursor.execute("""
    CREATE TABLE order_payments (
        order_id TEXT,
        payment_sequential INTEGER,
        payment_type TEXT,
        payment_installments INTEGER,
        payment_value REAL
    );
    """)

    # 8. Category translation table
    cursor.execute("""
    CREATE TABLE product_category_translation (
        product_category_name TEXT PRIMARY KEY,
        product_category_name_english TEXT
    );
    """)

    # --- Seed Test Records ---
    # Order 1: Delivered on time (delivered 2018-02-10 < estimated 2018-02-15) -> is_delayed = 0
    # Dispatch time: 2018-02-03 10:00 to 2018-02-04 10:00 = 24.0 hours
    # Transit time: 2018-02-04 10:00 to 2018-02-10 10:00 = 144.0 hours
    cursor.execute("""
    INSERT INTO orders VALUES (
        'ORD_ON_TIME', 'CUST_1', 'delivered',
        '2018-02-03 08:00:00', '2018-02-03 10:00:00',
        '2018-02-04 10:00:00', '2018-02-10 10:00:00',
        '2018-02-15 00:00:00'
    );
    """)

    # Order 2: Delivered late (delivered 2018-03-25 > estimated 2018-03-20) -> is_delayed = 1
    cursor.execute("""
    INSERT INTO orders VALUES (
        'ORD_DELAYED', 'CUST_2', 'delivered',
        '2018-03-01 09:00:00', '2018-03-01 11:00:00',
        '2018-03-03 11:00:00', '2018-03-25 15:00:00',
        '2018-03-20 00:00:00'
    );
    """)

    # Order 3: Shipped/Canceled (not delivered) -> Should be excluded strictly
    cursor.execute("""
    INSERT INTO orders VALUES (
        'ORD_CANCELED', 'CUST_3', 'canceled',
        '2018-04-01 09:00:00', '2018-04-01 11:00:00',
        NULL, NULL, '2018-04-10 00:00:00'
    );
    """)

    # Products & Translations
    cursor.execute("INSERT INTO product_category_translation VALUES ('beleza_saude', 'health_beauty');")
    cursor.execute("INSERT INTO products VALUES ('PROD_1', 'beleza_saude', 10, 50, 2, 500.0, 20.0, 10.0, 15.0);")

    # Order Items (2 items for ORD_ON_TIME, 1 item for ORD_DELAYED)
    cursor.execute("INSERT INTO order_items VALUES ('ORD_ON_TIME', 1, 'PROD_1', 'SELLER_1', '2018-02-08', 100.0, 15.0);")
    cursor.execute("INSERT INTO order_items VALUES ('ORD_ON_TIME', 2, 'PROD_1', 'SELLER_1', '2018-02-08', 50.0, 10.0);")
    cursor.execute("INSERT INTO order_items VALUES ('ORD_DELAYED', 1, 'PROD_1', 'SELLER_2', '2018-03-05', 200.0, 25.0);")

    # Customers & Sellers
    cursor.execute("INSERT INTO customers VALUES ('CUST_1', 'U1', 1001, 'Sao Paulo', 'SP');")
    cursor.execute("INSERT INTO customers VALUES ('CUST_2', 'U2', 2002, 'Rio de Janeiro', 'RJ');")
    cursor.execute("INSERT INTO sellers VALUES ('SELLER_1', 1001, 'Sao Paulo', 'SP');")
    cursor.execute("INSERT INTO sellers VALUES ('SELLER_2', 3003, 'Belo Horizonte', 'MG');")

    # Geolocation
    cursor.execute("INSERT INTO geolocation VALUES (1001, -23.5505, -46.6333, 'Sao Paulo', 'SP');")
    cursor.execute("INSERT INTO geolocation VALUES (2002, -22.9068, -43.1729, 'Rio de Janeiro', 'RJ');")
    cursor.execute("INSERT INTO geolocation VALUES (3003, -19.9167, -43.9345, 'Belo Horizonte', 'MG');")

    # Payments
    cursor.execute("INSERT INTO order_payments VALUES ('ORD_ON_TIME', 1, 'credit_card', 2, 175.0);")
    cursor.execute("INSERT INTO order_payments VALUES ('ORD_DELAYED', 1, 'boleto', 1, 225.0);")

    conn.commit()
    yield conn
    conn.close()


def test_order_analytical_base_table_query(mock_db):
    """Verifies that the ABT query filters non-delivered orders and computes metrics accurately."""
    df = pd.read_sql_query(ORDER_ANALYTICAL_BASE_TABLE_QUERY, mock_db)

    # 1. Filter Check: Must only contain the 2 delivered orders
    assert len(df) == 2, f"Expected 2 delivered orders, got {len(df)}"
    assert "ORD_CANCELED" not in df["order_id"].values

    # 2. Target Variable Verification
    on_time_row = df[df["order_id"] == "ORD_ON_TIME"].iloc[0]
    delayed_row = df[df["order_id"] == "ORD_DELAYED"].iloc[0]

    assert on_time_row["is_delayed"] == 0, "ORD_ON_TIME should have is_delayed = 0"
    assert delayed_row["is_delayed"] == 1, "ORD_DELAYED should have is_delayed = 1"

    # 3. Item Aggregation Check
    assert on_time_row["total_items"] == 2
    assert on_time_row["total_order_value"] == 150.0  # 100 + 50
    assert on_time_row["total_freight_value"] == 25.0  # 15 + 10
    assert on_time_row["freight_to_price_ratio"] == pytest.approx(25.0 / 150.0, abs=1e-4)

    # 4. Dispatch and Transit Durations Check
    # ORD_ON_TIME: Approved 2018-02-03 10:00, Carrier 2018-02-04 10:00 -> 24.0 hours
    assert on_time_row["dispatch_time_hours"] == pytest.approx(24.0, abs=0.1)
    # Transit: Carrier 2018-02-04 10:00, Delivered 2018-02-10 10:00 -> 144.0 hours
    assert on_time_row["transit_time_hours"] == pytest.approx(144.0, abs=0.1)

    # 5. Geolocation & English Category Check
    assert on_time_row["primary_category_name"] == "health_beauty"
    assert on_time_row["customer_lat"] == pytest.approx(-23.5505, abs=1e-4)
    assert on_time_row["is_cross_state_shipment"] == 0  # SP to SP
    assert delayed_row["is_cross_state_shipment"] == 1  # MG to RJ

