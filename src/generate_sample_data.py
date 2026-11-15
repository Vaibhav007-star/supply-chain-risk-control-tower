"""
Realistic Sample Data Generator for Olist Brazilian E-Commerce Dataset.
Generates realistic CSVs conforming to the exact Kaggle Olist schema,
allowing immediate end-to-end pipeline execution, model training, and app testing.
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Brazilian state coordinates & city anchors
BRAZIL_REGIONS = [
    {"state": "SP", "city": "sao paulo", "lat": -23.5505, "lng": -46.6333, "base_zip": 1000},
    {"state": "RJ", "city": "rio de janeiro", "lat": -22.9068, "lng": -43.1729, "base_zip": 20000},
    {"state": "MG", "city": "belo horizonte", "lat": -19.9167, "lng": -43.9345, "base_zip": 30000},
    {"state": "RS", "city": "porto alegre", "lat": -30.0346, "lng": -51.2177, "base_zip": 90000},
    {"state": "PR", "city": "curitiba", "lat": -25.4284, "lng": -49.2733, "base_zip": 80000},
    {"state": "BA", "city": "salvador", "lat": -12.9714, "lng": -38.5014, "base_zip": 40000},
    {"state": "PE", "city": "recife", "lat": -8.0476, "lng": -34.8770, "base_zip": 50000},
    {"state": "CE", "city": "fortaleza", "lat": -3.7172, "lng": -38.5433, "base_zip": 60000},
    {"state": "AM", "city": "manaus", "lat": -3.1190, "lng": -60.0217, "base_zip": 69000},
    {"state": "DF", "city": "brasilia", "lat": -15.7975, "lng": -47.8919, "base_zip": 70000},
]

CATEGORIES = [
    ("beleza_saude", "health_beauty", 450, 20, 15, 10),
    ("relogios_presentes", "watches_gifts", 350, 15, 10, 8),
    ("cama_mesa_banho", "bed_bath_table", 1800, 40, 30, 25),
    ("esporte_lazer", "sports_leisure", 1200, 35, 25, 20),
    ("informatica_acessorios", "computers_accessories", 850, 25, 20, 15),
    ("moveis_decoracao", "furniture_decor", 4500, 60, 50, 40),
    ("utilidades_domesticas", "housewares", 1400, 30, 25, 20),
    ("telefonia", "telephony", 300, 18, 12, 5),
    ("automotivo", "auto", 2200, 45, 30, 25),
    ("brinquedos", "toys", 700, 25, 20, 18),
]

PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]


def generate_synthetic_olist_data(n_orders: int = 5000, seed: int = 42):
    """Generates synthetic dataset mirroring the Olist Brazilian E-Commerce data."""
    np.random.seed(seed)
    random.seed(seed)

    print(f"Generating realistic synthetic Olist dataset ({n_orders:,} orders)...")

    # 1. Product Category Translations
    cat_df = pd.DataFrame([
        {"product_category_name": c[0], "product_category_name_english": c[1]}
        for c in CATEGORIES
    ])
    cat_df.to_csv(RAW_DIR / "product_category_name_translation.csv", index=False)

    # 2. Products
    products = []
    for i in range(150):
        prod_id = f"prod_{i:04d}"
        c = random.choice(CATEGORIES)
        weight_jitter = max(100, int(c[2] * np.random.uniform(0.7, 1.4)))
        l = max(10, int(c[3] * np.random.uniform(0.8, 1.3)))
        h = max(5, int(c[4] * np.random.uniform(0.8, 1.3)))
        w = max(5, int(c[5] * np.random.uniform(0.8, 1.3)))
        products.append({
            "product_id": prod_id,
            "product_category_name": c[0],
            "product_name_lenght": random.randint(30, 60),
            "product_description_lenght": random.randint(150, 800),
            "product_photos_qty": random.randint(1, 6),
            "product_weight_g": weight_jitter,
            "product_length_cm": l,
            "product_height_cm": h,
            "product_width_cm": w,
        })
    products_df = pd.DataFrame(products)
    products_df.to_csv(RAW_DIR / "olist_products_dataset.csv", index=False)

    # 3. Sellers
    sellers = []
    for i in range(80):
        seller_id = f"seller_{i:04d}"
        reg = random.choice(BRAZIL_REGIONS)
        zip_prefix = reg["base_zip"] + random.randint(1, 999)
        sellers.append({
            "seller_id": seller_id,
            "seller_zip_code_prefix": zip_prefix,
            "seller_city": reg["city"],
            "seller_state": reg["state"],
            "seller_lat": reg["lat"] + np.random.normal(0, 0.05),
            "seller_lng": reg["lng"] + np.random.normal(0, 0.05),
        })
    sellers_df = pd.DataFrame(sellers)
    sellers_df[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]].to_csv(
        RAW_DIR / "olist_sellers_dataset.csv", index=False
    )

    # 4. Customers
    customers = []
    for i in range(n_orders):
        cust_id = f"cust_{i:06d}"
        reg = random.choice(BRAZIL_REGIONS)
        zip_prefix = reg["base_zip"] + random.randint(1, 999)
        customers.append({
            "customer_id": cust_id,
            "customer_unique_id": f"uniq_{cust_id}",
            "customer_zip_code_prefix": zip_prefix,
            "customer_city": reg["city"],
            "customer_state": reg["state"],
            "customer_lat": reg["lat"] + np.random.normal(0, 0.08),
            "customer_lng": reg["lng"] + np.random.normal(0, 0.08),
        })
    customers_df = pd.DataFrame(customers)
    customers_df[["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"]].to_csv(
        RAW_DIR / "olist_customers_dataset.csv", index=False
    )

    # 5. Geolocation Raw Table
    geo_records = []
    for s in sellers:
        geo_records.append({
            "geolocation_zip_code_prefix": s["seller_zip_code_prefix"],
            "geolocation_lat": s["seller_lat"],
            "geolocation_lng": s["seller_lng"],
            "geolocation_city": s["seller_city"],
            "geolocation_state": s["seller_state"],
        })
    for c in customers:
        geo_records.append({
            "geolocation_zip_code_prefix": c["customer_zip_code_prefix"],
            "geolocation_lat": c["customer_lat"],
            "geolocation_lng": c["customer_lng"],
            "geolocation_city": c["customer_city"],
            "geolocation_state": c["customer_state"],
        })
    geo_df = pd.DataFrame(geo_records)
    geo_df.to_csv(RAW_DIR / "olist_geolocation_dataset.csv", index=False)

    # 6. Orders, Items, Payments
    start_date = datetime(2017, 1, 1, 8, 0, 0)
    orders = []
    order_items = []
    payments = []
    reviews = []

    # Assign each seller a latent delay tendency (some sellers are high risk)
    seller_risk = {s["seller_id"]: np.clip(np.random.beta(1.5, 10), 0.02, 0.45) for s in sellers}

    for i in range(n_orders):
        order_id = f"order_{i:06d}"
        cust = customers[i]
        
        # Chronological order progression from Jan 2017 to August 2018
        purchase_time = start_date + timedelta(
            days=int((i / n_orders) * 580),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        approved_time = purchase_time + timedelta(hours=random.randint(1, 24))

        # Select primary seller and products
        seller = random.choice(sellers)
        n_items = 1 if np.random.rand() > 0.15 else random.randint(2, 4)
        order_val = 0.0
        freight_val = 0.0

        for item_idx in range(1, n_items + 1):
            prod = random.choice(products)
            price = round(float(np.random.gamma(shape=3.0, scale=35.0)), 2)
            price = max(15.0, price)
            is_cross = 1 if cust["customer_state"] != seller["seller_state"] else 0
            freight = round(12.0 + (is_cross * 18.0) + (prod["product_weight_g"] / 1000.0 * 2.5), 2)
            
            order_val += price
            freight_val += freight

            order_items.append({
                "order_id": order_id,
                "order_item_id": item_idx,
                "product_id": prod["product_id"],
                "seller_id": seller["seller_id"],
                "shipping_limit_date": (approved_time + timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "price": price,
                "freight_value": freight,
            })

        # Promised SLA buffer (typically 15-30 days in Brazil e-commerce)
        estimated_delivery_days = random.randint(14, 30)
        estimated_delivery = approved_time + timedelta(days=estimated_delivery_days)

        # Warehouse dispatch lag (approval to carrier handoff)
        # Slower dispatch for risky sellers or multi-item orders
        base_dispatch_hours = random.randint(8, 48)
        if seller_risk[seller["seller_id"]] > 0.20:
            base_dispatch_hours += random.randint(24, 96)
        carrier_handoff = approved_time + timedelta(hours=base_dispatch_hours)

        # Carrier transit time
        is_cross_state = cust["customer_state"] != seller["seller_state"]
        base_transit_days = random.randint(2, 7) if not is_cross_state else random.randint(6, 18)
        
        # Chance of delay depends on: cross-state, seller risk, warehouse dispatch delay, remote states
        remote_state = cust["customer_state"] in ["AM", "CE", "PE", "BA"]
        p_delay = seller_risk[seller["seller_id"]]
        if is_cross_state:
            p_delay += 0.08
        if remote_state:
            p_delay += 0.12
        if base_dispatch_hours > 72:
            p_delay += 0.18
        p_delay = min(0.85, p_delay)

        is_delay = np.random.rand() < p_delay
        if is_delay:
            transit_days = estimated_delivery_days + random.randint(1, 10) - int(base_dispatch_hours / 24)
        else:
            transit_days = max(1, estimated_delivery_days - random.randint(2, 9) - int(base_dispatch_hours / 24))

        actual_delivery = carrier_handoff + timedelta(days=transit_days)

        orders.append({
            "order_id": order_id,
            "customer_id": cust["customer_id"],
            "order_status": "delivered",
            "order_purchase_timestamp": purchase_time.strftime("%Y-%m-%d %H:%M:%S"),
            "order_approved_at": approved_time.strftime("%Y-%m-%d %H:%M:%S"),
            "order_delivered_carrier_date": carrier_handoff.strftime("%Y-%m-%d %H:%M:%S"),
            "order_delivered_customer_date": actual_delivery.strftime("%Y-%m-%d %H:%M:%S"),
            "order_estimated_delivery_date": estimated_delivery.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Payment
        pay_type = random.choice(PAYMENT_TYPES)
        payments.append({
            "order_id": order_id,
            "payment_sequential": 1,
            "payment_type": pay_type,
            "payment_installments": random.randint(1, 6) if pay_type == "credit_card" else 1,
            "payment_value": round(order_val + freight_val, 2),
        })

        # Review (for completeness in schema; excluded from features)
        score = 1 if is_delay else random.choice([4, 5, 5, 4, 3])
        reviews.append({
            "review_id": f"rev_{i:06d}",
            "order_id": order_id,
            "review_score": score,
            "review_comment_title": None,
            "review_comment_message": "Entrega demorou" if is_delay else "Excelente produto",
            "review_creation_date": (actual_delivery + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "review_answer_timestamp": (actual_delivery + timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
        })

    pd.DataFrame(orders).to_csv(RAW_DIR / "olist_orders_dataset.csv", index=False)
    pd.DataFrame(order_items).to_csv(RAW_DIR / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame(payments).to_csv(RAW_DIR / "olist_order_payments_dataset.csv", index=False)
    pd.DataFrame(reviews).to_csv(RAW_DIR / "olist_order_reviews_dataset.csv", index=False)

    print(f"Sample dataset generated successfully in '{RAW_DIR}'.")


if __name__ == "__main__":
    generate_synthetic_olist_data()

