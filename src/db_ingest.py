"""
Database Ingestion Module for Olist Brazilian E-Commerce Dataset.
Loads raw CSVs into an optimized SQLite database with indexes and clean schemas.
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("db_ingest")

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_DB_PATH = Path("data/processed/olist.db")

# Mapping of file names (with or without prefix) to table names
DATASET_FILES: Dict[str, str] = {
    "olist_orders_dataset.csv": "orders",
    "olist_order_items_dataset.csv": "order_items",
    "olist_products_dataset.csv": "products",
    "olist_customers_dataset.csv": "customers",
    "olist_sellers_dataset.csv": "sellers",
    "olist_geolocation_dataset.csv": "geolocation_raw",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "product_category_name_translation.csv": "product_category_translation",
}

# Timestamp columns per table for explicit datetime parsing
DATETIME_COLS: Dict[str, list] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}


def get_db_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates directory if needed and returns SQLite connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    # Enable WAL mode and synchronous normal for high performance
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ingest_csv_to_sqlite(
    csv_path: Path,
    table_name: str,
    conn: sqlite3.Connection,
    chunksize: int = 50_000
) -> int:
    """Reads a CSV in chunks and writes to SQLite table."""
    logger.info(f"Ingesting '{csv_path.name}' into table '{table_name}'...")
    parse_dates = DATETIME_COLS.get(table_name, False)

    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(csv_path, parse_dates=parse_dates, chunksize=chunksize, low_memory=False):
        # Format datetimes as ISO strings for SQLite compatibility
        if parse_dates and isinstance(parse_dates, list):
            for col in parse_dates:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str).replace({"NaT": None, "nan": None})

        if_exists = "replace" if first_chunk else "append"
        chunk.to_sql(table_name, conn, if_exists=if_exists, index=False)
        total_rows += len(chunk)
        first_chunk = False

    logger.info(f"Successfully loaded {total_rows:,} rows into '{table_name}'.")
    return total_rows


def build_deduplicated_geolocation_table(conn: sqlite3.Connection) -> None:
    """
    The raw Olist geolocation file contains over 1,000,000 rows with multiple
    latitude/longitude coordinates per 5-digit zip code prefix.
    Joining on raw zip code creates a Cartesian product.
    This creates an aggregated `geolocation` table with median coordinates.
    """
    logger.info("Building deduplicated, median-aggregated 'geolocation' table...")
    create_geo_sql = """
    DROP TABLE IF EXISTS geolocation;
    CREATE TABLE geolocation AS
    SELECT
        geolocation_zip_code_prefix AS zip_code_prefix,
        ROUND(AVG(geolocation_lat), 6) AS geolocation_lat,
        ROUND(AVG(geolocation_lng), 6) AS geolocation_lng,
        geolocation_city,
        geolocation_state
    FROM (
        SELECT
            geolocation_zip_code_prefix,
            geolocation_lat,
            geolocation_lng,
            geolocation_city,
            geolocation_state,
            ROW_NUMBER() OVER (
                PARTITION BY geolocation_zip_code_prefix 
                ORDER BY geolocation_city DESC
            ) as rn
        FROM geolocation_raw
    )
    GROUP BY geolocation_zip_code_prefix;
    """
    conn.executescript(create_geo_sql)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM geolocation;")
    count = cursor.fetchone()[0]
    logger.info(f"Created clean 'geolocation' table with {count:,} unique zip codes.")


def create_indexes(conn: sqlite3.Connection) -> None:
    """Creates indexes on primary and foreign keys for high-performance joins."""
    logger.info("Creating relational database indexes...")
    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(order_status);",
        "CREATE INDEX IF NOT EXISTS idx_orders_purchase_ts ON orders(order_purchase_timestamp);",
        
        "CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_seller_id ON order_items(seller_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);",
        
        "CREATE INDEX IF NOT EXISTS idx_products_product_id ON products(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_customers_zip ON customers(customer_zip_code_prefix);",
        "CREATE INDEX IF NOT EXISTS idx_sellers_seller_id ON sellers(seller_id);",
        "CREATE INDEX IF NOT EXISTS idx_sellers_zip ON sellers(seller_zip_code_prefix);",
        "CREATE INDEX IF NOT EXISTS idx_geo_zip ON geolocation(zip_code_prefix);",
        "CREATE INDEX IF NOT EXISTS idx_payments_order_id ON order_payments(order_id);"
    ]
    for stmt in index_statements:
        conn.execute(stmt)
    conn.commit()
    logger.info("Indexes successfully created.")


def run_ingestion_pipeline(
    raw_dir: Path = DEFAULT_RAW_DIR,
    db_path: Path = DEFAULT_DB_PATH
) -> None:
    """Executes the full ingestion pipeline from raw directory to SQLite DB."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory '{raw_dir}' does not exist.")

    conn = get_db_connection(db_path)
    try:
        found_files = 0
        for filename, table_name in DATASET_FILES.items():
            file_path = raw_dir / filename
            if file_path.exists():
                ingest_csv_to_sqlite(file_path, table_name, conn)
                found_files += 1
            else:
                logger.warning(f"File not found: {file_path}. Skipping.")

        if found_files == 0:
            logger.error(f"No Olist CSV files found in {raw_dir}. Place raw files there first.")
            return

        # Build clean geolocation table if raw was ingested
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='geolocation_raw';")
        if cursor.fetchone():
            build_deduplicated_geolocation_table(conn)

        create_indexes(conn)

        # Print summary
        logger.info("--- Data Warehouse Table Summary ---")
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()
        for (tbl,) in tables:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {tbl};").fetchone()[0]
            logger.info(f"Table: {tbl:<30} | Rows: {cnt:,}")

    finally:
        conn.close()
        logger.info(f"Database ready at: {db_path.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Olist CSVs into SQLite Data Warehouse")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Path to raw CSV directory")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Path to output SQLite database")
    args = parser.parse_args()

    run_ingestion_pipeline(args.raw_dir, args.db_path)

