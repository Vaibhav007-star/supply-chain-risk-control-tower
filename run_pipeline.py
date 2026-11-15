"""
Master Execution Pipeline Script.
Runs the entire end-to-end workflow:
1. Validates / Generates raw data files
2. Executes SQLite database ingestion & indexing
3. Runs leak-free feature engineering, time-split, model benchmarking & calibration
4. Executes automated test suite
5. Provides command to launch Streamlit Control Tower
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("master_runner")

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "olist.db"


def run_command_checked(cmd: list, desc: str):
    logger.info(f"==> Step: {desc}")
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error(f"Error during '{desc}'. Exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    logger.info("=====================================================================")
    logger.info("  E-Commerce Delivery Delay & Supply Chain Risk Predictor Pipeline  ")
    logger.info("=====================================================================")

    # Step 1: Check if raw data exists
    has_orders = (RAW_DIR / "olist_orders_dataset.csv").exists()
    if not has_orders:
        logger.info("Raw Olist data not found in data/raw. Generating synthetic sample dataset...")
        run_command_checked([sys.executable, "src/generate_sample_data.py"], "Generate Synthetic Sample Data")
    else:
        logger.info(f"Raw Olist dataset detected in '{RAW_DIR}'.")

    # Step 2: Database Ingestion
    run_command_checked([sys.executable, "src/db_ingest.py"], "SQLite Data Warehouse Ingestion & Indexing")

    # Step 3: Model Training, Calibration & Explainability
    run_command_checked([sys.executable, "-m", "src.train"], "Model Benchmarking, Calibration & SHAP")

    # Step 4: Run Test Suite
    run_command_checked([sys.executable, "-m", "pytest", "tests/", "-v"], "Automated Pytest Suite")

    logger.info("=====================================================================")
    logger.info("  All Pipeline Phases Completed Successfully!                       ")
    logger.info("=====================================================================")
    logger.info("To launch the Streamlit Operations Control Tower, execute:")
    logger.info("    streamlit run app.py")


if __name__ == "__main__":
    main()
