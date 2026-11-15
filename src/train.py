"""
Model Training, Benchmarking, Probability Calibration & Explainability Pipeline.
Performs chronological time-split, benchmarks LogReg, RF, XGBoost, LightGBM,
calibrates probabilities, optimizes operational decision threshold, and generates SHAP explainers.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.db_ingest import get_db_connection, DEFAULT_DB_PATH
from src.queries import load_analytical_base_table
from src.features import (
    engineer_features,
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
)
from src.evaluate import evaluate_predictions, optimize_classification_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("train_pipeline")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(
    numeric_features: list,
    categorical_features: list
) -> ColumnTransformer:
    """Constructs scikit-learn ColumnTransformer for numerical scaling and categorical encoding."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str = "order_purchase_timestamp",
    train_frac: float = 0.70,
    val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically into Train, Validation, and Test sets.
    Simulates production deployment where model trains on historical data and predicts future orders.
    """
    sorted_df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()

    logger.info(f"Chronological Split Summary:")
    logger.info(f"  Train:      {len(train_df):,} orders | [{train_df[timestamp_col].min()} to {train_df[timestamp_col].max()}] | Delay Rate: {train_df['is_delayed'].mean():.2%}")
    logger.info(f"  Validation: {len(val_df):,} orders | [{val_df[timestamp_col].min()} to {val_df[timestamp_col].max()}] | Delay Rate: {val_df['is_delayed'].mean():.2%}")
    logger.info(f"  Test:       {len(test_df):,} orders | [{test_df[timestamp_col].min()} to {test_df[timestamp_col].max()}] | Delay Rate: {test_df['is_delayed'].mean():.2%}")

    return train_df, val_df, test_df


def benchmark_candidate_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    preprocessor: ColumnTransformer
) -> Tuple[str, Any, pd.DataFrame]:
    """Trains and benchmarks Logistic Regression, Random Forest, XGBoost, and LightGBM."""
    logger.info("Transforming feature matrices with ColumnTransformer...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # Calculate class imbalance ratio
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = float(neg_count / max(1, pos_count))

    candidate_models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            num_leaves=31,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
    }

    benchmark_records = []
    trained_models = {}

    for name, model in candidate_models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_proc, y_train)
        val_probs = model.predict_proba(X_val_proc)[:, 1]
        metrics = evaluate_predictions(y_val, val_probs, threshold=0.50)
        metrics["model_name"] = name
        benchmark_records.append(metrics)
        trained_models[name] = model

    benchmark_df = pd.DataFrame(benchmark_records).sort_values(by="pr_auc", ascending=False)
    logger.info("\n--- Model Benchmark Leaderboard (Validation Set) ---\n" + benchmark_df.to_string(index=False))

    best_model_name = benchmark_df.iloc[0]["model_name"]
    best_model = trained_models[best_model_name]
    logger.info(f"Winning Candidate Model: {best_model_name} (PR-AUC: {benchmark_df.iloc[0]['pr_auc']:.4f})")

    return best_model_name, best_model, benchmark_df


def train_and_evaluate_pipeline(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Runs the full Phase 3 machine learning, calibration, explainability, and export flow."""
    conn = get_db_connection(db_path)
    try:
        raw_abt_df = load_analytical_base_table(conn)
    finally:
        conn.close()

    if len(raw_abt_df) == 0:
        raise ValueError("Analytical Base Table is empty. Please run Phase 1 database ingestion first.")

    logger.info(f"Loaded {len(raw_abt_df):,} delivered order records from SQLite DW.")

    # 1. Chronological Train/Val/Test Split
    train_df, val_df, test_df = chronological_split(raw_abt_df)

    # 2. Leak-Free Feature Engineering
    logger.info("Computing leak-free training features and seller Bayesian prior lookup...")
    train_feat, seller_summary = engineer_features(train_df)
    global_prior = float(train_df["is_delayed"].mean())

    val_feat, _ = engineer_features(
        val_df,
        seller_summary_lookup=seller_summary,
        global_delay_prior=global_prior
    )
    test_feat, _ = engineer_features(
        test_df,
        seller_summary_lookup=seller_summary,
        global_delay_prior=global_prior
    )

    # Identify numeric and categorical columns
    categorical_cols = ["customer_state", "seller_state", "primary_category_name", "dominant_payment_type"]
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in categorical_cols]

    X_train = train_feat[FEATURE_COLUMNS]
    y_train = train_feat["is_delayed"].values

    X_val = val_feat[FEATURE_COLUMNS]
    y_val = val_feat["is_delayed"].values

    X_test = test_feat[FEATURE_COLUMNS]
    y_test = test_feat["is_delayed"].values

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # 3. Benchmark Models
    best_name, best_estimator, benchmark_df = benchmark_candidate_models(
        X_train, y_train, X_val, y_val, preprocessor
    )

    # 4. Probability Calibration (Isotonic / Sigmoid on Validation Set)
    logger.info("Calibrating winning model probabilities on validation set...")
    X_val_proc = preprocessor.transform(X_val)
    
    # Modern scikit-learn (1.4+) uses FrozenEstimator for prefit models
    try:
        from sklearn.frozen import FrozenEstimator
        calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(best_estimator), method="sigmoid")
    except (ImportError, TypeError):
        try:
            calibrator = CalibratedClassifierCV(estimator=best_estimator, method="sigmoid", cv="prefit")
        except Exception:
            calibrator = CalibratedClassifierCV(estimator=best_estimator, method="sigmoid", cv=None)

    calibrator.fit(X_val_proc, y_val)

    # Evaluate uncalibrated vs calibrated Brier score on validation set
    raw_val_probs = best_estimator.predict_proba(X_val_proc)[:, 1]
    cal_val_probs = calibrator.predict_proba(X_val_proc)[:, 1]

    uncal_brier = evaluate_predictions(y_val, raw_val_probs)["brier_score"]
    cal_brier = evaluate_predictions(y_val, cal_val_probs)["brier_score"]
    logger.info(f"Brier Score Improvement: {uncal_brier:.4f} (Uncalibrated) -> {cal_brier:.4f} (Calibrated)")

    # 5. Optimal Operational Threshold Tuning
    logger.info("Optimizing operational decision threshold for supply chain delay intervention...")
    best_threshold, sweep_df, optimal_val_metrics = optimize_classification_threshold(
        y_val, cal_val_probs, min_recall=0.70
    )
    logger.info(f"Optimal Threshold Selected: {best_threshold:.3f} | Recall: {optimal_val_metrics['recall']:.2%} | F1: {optimal_val_metrics['f1_score']:.4f}")

    # 6. Unseen Test Set Final Evaluation
    logger.info("Evaluating calibrated pipeline on unseen future Test Set...")
    X_test_proc = preprocessor.transform(X_test)
    test_cal_probs = calibrator.predict_proba(X_test_proc)[:, 1]

    default_test_metrics = evaluate_predictions(y_test, test_cal_probs, threshold=0.50)
    tuned_test_metrics = evaluate_predictions(y_test, test_cal_probs, threshold=best_threshold)

    logger.info(f"Test Set Performance (Default 0.50 Threshold): ROC-AUC={default_test_metrics['roc_auc']:.4f}, PR-AUC={default_test_metrics['pr_auc']:.4f}, Recall={default_test_metrics['recall']:.2%}")
    logger.info(f"Test Set Performance (Tuned {best_threshold:.3f} Threshold): Recall={tuned_test_metrics['recall']:.2%}, F1={tuned_test_metrics['f1_score']:.4f}")

    # 7. SHAP Global & Local Explainability
    logger.info("Computing SHAP TreeExplainer for feature attribution...")
    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_names

    # If best model is tree-based, compute SHAP
    shap_summary = {}
    explainer = None
    if best_name in ["LightGBM", "XGBoost", "RandomForest"]:
        try:
            explainer = shap.TreeExplainer(best_estimator)
            # Sample up to 500 rows for fast SHAP computation
            sample_size = min(500, len(X_test_proc))
            sample_proc = X_test_proc[:sample_size]
            shap_vals = explainer.shap_values(sample_proc)
            
            # Handle binary class shap shape (some shap versions return list of 2 arrays, some return 2D or 3D)
            if isinstance(shap_vals, list):
                shap_vals_matrix = shap_vals[1]
            elif hasattr(shap_vals, "shape") and len(shap_vals.shape) == 3:
                shap_vals_matrix = shap_vals[:, :, 1]
            else:
                shap_vals_matrix = shap_vals

            mean_abs_shap = np.mean(np.abs(shap_vals_matrix), axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:15]
            shap_summary = {
                all_feature_names[i]: round(float(mean_abs_shap[i]), 4)
                for i in top_indices
            }
            logger.info(f"Top 5 Global Delay Drivers (SHAP): {list(shap_summary.items())[:5]}")
        except Exception as e:
            logger.warning(f"SHAP computation note: {e}")

    # 8. Enrich Test Set with Predictions for Control Tower High-Risk Queue (Tab 2)
    test_export_df = test_feat.copy()
    test_export_df["delay_probability"] = np.round(test_cal_probs, 4)
    test_export_df["predicted_delayed"] = (test_cal_probs >= best_threshold).astype(int)
    test_export_df["risk_tier"] = pd.cut(
        test_export_df["delay_probability"],
        bins=[0.0, 0.20, 0.50, 0.75, 1.0],
        labels=["Low", "Medium", "High", "Critical"]
    )
    test_export_path = MODELS_DIR / "test_predictions.parquet"
    test_export_df.to_parquet(test_export_path, index=False)
    test_export_df.to_csv(MODELS_DIR / "test_predictions.csv", index=False)
    logger.info(f"Saved test set predictions to {test_export_path}")

    # 9. Save Production Artifacts
    full_pipeline = {
        "preprocessor": preprocessor,
        "best_estimator": best_estimator,
        "calibrator": calibrator,
        "best_model_name": best_name,
        "best_threshold": best_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "all_transformed_feature_names": all_feature_names,
        "seller_summary": seller_summary,
        "global_prior": global_prior,
    }
    joblib.dump(full_pipeline, MODELS_DIR / "calibrated_pipeline.joblib")

    # Save metadata JSON for dashboard display
    metadata = {
        "best_model_name": best_name,
        "best_threshold": best_threshold,
        "benchmark_leaderboard": benchmark_df.to_dict(orient="records"),
        "test_metrics_default": default_test_metrics,
        "test_metrics_tuned": tuned_test_metrics,
        "calibration_brier_improvement": {
            "uncalibrated": round(uncal_brier, 4),
            "calibrated": round(cal_brier, 4),
        },
        "top_shap_features": shap_summary,
        "dataset_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        }
    }
    with open(MODELS_DIR / "model_card.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Pipeline artifacts successfully saved to 'models/' directory.")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_pipeline()
