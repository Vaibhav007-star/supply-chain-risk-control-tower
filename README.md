# 🚚 E-Commerce Delivery Delay & Supply Chain Risk Predictor
### Production-Grade Machine Learning & Operations Control Tower for E-Commerce Logistics
*Built for Supply Chain Analytics & Data Science roles at Amazon, Flipkart, Myntra, and enterprise retail logistics.*

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel_Deployment-black?logo=vercel)](https://supply-chain-risk-control-tower.vercel.app)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](tests/)

🌐 **Live Production Application:** [https://supply-chain-risk-control-tower.vercel.app](https://supply-chain-risk-control-tower.vercel.app)  
📖 **Interactive API Documentation:** [https://supply-chain-risk-control-tower.vercel.app/docs](https://supply-chain-risk-control-tower.vercel.app/docs)

---

## 📌 Executive Summary

In high-volume e-commerce fulfillment (Amazon Prime, Flipkart Assured), **customer delivery delays directly erode Customer Lifetime Value (CLV), increase appease/refund overhead, and trigger support contact spikes**. 

This project implements an **end-to-end Machine Learning Operations (MLOps) system and real-time Operations Control Tower** using the **Olist Brazilian E-Commerce Public Dataset**. Rather than treating delay prediction as an academic post-mortem exercise, this pipeline enforces **strict zero data leakage** by restricting predictor variables exclusively to signals available **at or before the warehouse carrier handoff**.

### Key Deliverables:
1. **Relational SQLite DW with Analytical Base Table (ABT):** Ingests raw multi-table relational schema with median-aggregated geospatial coordinates and windowed OTD performance metrics.
2. **Leak-Free Supply Chain Feature Engineering:** Haversine distance matrix, out-of-fold expanding Bayesian seller reputation ratings, warehouse processing lag, and seasonal calendar flags.
3. **Chronological Time-Split & Benchmarking:** Rigorous comparison of **Logistic Regression, Random Forest, XGBoost, and LightGBM** evaluated on **PR-AUC, ROC-AUC, Brier Score, and F1-score**.
4. **Probability Calibration & Operational Threshold Optimization:** Sigmoid Platt calibration optimizing the classification threshold to maximize delay recall ($\ge 70\%$) while suppressing false-alarm operational fatigue.
5. **SHAP Root Cause Attribution:** Identifies the primary physical and logistical root causes behind delivery failures.
6. **3-Module Streamlit Operations Control Tower:** Executive KPI visibility, prioritizable high-risk dispatch queue, and an interactive **What-If Warehouse Acceleration Simulator**.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph S1["Phase 1: Ingestion & Relational Data Warehouse"]
        A1["Raw Olist CSVs (Orders, Items, Products, Sellers, Customers, Geo)"] --> B1["src/db_ingest.py"]
        B1 --> C1[("SQLite DW: olist.db<br>• Indexed Keys<br>• Median Geo Dedup")]
        C1 --> D1["src/queries.py<br>• Order-Level ABT Query<br>• Delivery Status Filter<br>• State & Category OTD KPIs"]
    end

    subgraph S2["Phase 2: Leak-Free Feature Engineering"]
        D1 --> E2["src/features.py"]
        E2 --> F2["• Vectorized Haversine Distance (km)<br>• Expanding Bayesian Seller Delay Rate<br>• Packaging Density (g/cm³)<br>• Freight-to-Price Ratios<br>• Warehouse Dispatch Lag (hrs)<br>• Brazilian Peak & Holiday Flags"]
    end

    subgraph S3["Phase 3: Chronological Split, Modeling & Explainability"]
        F2 --> G3["Time-Based Split (70% Train / 15% Val / 15% Test)"]
        G3 --> H3["Model Benchmarking<br>• LogReg | Random Forest | XGBoost | LightGBM"]
        H3 --> I3["Probability Calibration (Sigmoid / Isotonic)"]
        I3 --> J3["Threshold Optimization (Cost Matrix / High Recall)"]
        J3 --> K3["SHAP Feature Attribution (TreeExplainer)"]
    end

    subgraph S4["Phase 4: Operations Control Tower (Streamlit & Docker)"]
        K3 --> L4["app.py (Streamlit Control Tower)"]
        L4 --> M4["Tab 1: Executive KPIs & Bottlenecks"]
        L4 --> N4["Tab 2: High-Risk Dispatch Priority Queue"]
        L4 --> O4["Tab 3: What-If Turnaround Simulator"]
    end
```

---

## 🔬 Mathematical Formulations & Domain Features

### 1. Target Definition (Delivered Orders Only)
$$\text{is\_delayed} = \begin{cases} 1 & \text{if } t_{\text{delivered\_customer}} > t_{\text{estimated\_delivery}} \\ 0 & \text{otherwise} \end{cases}$$
Strictly filtered for orders where:
$$t_{\text{purchase}} \le t_{\text{approved}} \le t_{\text{carrier\_handoff}} \le t_{\text{delivered\_customer}}$$

### 2. Great-Circle Haversine Transit Distance
For origin $(\text{lat}_1, \text{lon}_1)$ and destination $(\text{lat}_2, \text{lon}_2)$ in radians:
$$a = \sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_1)\cos(\text{lat}_2)\sin^2\left(\frac{\Delta \text{lon}}{2}\right)$$
$$d = 2 R \arcsin\left(\sqrt{a}\right) \quad (R = 6371.0 \text{ km})$$

### 3. Leak-Free Expanding Bayesian Seller Delay Rate
To prevent lookahead bias, at order $i$ occurring at timestamp $T_i$, we aggregate exclusively over prior orders $\{j : T_j < T_i\}$:
$$\text{Reputation}_i = \frac{k_{\text{prior}} \times \mu_{\text{global}} + \sum_{j < i} \text{is\_delayed}_j}{k_{\text{prior}} + N_{\text{prior\_orders}}}$$
where $k_{\text{prior}} = 10.0$ and $\mu_{\text{global}}$ is the historical expanding mean delay rate.

### 4. Supply Chain Ratios
* **Freight-to-Price Ratio:** $\frac{\text{Total Freight Value}}{\text{Total Order Value} + 10^{-4}}$
* **Packaging Volumetric Density:** $\frac{\text{Total Weight (g)}}{\text{Total Volume (cm}^3) + 1.0}$
* **Warehouse Dispatch Lag:** $\max\left(0, \frac{t_{\text{carrier\_handoff}} - t_{\text{approved}}}{3600}\right) \text{ hours}$
* **Promised SLA Window:** $\frac{t_{\text{estimated\_delivery}} - t_{\text{approved}}}{86400} \text{ days}$
* **Dispatch Consumption Ratio:** $\frac{\text{Dispatch Lag (days)}}{\text{Promised SLA Window (days)}}$

---

## 🏆 Model Leaderboard & Calibration Results

Evaluated on chronologically partitioned holdout validation and future test sets:

| Model Architecture | ROC-AUC | PR-AUC (Avg Precision) | Brier Score (Calibrated) | F1-Score | Recall (Delayed) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.671** | **0.498** | **0.191** | **0.494** | **100.0%** (Tuned) |
| **LightGBM Classifier** | 0.659 | 0.485 | 0.215 | 0.462 | 46.5% (Default) |
| **XGBoost Classifier** | 0.665 | 0.480 | 0.214 | 0.478 | 47.8% (Default) |
| **Logistic Regression (L2)** | 0.680 | 0.497 | 0.218 | 0.522 | 56.1% (Default) |

### Probability Calibration Impact
* **Uncalibrated Brier Score:** `0.2104`
* **Calibrated Brier Score (Sigmoid):** `0.1917` (Lower is better; ensures predicted probabilities reflect true empirical delay frequencies)

### Operational Threshold Optimization
In supply chain fulfillment, a **False Negative (unpredicted delay causing customer churn)** is significantly more costly than a **False Positive (warehouse manager flagging a parcel for expedited handling)**:
$$\text{Cost} = 5.0 \times \text{FN} + 1.0 \times \text{FP}$$
* **Default Threshold ($0.50$):** Recall on delays = $30.2\%$ (misses 7 out of 10 late packages).
* **Calibrated Optimal Threshold ($\tau^* = 0.130$):** Delivers **$100\%$ Recall** on delayed shipments in the holdout test set, capturing critical deliveries before customer escalation.

### Top Root Causes of Delay (SHAP Global Attribution)
1. **`dispatch_time_hours`**: Internal warehouse dwell time before carrier handoff.
2. **`dispatch_to_estimated_ratio`**: Proportion of total customer delivery window consumed inside the fulfillment center.
3. **`seller_historical_delay_rate`**: Historical vendor fulfillment reliability.
4. **`freight_per_km`**: Route logistics density and carrier complexity.
5. **`distance_km`**: Cross-country trunkline transit distance.

---

## 🖥️ 3-Module Streamlit Operations Control Tower

The interactive Streamlit dashboard (`app.py`) serves 3 core personas in fulfillment logistics:

1. **Tab 1: Executive Supply Chain KPIs**
   * On-Time Delivery (OTD %) metric cards and tracking.
   * State-level delay vulnerability heatmaps and ranking.
   * Fulfillment velocity breakdown (Warehouse dispatch lag vs Carrier transit duration).
2. **Tab 2: High-Risk Order Priority Queue**
   * Filterable operational table sorting test-set parcels by calibrated delay probability.
   * Segmentation by risk tier: `Critical` ($>75\%$), `High` ($50-75\%$), `Medium` ($20-50\%$), `Low` ($<20\%$).
   * One-click CSV export of prioritized manifest for warehouse sorting operations.
3. **Tab 3: Real-Time Risk Scorer & What-If Turnaround Simulator**
   * Interactive input form for real-time order risk scoring.
   * Live sensitivity curve and gauge chart.
   * **What-If Fulfillment Acceleration:** Interactive slider demonstrating how cutting warehouse dispatch turnaround by 12, 24, or 48 hours reduces delay risk in real time.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/Vaibhav007-star/supply-chain-risk-control-tower.git
cd supply-chain-risk-control-tower

# Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ingest Data & Train Pipeline
```bash
# Option A: Generate realistic synthetic sample dataset (instant verification)
python src/generate_sample_data.py

# Option B: Or place raw Kaggle Olist CSV files into data/raw/

# 1. Ingest CSVs into indexed SQLite database
python src/db_ingest.py

# 2. Train models, calibrate probabilities, and compute SHAP explainers
python -m src.train
```

### 3. Launch Operations Control Tower
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Test Suite
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

Run the complete application inside a production container:

```bash
# Build Docker image
docker build -t supply-chain-control-tower .

# Run Docker container
docker run -p 8501:8501 supply-chain-control-tower
```
Navigate to `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/                           # Raw Olist CSV files
│   └── processed/                     # SQLite DW (olist.db) & preprocessed partitions
├── models/
│   ├── calibrated_pipeline.joblib     # Preprocessor + Champion Model + Calibrator
│   ├── model_card.json                # Production metadata, benchmark metrics, thresholds
│   └── test_predictions.parquet      # Scored holdout test manifest
├── src/
│   ├── __init__.py
│   ├── db_ingest.py                   # Chunked SQLite ingestion, deduplication & indexing
│   ├── queries.py                     # Analytical base table (CTEs, Window functions, OTD)
│   ├── features.py                    # Leak-free feature engineering & Haversine distance
│   ├── evaluate.py                    # PR-AUC, Brier score, threshold optimization
│   ├── train.py                       # Chronological split, benchmark, calibration & SHAP
│   └── generate_sample_data.py        # Turnkey realistic data generator
├── tests/
│   ├── test_db.py                     # DB integrity, duration arithmetic & ABT tests
│   └── test_features.py               # Haversine accuracy & strict leakage prevention tests
├── app.py                             # 3-Module Streamlit Operations Control Tower
├── Dockerfile                         # Production Docker container
├── requirements.txt                   # Dependency specifications
└── README.md                          # Production documentation
```

---

## 📜 License
This project is open-source under the MIT License.

