# SmartSneaks AI — Data Warehouse & Mining Perspective

Project snapshot
- A Streamlit application that visualizes and analyzes sneaker sales in Tamil Nadu.
- Combines a lightweight data warehouse approach (ETL, stored datasets, star-schema modeling) with data mining techniques (forecasting, anomaly detection, clustering, authenticity checks).
- Purpose: enable analytics-ready storage of sales data and produce operational insights for product managers, data engineers, and data scientists.

Key goals
- Centralize sales-related data into a coherent data warehouse structure.
- Provide repeatable ETL to transform raw sales, product, and geolocation sources into analytics tables.
- Support mining tasks: demand forecasting, anomaly detection, segmentation, and authenticity classification.
- Serve interactive dashboards and reports via Streamlit for business consumption.

Repository layout (high level)
- app.py — Streamlit front-end (dashboard, authenticity checker, AI insights).
- data/ — raw and processed datasets (CSV/Parquet).
- etl/ — ETL scripts, ingestion and transformation logic.
- models/ — trained model artifacts (forecast, anomaly, classifier).
- notebooks/ — exploratory analysis and model experiments.
- tests/ — unit tests for ETL and core logic.
- requirements.txt / venv/ — environment dependencies.

Data sources (typical)
- Transactional sales logs: order_id, sku, timestamp, qty, price, city, store_id.
- Product catalog: sku, brand, model, category, launch_date.
- External enrichment: city lat/long, population, seasonal indicators.
- Image inputs: uploaded shoe photos for authenticity checks.

Data warehouse design (recommended)
- Modeling approach: star schema (fact_sales + dimensions).
- Fact table
  - fact_sales(sale_id, sku, store_id, date_id, qty, revenue, cost, customer_id, channel)
- Dimensions
  - dim_date(date_id, date, day, week, month, quarter, year, is_holiday)
  - dim_product(sku, brand, model, category, release_date)
  - dim_store(store_id, city, state, lat, lon)
  - dim_customer(customer_id, segment, created_at)
- Materialized aggregates
  - daily_brand_sales, city_daily_metrics, sku_popularity_rank

ETL pipeline (recommended flow)
1. Ingest raw files from data/ or external source (APIs, S3).
2. Validate and clean (schema validation, null handling, deduplication).
3. Enrich (map city coordinates, derive calendar features, add promotions).
4. Transform into dimension records and grain-aligned fact rows.
5. Load into warehouse storage (Parquet files, SQLite/Postgres, or cloud DW).
6. Run incremental updates and backfills; persist model inputs.

Tools & storage recommendations
- Local / dev: Parquet files + SQLite for metadata; Pandas + PyArrow.
- Production: Postgres or cloud DW (BigQuery / Redshift / Synapse) + S3/ADLS.
- Orchestration: Airflow or Prefect for scheduling ETL & retraining jobs.
- Models: joblib / ONNX for serialization; models/ holds artifacts.

Data quality & governance
- Schema tests (pytest + great_expectations for expectations).
- Row-count and null checks after each ETL run.
- Timestamped snapshots and versioning for reproducibility.
- Access controls and encryption for PII.

Data mining & analytics components
- Forecasting (demand forecaster)
  - Input: aggregated sku-city-day timeseries
  - Techniques: Prophet / SARIMAX / Gradient-boosted regression or simple linear models inside a pipeline
  - Output: per-sku demand forecast with confidence bands
- Anomaly detection
  - Technique: IsolationForest on time-windowed features or robust z-score
  - Use case: detect sudden spikes/drops in daily revenue or returns
- Clustering / segmentation
  - Technique: KMeans or hierarchical clustering on recency-frequency-monetary (RFM) + product preferences
  - Output: customer segments for targeting
- Authenticity / image checks
  - Lightweight local model + heuristics to flag likely counterfeit listings
  - Can be swapped with a remote classifier or CV model (YOLO, EfficientNet) in production

Serving & visualization
- Streamlit app (app.py) provides:
  - Dashboard: filters (brand, city), maps, KPIs, trend charts.
  - Authenticity Checker: upload image -> model verdict + explanation.
  - AI Insights: run forecasting, anomaly detection, and present actionable recommendations.
- Local run (Windows, in repo root):
````bash
python -m venv venv
.\venv\Scripts\activate
pip install -r [requirements.txt](http://_vscodecontentref_/0)
python -m streamlit run [app.py](http://_vscodecontentref_/1)