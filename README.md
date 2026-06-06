# Real-Time Fraud / Transaction-Risk Platform

A production-grade ML system for fraud detection demonstrating the full MLOps lifecycle:
ingest → feature engineering → model training → experiment tracking → containerised API →
CI/CD → drift monitoring → automated retraining.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Data Layer                                       │
│  PaySim CSV (6M rows)  →  PySpark feature pipeline  →  Parquet           │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                         Training Pipeline                                 │
│  LogReg (baseline)  →  XGBoost / LightGBM  →  MLflow registry           │
│  Airflow DAG: check retrain flag → train → validate → promote @champion  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                         Serving Layer                                     │
│  FastAPI  →  Docker  →  AWS ECS Fargate  (POST /predict, GET /health)    │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│                         Monitoring Loop                                   │
│  Predictions → Postgres → Evidently drift report →                       │
│  Alert (SNS) + retrain_requests flag → Airflow retrain DAG               │
└──────────────────────────────────────────────────────────────────────────┘
```

## Stack

| Layer | Technology |
|---|---|
| Data | PaySim (Kaggle), Postgres, PySpark |
| Feature engineering | Pandas (dev), PySpark (production) |
| ML models | Logistic Regression (baseline), XGBoost, LightGBM, optional TF/Keras DNN |
| Experiment tracking | MLflow (self-hosted) |
| Orchestration | Apache Airflow |
| API | FastAPI + Pydantic v2 |
| Containerisation | Docker, Docker Compose |
| Cloud deploy | AWS ECS Fargate + ALB, RDS Postgres, S3, ECR |
| CI/CD | GitHub Actions |
| Drift monitoring | Evidently |
| IaC | Terraform |

## Operating Metric: F2 Score

**Why not accuracy?**
PaySim fraud rate is ~0.13%. A model that classifies everything as legitimate scores 99.87% accuracy and catches zero fraud. Accuracy is useless here.

**Why F2?**
`F2 = (5 * Precision * Recall) / (4 * Precision + Recall)`

F2 weights recall twice as heavily as precision. This encodes the cost asymmetry: missing a
fraudulent transaction (false negative) is far more damaging than a false alarm (false positive).
Every evaluation, threshold choice, and retraining trigger in this system is driven by F2.

**Why not AUC-ROC?**
Under extreme class imbalance, AUC-ROC is overly optimistic. Average Precision (area under the
PR curve) is used for model comparison during training; F2 at the operating threshold is the
go/no-go promotion criterion.

**Threshold selection:**
After training, we sweep decision thresholds from 0.01 to 0.99 and select the value that
maximises F2 on the validation fold. This threshold is stored alongside the model artifact and
used at inference time. See `notebooks/01_threshold_selection.ipynb`.

**Promotion bar:** F2 ≥ 0.70 on held-out test set (20% stratified split).

## Class Imbalance Strategy

1. **Class weights** (primary): `class_weight="balanced"` for LogReg/LGBM; `scale_pos_weight`
   for XGBoost. Simpler, no synthetic data, usually competitive with SMOTE on large datasets.
2. **SMOTE comparison**: run `make train-xgb-smote` to log a SMOTE-augmented run to MLflow.
   Compare F2 on the same test fold. The comparison itself is a documented decision.

> In PaySim experiments, class weights match or beat SMOTE because the dataset is large enough
> for the tree models to find signal without synthetic oversampling.

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Download PaySim dataset (requires Kaggle API credentials)
make download-data

# 3. Generate 1k stratified sample for CI
python scripts/make_sample.py

# 4. Train baseline model
make train-baseline

# 5. Start the full stack (Postgres + MLflow + API)
make docker-up

# 6. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"step":1,"type":"TRANSFER","amount":181.0,"oldbalanceOrg":181.0,
       "newbalanceOrig":0.0,"oldbalanceDest":0.0,"newbalanceDest":0.0}'

# 7. Run CI checks locally
make ci-local
```

## Model Results

| Model | F2 (test) | Avg Precision | Notes |
|---|---|---|---|
| Logistic Regression | — | — | Baseline |
| XGBoost | — | — | Primary model |
| LightGBM | — | — | Challenger |
| XGBoost + SMOTE | — | — | Comparison only |

*Results populated after running `make train-xgb`*

## Drift Monitoring

The drift monitor runs nightly (via Airflow). It compares the last 7 days of prediction inputs
against the training reference distribution using Evidently's `DataDriftPreset`.

Alerts are fired when:
- More than 30% of input features have drifted (Kolmogorov-Smirnov test)
- F2 on labelled predictions drops by > 0.05 from the champion model's baseline

A `retrain_requests` row is inserted; the Airflow DAG picks it up and conditionally promotes
a new model if the retrained version exceeds the F2 bar.

## Fairness

FPR (false positive rate) is computed per transaction type (`type` column) as a proxy for
fairness segmentation — PaySim does not include demographic features. The results are reported
in `training/evaluate.py::fairness_report()` and noted as an open limitation.

## Responsible AI

- **Scope**: This model flags transactions for human review. It does not autonomously block.
- **Data limitations**: PaySim is synthetic; real-world generalization requires revalidation.
- **Threshold trade-off**: F2-optimised threshold prioritises recall. False positives create
  friction for legitimate users and should be monitored.
- **Drift lag**: Evidently detects distribution shift; ground-truth label delay means performance
  degradation is detected with a lag of days, not minutes.
- **Fairness gap**: Sliced FPR analysis requires demographic columns not present in PaySim.
  Any production deployment must rerun fairness analysis on real labelled data.

## Limitations / What I Would Do Next

1. **Online feature store**: replace batch parquet with Feast or Tecton for sub-millisecond
   feature lookup in inference.
2. **Real-time streaming**: replace the batch Spark pipeline with Spark Structured Streaming
   or Flink for true real-time risk scoring.
3. **Graph features**: fraudulent networks have strong graph structure (shared devices, IPs).
   A GraphSAGE layer on the transaction graph would likely improve recall significantly.
4. **Label delay mitigation**: implement a delayed-label pipeline that backfills confirmed fraud
   labels from a dispute system and triggers drift checks on those, not just distribution shift.
5. **A/B deployment**: shadow-mode deployment of challenger models with traffic splitting
   before full promotion.
6. **Demographic fairness**: partner with a dataset that includes protected attributes to
   run proper equalized-odds analysis.
