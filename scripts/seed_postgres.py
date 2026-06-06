"""Create database tables and optionally seed with sample prediction logs."""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    transaction_id  UUID NOT NULL DEFAULT gen_random_uuid(),
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fraud_probability FLOAT NOT NULL,
    is_fraud        BOOLEAN NOT NULL,
    threshold       FLOAT NOT NULL,
    input_features  JSONB
);

CREATE TABLE IF NOT EXISTS retrain_requests (
    id          BIGSERIAL PRIMARY KEY,
    reason      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status      TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id           BIGSERIAL PRIMARY KEY,
    report_date  DATE NOT NULL,
    drift_score  FLOAT,
    f2_current   FLOAT,
    f2_baseline  FLOAT,
    raw_report   JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions(predicted_at);
CREATE INDEX IF NOT EXISTS idx_retrain_status ON retrain_requests(status);
"""

if __name__ == "__main__":
    url = os.getenv("DATABASE_URL", "postgresql://fraud_user:changeme@localhost:5432/fraud")
    conn = psycopg2.connect(url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.close()
    print("Tables created successfully.")
