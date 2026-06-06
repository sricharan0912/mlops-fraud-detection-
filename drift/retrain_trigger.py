"""Write a retrain request to Postgres so the Airflow DAG can pick it up."""

from __future__ import annotations

import os

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud_user:changeme@localhost:5433/fraud")


def set_retrain_flag(reason: str):
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO retrain_requests (reason, status) VALUES (%s, 'pending')",
            (reason,),
        )
    conn.commit()
    conn.close()
    print(f"Retrain flag written: {reason}")


def get_pending_retrains() -> list[dict]:
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, reason, created_at FROM retrain_requests WHERE status = 'pending' ORDER BY created_at"
        )
        rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "reason": r[1], "created_at": r[2]} for r in rows]


def mark_retrain_done(retrain_id: int):
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE retrain_requests SET status = 'done' WHERE id = %s",
            (retrain_id,),
        )
    conn.commit()
    conn.close()
