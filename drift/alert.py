"""Check drift metrics against thresholds and fire alerts."""

from __future__ import annotations

import json
import os

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
F2_DROP_THRESHOLD = float(os.getenv("F2_DROP_THRESHOLD", "0.05"))


def check_and_alert(metrics: dict, baseline_f2: float = None) -> bool:
    drift_triggered = metrics.get("drift_score", 0.0) > DRIFT_THRESHOLD
    perf_triggered = False

    if baseline_f2 is not None and "f2_current" in metrics:
        perf_triggered = (baseline_f2 - metrics["f2_current"]) > F2_DROP_THRESHOLD

    if drift_triggered or perf_triggered:
        _send_alert(metrics, drift_triggered, perf_triggered)
        _write_retrain_flag(
            reason=f"drift={drift_triggered}, perf_drop={perf_triggered}"
        )
        return True

    print("No alert triggered. Model appears healthy.")
    return False


def _send_alert(metrics: dict, drift: bool, perf: bool):
    sns_topic = os.getenv("ALERT_SNS_TOPIC_ARN")
    message = json.dumps({
        "alert": "Fraud Model Drift Alert",
        "drift_triggered": drift,
        "perf_degradation": perf,
        "metrics": metrics,
    }, indent=2)

    if sns_topic:
        try:
            import boto3
            boto3.client("sns").publish(
                TopicArn=sns_topic,
                Subject="Fraud Model Drift Alert",
                Message=message,
            )
            print(f"SNS alert sent to {sns_topic}")
        except Exception as exc:
            print(f"WARNING: SNS alert failed: {exc}")
    else:
        print("=" * 60)
        print("ALERT: Drift detected — retrain required")
        print(message)
        print("=" * 60)


def _write_retrain_flag(reason: str):
    from drift.retrain_trigger import set_retrain_flag
    try:
        set_retrain_flag(reason)
    except Exception as exc:
        print(f"WARNING: Could not write retrain flag: {exc}")
