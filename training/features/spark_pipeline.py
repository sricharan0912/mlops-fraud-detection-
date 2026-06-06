"""PySpark feature engineering pipeline.

Replaces pandas_pipeline.py for production-scale processing of the full
6M-row PaySim dataset. Outputs parquet to data/processed/features.parquet.

Install spark extra: pip install fraud-platform[spark]

Usage:
    python training/features/spark_pipeline.py \
        --input data/raw/PS_20174392719_1491204439457_log.csv \
        --output data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import pathlib

LABEL_COL = "isFraud"


def create_spark_session():
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName("FraudFeatureEngineering")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "50")  # reduce for local dev (default=200)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def engineer_features_spark(df):
    from pyspark.sql import functions as F

    return (
        df.withColumn(
            "balance_diff_orig",
            F.col("oldbalanceOrg") - F.col("newbalanceOrig"),
        )
        .withColumn(
            "balance_diff_dest",
            F.col("newbalanceDest") - F.col("oldbalanceDest"),
        )
        .withColumn("amount_log", F.log1p(F.col("amount")))
    )


def build_spark_ml_pipeline(feature_cols: list[str]):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler

    type_indexer = StringIndexer(
        inputCol="type", outputCol="type_idx", handleInvalid="keep"
    )
    numeric_cols = [c for c in feature_cols if c != "type"]
    assembler = VectorAssembler(
        inputCols=numeric_cols + ["type_idx"], outputCol="features_raw"
    )
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features", withStd=True, withMean=True
    )
    return Pipeline(stages=[type_indexer, assembler, scaler])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--output", default="data/processed/features.parquet")
    args = parser.parse_args()

    spark = create_spark_session()
    print(f"Spark version: {spark.version}")

    print(f"Reading {args.input} ...")
    df = spark.read.csv(args.input, header=True, inferSchema=True)
    print(f"  Rows: {df.count():,}")

    df = engineer_features_spark(df)

    feature_cols = [
        "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "balance_diff_orig", "balance_diff_dest", "amount_log", "type",
    ]

    pipeline = build_spark_ml_pipeline(feature_cols)
    model = pipeline.fit(df)
    transformed = model.transform(df)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as parquet; coalesce to avoid many small files
    (
        transformed
        .select("features", LABEL_COL, "type", "amount", "step")
        .coalesce(4)
        .write
        .mode("overwrite")
        .parquet(str(out_path))
    )

    print(f"Parquet written to {out_path}")
    spark.stop()


if __name__ == "__main__":
    main()
