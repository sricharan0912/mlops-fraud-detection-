"""
Generate data/sample/paysim_sample_1k.csv — stratified 500 fraud / 500 legit.
Run once after downloading the full dataset.
"""

import pandas as pd
import pathlib

SRC = "data/raw/PS_20174392719_1491204439457_log.csv"
DST = "data/sample/paysim_sample_1k.csv"
N_PER_CLASS = 500
RANDOM_STATE = 42


def main():
    print(f"Reading {SRC} ...")
    df = pd.read_csv(SRC)
    fraud = df[df["isFraud"] == 1].sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
    legit = df[df["isFraud"] == 0].sample(n=N_PER_CLASS, random_state=RANDOM_STATE)
    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=RANDOM_STATE)
    pathlib.Path("data/sample").mkdir(parents=True, exist_ok=True)
    sample.to_csv(DST, index=False)
    print(f"Saved {len(sample)} rows to {DST}")
    print(sample["isFraud"].value_counts())


if __name__ == "__main__":
    main()
