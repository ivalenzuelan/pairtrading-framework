#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd

RAW_ROOT  = Path("data/raw2")        # ← your organised folders
PROC_ROOT = Path("data/processed2")
KEEP      = ["open_time", "close", "volume"]        # final column order

def process_one(csv_path: Path, interval: str):
    symbol = csv_path.stem.split("_")[0].upper()     # btcusdt → BTCUSDT
    df = pd.read_csv(csv_path, parse_dates=["open_time"])

    # ── ensure required columns ────────────────────────────────────────────────
    if "volume" not in df.columns:
        df["volume"] = pd.NA                 # placeholder so the column exists

    df = (df
          .loc[:, KEEP]                      # drop everything else
          .sort_values("open_time")
          .drop_duplicates("open_time")
          .set_index("open_time")
          .ffill(limit=2)                    # forward-fill ≤2 missing rows
    )

    out_dir = PROC_ROOT / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{symbol}.parquet")

    print(f"✔ {csv_path.name:35s} → {interval}/{symbol}.parquet  ({len(df):,} rows)")

def main():
    for interval_dir in RAW_ROOT.iterdir():
        if not interval_dir.is_dir():
            continue
        interval = interval_dir.name
        for csv_file in interval_dir.glob("*.csv"):
            process_one(csv_file, interval)

if __name__ == "__main__":
    main()
