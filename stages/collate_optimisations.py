#!/usr/bin/env python3
"""
Collate battery simulation results
Author: Daniel Bairstow

Aggregates the output of all simulation runs from `simulate.py` into
single consolidated CSVs, grouped by location, SOC bounds, and cooling status.
"""

import os
import argparse
from sched import scheduler
import pandas as pd
from pathlib import Path
import re


def extract_metadata(path: Path):
    """Extract location, min_soc, max_soc, and cooling status from simulation path."""
    pattern = r"(?P<location>[A-Za-z_]+)_min_soc=(?P<min>[\d\.]+)_max_soc=(?P<max>[\d\.]+).csv"
    match = re.search(pattern, path.name)
    if not match:
        return None
    return {
        "location": match.group("location"),
        "min_soc": float(match.group("min")),
        "max_soc": float(match.group("max")),
    }


def collate_results(input_dir: Path, output_dir: Path):
    schedule_list = []
    for file in Path("results/schedule").glob("Brisbane_*.csv"):
        meta = extract_metadata(file)
        if meta is None:
            continue

        schedule_df = pd.read_csv(file)
        for k, v in meta.items():
            schedule_df[k] = v
        schedule_df["Name"] = schedule_df.apply(lambda row: f"{row["min_soc"]}-{row["max_soc"]}", axis=1)
        schedule_df["Datetime"] = pd.Timestamp("2024-01-01 00:00:00") + pd.to_timedelta(schedule_df["Time_s"], unit="s")

        # results_list.append(results_df)
        schedule_list.append(schedule_df.drop(columns=["Temperature_C", "location"]))

    df = pd.concat(schedule_list, ignore_index=True)

    # Combine and export
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(schedule_list, ignore_index=True).to_csv(output_dir /    "schedule_collated.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate simulation outputs")
    parser.add_argument("--input-dir", required=True, help="Directory containing simulation folders")
    parser.add_argument("--output-dir", required=True, help="Directory to write collated CSVs")
    args = parser.parse_args()

    collate_results(Path(args.input_dir), Path(args.output_dir))
