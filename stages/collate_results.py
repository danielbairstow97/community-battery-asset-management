#!/usr/bin/env python3
"""
Collate battery simulation results
Author: Daniel Bairstow

Aggregates the output of all simulation runs from `simulate.py` into
single consolidated CSVs, grouped by location, SOC bounds, and cooling status.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import re


def extract_metadata(path: Path):
    """Extract location, min_soc, max_soc, and cooling status from folder name."""
    pattern = r"(?P<location>[A-Za-z_]+)_min_soc=(?P<min>[\d\.]+)_max_soc=(?P<max>[\d\.]+)_cooling_disabled=(?P<cooling>\w+)"
    match = re.search(pattern, path.name)
    if not match:
        return None
    return {
        "location": match.group("location"),
        "min_soc": float(match.group("min")),
        "max_soc": float(match.group("max")),
        "disable_cooling": match.group("cooling").lower() == "true",
    }


def collate_results(input_dir: Path, output_dir: Path):
    summary_list = []
    daily_list = []
    yearly_list = []

    for folder in input_dir.iterdir():
        print(f"Collating: {folder}")

        meta = extract_metadata(folder)
        if meta is None:
            print("Meta variables could not be extracted")
            continue

        summary_file = folder / "summary.csv"
        daily_file = folder / "daily_results.csv"
        yearly_file = folder / "yearly_results.csv"

        # Load each and add metadata columns
        summary_df = pd.read_csv(summary_file)
        daily_results_df = pd.read_csv(daily_file)
        yearly_results_df = pd.read_csv(yearly_file)

        yearly_results_df['discounted_revenue'] = yearly_results_df['revenue']/(1.025**(yearly_results_df.index))

        for df in [daily_results_df, yearly_results_df, summary_df]:
            for k, v in meta.items():
                df[k] = v

            df['Available Capacity'] = df["max_soc"] - df["min_soc"]
            if not meta["disable_cooling"]:
                df["location"] = "Temperature Regulated"




        summary_list.append(summary_df)
        daily_list.append(daily_results_df)
        yearly_list.append(yearly_results_df)




    # Combine and export
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(daily_list, ignore_index=True).to_csv(output_dir / "daily_collated.csv", index=False)
    pd.concat(yearly_list, ignore_index=True).to_csv(output_dir / "yearly_collated.csv", index=False)
    pd.concat(summary_list, ignore_index=True).to_csv(output_dir / "summary_collated.csv", index=False)

    print(f"✅ Collated results written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate simulation outputs")
    parser.add_argument("--input-dir", required=True, help="Directory containing simulation folders")
    parser.add_argument("--output-dir", required=True, help="Directory to write collated CSVs")
    args = parser.parse_args()

    collate_results(Path(args.input_dir), Path(args.output_dir))
