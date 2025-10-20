import argparse
import dvc.api

from nemosis import dynamic_data_compiler

import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)

    start_time = '2024/01/01 00:00:00'
    end_time = '2025/01/01 00:00:00'
    table = 'DISPATCHPRICE'

    args = parser.parse_args()
    output_dir = Path(args.output)
    cache_dir = output_dir.joinpath("cache")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)


    price_data = dynamic_data_compiler(start_time, end_time, table, cache_dir, filter_cols=["REGIONID"], filter_values=[["QLD1"]])
    price_data.to_csv(output_dir.joinpath("dispatch.csv"))