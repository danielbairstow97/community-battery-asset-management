import argparse
import dvc.api

import pandas as pd
import pvlib
from pvlib import location

import os
import pytz
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    params = dvc.api.params_show()

    location_info = params['locations'][args.location]
    tmy_df, _ = pvlib.iotools.get_pvgis_tmy(location_info['latitude'], location_info['longitude'])
    tmy_df = tmy_df[["temp_air", "relative_humidity"]].reset_index()
    tmy_df['time(UTC)'] = tmy_df['time(UTC)'].apply(lambda x: x.replace(year=2024))
    tmy_df = tmy_df.sort_values(by="time(UTC)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tmy_df.to_csv(args.output, index=False)