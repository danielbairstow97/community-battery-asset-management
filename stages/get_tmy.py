import argparse
import dvc.api

import pandas as pd
import pvlib
from pvlib import location

import os
import pytz
from pathlib import Path

DISPATCH_TZ = "Australia/Brisbane"

def get_tmy(longitude: float, latitude: float)->pd.DataFrame:
    lc = location.Location(latitude=latitude, longitude=longitude)
    aest_tz = pytz.timezone(DISPATCH_TZ)

    tmy_df, _ = pvlib.iotools.get_pvgis_tmy(latitude, longitude)
    tmy_df.index = tmy_df.index.tz_convert(aest_tz)
    tmy_df = tmy_df[["temp_air", "relative_humidity"]].reset_index().rename(columns={"time(UTC)": "Datetime"})
    tmy_df['Datetime'] = tmy_df['Datetime'].apply(lambda x: x.replace(year=1990))
    tmy_df = tmy_df.sort_values(by="Datetime")

    return tmy_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()
    params = dvc.api.params_show()

    location_info = params['locations'][args.location]
    location_tmy = get_tmy(location_info["longitude"], location_info["latitude"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    location_tmy.to_csv(args.output)