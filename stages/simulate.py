#!/usr/bin/env python3
"""
Community Battery Asset Management Loop
Author: Daniel Bairstow
Description:
    Iteratively optimises battery dispatch using Gurobi against energy price data,
    simulates degradation (rainflow-based or physics-based), and repeats until a
    target state of health (SoH) or lifetime (years) is reached.

Usage:
    python battery_asset_loop.py \
        --price-file prices_hourly.csv \
        --rainflow True \
        --target-soh 0.7 \
        --target-lifetime 15 \
        --min-soc 0.1 \
        --max-soc 1.0
"""

import argparse
from anyio import Path
import pandas as pd
import numpy as np
import os
from gurobipy import Model, GRB, quicksum
import rainflow

import os
import sys

import argparse
import pandas as pd
import pybamm
import numpy as np
import pytz

from ast import mod
import pandas as pd
from blast import models
import numpy as np


def simulate_battery(schedule, dt):
    data = {
        "Time_s": schedule["Time_s"].to_numpy(),
        "SOC": schedule["SOC_%"].to_numpy(),
        "Temperature_C": schedule["Temperature_C"].to_numpy()
    }
    batt = models.Lfp_Gr_250AhPrismatic(degradation_scalar=1.1901)
    batt.simulate_battery_life(data, threshold_capacity=0.7, is_conserve_energy_throughput=False)

    dSOC = schedule['SOC'].diff().fillna(0)
    efc = 0.5 * dSOC.abs().sum() * ((365*24*3600) / data['Time_s'][-1])

    batt_soc = batt.stressors['soc'][1:]
    soc = batt_soc.tolist()
    batt_dod = batt.stressors['dod'][1:]
    dod = batt_dod.tolist()
    efc_subcycle = batt.stressors['delta_efc'][1:].tolist()
    t_subcycle = batt.stressors['delta_t_days'][1:].tolist()
    dod_rms =  np.sqrt(np.mean(batt_dod**2))
    dod_median = np.median(batt_dod)
    dod_95 = np.percentile(batt_dod, 95)
    soc_mean = np.mean(batt_soc)
    soc_median = np.median(batt_soc)

    results = pd.DataFrame({
        "Relative Capacity": batt.outputs['q'],
        "EFC": batt.stressors["efc"],
        "T (Days)": batt.stressors['t_days'],
    })

    summary = {
        'EFCs/year': [efc],
        'Mean SOC': [soc_mean],
        'Median SOC': [soc_median],
        'DOD RMS': [dod_rms],
        'Median DOD': [dod_median],
        'DOD 95th percentile': [dod_95],
        "Lifetime": [batt.stressors['t_days'][-1]],
    }

    # Convert degradation time to seconds to match the dispatch dataframe
    df_deg = results.copy()
    df_disp = schedule.copy()
    df_deg["Time_s"] = df_deg["T (Days)"] * 24 * 3600  # convert days → seconds

    # 2. Determine total simulated lifetime in seconds (until 0.7 SoH)
    lifetime_s = df_deg["Time_s"].iloc[-1]  # e.g., 3294 days → ~284 million seconds

    # 3. Find how long one dispatch cycle lasts
    period_s = df_disp["Time_s"].iloc[-1]  # duration of one dispatch cycle

    # 4. Compute how many repeats needed to reach the lifetime duration
    n_repeats = int(np.ceil(lifetime_s / period_s))

    # 5. Repeat the dispatch schedule
    df_disp = pd.concat(
        [df_disp.assign(Time_s=df_disp["Time_s"] + i * (period_s + df_disp["Time_s"].iloc[1]))
        for i in range(n_repeats)],
        ignore_index=True
    )

    # Interpolate the relative capacity to the dispatch time base
    df_disp["Relative Capacity"] = np.interp(
        df_disp["Time_s"],
        df_deg["Time_s"],
        df_deg["Relative Capacity"]
    )
    df_disp["EFC"] = np.interp(
        df_disp["Time_s"],
        df_deg["Time_s"],
        df_deg["EFC"]
    )

    # Optional: compute degraded net power (useful for lifetime revenue estimation)
    df_disp["degraded_P_net"] = df_disp["P_net"] * df_disp["Relative Capacity"]
    df_disp["degraded_P_chg"] = -df_disp["P_net"].clip(upper=0)
    df_disp["degraded_P_dis"] = df_disp["P_net"].clip(lower=0)

    # Optional: compute revenue
    df_disp["revenue"] = df_disp["degraded_P_net"] * dt / 1000 * df_disp["price"]
    df_disp["Datetime"] = pd.Timestamp("2025-01-01 00:00:00") + pd.to_timedelta(df_disp["Time_s"], unit="s")
    df_disp = df_disp.drop(columns=["P_charge", "P_discharge", "P_net"])

    summary["Lifetime Value"] = [df_disp["revenue"].sum()]


    daily_results = df_disp.groupby(df_disp["Datetime"].dt.date).aggregate({
        "revenue": "sum",
        "EFC": "last",
        "degraded_P_dis": "sum",
        "degraded_P_chg": "sum",
        "Relative Capacity": "last"
    })

    yearly_results = df_disp.groupby(df_disp["Datetime"].dt.year).aggregate({
        "revenue": "sum",
        "EFC": "last",
        "degraded_P_dis": "sum",
        "degraded_P_chg": "sum",
        "Relative Capacity": "last"
    })


    return results, summary, df_disp, daily_results, yearly_results


# ----------------------------
# Main entry
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative community battery dispatch & degradation model")
    parser.add_argument("--disable-cooling", type=lambda x: str(x).lower() == "true", default=True,help="Disable cooling")
    parser.add_argument("--schedule", type=str, default=None,
                        help="Schedule to simulate battery operation with")
    parser.add_argument("--output", default=None,
                        help="Output directory of results")
    args = parser.parse_args()

    dt = 5/60.0
    schedule = pd.read_csv(args.schedule)
    schedule["Time_s"] = schedule.index * 5 * 60

    if not args.disable_cooling:
        schedule['Temperature_C'] = 25

    results, summary, df_disp, daily_results, yearly_results = simulate_battery(schedule, dt)

    print(yearly_results)

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(results).to_csv(output_dir.joinpath("results.csv"))
    pd.DataFrame(summary).to_csv(output_dir.joinpath("summary.csv"))
    df_disp.to_csv(output_dir.joinpath("degraded_schedule.csv"))
    daily_results.to_csv(output_dir.joinpath("daily_results.csv"), index=True)
    yearly_results.to_csv(output_dir.joinpath("yearly_results.csv"), index=True)


