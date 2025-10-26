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

import os

import argparse
import pandas as pd
import numpy as np
import pytz

os.environ["GRB_LICENSE_FILE"] = "./gurobi.lic"

# ----------------------------
# Constants (from datasheet https://www.pixii.com/wp-content/uploads/PDFs/7226/datasheet/en/2025-10-01_Ver.1.5_PowerShaper%20XL_60kW_225kWh_Aircon_LFP_EN.pdf)
# ----------------------------
INVERTER_EFF = 0.969
P_MAX = 60.0          # kW AC
E_MAX_USABLE = 202.6  # kWh usable
E_MAX = 225.1 # kWh
CYCLE_LIFE_90DOD = 7600  # cycles to 70% SoH @90% DoD
EOL_SOH = 0.7
CYCLE_EFF = 0.5  # rough initial assumption for scaling (DoD^-0.5)
DISPATCH_TZ = "Australia/Brisbane"

NUM_CELLS = 16
V_CELL = 3.2
C_RATE = 0.5

def optimise_dispatch(prices, capacity_kwh, soc_min, soc_max, dt):
    """Solve 1-year price arbitrage optimisation for given usable capacity."""
    degraded_max = soc_max * capacity_kwh
    degraded_min = soc_min * capacity_kwh

    P_max = C_RATE * capacity_kwh

    m = Model("Battery_optimisation")
    m.Params.LogToConsole = 0  # silence output

    T = len(prices)
    P_chg = m.addVars(T, lb=0.0, ub=P_max, name="P_charge")
    P_dis = m.addVars(T, lb=0.0, ub=P_max, name="P_discharge")
    soc = m.addVars(T + 1, lb=degraded_min, ub=degraded_max, name="soc")

    # Initial SoC
    m.addConstr(soc[0] == 0.5 * (degraded_min + degraded_max))

    for t in range(T):
        m.addConstr(soc[t + 1] == soc[t] + (INVERTER_EFF * P_chg[t] - (1 / INVERTER_EFF) * P_dis[t]) * dt)
        m.addConstr(P_dis[t] + P_chg[t] <= P_max)
        m.addConstr(P_dis[t] / INVERTER_EFF <= soc[t] - degraded_min)
        m.addConstr(P_chg[t] <= degraded_max - soc[t])

    # Objective: maximize arbitrage profit
    m.setObjective(
        quicksum(prices.iloc[t].item() / 1000 * (P_dis[t] - P_chg[t]) * dt for t in range(T)),
        GRB.MAXIMIZE,
    )
    m.optimize()

    schedule = pd.DataFrame({
        "price": np.asarray(prices).flatten(),
        "SOC": [soc[t].X for t in range(T)],
        "P_charge": [P_chg[t].X for t in range(T)],
        "P_discharge": [P_dis[t].X for t in range(T)],
        "P_net": [P_dis[t].X - P_chg[t].X for t in range(T)],
    }, index=prices.index)
    schedule["SOC_%"] = schedule["SOC"]/E_MAX

    return schedule, m.objVal

def build_temperature_function(tmy_df, price_df, dt):
    """
    Align hourly TMY temperature (and optionally humidity) with 5-min price data
    and return smooth interpolation functions in Kelvin.
    """

    # --- 1. Prepare and clean TMY ---
    tmy_df = tmy_df.copy()
    tmy_df.index = pd.to_datetime(tmy_df.index)
    tmy_df["DOY"] = tmy_df.index.dayofyear
    tmy_df["TOD"] = tmy_df.index.hour + tmy_df.index.minute / 60.0
    tmy_df.rename(columns={"temp_air": "Temperature_C"}, inplace=True)

    # --- 2. Build daily/hour pattern from TMY ---
    tmy_daily = tmy_df.set_index(["DOY", "TOD"])

    # --- 3. Extract day/time-of-day pattern for price timestamps ---
    price_pattern = price_df.copy()
    price_pattern["DOY"] = price_pattern.index.dayofyear
    price_pattern["TOD"] = price_pattern.index.hour + price_pattern.index.minute / 60.0

    # --- 4. Join TMY data ---
    merged_temp = (
        price_pattern.set_index(["DOY", "TOD"])
        .join(tmy_daily[["Temperature_C"]], on=["DOY", "TOD"], how="left")
        .interpolate(method="linear", limit_direction="both")
        .reset_index(drop=True)
    )

    # --- 5. Align with price_df index ---
    merged_temp.index = price_df.index  # now safe; same length & order

    # --- 6. Convert to Kelvin & build interpolators ---
    t_seconds = np.arange(0, len(merged_temp) * dt * 3600, dt * 3600)
    temp_K = merged_temp["Temperature_C"].to_numpy() + 273.15

    print(f"✓ Aligned TMY with {len(price_df)} dispatch intervals.")
    print(f"Temperature range: {merged_temp['Temperature_C'].min():.1f}–{merged_temp['Temperature_C'].max():.1f} °C")

    return temp_K, merged_temp

def run_simulation(args):
    """Main simulation loop."""
    aest_tz = pytz.timezone("Australia/Brisbane")

    output_file = Path(args.output)

    price_df = pd.read_csv(args.dispatch, parse_dates=["SETTLEMENTDATE"])
    price_df = price_df.set_index("SETTLEMENTDATE").sort_index().tz_localize(aest_tz)[["RRP"]]
    dt = 5.0/60.0

    tmy_df = pd.read_csv(args.tmy, parse_dates=["time(UTC)"])
    tmy_df["time(UTC)"] = pd.to_datetime(tmy_df['time(UTC)'])
    tmy_df = tmy_df.rename(columns={"time(UTC)": "Datetime"})
    tmy_df = tmy_df.set_index("Datetime")
    tmy_df =  tmy_df.tz_convert(aest_tz)

    ambient_temp, merged_temp = build_temperature_function(tmy_df, price_df, dt)

    schedule, profit = optimise_dispatch(
        price_df["RRP"], E_MAX, args.min_soc, args.max_soc, dt
    )
    # Add extra fields
    schedule = schedule.join(merged_temp[["Temperature_C"]])
    earliest_datetime = schedule.index.min()
    time_difference = schedule.index - earliest_datetime
    schedule["Time_s"] = time_difference.total_seconds()

    schedule.to_csv(output_file, index=False)
    return


# ----------------------------
# Main entry
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative community battery dispatch & degradation model")
    parser.add_argument("--dispatch", required=True, help="CSV of prices ($/kWh) with datetime index")
    parser.add_argument("--tmy", required=True, help="CSV of hourly TMY temperature data")
    parser.add_argument("--min-soc", type=float, required=True, help="Minimum SoC fraction (e.g., 0.2 for 20%)")
    parser.add_argument("--max-soc", type=float, required=True, help="Maximum SoC fraction (e.g., 0.8 for 80%)")
    parser.add_argument("--target-soh", type=float, default=EOL_SOH,
                        help="Target SoH to stop at (e.g., 0.7 = 70%)")
    parser.add_argument("--disable-cooling", type=lambda x: str(x).lower() == "true", default=True,
                        help="Use rainflow-based degradation estimation (True/False)")
    parser.add_argument("--target-lifetime", type=int, default=None,
                        help="Target lifetime in years to simulate")
    parser.add_argument("--output", default=None,
                        help="Output file of results")

    args = parser.parse_args()

    run_simulation(args)
