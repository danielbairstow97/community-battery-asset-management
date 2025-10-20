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
import pandas as pd
import numpy as np
import os
from gurobipy import Model, GRB
import rainflow

import argparse
import pandas as pd
import pybamm
import numpy as np
import sys

# ----------------------------
# Constants (from datasheet https://www.pixii.com/wp-content/uploads/PDFs/7226/datasheet/en/2025-10-01_Ver.1.5_PowerShaper%20XL_60kW_225kWh_Aircon_LFP_EN.pdf)
# ----------------------------
INVERTER_EFF = 0.969
P_MAX = 60.0          # kW AC
E_MAX_USABLE = 202.6  # kWh usable
CYCLE_LIFE_90DOD = 7600  # cycles to 70% SoH @90% DoD
EOL_SOH = 0.7
CYCLE_EFF = 0.5  # rough initial assumption for scaling (DoD^-0.5)

NUM_CELLS = 16
V_CELL = 3.2
C_RATE = 0.5



# ----------------------------
# Helper functions
# ----------------------------
def get_arrhenius(temperature_C):
    """
    Apply a temperature-dependent multiplier to the degradation rate.
    For simplicity: Arrhenius-like acceleration factor per degree.
    """
    T_ref = 25  # reference temperature (°C)
    k_ref = 1.0
    Ea = 30000  # activation energy (J/mol)
    R = 8.314   # gas constant

    # Convert to Kelvin
    T_K = temperature_C + 273.15
    T_ref_K = T_ref + 273.15

    # Arrhenius factor
    temp_factor = np.exp((Ea/R) * (1/T_ref_K - 1/T_K))
    return temp_factor.mean()  # average factor across profile

def optimise_dispatch(prices, capacity_kwh, soh, soc_min, soc_max, dt):
    """Solve 1-year price arbitrage optimisation for given usable capacity."""
    P_max = C_RATE * capacity_kwh * soh

    m = Model()
    m.Params.LogToConsole = 0  # silence output

    T = len(prices)
    P_ch = m.addVars(T, lb=0.0, ub=P_MAX, name="P_charge")
    P_dis = m.addVars(T, lb=0.0, ub=P_MAX, name="P_discharge")
    P_balance = m.addVars(T, lb=0.0, ub=P_MAX, name="P_balance")
    soc = m.addVars(T + 1, lb=soc_min * capacity_kwh, ub=soc_max * capacity_kwh, name="soc")

    # initial SoC
    m.addConstr(soc[0] == 0.5 * capacity_kwh)

    for t in range(T):
        m.addConstr(soc[t + 1] == soc[t] + (INVERTER_EFF * P_ch[t] - (1 / INVERTER_EFF) * P_dis[t]) * dt)
    m.addConstrs(P_dis[t] + P_ch[t] <= P_MAX for t in range(T))
    m.addConstrs(P_chg[t] <= P_max for t in range(T))
    m.addConstrs(P_dis[t] <= P_max for t in range(T))

    revenue = sum(prices.iloc[t] * (P_dis[t] - P_ch[t]) * dt for t in range(T))
    m.setObjective(revenue, GRB.MAXIMIZE)
    m.optimize()

    schedule = pd.DataFrame({
        "timestamp": prices.index,
        "price": prices
        "P_charge": [P_ch[t].X for t in range(T)],
        "P_discharge": [P_dis[t].X for t in range(T)],
        "P_net": [P_dis[t].X - P_chg[t].X for t in range(T)],
    })
    return schedule, m.objVal


def estimate_degradation(schedule, capacity_kwh, soc_min, soc_max):
    """Approximate degradation from rainflow analysis on SoC profile."""
    dt = 1.0  # hr
    soc = np.zeros(len(schedule) + 1)
    for t in range(len(schedule)):
        net_power_ac = schedule["P_discharge"][t] - schedule["P_charge"][t]
        net_power_dc = net_power_ac / INVERTER_EFF
        soc[t + 1] = np.clip(
            soc[t] + (INVERTER_EFF * max(0, -net_power_dc) - (1 / INVERTER_EFF) * max(0, net_power_dc)) * dt / capacity_kwh,
            soc_min, soc_max
        )

    # Compute equivalent cycles using rainflow
    cycles = rainflow.count_cycles(soc)
    equiv_full_cycles = sum(abs(depth) for depth in cycles)  # approximate sum of DoDs
    avg_dod = np.mean([abs(d) for d in cycles]) if len(cycles) > 0 else 0.5

    # Estimate cycle-based degradation
    N_ref = CYCLE_LIFE_90DOD * (0.9 / avg_dod) ** CYCLE_EFF  # scale with DoD
    cycle_deg = (1 - EOL_SOH) / N_ref * equiv_full_cycles

    # Add small calendar fade
    calendar_deg = 0.005  # 0.5%/year baseline
    deg_total = cycle_deg + calendar_deg
    soh_new = max(0, 1 - deg_total)
    return soh_new, deg_total, equiv_full_cycles




def run_pybamm_simulation(schedule, max_depth, max_soc, rainflow, target_lifetime, tmy_df, prev_solution=None,capacity=E_MAX_USABLE, dt=5/60, soh=1.0):
    """
    Main battery degradation simulation loop.
    """

    # Temperature interpolation
    temp_interp = np.interp1d(tmy_df.index.astype(np.int64) / 1e9, tmy_df["temperature"].values+273.15, fill_value="extrapolate")
    def ambient_temp(t): return temp_interp(t)

    # Convert schedule -> current profile
    time_s = np.arange(0, len(schedule) * dt * 3600, dt * 3600)
    current_profile = (schedule["P_net"].values) / (48)  # approximate scaling


    # Setup PyBaMM model
    model = pybamm.lithium_ion.DFN()
    # model = pybamm.lithium_ion.SPM()
    params = model.default_parameter_values
    params["Ambient temperature [K]"] = ambient_temp
    params["Nominal cell capacity [A.h]"] *= soh

    # Define simulation parameters
    params.update({
        "Upper voltage cut-off [V]": 4.2 * max_soc,
        "Lower voltage cut-off [V]": 2.5 + (0.5 * max_depth),
        "SEI film resistance [Ohm.m2]": 0.01,
        "SEI growth rate constant [m.s-1]": 2e-14
    })

    # Create experiment based on current profile
    experiment_steps = []
    for i, I in enumerate(current_profile):
        c_rate = I / (params["Nominal cell capacity [A.h]"] * 3600)
        direction = "Discharge" if I > 0 else "Charge"
        experiment_steps.append(f"{direction} at {abs(c_rate):.4f}C for {dt*60:.1f} seconds")

    experiment = pybamm.Experiment(experiment_steps)
    sim = pybamm.Simulation(model, experiment=experiment,       parameter_values=params)
    solution = sim.solve(initial_conditions=prev_solution)

    # Extract degradation metrics
    capacity_fade = solution["Capacity [A.h]"].data
    soh_new = capacity_fade[-1] / capacity_fade[0]
    current = solution["Current [A]"].data
    time = solution["Time [s]"].data
    throughput_Ah = np.trapz(np.abs(current), time) / 3600
    equiv_cycles = throughput_Ah / capacity_fade[0]

    degradation = 1 - soh_new
    return soh_new, degradation, equiv_cycles, solution


def run_simulation(args):
    """Main simulation loop."""
    prices = pd.read_csv(args.price_file, index_col=0, parse_dates=True).iloc[:, 0]
    years_elapsed = 0
    soh = 1.0
    dt = 5/60
    capacity = E_MAX_USABLE
    history = []

    while True:
        print(f"\n=== Year {years_elapsed + 1} | SoH = {soh:.3f} | Capacity = {capacity:.1f} kWh ===")

        schedule, profit = optimise_dispatch(prices, capacity, soh, args.min_soc, args.max_soc, dt)
        schedule.to_csv(f"schedule_year{years_elapsed + 1}.csv", index=False)

        if args.rainflow:
            soh_new, degradation, cycles = estimate_degradation(schedule, capacity, args.min_soc, args.max_soc)
        else:
            soh_new, degradation, cycles = run_pybamm_simulation(
                    schedule=schedule,
                    max_depth=args.max_depth,
                    max_soc=args.max_soc,
                    rainflow=args.rainflow,
                    target_soh=args.target_soh,
                    target_lifetime=args.target_lifetime,
                    tmy_df=tmy_df
                )
        print(f"Degradation this year: {degradation*100:.3f}% | SoH -> {soh_new:.3f} | Equivalent cycles serviced: {cycles_equiv:.3f}")
        years_elapsed += 1
        history.append({
            "year": years_elapsed,
            "profit_$": profit,
            "degradation": degradation,
            "SoH": soh_new,
            "capacity_kWh": capacity,
            "cycles_equiv": cycles
        })

        # Update capacity
        capacity = E_MAX_USABLE * soh_new
        soh = soh_new

        # stopping criteria
        if args.target_soh and soh <= args.target_soh:
            print(f"Target SoH reached ({soh:.3f} <= {args.target_soh}). Stopping.")
            break
        if args.target_lifetime and years_elapsed >= args.target_lifetime:
            print(f"Target lifetime reached ({years_elapsed} years). Stopping.")
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv("simulation_summary.csv", index=False)
    print("\nSimulation complete. Results saved to simulation_summary.csv.")
    return hist_df


# ----------------------------
# Main entry
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative community battery dispatch & degradation model")
    parser.add_argument("--price-file", required=True, help="CSV of hourly prices ($/kWh) with datetime index")
    parser.add_argument("--min-soc", type=float, default=0.1, help="Minimum SoC fraction (e.g., 0.2 for 20%)")
    parser.add_argument("--max-soc", type=float, default=1.0, help="Maximum SoC fraction (e.g., 0.8 for 80%)")
    parser.add_argument("--rainflow", type=lambda x: str(x).lower() == "true", default=True,
                        help="Use rainflow-based degradation estimation (True/False)")
    parser.add_argument("--target-soh", type=float, default=None,
                        help="Target SoH to stop at (e.g., 0.7 = 70%)")
    parser.add_argument("--target-lifetime", type=int, default=None,
                        help="Target lifetime in years to simulate")

    args = parser.parse_args()

    run_simulation(args)
