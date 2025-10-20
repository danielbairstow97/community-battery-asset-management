from gurobipy import Model, GRB
import pandas as pd

prices_df = pd.read_csv("prices_hourly.csv", index_col=0, parse_dates=True)  # $/kWh
dispatch_price = prices_df['RRP']

T = len(dispatch_price)
dt = 1.0  # hours
P_max = 60.0  # kW AC
E_max = 202.6  # kWh usable
eta_batt = 0.969
soc_min = 0.1
soc_max = 1.0

m = Model()
P_ch = m.addVars(T, lb=0.0, ub=P_max, name="P_charge")     # AC kW
P_dis = m.addVars(T, lb=0.0, ub=P_max, name="P_discharge") # AC kW
soc = m.addVars(T+1, lb=soc_min*E_max, ub=soc_max*E_max, name="soc") # in kWh

# initial SoC - choose e.g., 50% of usable:
m.addConstr(soc[0] == 0.5 * E_max)

for t in range(T):
    # convert AC to DC inside SoC update:
    m.addConstr(soc[t+1] == soc[t] + (eta_batt * P_ch[t] - (1/eta_batt) * P_dis[t]) * dt)

# objective: revenue = price($/kWh) * dispatch kWh (discharge) - cost of charging (grid buy)
obj = sum((dispatch_price.iloc[t] * (P_dis[t] - P_ch[t])) * dt for t in range(T))
m.setObjective(obj, GRB.MAXIMIZE)

m.optimize()

# extract schedule:
schedule = []
for t in range(T):
    schedule.append({"timestamp": dispatch_price.index[t],
                     "P_ac_charge": P_ch[t].X,
                     "P_ac_discharge": P_dis[t].X})
pd.DataFrame(schedule).to_csv("schedule_ac.csv", index=False)
