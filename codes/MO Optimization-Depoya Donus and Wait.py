import logging
import os
import sys
from datetime import datetime, time

import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


MODEL_NAME = "MR_MV_RD"

CASE_NAME   = "case1"
EPS_WAIT    = 9999
MAX_ROUTES  = 3


RESULTS_ROOT = r"C:\Users\Asus\Documents\GitHub\logistics-model2\internal-logistics-model2\results"
OUTPUT_DIR = os.path.join(RESULTS_ROOT, MODEL_NAME, CASE_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")


TIME_LIMIT = 600
MIP_GAP = 0.01

SHIFT_START = 0

WAIT_WEIGHT = 0.001


# Composite run identifier (model + case + max_routes + epsilon)
RUN_ID = f"{MODEL_NAME}_{CASE_NAME}_R{MAX_ROUTES}_eps{int(EPS_WAIT)}"

log_file = os.path.join(OUTPUT_DIR, f"{RUN_ID}_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

log.info("Run started | run_id: %s | log file: %s", RUN_ID, log_file)


def ready_to_min(v):
    """Convert ready_time cell value to minutes from midnight."""
    if pd.isna(v):
        return 0

    if isinstance(v, (pd.Timestamp, datetime, time)):
        return v.hour * 60 + v.minute

    if isinstance(v, (int, float)):
        return int(v)

    s = str(v).strip()

    if ":" in s:
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm)

    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return dt.hour * 60 + dt.minute

    return int(float(s))


def minutes_to_hhmm(minutes):
    """Format minutes as HH:MM string."""
    if minutes is None or pd.isna(minutes):
        return ""
    minutes = float(minutes)
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


DATA_PATH = r"C:\Users\Asus\Desktop\Er"

nodes = pd.read_excel(os.path.join(DATA_PATH, "nodes.xlsx"))
vehicles = pd.read_excel(os.path.join(DATA_PATH, "vehicles.xlsx"))
products = pd.read_excel(os.path.join(DATA_PATH, "products.xlsx"), sheet_name=CASE_NAME)

log.info("Products loaded | sheet=%s | rows=%d", CASE_NAME, len(products))


def read_distance_matrix(path, val_col):
    """Read OD distance/duration matrix into a dict keyed by (from, to)."""
    df = pd.read_excel(path, sheet_name=0)

    required = ["from_node", "to_node", val_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("%s missing columns: %s" % (os.path.basename(path), missing))

    df["from_node"] = df["from_node"].astype(str).str.strip()
    df["to_node"] = df["to_node"].astype(str).str.strip()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=required)
    df = df[df["from_node"] != df["to_node"]]

    return dict(zip(zip(df["from_node"], df["to_node"]), df[val_col].astype(float)))


c = read_distance_matrix(os.path.join(DATA_PATH, "distances - dakika.xlsx"), "duration_min")


# Node sets
nodes["node_id"] = nodes["node_id"].astype(str).str.strip()
N = nodes["node_id"].dropna().drop_duplicates().tolist()
Nw = [n for n in N if n != "h"]

# Vehicle set (multi-vehicle scenario — full fleet)
vehicles["vehicle_id"] = vehicles["vehicle_id"].astype(str).str.strip()
vehicles = vehicles.dropna(subset=["vehicle_id"]).drop_duplicates("vehicle_id", keep="first")
K = vehicles["vehicle_id"].tolist()

# Route set (multi-route)
R = list(range(1, MAX_ROUTES + 1))

# Vehicle capacities
q_vehicle = dict(zip(K, vehicles["capacity_m2"]))

# Product set and parameters
products["product_id"] = products["product_id"].astype(str).str.strip()
products["origin"] = products["origin"].astype(str).str.strip()
products["destination"] = products["destination"].astype(str).str.strip()
products = products.dropna(subset=["product_id"]).drop_duplicates("product_id", keep="first")

P = products["product_id"].tolist()

e = dict(zip(P, [ready_to_min(v) for v in products["ready_time"]]))
sl = dict(zip(P, products["load_time"]))
su = dict(zip(P, products["unload_time"]))
q_product = dict(zip(P, products["area_m2"]))
o = dict(zip(P, products["origin"]))
d = dict(zip(P, products["destination"]))

# Horizon and global bounds
T_max = 480
C_max = 11
e_min = 15
Q_max = 20

U = len(Nw)

# Tight Big-M coefficients
BIG_M_TIME       = T_max + C_max
BIG_M_PRECEDENCE = T_max
BIG_M_WAIT       = T_max + SHIFT_START - e_min
BIG_M_LOAD_LB    = Q_max
BIG_M_LOAD_UB    = Q_max

log.info("Data loaded | |N|=%d |Nw|=%d |K|=%d |R|=%d |P|=%d | U=%d",
         len(N), len(Nw), len(K), len(R), len(P), U)
log.info("Big-M | time=%.1f precedence=%.1f wait=%.1f load_lb=%.0f load_ub=%.0f",
         BIG_M_TIME, BIG_M_PRECEDENCE, BIG_M_WAIT, BIG_M_LOAD_LB, BIG_M_LOAD_UB)


m = gp.Model(RUN_ID)

# Decision variables
x     = m.addVars([(i, j, k, r) for i in N for j in N for k in K for r in R if i != j],
                  vtype=GRB.BINARY, name="x")           # arc traversal
f     = m.addVars(P, K, R, vtype=GRB.BINARY, name="f")  # product-to-(vehicle, route) assignment
w     = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="w")                       # waiting time per product
y     = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="y")                 # vehicle load at node
ta    = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="ta")                # arrival time
td    = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="td")                # departure time
ts    = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="ts")               # service start time
u     = m.addVars(Nw, K, R, vtype=GRB.INTEGER, lb=0, ub=U, name="u")               # MTZ position
delta = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="delta")  # net load change at node


route_duration = quicksum(ta["h", k, R[-1]] for k in K) - len(K) * SHIFT_START
total_wait     = quicksum(w[p] for p in P)

m.setObjective(route_duration + WAIT_WEIGHT * total_wait, GRB.MINIMIZE)

log.info("Objective set | primary: min route_duration | tie-break weight on wait: %g",
         WAIT_WEIGHT)


# Constraint (3): epsilon-bound on total waiting time
m.addConstr(total_wait <= EPS_WAIT, name="c3_epsilon_wait")
log.info("Constraint (3) added | total_wait <= %g", EPS_WAIT)


# Constraints (4)-(9): route structure
for k in K:
    for r in R:
        out_h = quicksum(x["h", j, k, r] for j in Nw if ("h", j, k, r) in x)
        in_h  = quicksum(x[j, "h", k, r] for j in Nw if (j, "h", k, r) in x)

        m.addConstr(out_h == in_h,  name=f"c4[{k},{r}]")
        m.addConstr(out_h <= 1,     name=f"c5[{k},{r}]")

        m.addConstr(
            quicksum(x[i, j, k, r] for i in N for j in N if (i, j, k, r) in x)
            <= (2 * len(P) + 1) * quicksum(f[p, k, r] for p in P),
            name=f"c6[{k},{r}]"
        )

for j in Nw:
    for k in K:
        for r in R:
            inflow  = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[j, i, k, r] for i in N if i != j and (j, i, k, r) in x)

            m.addConstr(
                inflow <= quicksum(f[p, k, r] for p in P if o[p] == j or d[p] == j),
                name=f"c7[{j},{k},{r}]"
            )
            m.addConstr(inflow == outflow, name=f"c8[{j},{k},{r}]")
            m.addConstr(inflow <= 1,       name=f"c9[{j},{k},{r}]")

log.info("Constraints (4)-(9) added | route structure")


# Constraints (10)-(12): product assignment and pickup/delivery visits
for p in P:
    m.addConstr(
        quicksum(f[p, k, r] for k in K for r in R) == 1,
        name=f"c10[{p}]"
    )

    op = o[p]
    dp = d[p]

    if op in N:
        for k in K:
            for r in R:
                m.addConstr(
                    quicksum(x[i, op, k, r] for i in N if i != op and (i, op, k, r) in x)
                    >= f[p, k, r],
                    name=f"c11[{p},{k},{r}]"
                )

    if dp in N:
        for k in K:
            for r in R:
                m.addConstr(
                    quicksum(x[i, dp, k, r] for i in N if i != dp and (i, dp, k, r) in x)
                    >= f[p, k, r],
                    name=f"c12[{p},{k},{r}]"
                )

log.info("Constraints (10)-(12) added | product assignment")


# Constraints (13)-(15): route timing (shift start, inter-route precedence)
for k in K:
    m.addConstr(td["h", k, R[0]] == SHIFT_START, name=f"c13[{k}]")

for k in K:
    for r in R[1:]:
        m.addConstr(td["h", k, r] >= ta["h", k, r - 1], name=f"c14[{k},{r}]")
        m.addConstr(ta["h", k, r] >= ta["h", k, r - 1], name=f"c15[{k},{r}]")

log.info("Constraints (13)-(15) added | route timing")


# Constraint (16): time propagation along selected arcs (Big-M linearization)
for i in N:
    for j in N:
        if i == j:
            continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    c_ij = c.get((i, j))
                    if c_ij is None:
                        log.warning("Missing distance for (%s, %s) | using 0.0", i, j)
                        c_ij = 0.0
                    m.addConstr(
                        ta[j, k, r] >= td[i, k, r]
                                       + c_ij * x[i, j, k, r]
                                       - BIG_M_TIME * (1 - x[i, j, k, r]),
                        name=f"c16[{i},{j},{k},{r}]"
                    )

log.info("Constraints (16) added | time consistency")


# Constraints (17)-(19): service start/end times and ready-time linking
for j in Nw:
    for k in K:
        for r in R:
            unload = quicksum(su[p] * f[p, k, r] for p in P if d[p] == j)
            load   = quicksum(sl[p] * f[p, k, r] for p in P if o[p] == j)

            m.addConstr(ts[j, k, r] >= ta[j, k, r] + unload, name=f"c17[{j},{k},{r}]")
            m.addConstr(td[j, k, r] >= ts[j, k, r] + load,   name=f"c19[{j},{k},{r}]")

for p in P:
    op = o[p]
    if op in Nw:
        for k in K:
            for r in R:
                m.addConstr(
                    ts[op, k, r] >= e[p] * f[p, k, r],
                    name=f"c18[{p},{k},{r}]"
                )

log.info("Constraints (17)-(19) added | service times")


# Constraints (20)-(22): pickup-delivery precedence, route closure, waiting
for p in P:
    op = o[p]
    dp = d[p]

    if op in N and dp in N:
        for k in K:
            for r in R:
                m.addConstr(
                    ta[dp, k, r] >= td[op, k, r]
                                    - BIG_M_PRECEDENCE * (1 - f[p, k, r]),
                    name=f"c20[{p},{k},{r}]"
                )

for k in K:
    for r in R:
        m.addConstr(ta["h", k, r] >= td["h", k, r], name=f"c21[{k},{r}]")

for p in P:
    dp = d[p]
    ep = e[p]

    for k in K:
        for r in R:
            m.addConstr(
                w[p] >= ta[dp, k, r] + su[p] - ep
                        - BIG_M_WAIT * (1 - f[p, k, r]),
                name=f"c22[{p},{k},{r}]"
            )

log.info("Constraints (20)-(22) added | precedence, route closure, waiting")


# Constraints (23)-(27): load propagation, capacity, depot reset
for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(q_product[p] * f[p, k, r] for p in P if o[p] == j)
            load_out = quicksum(q_product[p] * f[p, k, r] for p in P if d[p] == j)

            m.addConstr(
                delta[j, k, r] >= load_in - load_out,
                name=f"c23[{j},{k},{r}]"
            )
            m.addConstr(y[j, k, r] <= q_vehicle[k], name=f"c26[{j},{k},{r}]")

for i in Nw:
    for j in Nw:
        if i == j:
            continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        y[j, k, r] >= y[i, k, r] + delta[j, k, r]
                                      - BIG_M_LOAD_LB * (1 - x[i, j, k, r]),
                        name=f"c24[{i},{j},{k},{r}]"
                    )
                    m.addConstr(
                        y[j, k, r] <= y[i, k, r] + delta[j, k, r]
                                      + BIG_M_LOAD_UB * (1 - x[i, j, k, r]),
                        name=f"c25[{i},{j},{k},{r}]"
                    )

for k in K:
    for r in R:
        m.addConstr(y["h", k, r] == 0, name=f"c27[{k},{r}]")

log.info("Constraints (23)-(27) added | load propagation")


# Constraints (28)-(32): inter-route ordering, MTZ subtour elimination, pickup-before-delivery
# (28) is the symmetry-breaking constraint for multi-route: route r+1 only used if route r is used
for k in K:
    for r in R[:-1]:
        out_h_curr = quicksum(x["h", j, k, r]     for j in Nw if ("h", j, k, r)     in x)
        out_h_next = quicksum(x["h", j, k, r + 1] for j in Nw if ("h", j, k, r + 1) in x)
        m.addConstr(out_h_curr >= out_h_next, name=f"c28[{k},{r}]")

for i in Nw:
    for j in Nw:
        if i == j:
            continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        u[j, k, r] >= u[i, k, r] + 1
                                      - U * (1 - x[i, j, k, r]),
                        name=f"c29[{i},{j},{k},{r}]"
                    )

for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)

            m.addConstr(u[j, k, r] <= U * indeg, name=f"c30[{j},{k},{r}]")
            m.addConstr(u[j, k, r] >= indeg,     name=f"c31[{j},{k},{r}]")

for p in P:
    op = o[p]
    dp = d[p]

    if op in Nw and dp in Nw:
        for k in K:
            for r in R:
                m.addConstr(
                    u[dp, k, r] >= u[op, k, r] + 1
                                   - U * (1 - f[p, k, r]),
                    name=f"c32[{p},{k},{r}]"
                )

log.info("Constraints (28)-(32) added | MTZ and route ordering")


excel_filename  = f"result_{RUN_ID}_{timestamp}.xlsx"
excel_full_path = os.path.join(OUTPUT_DIR, excel_filename)
gurobi_log_path = os.path.join(OUTPUT_DIR, f"result_{RUN_ID}_{timestamp}.log")

m.setParam("TimeLimit", TIME_LIMIT)
m.setParam("MIPGap", MIP_GAP)
m.setParam("Presolve", 2)
m.setParam("LogFile", gurobi_log_path)
m.update()

log.info("Solver configured | TimeLimit=%ds | MIPGap=%g | gurobi_log=%s",
         TIME_LIMIT, MIP_GAP, gurobi_log_path)
log.info("Optimization starting...")

m.optimize()


if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and m.SolCount > 0:

    route_duration_val = sum(ta["h", k, R[-1]].X for k in K) - len(K) * SHIFT_START
    total_wait_val     = sum(w[p].X for p in P)

    log.info("Solution found | route_duration=%.2f | total_wait=%.2f",
             route_duration_val, total_wait_val)
    log.info("Objective=%.2f | runtime=%.2fs | MIP Gap=%.2f%%",
             m.objVal, m.Runtime, m.MIPGap * 100)

    log.info("Generating Excel output | %s", excel_full_path)

    with pd.ExcelWriter(excel_full_path, engine="openpyxl") as writer:

        # Sheet 1: optimization_results
        df_opt = pd.DataFrame([{
            "run_id":                   RUN_ID,
            "model":                    MODEL_NAME,
            "case":                     CASE_NAME,
            "max_routes":               MAX_ROUTES,
            "epsilon_wait":             EPS_WAIT,
            "timestamp":                timestamp,
            "objective":                "min_route_duration_plus_lex_wait",
            "obj_value":                m.objVal,
            "best_bound":               m.ObjBound,
            "mip_gap":                  m.MIPGap,
            "runtime":                  m.Runtime,
            "status":                   m.status,
            "route_duration_minutes":   route_duration_val,
            "route_duration_stamp":     minutes_to_hhmm(route_duration_val),
            "total_wait_minutes":       total_wait_val,
            "wait_slack_remaining":     EPS_WAIT - total_wait_val,
            "wait_weight_in_objective": WAIT_WEIGHT,
            "|N|":                      len(N),
            "|Nw|":                     len(Nw),
            "|K|":                      len(K),
            "|R|":                      len(R),
            "|P|":                      len(P),
            "KxR":                      len(K) * len(R),
            "U_(|Nw|)":                 U,
            "BIG_M_TIME":               BIG_M_TIME,
            "BIG_M_PRECEDENCE":         BIG_M_PRECEDENCE,
            "BIG_M_WAIT":               BIG_M_WAIT,
            "BIG_M_LOAD_LB":            BIG_M_LOAD_LB,
            "BIG_M_LOAD_UB":            BIG_M_LOAD_UB,
        }])
        df_opt.to_excel(writer, sheet_name="optimization_results", index=False)

        # Sheet 2: x_ijkr
        df_x = pd.DataFrame([
            {"var": "x", "i": i, "j": j, "k": k, "r": r, "val": int(x[i, j, k, r].X)}
            for i in N for j in N for k in K for r in R
            if i != j and (i, j, k, r) in x and x[i, j, k, r].X > 0.5
        ])
        df_x.to_excel(writer, sheet_name="x_ijkr", index=False)

        # Sheet 3: f_pkr
        df_f = pd.DataFrame([
            {"var": "f", "p": p, "k": k, "r": r, "val": int(f[p, k, r].X)}
            for p in P for k in K for r in R
            if f[p, k, r].X > 0.5
        ])
        df_f.to_excel(writer, sheet_name="f_pkr", index=False)

        # Sheet 4: u_jkr
        df_u = pd.DataFrame([
            {"var": "u", "j": j, "k": k, "r": r, "u": u[j, k, r].X}
            for j in Nw for k in K for r in R
            if u[j, k, r].X > 0.01
        ])
        df_u.to_excel(writer, sheet_name="u_jkr", index=False)

        # Sheet 5: z_kr (derived: route used or not)
        df_z = pd.DataFrame([
            {
                "var": "z", "k": k, "r": r,
                "z": int(any(("h", j, k, r) in x and x["h", j, k, r].X > 0.5 for j in Nw))
            }
            for k in K for r in R
        ])
        df_z.to_excel(writer, sheet_name="z_kr", index=False)

        # Sheet 6: w_p
        df_w = pd.DataFrame([
            {"var": "w", "p": p, "w_val": w[p].X}
            for p in P
        ])
        df_w.to_excel(writer, sheet_name="w_p", index=False)

        # Sheet 7: ta (arrival times)
        df_ta = pd.DataFrame([
            {
                "var": "ta", "node": j, "k": k, "r": r,
                "time": ta[j, k, r].X, "stamp": minutes_to_hhmm(ta[j, k, r].X)
            }
            for j in N for k in K for r in R
        ])
        df_ta.to_excel(writer, sheet_name="ta", index=False)

        # Sheet 8: td (departure times)
        df_td = pd.DataFrame([
            {
                "var": "td", "node": j, "k": k, "r": r,
                "time": td[j, k, r].X, "stamp": minutes_to_hhmm(td[j, k, r].X)
            }
            for j in N for k in K for r in R
        ])
        df_td.to_excel(writer, sheet_name="td", index=False)

        # Sheet 9: ts (service start times)
        df_ts = pd.DataFrame([
            {
                "var": "ts", "node": j, "k": k, "r": r,
                "time": ts[j, k, r].X, "stamp": minutes_to_hhmm(ts[j, k, r].X)
            }
            for j in Nw for k in K for r in R
        ])
        df_ts.to_excel(writer, sheet_name="ts", index=False)

        # Sheet 10: y_jkr (vehicle load at node)
        df_y = pd.DataFrame([
            {"var": "y", "node": j, "k": k, "r": r, "y_val": y[j, k, r].X}
            for j in N for k in K for r in R
            if abs(y[j, k, r].X) > 1e-6
        ])
        df_y.to_excel(writer, sheet_name="y_jkr", index=False)

        # Sheet 11: delta_jkr (net load change)
        df_delta = pd.DataFrame([
            {"var": "delta", "node": j, "k": k, "r": r, "delta_val": delta[j, k, r].X}
            for j in Nw for k in K for r in R
            if abs(delta[j, k, r].X) > 1e-6
        ])
        df_delta.to_excel(writer, sheet_name="delta_jkr", index=False)

        # Sheet 12: itinerary (route sequence reconstruction)
        itinerary_list = []

        for k in K:
            for r in R:
                current = "h"
                visited = {"h"}
                order = 1

                itinerary_list.append({
                    "k": k, "r": r, "order": order, "node": "h", "u": None,
                    "ta":       ta["h", k, r].X,
                    "ta_stamp": minutes_to_hhmm(ta["h", k, r].X),
                    "td":       td["h", k, r].X,
                    "td_stamp": minutes_to_hhmm(td["h", k, r].X),
                    "y_after":  y["h", k, r].X,
                    "route_duration":       route_duration_val,
                    "route_duration_stamp": minutes_to_hhmm(route_duration_val),
                })

                while True:
                    next_node = next(
                        (j for j in N
                         if j != current
                         and (current, j, k, r) in x
                         and x[current, j, k, r].X > 0.5),
                        None
                    )

                    if next_node is None:
                        break

                    order += 1

                    itinerary_list.append({
                        "k": k, "r": r, "order": order, "node": next_node,
                        "u":        u[next_node, k, r].X if next_node in Nw else None,
                        "ta":       ta[next_node, k, r].X,
                        "ta_stamp": minutes_to_hhmm(ta[next_node, k, r].X),
                        "td":       td[next_node, k, r].X,
                        "td_stamp": minutes_to_hhmm(td[next_node, k, r].X),
                        "y_after":  y[next_node, k, r].X if next_node in N else None,
                        "route_duration":       route_duration_val,
                        "route_duration_stamp": minutes_to_hhmm(route_duration_val),
                    })

                    if next_node == "h":
                        break
                    if next_node in visited:
                        break

                    visited.add(next_node)
                    current = next_node

        df_itinerary = pd.DataFrame(itinerary_list)
        df_itinerary.to_excel(writer, sheet_name="itinerary", index=False)

    log.info("Excel output written | %d sheets | %s", 12, excel_full_path)


    master_pareto_path = os.path.join(RESULTS_ROOT, "master_pareto.xlsx")

    new_row = pd.DataFrame([{
        "run_id":               RUN_ID,
        "model":                MODEL_NAME,
        "case":                 CASE_NAME,
        "max_routes":           MAX_ROUTES,
        "epsilon_wait":         EPS_WAIT,
        "route_duration":       route_duration_val,
        "total_wait":           total_wait_val,
        "obj_value":            m.objVal,
        "best_bound":           m.ObjBound,
        "mip_gap":              m.MIPGap,
        "runtime":              m.Runtime,
        "status":               m.status,
        "n_products":           len(P),
        "n_vehicles":           len(K),
        "timestamp":            timestamp,
    }])

    if os.path.exists(master_pareto_path):
        existing = pd.read_excel(master_pareto_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_excel(master_pareto_path, index=False)
    log.info("Master Pareto updated | %d total rows | %s", len(combined), master_pareto_path)

elif m.status == GRB.INFEASIBLE:
    log.error("Model is INFEASIBLE — computing IIS")
    m.computeIIS()
    iis_file = os.path.join(OUTPUT_DIR, f"infeasible_{RUN_ID}_{timestamp}.ilp")
    m.write(iis_file)
    log.error("IIS written | %s", iis_file)

else:
    log.warning("No solution found | status=%d", m.status)

log.info("Run complete | run_id=%s", RUN_ID)
