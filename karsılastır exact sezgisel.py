import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
import os
from datetime import datetime, time

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = r"C:\Users\Asus\Desktop\Er\\"
OUTPUT_DIR = r"C:\Users\Asus\Documents\GitHub\logistics-model2\internal-logistics-model2\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_LIMIT = 3600
MIP_GAP = 0.01
SHIFT_START = 0
RETURN_LIMIT = 9999

USE_FIRST_VEHICLE_ONLY = True
MAX_ROUTES_OVERRIDE = 1

PRODUCT_FILES = [
    "products_4part.xlsx",
    "products_5part.xlsx",
    "products_6part.xlsx",
    "products_10part.xlsx",
]

CASE_SHEETS = [
    "Case1_AllDistinct",
    "Case2_LOC-C",
    "Case3_SharedPickup",
]

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
out_path = os.path.join(OUTPUT_DIR, f"exact_vs_heuristic_12cases_{timestamp}.xlsx")


# ============================================================
# HELPERS
# ============================================================
def ready_to_min(v):
    if pd.isna(v):
        return 0
    if isinstance(v, (pd.Timestamp, datetime)):
        return int(v.hour) * 60 + int(v.minute)
    if isinstance(v, time):
        return int(v.hour) * 60 + int(v.minute)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)

    s = str(v).strip()
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return int(dt.hour) * 60 + int(dt.minute)

    if ":" in s:
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm)

    return int(float(s))


def minutes_to_hhmm(x):
    if x is None or pd.isna(x):
        return ""
    x = float(x)
    return f"{int(x // 60):02d}:{int(x % 60):02d}"


def safe_gap(model):
    try:
        return model.MIPGap
    except Exception:
        return None


def read_dist(path, val_col):
    df = pd.read_excel(path)
    df["from_node"] = df["from_node"].astype(str).str.strip()
    df["to_node"] = df["to_node"].astype(str).str.strip()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=["from_node", "to_node", val_col])
    df = df[df["from_node"] != df["to_node"]]
    return {
        (r["from_node"], r["to_node"]): float(r[val_col])
        for _, r in df.iterrows()
    }


# ============================================================
# COMMON DATA
# ============================================================
nodes = pd.read_excel(os.path.join(DATA_PATH, "nodes.xlsx"))
vehicles = pd.read_excel(os.path.join(DATA_PATH, "vehicles.xlsx"))
c = read_dist(os.path.join(DATA_PATH, "distances - dakika.xlsx"), "duration_min")

nodes["node_id"] = nodes["node_id"].astype(str).str.strip()
N = nodes["node_id"].dropna().drop_duplicates().tolist()
Nw = [n for n in N if n != "h"]
A = [(i, j) for (i, j) in c if i in N and j in N and i != j]

vehicles["vehicle_id"] = vehicles["vehicle_id"].astype(str).str.strip()
vehicles = vehicles.dropna(subset=["vehicle_id"]).drop_duplicates("vehicle_id")

if USE_FIRST_VEHICLE_ONLY:
    vehicles = vehicles.head(1)

K = vehicles["vehicle_id"].tolist()
q_vehicle = dict(zip(K, vehicles["capacity_m2"]))

if MAX_ROUTES_OVERRIDE is not None:
    MAX_ROUTES = int(MAX_ROUTES_OVERRIDE)
else:
    MAX_ROUTES = int(vehicles["max_routes"].max()) if "max_routes" in vehicles.columns else 1

R = list(range(1, MAX_ROUTES + 1))

T_max = 480
C_max = 11
e_min = 420
Q_max = 20

U = len(Nw)
M_16 = T_max + C_max
M_20 = T_max
M_22 = T_max + SHIFT_START - e_min
M_24 = Q_max
M_25 = Q_max


# ============================================================
# PRODUCT READER
# ============================================================
def load_products(product_file, sheet_name):
    path = os.path.join(DATA_PATH, product_file)
    df = pd.read_excel(path, sheet_name=sheet_name)

    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["origin"] = df["origin"].astype(str).str.strip()
    df["destination"] = df["destination"].astype(str).str.strip()
    df = df.dropna(subset=["product_id"]).drop_duplicates("product_id")

    P = df["product_id"].tolist()
    e = dict(zip(P, [ready_to_min(v) for v in df["ready_time"]]))
    sl = dict(zip(P, df["load_time"]))
    su = dict(zip(P, df["unload_time"]))
    q_product = dict(zip(P, df["area_m2"]))
    o = dict(zip(P, df["origin"]))
    d = dict(zip(P, df["destination"]))

    return df, P, e, sl, su, q_product, o, d


# ============================================================
# EXACT MODEL
# ============================================================
def solve_exact(product_file, case_name, P, e, sl, su, q_product, o, d):
    m = gp.Model(f"exact_{product_file}_{case_name}")

    x = m.addVars([(i, j, k, r) for (i, j) in A for k in K for r in R],
                  vtype=GRB.BINARY, name="x")
    f = m.addVars(P, K, R, vtype=GRB.BINARY, name="f")
    w = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
    y = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
    ta = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="ta")
    td = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="td")
    ts = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="ts")
    u = m.addVars(Nw, K, R, vtype=GRB.INTEGER, lb=0, ub=U, name="u")
    delta = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="delta")
    t_return = m.addVars(K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="t_return")

    obj1 = quicksum(t_return[k, r] - SHIFT_START for k in K for r in R)
    obj2 = quicksum(w[p] for p in P)
    m.setObjective(obj1 + 0.001 * obj2, GRB.MINIMIZE)

    for k in K:
        for r in R:
            out_h = quicksum(x["h", j, k, r] for j in Nw if ("h", j, k, r) in x)
            in_h = quicksum(x[j, "h", k, r] for j in Nw if (j, "h", k, r) in x)

            m.addConstr(out_h == in_h)
            m.addConstr(out_h <= 1)
            m.addConstr(t_return[k, r] <= RETURN_LIMIT)
            m.addConstr(td["h", k, r] == SHIFT_START if r == 1 else td["h", k, r] >= t_return[k, r - 1])
            m.addConstr(t_return[k, r] >= td["h", k, r])

            m.addConstr(
                quicksum(x[i, j, k, r] for (i, j) in A if (i, j, k, r) in x)
                <= (2 * len(P) + 1) * quicksum(f[p, k, r] for p in P)
            )

    for j in Nw:
        for k in K:
            for r in R:
                inflow = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
                outflow = quicksum(x[j, i, k, r] for i in N if i != j and (j, i, k, r) in x)

                m.addConstr(inflow <= quicksum(f[p, k, r] for p in P if o[p] == j or d[p] == j))
                m.addConstr(inflow == outflow)
                m.addConstr(inflow <= 1)

    for p in P:
        m.addConstr(quicksum(f[p, k, r] for k in K for r in R) == 1)

        for k in K:
            for r in R:
                if o[p] in N:
                    m.addConstr(
                        quicksum(x[i, o[p], k, r] for i in N if i != o[p] and (i, o[p], k, r) in x)
                        >= f[p, k, r]
                    )

                if d[p] in N:
                    m.addConstr(
                        quicksum(x[i, d[p], k, r] for i in N if i != d[p] and (i, d[p], k, r) in x)
                        >= f[p, k, r]
                    )

    for i in N:
        for j in N:
            if i == j:
                continue
            for k in K:
                for r in R:
                    if (i, j, k, r) in x:
                        m.addConstr(
                            ta[j, k, r] >= td[i, k, r] + c.get((i, j), 0.0) - M_16 * (1 - x[i, j, k, r])
                        )

    for i in Nw:
        for k in K:
            for r in R:
                if (i, "h", k, r) in x:
                    m.addConstr(
                        t_return[k, r] >= td[i, k, r] + c.get((i, "h"), 0.0) - M_16 * (1 - x[i, "h", k, r])
                    )

    for j in Nw:
        for k in K:
            for r in R:
                m.addConstr(ts[j, k, r] >= ta[j, k, r] + quicksum(su[p] * f[p, k, r] for p in P if d[p] == j))
                m.addConstr(td[j, k, r] >= ts[j, k, r] + quicksum(sl[p] * f[p, k, r] for p in P if o[p] == j))
                m.addConstr(y[j, k, r] <= q_vehicle[k])

    for p in P:
        for k in K:
            for r in R:
                if o[p] in Nw:
                    m.addConstr(ts[o[p], k, r] >= e[p] * f[p, k, r])

                if o[p] in N and d[p] in N:
                    m.addConstr(ta[d[p], k, r] >= td[o[p], k, r] - M_20 * (1 - f[p, k, r]))

                m.addConstr(w[p] >= ta[d[p], k, r] + su[p] - e[p] - M_22 * (1 - f[p, k, r]))

    for j in Nw:
        for k in K:
            for r in R:
                m.addConstr(
                    delta[j, k, r]
                    ==
                    quicksum(q_product[p] * f[p, k, r] for p in P if o[p] == j)
                    -
                    quicksum(q_product[p] * f[p, k, r] for p in P if d[p] == j)
                )

    for k in K:
        for r in R:
            m.addConstr(y["h", k, r] == 0)

    for i in Nw:
        for j in Nw:
            if i == j:
                continue
            for k in K:
                for r in R:
                    if (i, j, k, r) in x:
                        m.addConstr(y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M_24 * (1 - x[i, j, k, r]))
                        m.addConstr(y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M_25 * (1 - x[i, j, k, r]))
                        m.addConstr(u[j, k, r] >= u[i, k, r] + 1 - U * (1 - x[i, j, k, r]))

    for k in K:
        for r in R[:-1]:
            m.addConstr(
                quicksum(x["h", j, k, r] for j in Nw if ("h", j, k, r) in x)
                >=
                quicksum(x["h", j, k, r + 1] for j in Nw if ("h", j, k, r + 1) in x)
            )

    for j in Nw:
        for k in K:
            for r in R:
                indeg = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
                m.addConstr(u[j, k, r] <= U * indeg)
                m.addConstr(u[j, k, r] >= indeg)

    for p in P:
        if o[p] in Nw and d[p] in Nw:
            for k in K:
                for r in R:
                    m.addConstr(u[d[p], k, r] >= u[o[p], k, r] + 1 - U * (1 - f[p, k, r]))

    m.setParam("TimeLimit", TIME_LIMIT)
    m.setParam("MIPGap", MIP_GAP)
    m.setParam("OutputFlag", 0)
    m.optimize()

    if m.SolCount == 0:
        return {
            "status": m.status,
            "f1": None,
            "f2": None,
            "runtime": m.Runtime,
            "gap": None,
            "itinerary": pd.DataFrame()
        }

    f1 = sum(t_return[k, r].X - SHIFT_START for k in K for r in R)
    f2 = sum(w[p].X for p in P)

    rows = []
    for k in K:
        for r in R:
            cur = "h"
            visited = {"h"}
            order = 1

            rows.append({
                "method": "exact",
                "file": product_file,
                "case": case_name,
                "k": k,
                "r": r,
                "order": order,
                "node": "h",
                "ta": ta["h", k, r].X,
                "td": td["h", k, r].X,
                "stamp": minutes_to_hhmm(td["h", k, r].X),
                "t_return": t_return[k, r].X
            })

            while True:
                nxt = next(
                    (
                        j for j in N
                        if j != cur and (cur, j, k, r) in x and x[cur, j, k, r].X > 0.5
                    ),
                    None
                )

                if nxt is None:
                    break

                order += 1
                rows.append({
                    "method": "exact",
                    "file": product_file,
                    "case": case_name,
                    "k": k,
                    "r": r,
                    "order": order,
                    "node": nxt,
                    "ta": ta[nxt, k, r].X,
                    "td": td[nxt, k, r].X,
                    "stamp": minutes_to_hhmm(ta[nxt, k, r].X),
                    "t_return": t_return[k, r].X
                })

                if nxt == "h" or nxt in visited:
                    break

                visited.add(nxt)
                cur = nxt

    return {
        "status": m.status,
        "f1": f1,
        "f2": f2,
        "runtime": m.Runtime,
        "gap": safe_gap(m),
        "itinerary": pd.DataFrame(rows)
    }


# ============================================================
# HEURISTIC
# ============================================================
def solve_heuristic(product_file, case_name, P, e, sl, su, q_product, o, d):
    k = K[0]
    r = R[0]

    current = "h"
    current_time = SHIFT_START
    load = 0.0

    unpicked = set(P)
    onboard = set()
    delivered = set()
    visited_nodes = {"h"}

    rows = []
    order = 1
    total_wait = 0.0
    feasible = True

    rows.append({
        "method": "heuristic",
        "file": product_file,
        "case": case_name,
        "k": k,
        "r": r,
        "order": order,
        "node": "h",
        "operation": "start",
        "time": current_time,
        "stamp": minutes_to_hhmm(current_time),
        "load_after": load
    })

    while len(delivered) < len(P):
        candidate_nodes = set()

        for p in unpicked:
            if o[p] not in visited_nodes and o[p] in Nw:
                candidate_nodes.add(o[p])

        for p in onboard:
            if d[p] not in visited_nodes and d[p] in Nw:
                candidate_nodes.add(d[p])

        if not candidate_nodes:
            feasible = False
            break

        best_node = None
        best_score = float("inf")

        for node in candidate_nodes:
            travel = c.get((current, node), 999999)
            arrival = current_time + travel

            pickups_here = [p for p in unpicked if o[p] == node]
            deliveries_here = [p for p in onboard if d[p] == node]

            cap_needed = sum(q_product[p] for p in pickups_here)
            if load + cap_needed > q_vehicle[k]:
                continue

            ready_wait = 0.0
            if pickups_here:
                ready_wait = max(0.0, max(e[p] for p in pickups_here) - arrival)

            score = travel + ready_wait

            if score < best_score:
                best_score = score
                best_node = node

        if best_node is None:
            feasible = False
            break

        travel = c.get((current, best_node), 999999)
        arrival = current_time + travel

        deliveries_here = [p for p in list(onboard) if d[p] == best_node]
        unload_time = sum(su[p] for p in deliveries_here)

        for p in deliveries_here:
            total_wait += max(0.0, arrival + su[p] - e[p])
            onboard.remove(p)
            delivered.add(p)
            load -= q_product[p]

        service_start = arrival + unload_time

        pickups_here = [p for p in list(unpicked) if o[p] == best_node]
        if pickups_here:
            service_start = max(service_start, max(e[p] for p in pickups_here))

        load_time = sum(sl[p] for p in pickups_here)

        for p in pickups_here:
            unpicked.remove(p)
            onboard.add(p)
            load += q_product[p]

        depart = service_start + load_time

        order += 1
        rows.append({
            "method": "heuristic",
            "file": product_file,
            "case": case_name,
            "k": k,
            "r": r,
            "order": order,
            "node": best_node,
            "operation": "delivery+pickup" if deliveries_here and pickups_here else ("delivery" if deliveries_here else "pickup"),
            "delivered_products": ",".join(deliveries_here),
            "picked_products": ",".join(pickups_here),
            "arrival": arrival,
            "arrival_stamp": minutes_to_hhmm(arrival),
            "service_start": service_start,
            "service_start_stamp": minutes_to_hhmm(service_start),
            "departure": depart,
            "departure_stamp": minutes_to_hhmm(depart),
            "load_after": load
        })

        visited_nodes.add(best_node)
        current = best_node
        current_time = depart

    return_travel = c.get((current, "h"), 999999)
    t_return = current_time + return_travel

    order += 1
    rows.append({
        "method": "heuristic",
        "file": product_file,
        "case": case_name,
        "k": k,
        "r": r,
        "order": order,
        "node": "h",
        "operation": "return",
        "arrival": t_return,
        "arrival_stamp": minutes_to_hhmm(t_return),
        "load_after": load
    })

    if len(delivered) < len(P) or abs(load) > 1e-6:
        feasible = False

    return {
        "status": "FEASIBLE" if feasible else "INCOMPLETE",
        "f1": t_return - SHIFT_START,
        "f2": total_wait,
        "runtime": None,
        "gap": None,
        "itinerary": pd.DataFrame(rows),
        "served": len(delivered),
        "total": len(P)
    }


# ============================================================
# RUN 12 CASES
# ============================================================
summary_rows = []
exact_itins = []
heur_itins = []

for product_file in PRODUCT_FILES:
    for case_name in CASE_SHEETS:
        print(f"\nRUNNING: {product_file} | {case_name}")

        try:
            _, P, e, sl, su, q_product, o, d = load_products(product_file, case_name)

            exact = solve_exact(product_file, case_name, P, e, sl, su, q_product, o, d)
            heur = solve_heuristic(product_file, case_name, P, e, sl, su, q_product, o, d)

            exact_f1 = exact["f1"]
            exact_f2 = exact["f2"]
            heur_f1 = heur["f1"]
            heur_f2 = heur["f2"]

            gap_f1 = None if exact_f1 in [None, 0] else 100 * (heur_f1 - exact_f1) / exact_f1
            gap_f2 = None if exact_f2 in [None, 0] else 100 * (heur_f2 - exact_f2) / exact_f2

            summary_rows.append({
                "file": product_file,
                "case": case_name,
                "num_parts": len(P),

                "exact_status": exact["status"],
                "exact_f1_return": exact_f1,
                "exact_f2_wait": exact_f2,
                "exact_runtime_s": exact["runtime"],
                "exact_gap": exact["gap"],

                "heuristic_status": heur["status"],
                "heuristic_f1_return": heur_f1,
                "heuristic_f2_wait": heur_f2,
                "heuristic_served": heur["served"],
                "heuristic_total": heur["total"],

                "heuristic_gap_to_exact_f1_pct": gap_f1,
                "heuristic_gap_to_exact_f2_pct": gap_f2
            })

            if not exact["itinerary"].empty:
                exact_itins.append(exact["itinerary"])

            if not heur["itinerary"].empty:
                heur_itins.append(heur["itinerary"])

        except Exception as ex:
            summary_rows.append({
                "file": product_file,
                "case": case_name,
                "error": str(ex)
            })
            print(f"ERROR: {product_file} | {case_name} -> {ex}")


df_summary = pd.DataFrame(summary_rows)
df_exact_itin = pd.concat(exact_itins, ignore_index=True) if exact_itins else pd.DataFrame()
df_heur_itin = pd.concat(heur_itins, ignore_index=True) if heur_itins else pd.DataFrame()

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="summary_12_cases", index=False)
    df_exact_itin.to_excel(writer, sheet_name="exact_itinerary", index=False)
    df_heur_itin.to_excel(writer, sheet_name="heuristic_itinerary", index=False)

print("\n" + "=" * 80)
print("BATCH COMPLETE")
print(f"Output: {out_path}")
print("=" * 80)

try:
    os.system(f'explorer /select,"{os.path.abspath(out_path)}"')
except Exception:
    pass