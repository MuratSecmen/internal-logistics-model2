import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import re, os
from datetime import datetime, time, timedelta
import sys

# =====================================================================
# TERMINAL ÇIKTISINI DOSYAYA KAYDET
# =====================================================================
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
terminal_log_path = os.path.join(r"C:\Users\Asus\Desktop", f"terminal_output_{timestamp}.txt")
terminal_log_file = open(terminal_log_path, 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = TeeOutput(original_stdout, terminal_log_file)

print(f"Terminal çıktısı kaydediliyor: {terminal_log_path}\n")

# =====================================================================
# PARAMETERS
# =====================================================================
TIME_LIMIT = 600
MIP_GAP    = 0.03
THREADS    = 6
EPS_WAIT   = 150

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def ready_to_min(v):
    if pd.isna(v): return 0
    if isinstance(v, (pd.Timestamp, datetime)): return int(v.hour)*60 + int(v.minute)
    if isinstance(v, time): return int(v.hour)*60 + int(v.minute)
    if isinstance(v, (int, float)) and not isinstance(v, bool): return int(v)
    s = str(v).strip()
    dt = pd.to_datetime(s, errors='coerce')
    if pd.notna(dt): return int(dt.hour)*60 + int(dt.minute)
    if ":" in s:
        hh, mm = s.split(":")
        return int(hh)*60 + int(mm)
    return int(float(s))

def minutes_to_hhmm(minutes):
    if pd.isna(minutes) or minutes is None:
        return ''
    minutes = float(minutes)
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

# =====================================================================
# DATA LOADING
# =====================================================================
data_path   = r"C:\Users\Asus\Desktop\Er\\"
desktop_dir = r"C:\Users\Asus\Desktop"

nodes    = pd.read_excel(os.path.join(data_path, "nodes.xlsx"))
vehicles = pd.read_excel(os.path.join(data_path, "vehicles.xlsx"))
products = pd.read_excel(os.path.join(data_path, "products.xlsx")).head(50)

def _read_dist(path, val_col):
    df = pd.read_excel(path, sheet_name=0)
    need = ['from_node', 'to_node', val_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{os.path.basename(path)} dosyasında eksik: {miss}")
    df['from_node'] = df['from_node'].astype(str).str.strip()
    df['to_node']   = df['to_node'].astype(str).str.strip()
    df = df.dropna(subset=['from_node', 'to_node', val_col])
    df = df[df['from_node'] != df['to_node']]
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=[val_col])
    return {(r['from_node'], r['to_node']): float(r[val_col]) for _, r in df.iterrows()}

c = _read_dist(os.path.join(data_path, "distances - dakika.xlsx"), "duration_min")

# =====================================================================
# SETS & PARAMETERS
# =====================================================================
nodes['node_id'] = nodes['node_id'].astype(str).str.strip()
N  = nodes['node_id'].dropna().drop_duplicates().tolist()
Nw = [n for n in N if n != 'h']

vehicles['vehicle_id'] = vehicles['vehicle_id'].astype(str).str.strip()
vehicles = vehicles.dropna(subset=['vehicle_id']).drop_duplicates('vehicle_id', keep='first')
K = vehicles['vehicle_id'].tolist()
q_vehicle = dict(zip(K, vehicles['capacity_m2']))

MAX_ROUTES = int(vehicles['max_routes'].max()) if 'max_routes' in vehicles.columns else 5
R = list(range(1, MAX_ROUTES + 1))

products['product_id']  = products['product_id'].astype(str).str.strip()
products['origin']      = products['origin'].astype(str).str.strip()
products['destination'] = products['destination'].astype(str).str.strip()
products = products.dropna(subset=['product_id']).drop_duplicates('product_id', keep='first')

P = products['product_id'].tolist()

e  = dict(zip(P, [ready_to_min(v) for v in products['ready_time']]))
sl = dict(zip(P, products['load_time']))
su = dict(zip(P, products['unload_time']))
q_product = dict(zip(P, products['area_m2']))
o  = dict(zip(P, products['origin']))
d  = dict(zip(P, products['destination']))

T_max = 480
C_max = 11
e_min = 435
Q_max = 20
# epsilon = 0.1
U = len(Nw)

M_16 = T_max - e_min + C_max
M_20 = T_max - e_min
M_22 = T_max
M_24 = Q_max
M_25 = Q_max

print("\n" + "="*80)
print("DATA LOADED")
print("="*80)
print(f"|N|={len(N)}, |Nw|={len(Nw)}, |K|={len(K)}, |R|={len(R)}, |P|={len(P)}")
print(f"TIGHT BIG-M:")
print(f"  M_16={M_16:.1f}  M_20={M_20:.1f}  M_22={M_22:.1f}")
print(f"  M_24={M_24:.0f}  M_25={M_25:.0f}  U={U}")
print("="*80 + "\n")

# =====================================================================
# MODEL CREATION
# =====================================================================
m = gp.Model("Internal_Logistics_Model_Constraints_c3_to_c36")

# =====================================================================
# DECISION VARIABLES
# =====================================================================
x = m.addVars([(i, j, k, r) for i in N for j in N for k in K for r in R 
               if i != j], vtype=GRB.BINARY, name="x")
f = m.addVars(P, K, R, vtype=GRB.BINARY, name="f")
w = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
y = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
ta = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="ta")
td = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="td")
ts = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="ts")
u = m.addVars(Nw, K, R, vtype=GRB.INTEGER, lb=0, ub=U, name="u")
delta = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="delta")

# =====================================================================
# OBJECTIVE FUNCTIONS (1)-(2)
# =====================================================================
obj1 = quicksum(ta['h', k, MAX_ROUTES] for k in K)
obj2 = quicksum(w[p] for p in P)
m.setObjective(obj1 + 0.001*obj2, GRB.MINIMIZE)

print("="*80)
print("OBJECTIVE FUNCTIONS")
print("="*80)
print("(1) Primary: min Σ_k ta_h,k,|R|  (total arrival times)")
print("(2) Secondary: min Σ_p w_p  (total waiting time)")
print("    Weighted: 0.001 × secondary")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (3): EPSILON-CONSTRAINT ON WAITING TIME
# =====================================================================
print("="*80)
print("CONSTRAINT (3): EPSILON-CONSTRAINT")
print("="*80)
m.addConstr(quicksum(w[p] for p in P) <= EPS_WAIT, name="c3")
print(f"(3) Σ_p w_p <= {EPS_WAIT}")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINTS (4)-(9): ROUTE STRUCTURE
# =====================================================================
print("="*80)
print("CONSTRAINTS (4)-(9): ROUTE STRUCTURE")
print("="*80)

# (4) Route closure: out-flow = in-flow
for k in K:
    for r in R:
        out_h = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        in_h  = quicksum(x[(j, 'h', k, r)] for j in Nw if (j, 'h', k, r) in x)
        m.addConstr(out_h == in_h, name=f"c4[{k},{r}]")

print("(4) Route closure: Σ_j x_hj = Σ_j x_jh  (k,r)")

# (5) At most 1 departure per route
for k in K:
    for r in R:
        out_h = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        m.addConstr(out_h <= 1, name=f"c5[{k},{r}]")

print("(5) At most 1 departure: Σ_j x_hj <= 1  (k,r)")

# (6) Route activation: routing justified by assignments
for k in K:
    for r in R:
        lhs = quicksum(x[(i, j, k, r)] for i in N for j in N if (i, j, k, r) in x)
        rhs = quicksum(f[p, k, r] for p in P)
        m.addConstr(lhs <= (2*len(P)+1)*rhs, name=f"c6[{k},{r}]")

print("(6) Route activation: Σ_i,j x_ij <= (2|P|+1) Σ_p f_p  (k,r)")

# (7) Out-flow coupling: visit only if product origin/destination
for j in Nw:
    for k in K:
        for r in R:
            lhs = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            rhs = quicksum(f[p, k, r] for p in P if (o[p] == j or d[p] == j))
            m.addConstr(lhs <= rhs, name=f"c7[{j},{k},{r}]")

print("(7) Out-flow coupling: Σ_i x_ij <= Σ_p I(o_p=j ∨ d_p=j) f_p  (j,k,r)")

# (8) Flow conservation
for j in Nw:
    for k in K:
        for r in R:
            inflow  = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[(j, i, k, r)] for i in N if i != j and (j, i, k, r) in x)
            m.addConstr(inflow == outflow, name=f"c8[{j},{k},{r}]")

print("(8) Flow conservation: Σ_i x_ij = Σ_i x_ji  (j,k,r)")

# (9) At most once per station per route
for j in Nw:
    for k in K:
        for r in R:
            inflow  = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            m.addConstr(inflow <= 1, name=f"c9[{j},{k},{r}]")

print("(9) At most once: Σ_i x_ij <= 1  (j,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINTS (10)-(12): PRODUCT ASSIGNMENT & PICKUP/DELIVERY
# =====================================================================
print("="*80)
print("CONSTRAINTS (10)-(12): PRODUCT ASSIGNMENT")
print("="*80)

# (10) Each product assigned exactly once
for p in P:
    m.addConstr(quicksum(f[p, k, r] for k in K for r in R) == 1, name=f"c10[{p}]")

print("(10) Assignment: Σ_k,r f_p = 1  (p)")

# (11) Pickup node must be visited
for p in P:
    op = o[p]
    if op in N:
        for k in K:
            for r in R:
                lhs = quicksum(x[(i, op, k, r)] for i in N if i != op and (i, op, k, r) in x)
                m.addConstr(lhs >= f[p, k, r], name=f"c11[{p},{k},{r}]")

print("(11) Pickup visit: Σ_i x_i,o_p >= f_p  (p,k,r)")

# (12) Delivery node must be visited
for p in P:
    dp = d[p]
    if dp in N:
        for k in K:
            for r in R:
                lhs = quicksum(x[(i, dp, k, r)] for i in N if i != dp and (i, dp, k, r) in x)
                m.addConstr(lhs >= f[p, k, r], name=f"c12[{p},{k},{r}]")

print("(12) Delivery visit: Σ_i x_i,d_p >= f_p  (p,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINTS (13)-(15): ROUTE TIMING
# =====================================================================
print("="*80)
print("CONSTRAINTS (13)-(15): ROUTE TIMING")
print("="*80)

# (13) First departure
for k in K:
    m.addConstr(td['h', k, 1] == 420, name=f"c13[{k}]")

print("(13) First departure: td_h,k,1 = 0  (k)")

# (14) Successive route departure >= previous arrival
for k in K:
    for r in R[1:]:
        m.addConstr(td['h', k, r] >= ta['h', k, r-1], name=f"c14[{k},{r}]")

print("(14) Route sequence (departure): td_h,k,r >= ta_h,k,r-1 + ε  (k,r>1)")

# (15) Arrival time monotonicity
for k in K:
    for r in R[1:]:
        m.addConstr(ta['h', k, r] >= ta['h', k, r-1], name=f"c15[{k},{r}]")

print("(15) Arrival monotonicity: ta_h,k,r >= ta_h,k,r-1  (k,r>1)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (16): TIME CONSISTENCY (TIGHT BIG-M = M_16 = 56) 🔥
# =====================================================================
print("="*80)
print("CONSTRAINT (16): TIME CONSISTENCY (TIGHT BIG-M)")
print("="*80)

for i in N:
    for j in N:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    c_ij = c.get((i, j), 0.0)
                    m.addConstr(
                        ta[j, k, r] >= td[i, k, r] + c_ij * x[(i, j, k, r)] - M_16 * (1 - x[(i, j, k, r)]),
                        name=f"c16[{i},{j},{k},{r}]"
                    )

print(f"(16) Time consistency: ta_j >= td_i + c_ij·x_ij - {M_16}·(1-x_ij)  (i,j≠i,k,r)")
print(f"     TIGHT BIG-M M_16 = T_max - e_min + C_max = {M_16}")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (17): SERVICE START
# =====================================================================
print("="*80)
print("CONSTRAINT (17): SERVICE START (UNLOADING)")
print("="*80)

for j in Nw:
    for k in K:
        for r in R:
            unload = quicksum(su[p] * f[p, k, r] for p in P if d[p] == j)
            m.addConstr(ts[j, k, r] >= ta[j, k, r] + unload, name=f"c17[{j},{k},{r}]")

print("(17) Service start: ts_j >= ta_j + Σ_p s_u_p·f_p  (j,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (18): READY TIME
# =====================================================================
print("="*80)
print("CONSTRAINT (18): READY TIME")
print("="*80)

for p in P:
    op = o[p]
    if op in Nw:
        ep = e[p]
        for k in K:
            for r in R:
                m.addConstr(ts[op, k, r] >= ep * f[p, k, r], name=f"c18[{p},{k},{r}]")

print("(18) Ready time: ts_o_p >= e_p·f_p  (p,k,r where o_p∈Nw)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (19): DEPARTURE TIME (LOADING)
# =====================================================================
print("="*80)
print("CONSTRAINT (19): DEPARTURE TIME (LOADING)")
print("="*80)

for j in Nw:
    for k in K:
        for r in R:
            load = quicksum(sl[p] * f[p, k, r] for p in P if o[p] == j)
            m.addConstr(td[j, k, r] >= ts[j, k, r] + load, name=f"c19[{j},{k},{r}]")

print("(19) Departure time: td_j >= ts_j + Σ_p s_l_p·f_p  (j,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (20): PICKUP-DELIVERY PRECEDENCE
# =====================================================================
print("="*80)
print("CONSTRAINT (20): PICKUP-DELIVERY PRECEDENCE (TIGHT BIG-M)")
print("="*80)

for p in P:
    op, dp = o[p], d[p]
    if (op in N) and (dp in N):
        for k in K:
            for r in R:
                m.addConstr(
                    ta[dp, k, r] >= td[op, k, r] - M_20 * (1 - f[p, k, r]),
                    name=f"c20[{p},{k},{r}]"
                )

print(f"(20) Pickup-delivery: ta_d_p >= td_o_p - {M_20}·(1-f_p)  (p,k,r)")
print(f"     TIGHT BIG-M M_20 = T_max - e_min = {M_20}")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (21): DEPOT ARRIVAL >= DEPARTURE
# =====================================================================
print("="*80)
print("CONSTRAINT (21): DEPOT TIMING")
print("="*80)

for k in K:
    for r in R:
        m.addConstr(ta['h', k, r] >= td['h', k, r], name=f"c21[{k},{r}]")

print("(21) Depot arrival: ta_h >= td_h  (k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (22): WAITING TIME DEFINITION
# =====================================================================
print("="*80)
print("CONSTRAINT (22): WAITING TIME DEFINITION (TIGHT BIG-M)")
print("="*80)

for p in P:
    dp = d[p]
    ep = e[p]
    for k in K:
        for r in R:
            m.addConstr(
                w[p] >= ta[dp, k, r] - ep - M_22 * (1 - f[p, k, r]),
                name=f"c22[{p},{k},{r}]"
            )

print(f"(22) Waiting time: w_p >= ta_d_p - e_p - {M_22}·(1-f_p)  (p,k,r)")
print(f"     TIGHT BIG-M M_22 = T_max = {M_22}")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (23): LOAD CHANGE
# =====================================================================
print("="*80)
print("CONSTRAINT (23): LOAD CHANGE DEFINITION")
print("="*80)

for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(q_product[p] * f[p, k, r] for p in P if o[p] == j)
            load_out = quicksum(q_product[p] * f[p, k, r] for p in P if d[p] == j)
            m.addConstr(delta[j, k, r] >= load_in - load_out, name=f"c23[{j},{k},{r}]")

print("(23) Load change: Δ_j >= Σ_p q_p·f_p(o_p=j) - Σ_p q_p·f_p(d_p=j)  (j,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINTS (24)-(25): LOAD FLOW
# =====================================================================
print("="*80)
print("CONSTRAINTS (24)-(25): LOAD FLOW (TIGHT BIG-M)")
print("="*80)

for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M_24 * (1 - x[(i, j, k, r)]),
                        name=f"c24[{i},{j},{k},{r}]"
                    )
                    m.addConstr(
                        y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M_25 * (1 - x[(i, j, k, r)]),
                        name=f"c25[{i},{j},{k},{r}]"
                    )

print(f"(24) Load lower: y_j >= y_i + Δ_j - {M_24}·(1-x_ij)  (i,j≠i,k,r)")
print(f"(25) Load upper: y_j <= y_i + Δ_j + {M_25}·(1-x_ij)  (i,j≠i,k,r)")
print(f"     TIGHT BIG-M M_24=M_25 = Q_max = {M_24}")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (26): VEHICLE CAPACITY
# =====================================================================
print("="*80)
print("CONSTRAINT (26): VEHICLE CAPACITY")
print("="*80)

for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(y[j, k, r] <= q_vehicle[k], name=f"c26[{j},{k},{r}]")

print("(26) Capacity: y_j <= q_k  (j,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (27): DEPOT LOAD = 0
# =====================================================================
print("="*80)
print("CONSTRAINT (27): DEPOT LOAD")
print("="*80)

for k in K:
    for r in R:
        m.addConstr(y['h', k, r] == 0, name=f"c27[{k},{r}]")

print("(27) Depot load: y_h = 0  (k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (28): ROUTE ACTIVATION MONOTONICITY
# =====================================================================
print("="*80)
print("CONSTRAINT (28): ROUTE MONOTONICITY")
print("="*80)

for k in K:
    for r in R[:-1]:
        lhs = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        rhs = quicksum(x[('h', j, k, r+1)] for j in Nw if ('h', j, k, r+1) in x)
        m.addConstr(lhs >= rhs, name=f"c28[{k},{r}]")

print("(28) Route monotonicity: Σ_j x_hj^r >= Σ_j x_hj^(r+1)  (k,r<|R|)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (29): MTZ SUBTOUR ELIMINATION
# =====================================================================
print("="*80)
print("CONSTRAINT (29): MTZ SUBTOUR ELIMINATION")
print("="*80)

for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        u[j, k, r] >= u[i, k, r] + 1 - U * (1 - x[(i, j, k, r)]),
                        name=f"c29[{i},{j},{k},{r}]"
                    )

print(f"(29) MTZ subtour: u_j >= u_i + 1 - {U}·(1-x_ij)  (i,j≠i,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (30): MTZ UPPER BOUND
# =====================================================================
print("="*80)
print("CONSTRAINT (30): MTZ UPPER BOUND")
print("="*80)

for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            m.addConstr(u[j, k, r] <= U * indeg, name=f"c30[{j},{k},{r}]")

print(f"(30) MTZ upper: u_j <= {U} · Σ_i x_ij  (j,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (31): MTZ LOWER BOUND
# =====================================================================
print("="*80)
print("CONSTRAINT (31): MTZ LOWER BOUND")
print("="*80)

for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            m.addConstr(u[j, k, r] >= indeg, name=f"c31[{j},{k},{r}]")

print(f"(31) MTZ lower: u_j >= Σ_i x_ij  (j,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (32): DELIVERY-AFTER-PICKUP PRECEDENCE
# =====================================================================
print("="*80)
print("CONSTRAINT (32): DELIVERY-AFTER-PICKUP")
print("="*80)

for p in P:
    op, dp = o[p], d[p]
    if (op in Nw) and (dp in Nw):
        for k in K:
            for r in R:
                m.addConstr(
                    u[dp, k, r] >= u[op, k, r] + 1 - U * (1 - f[p, k, r]),
                    name=f"c32[{p},{k},{r}]"
                )

print(f"(32) Delivery after pickup: u_d_p >= u_o_p + 1 - {U}·(1-f_p)  (p,k,r)")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (33): LOAD CHANGE DOMAIN
# =====================================================================
print("="*80)
print("CONSTRAINT (33): LOAD CHANGE DOMAIN")
print("="*80)
# delta is already defined as continuous (-infinity to +infinity)
print("(33) Load change domain: Δ_j ∈ ℝ  (j,k,r)")
print("     [implicitly satisfied by variable definition]")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (34): NON-NEGATIVITY
# =====================================================================
print("="*80)
print("CONSTRAINT (34): NON-NEGATIVITY")
print("="*80)
# ta, td, ts, w, y already defined with lb=0
print("(34) Non-negativity: ta, td, ts, w, y >= 0")
print("     [implicitly satisfied by variable definition]")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (35): INTEGER BOUNDS
# =====================================================================
print("="*80)
print("CONSTRAINT (35): INTEGER BOUNDS FOR MTZ")
print("="*80)
# u already defined as integer with ub=U
print(f"(35) MTZ bounds: u_j ∈ {{0, 1, ..., {U}}}")
print("     [implicitly satisfied by variable definition]")
print("="*80 + "\n")

# =====================================================================
# CONSTRAINT (36): BINARY DECLARATIONS
# =====================================================================
print("="*80)
print("CONSTRAINT (36): BINARY DECLARATIONS")
print("="*80)
# x, f already defined as binary
print("(36) Binary: x_ijkr, f_pkr ∈ {0, 1}")
print("     [implicitly satisfied by variable definition]")
print("="*80 + "\n")

# =====================================================================
# SOLVER SETUP & OPTIMIZATION
# =====================================================================
print("="*80)
print("SOLVER SETUP")
print("="*80)

os.makedirs('results', exist_ok=True)
excel_path = os.path.join('results', f"result_internal_logistics_{timestamp}.xlsx")
log_path   = os.path.join(desktop_dir, f"result_internal_logistics_{timestamp}.txt")

m.setParam('TimeLimit', TIME_LIMIT)
m.setParam('MIPGap', MIP_GAP)
m.setParam('Threads', THREADS)
m.setParam('Presolve', 2)
m.setParam('LogFile', log_path)
m.update()

print(f"Time Limit: {TIME_LIMIT}s")
print(f"MIP Gap: {MIP_GAP}")
print(f"Threads: {THREADS}")
print(f"Log file: {log_path}")
print("="*80 + "\n")

print("OPTIMIZATION STARTING...\n")
m.optimize()

# =====================================================================
# RESULTS PROCESSING
# =====================================================================
if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    print("\n" + "="*80)
    print("SOLUTION FOUND")
    print("="*80)
    
    if m.status == GRB.TIME_LIMIT:
        print(f"Time limit exceeded ({TIME_LIMIT}s)")
        print(f"Best solution found: {m.objVal if m.SolCount > 0 else 'NONE'}")
    
    total_wait = sum(w[p].X for p in P if w[p].X is not None)
    total_arrival = sum(ta['h', k, r].X for k in K for r in R if ta['h', k, r].X is not None)
    
    print(f"\nResults:")
    print(f"  Primary Objective (Σ ta_hkr): {total_arrival:.2f} min")
    print(f"  Secondary Objective (Σ w_p): {total_wait:.2f} min")
    print(f"  Objective Value: {m.objVal:.2f}")
    print(f"  Runtime: {m.Runtime:.2f}s")
    print(f"  MIP Gap: {m.MIPGap*100:.2f}%")
    print("="*80 + "\n")

elif m.status == GRB.INFEASIBLE:
    print("\n" + "="*80)
    print("MODEL IS INFEASIBLE")
    print("="*80)
    m.computeIIS()
    iis_file = f"infeasible_internal_logistics_{timestamp}.ilp"
    m.write(iis_file)
    print(f"IIS file: {iis_file}")

else:
    print(f"\nNo solution found. Status = {m.status}")

print("\n" + "="*80)
print("PROGRAM COMPLETE")
print(f"Terminal output: {terminal_log_path}")
print("="*80)

sys.stdout = original_stdout
terminal_log_file.close()
