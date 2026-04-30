import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import os
from datetime import datetime, time
import sys

# =====================================================================
# OUTPUT & LOG
# =====================================================================
OUTPUT_DIR = r"C:\Users\Asus\Documents\GitHub\logistics-model2\internal-logistics-model2\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
terminal_log_path = os.path.join(OUTPUT_DIR, f"terminal_output_{timestamp}.txt")
terminal_log_file = open(terminal_log_path, 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = TeeOutput(original_stdout, terminal_log_file)

print(f"Terminal çıktısı kaydediliyor: {terminal_log_path}\n")

# =====================================================================
# PARAMETERS
# =====================================================================
TIME_LIMIT = 600
MIP_GAP = 0.01
SHIFT_START = 0
EPS_WAIT = 9999
WAIT_WEIGHT = 0.001

# =====================================================================
# HELPERS
# =====================================================================
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
    dt = pd.to_datetime(s, errors='coerce')

    if pd.notna(dt):
        return int(dt.hour) * 60 + int(dt.minute)

    if ":" in s:
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm)

    return int(float(s))


def minutes_to_hhmm(minutes):
    if minutes is None or pd.isna(minutes):
        return ''
    minutes = float(minutes)
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def safe_mipgap(model):
    try:
        return model.MIPGap
    except Exception:
        return None


def write_df(writer, df, sheet_name, columns=None):
    if df is None or df.empty:
        df = pd.DataFrame(columns=columns if columns is not None else ["empty"])
    df.to_excel(writer, sheet_name=sheet_name, index=False)


# =====================================================================
# DATA LOADING
# =====================================================================
data_path = r"C:\Users\Asus\Desktop\Er\\"

nodes = pd.read_excel(os.path.join(data_path, "nodes.xlsx"))
vehicles = pd.read_excel(os.path.join(data_path, "vehicles.xlsx"))
products = pd.read_excel(os.path.join(data_path, "products.xlsx"))

def _read_dist(path, val_col):
    df = pd.read_excel(path, sheet_name=0)

    need = ['from_node', 'to_node', val_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{os.path.basename(path)} dosyasında eksik kolonlar: {miss}")

    df['from_node'] = df['from_node'].astype(str).str.strip()
    df['to_node'] = df['to_node'].astype(str).str.strip()
    df = df.dropna(subset=['from_node', 'to_node', val_col])
    df = df[df['from_node'] != df['to_node']]
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=[val_col])

    return {
        (r['from_node'], r['to_node']): float(r[val_col])
        for _, r in df.iterrows()
    }

c = _read_dist(os.path.join(data_path, "distances - dakika.xlsx"), "duration_min")

# =====================================================================
# SETS & PARAMETERS
# =====================================================================
nodes['node_id'] = nodes['node_id'].astype(str).str.strip()
N = nodes['node_id'].dropna().drop_duplicates().tolist()
Nw = [n for n in N if n != 'h']

vehicles['vehicle_id'] = vehicles['vehicle_id'].astype(str).str.strip()
vehicles = vehicles.dropna(subset=['vehicle_id']).drop_duplicates('vehicle_id', keep='first')

vehicles = vehicles.head(1)
K = vehicles['vehicle_id'].tolist()

MAX_ROUTES = 1
R = [1]

q_vehicle = dict(zip(K, vehicles['capacity_m2']))

products['product_id'] = products['product_id'].astype(str).str.strip()
products['origin'] = products['origin'].astype(str).str.strip()
products['destination'] = products['destination'].astype(str).str.strip()
products = products.dropna(subset=['product_id']).drop_duplicates('product_id', keep='first')

P = products['product_id'].tolist()

e = dict(zip(P, [ready_to_min(v) for v in products['ready_time']]))
sl = dict(zip(P, products['load_time']))
su = dict(zip(P, products['unload_time']))
q_product = dict(zip(P, products['area_m2']))
o = dict(zip(P, products['origin']))
d = dict(zip(P, products['destination']))

T_max = 480
C_max = 11
e_min = 15
Q_max = 20

U = len(Nw)

M_16 = T_max + C_max
M_20 = T_max
M_22 = T_max + SHIFT_START - e_min
M_24 = Q_max
M_25 = Q_max

print("\n" + "=" * 80)
print("DATA LOADED")
print("=" * 80)
print(f"|N|={len(N)}, |Nw|={len(Nw)}, |K|={len(K)}, |R|={len(R)}, |P|={len(P)}")
print(f"K={K}")
print(f"R={R}")
print("TIGHT BIG-M:")
print(f"  M_16={M_16:.1f}  M_20={M_20:.1f}  M_22={M_22:.1f}")
print(f"  M_24={M_24:.0f}  M_25={M_25:.0f}  U={U}")
print("=" * 80 + "\n")

# =====================================================================
# MODEL CREATION
# =====================================================================
m = gp.Model("Internal_Logistics_Route_Duration_Primary_Model")

# =====================================================================
# DECISION VARIABLES
# =====================================================================
x = m.addVars(
    [(i, j, k, r) for i in N for j in N for k in K for r in R if i != j],
    vtype=GRB.BINARY,
    name="x"
)

f = m.addVars(P, K, R, vtype=GRB.BINARY, name="f")
w = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
y = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
ta = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="ta")
td = m.addVars(N, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="td")
ts = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=0.0, name="ts")
u = m.addVars(Nw, K, R, vtype=GRB.INTEGER, lb=0, ub=U, name="u")
delta = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="delta")

# =====================================================================
# OBJECTIVE
# =====================================================================
route_duration = quicksum(ta['h', k, MAX_ROUTES] for k in K) - len(K) * SHIFT_START
total_wait = quicksum(w[p] for p in P)

m.setObjective(route_duration + WAIT_WEIGHT * total_wait, GRB.MINIMIZE)

print("=" * 80)
print("OBJECTIVE FUNCTION")
print("=" * 80)
print("Primary objective: min route_duration")
print(f"Secondary weighted term: + {WAIT_WEIGHT} × total_wait")
print("route_duration = Σ_k ta_h,k,|R| - |K| × SHIFT_START")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINT (3): EPSILON-CONSTRAINT ON WAITING TIME
# =====================================================================
print("=" * 80)
print("CONSTRAINT (3): EPSILON-CONSTRAINT ON WAITING TIME")
print("=" * 80)

m.addConstr(total_wait <= EPS_WAIT, name="c3_epsilon_wait")

print(f"(3) total_wait <= {EPS_WAIT}")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINTS (4)-(9): ROUTE STRUCTURE
# =====================================================================
print("=" * 80)
print("CONSTRAINTS (4)-(9): ROUTE STRUCTURE")
print("=" * 80)

for k in K:
    for r in R:
        out_h = quicksum(x['h', j, k, r] for j in Nw if ('h', j, k, r) in x)
        in_h = quicksum(x[j, 'h', k, r] for j in Nw if (j, 'h', k, r) in x)

        m.addConstr(out_h == in_h, name=f"c4[{k},{r}]")
        m.addConstr(out_h <= 1, name=f"c5[{k},{r}]")

        m.addConstr(
            quicksum(x[i, j, k, r] for i in N for j in N if (i, j, k, r) in x)
            <=
            (2 * len(P) + 1) * quicksum(f[p, k, r] for p in P),
            name=f"c6[{k},{r}]"
        )

for j in Nw:
    for k in K:
        for r in R:
            inflow = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[j, i, k, r] for i in N if i != j and (j, i, k, r) in x)

            m.addConstr(
                inflow <= quicksum(f[p, k, r] for p in P if o[p] == j or d[p] == j),
                name=f"c7[{j},{k},{r}]"
            )

            m.addConstr(inflow == outflow, name=f"c8[{j},{k},{r}]")
            m.addConstr(inflow <= 1, name=f"c9[{j},{k},{r}]")

print("(4)-(9) Route structure constraints added.")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINTS (10)-(12): PRODUCT ASSIGNMENT
# =====================================================================
print("=" * 80)
print("CONSTRAINTS (10)-(12): PRODUCT ASSIGNMENT")
print("=" * 80)

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

print("(10)-(12) Assignment and pickup/delivery visit constraints added.")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINTS (13)-(15): ROUTE TIMING
# =====================================================================
print("=" * 80)
print("CONSTRAINTS (13)-(15): ROUTE TIMING")
print("=" * 80)

for k in K:
    m.addConstr(td['h', k, 1] == SHIFT_START, name=f"c13[{k}]")

for k in K:
    for r in R[1:]:
        m.addConstr(td['h', k, r] >= ta['h', k, r - 1], name=f"c14[{k},{r}]")
        m.addConstr(ta['h', k, r] >= ta['h', k, r - 1], name=f"c15[{k},{r}]")

print("(13)-(15) Route timing constraints added.")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINT (16): TIME CONSISTENCY
# =====================================================================
print("=" * 80)
print("CONSTRAINT (16): TIME CONSISTENCY")
print("=" * 80)

for i in N:
    for j in N:
        if i == j:
            continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    c_ij = c.get((i, j), 0.0)
                    m.addConstr(
                        ta[j, k, r]
                        >=
                        td[i, k, r] + c_ij * x[i, j, k, r] - M_16 * (1 - x[i, j, k, r]),
                        name=f"c16[{i},{j},{k},{r}]"
                    )

print("(16) Time consistency constraints added.")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINTS (17)-(19): SERVICE TIMES
# =====================================================================
print("=" * 80)
print("CONSTRAINTS (17)-(19): SERVICE TIMES")
print("=" * 80)

for j in Nw:
    for k in K:
        for r in R:
            unload = quicksum(su[p] * f[p, k, r] for p in P if d[p] == j)
            load = quicksum(sl[p] * f[p, k, r] for p in P if o[p] == j)

            m.addConstr(ts[j, k, r] >= ta[j, k, r] + unload, name=f"c17[{j},{k},{r}]")
            m.addConstr(td[j, k, r] >= ts[j, k, r] + load, name=f"c19[{j},{k},{r}]")

for p in P:
    op = o[p]
    if op in Nw:
        for k in K:
            for r in R:
                m.addConstr(
                    ts[op, k, r] >= e[p] * f[p, k, r],
                    name=f"c18[{p},{k},{r}]"
                )

print("(17)-(19) Service and ready-time constraints added.")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINTS (20)-(22): PRECEDENCE, ROUTE TIMING, WAITING
# =====================================================================
print("=" * 80)
print("CONSTRAINTS (20)-(22): PRECEDENCE / ROUTE TIMING / WAITING")
print("=" * 80)

for p in P:
    op = o[p]
    dp = d[p]

    if op in N and dp in N:
        for k in K:
            for r in R:
                m.addConstr(
                    ta[dp, k, r] >= td[op, k, r] - M_20 * (1 - f[p, k, r]),
                    name=f"c20[{p},{k},{r}]"
                )

for k in K:
    for r in R:
        m.addConstr(ta['h', k, r] >= td['h', k, r], name=f"c21[{k},{r}]")

for p in P:
    dp = d[p]
    ep = e[p]

    for k in K:
        for r in R:
            m.addConstr(
                w[p] >= ta[dp, k, r] + su[p] - ep - M_22 * (1 - f[p, k, r]),
                name=f"c22[{p},{k},{r}]"
            )

print("(20)-(22) Precedence, route timing and waiting constraints added.")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINTS (23)-(27): LOAD
# =====================================================================
print("=" * 80)
print("CONSTRAINTS (23)-(27): LOAD")
print("=" * 80)

for j in Nw:
    for k in K:
        for r in R:
            load_in = quicksum(q_product[p] * f[p, k, r] for p in P if o[p] == j)
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
                        y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M_24 * (1 - x[i, j, k, r]),
                        name=f"c24[{i},{j},{k},{r}]"
                    )

                    m.addConstr(
                        y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M_25 * (1 - x[i, j, k, r]),
                        name=f"c25[{i},{j},{k},{r}]"
                    )

for k in K:
    for r in R:
        m.addConstr(y['h', k, r] == 0, name=f"c27[{k},{r}]")

print("(23)-(27) Load constraints added.")
print("=" * 80 + "\n")

# =====================================================================
# CONSTRAINTS (28)-(32): ROUTE ORDER / MTZ / PRECEDENCE
# =====================================================================
print("=" * 80)
print("CONSTRAINTS (28)-(32): MTZ AND ROUTE ORDER")
print("=" * 80)

for k in K:
    for r in R[:-1]:
        lhs = quicksum(x['h', j, k, r] for j in Nw if ('h', j, k, r) in x)
        rhs = quicksum(x['h', j, k, r + 1] for j in Nw if ('h', j, k, r + 1) in x)
        m.addConstr(lhs >= rhs, name=f"c28[{k},{r}]")

for i in Nw:
    for j in Nw:
        if i == j:
            continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        u[j, k, r] >= u[i, k, r] + 1 - U * (1 - x[i, j, k, r]),
                        name=f"c29[{i},{j},{k},{r}]"
                    )

for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)

            m.addConstr(u[j, k, r] <= U * indeg, name=f"c30[{j},{k},{r}]")
            m.addConstr(u[j, k, r] >= indeg, name=f"c31[{j},{k},{r}]")

for p in P:
    op = o[p]
    dp = d[p]

    if op in Nw and dp in Nw:
        for k in K:
            for r in R:
                m.addConstr(
                    u[dp, k, r] >= u[op, k, r] + 1 - U * (1 - f[p, k, r]),
                    name=f"c32[{p},{k},{r}]"
                )

print("(28)-(32) MTZ and pickup-before-delivery constraints added.")
print("=" * 80 + "\n")

# =====================================================================
# SOLVER SETUP
# =====================================================================
print("=" * 80)
print("SOLVER SETUP")
print("=" * 80)

excel_filename = f"result_route_duration_primary_{timestamp}.xlsx"
excel_full_path = os.path.join(OUTPUT_DIR, excel_filename)
log_path = os.path.join(OUTPUT_DIR, f"result_route_duration_primary_{timestamp}.txt")

m.setParam('TimeLimit', TIME_LIMIT)
m.setParam('MIPGap', MIP_GAP)
m.setParam('Presolve', 2)
m.setParam('LogFile', log_path)
m.update()

print(f"Time Limit: {TIME_LIMIT}s")
print(f"MIP Gap: {MIP_GAP}")
print(f"Log file: {log_path}")
print("=" * 80 + "\n")

# =====================================================================
# OPTIMIZATION
# =====================================================================
print("OPTIMIZATION STARTING...\n")
m.optimize()

# =====================================================================
# RESULTS PROCESSING
# =====================================================================
if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) and m.SolCount > 0:

    route_duration_val = sum(ta['h', k, MAX_ROUTES].X for k in K) - len(K) * SHIFT_START
    total_wait_val = sum(w[p].X for p in P)

    print("\n" + "=" * 80)
    print("SOLUTION FOUND")
    print("=" * 80)
    print(f"Route duration: {route_duration_val:.2f}")
    print(f"Total wait: {total_wait_val:.2f}")
    print(f"Objective value: {m.objVal:.4f}")
    print(f"Runtime: {m.Runtime:.2f}s")

    gap = safe_mipgap(m)
    if gap is not None:
        print(f"MIP Gap: {gap * 100:.2f}%")
    else:
        print("MIP Gap: N/A")

    print("=" * 80 + "\n")

    # =================================================================
    # EXCEL OUTPUT - 12 SHEETS
    # =================================================================
    print("=" * 80)
    print("GENERATING EXCEL OUTPUT")
    print("=" * 80)
    print(f"Output file: {excel_full_path}\n")

    with pd.ExcelWriter(excel_full_path, engine='openpyxl') as writer:

        df_opt = pd.DataFrame([{
            'model': 'route_duration_primary_wait_weighted',
            'objective': 'min_route_duration_plus_0.001_total_wait',
            'obj_value': m.objVal,
            'best_bound': m.ObjBound,
            'mip_gap': safe_mipgap(m),
            'runtime': m.Runtime,
            'status': m.status,
            'route_duration_minutes': route_duration_val,
            'route_duration_stamp': minutes_to_hhmm(route_duration_val),
            'total_wait_minutes': total_wait_val,
            'epsilon_wait_upper': EPS_WAIT,
            'wait_slack_remaining': EPS_WAIT - total_wait_val,
            'wait_weight_in_objective': WAIT_WEIGHT,
            '|N|': len(N),
            '|Nw|': len(Nw),
            '|K|': len(K),
            '|R|': len(R),
            '|P|': len(P),
            'KxR': len(K) * len(R),
            'U_(|Nw|)': U,
            'M_16': M_16,
            'M_20': M_20,
            'M_22': M_22,
            'M_24': M_24,
            'M_25': M_25
        }])
        write_df(writer, df_opt, 'optimization_results')

        df_x = pd.DataFrame([
            {
                'var': 'x',
                'i': i,
                'j': j,
                'k': k,
                'r': r,
                'val': int(x[i, j, k, r].X)
            }
            for i in N
            for j in N
            for k in K
            for r in R
            if i != j and (i, j, k, r) in x and x[i, j, k, r].X > 0.5
        ])
        write_df(writer, df_x, 'x_ijkr', ['var', 'i', 'j', 'k', 'r', 'val'])

        df_f = pd.DataFrame([
            {
                'var': 'f',
                'p': p,
                'k': k,
                'r': r,
                'val': int(f[p, k, r].X)
            }
            for p in P
            for k in K
            for r in R
            if f[p, k, r].X > 0.5
        ])
        write_df(writer, df_f, 'f_pkr', ['var', 'p', 'k', 'r', 'val'])

        df_u = pd.DataFrame([
            {
                'var': 'u',
                'j': j,
                'k': k,
                'r': r,
                'u': u[j, k, r].X
            }
            for j in Nw
            for k in K
            for r in R
            if u[j, k, r].X > 0.01
        ])
        write_df(writer, df_u, 'u_jkr', ['var', 'j', 'k', 'r', 'u'])

        df_z = pd.DataFrame([
            {
                'var': 'z',
                'k': k,
                'r': r,
                'z': int(any(
                    ('h', j, k, r) in x and x['h', j, k, r].X > 0.5
                    for j in Nw
                ))
            }
            for k in K
            for r in R
        ])
        write_df(writer, df_z, 'z_kr', ['var', 'k', 'r', 'z'])

        df_w = pd.DataFrame([
            {
                'var': 'w',
                'p': p,
                'w_val': w[p].X
            }
            for p in P
        ])
        write_df(writer, df_w, 'w_p', ['var', 'p', 'w_val'])

        df_ta = pd.DataFrame([
            {
                'var': 'ta',
                'node': j,
                'k': k,
                'r': r,
                'time': ta[j, k, r].X,
                'stamp': minutes_to_hhmm(ta[j, k, r].X)
            }
            for j in N
            for k in K
            for r in R
        ])
        write_df(writer, df_ta, 'ta', ['var', 'node', 'k', 'r', 'time', 'stamp'])

        df_td = pd.DataFrame([
            {
                'var': 'td',
                'node': j,
                'k': k,
                'r': r,
                'time': td[j, k, r].X,
                'stamp': minutes_to_hhmm(td[j, k, r].X)
            }
            for j in N
            for k in K
            for r in R
        ])
        write_df(writer, df_td, 'td', ['var', 'node', 'k', 'r', 'time', 'stamp'])

        df_ts = pd.DataFrame([
            {
                'var': 'ts',
                'node': j,
                'k': k,
                'r': r,
                'time': ts[j, k, r].X,
                'stamp': minutes_to_hhmm(ts[j, k, r].X)
            }
            for j in Nw
            for k in K
            for r in R
        ])
        write_df(writer, df_ts, 'ts', ['var', 'node', 'k', 'r', 'time', 'stamp'])

        df_y = pd.DataFrame([
            {
                'var': 'y',
                'node': j,
                'k': k,
                'r': r,
                'y_val': y[j, k, r].X
            }
            for j in N
            for k in K
            for r in R
            if abs(y[j, k, r].X) > 1e-6
        ])
        write_df(writer, df_y, 'y_jkr', ['var', 'node', 'k', 'r', 'y_val'])

        df_delta = pd.DataFrame([
            {
                'var': 'delta',
                'node': j,
                'k': k,
                'r': r,
                'delta_val': delta[j, k, r].X
            }
            for j in Nw
            for k in K
            for r in R
            if abs(delta[j, k, r].X) > 1e-6
        ])
        write_df(writer, df_delta, 'delta_jkr', ['var', 'node', 'k', 'r', 'delta_val'])

        itinerary_list = []

        for k in K:
            for r in R:
                current = 'h'
                visited = {'h'}
                order = 1

                itinerary_list.append({
                    'k': k,
                    'r': r,
                    'order': order,
                    'node': 'h',
                    'u': None,
                    'ta': ta['h', k, r].X,
                    'ta_stamp': minutes_to_hhmm(ta['h', k, r].X),
                    'td': td['h', k, r].X,
                    'td_stamp': minutes_to_hhmm(td['h', k, r].X),
                    'y_after': y['h', k, r].X,
                    'route_duration': route_duration_val,
                    'route_duration_stamp': minutes_to_hhmm(route_duration_val)
                })

                while True:
                    next_node = next(
                        (
                            j for j in N
                            if j != current
                            and (current, j, k, r) in x
                            and x[current, j, k, r].X > 0.5
                        ),
                        None
                    )

                    if next_node is None:
                        break

                    order += 1

                    itinerary_list.append({
                        'k': k,
                        'r': r,
                        'order': order,
                        'node': next_node,
                        'u': u[next_node, k, r].X if next_node in Nw else None,
                        'ta': ta[next_node, k, r].X,
                        'ta_stamp': minutes_to_hhmm(ta[next_node, k, r].X),
                        'td': td[next_node, k, r].X,
                        'td_stamp': minutes_to_hhmm(td[next_node, k, r].X),
                        'y_after': y[next_node, k, r].X if next_node in N else None,
                        'route_duration': route_duration_val,
                        'route_duration_stamp': minutes_to_hhmm(route_duration_val)
                    })

                    if next_node == 'h':
                        break

                    if next_node in visited:
                        break

                    visited.add(next_node)
                    current = next_node

        df_itinerary = pd.DataFrame(itinerary_list)
        write_df(
            writer,
            df_itinerary,
            'itinerary',
            [
                'k', 'r', 'order', 'node', 'u',
                'ta', 'ta_stamp', 'td', 'td_stamp',
                'y_after', 'route_duration', 'route_duration_stamp'
            ]
        )

    print(f"✓ Excel output successfully generated with 12 sheets")
    print(f"  File: {excel_full_path}")

elif m.status == GRB.INFEASIBLE:
    print("\n" + "=" * 80)
    print("MODEL IS INFEASIBLE")
    print("=" * 80)

    m.computeIIS()
    iis_file = os.path.join(OUTPUT_DIR, f"infeasible_route_duration_primary_{timestamp}.ilp")
    m.write(iis_file)

    print(f"IIS file: {iis_file}")

else:
    print(f"\nNo solution found. Status = {m.status}")

print("\n" + "=" * 80)
print("PROGRAM COMPLETE")
print(f"Terminal output: {terminal_log_path}")
print("=" * 80)

sys.stdout = original_stdout
terminal_log_file.close()

try:
    os.system(f'explorer /select,"{excel_full_path}"')
except Exception:
    pass
