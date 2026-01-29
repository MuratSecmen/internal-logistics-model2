import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import re, os
from datetime import datetime, time, timedelta

TIME_LIMIT = 1200
MIP_GAP    = 0.03
THREADS    = 6
EPS_WAIT   = 158

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

data_path   = r"C:\Users\Asus\Desktop\Er\\"
desktop_dir = r"C:\Users\Asus\Desktop"

nodes    = pd.read_excel(os.path.join(data_path, "nodes.xlsx"))
vehicles = pd.read_excel(os.path.join(data_path, "vehicles.xlsx"))
products = pd.read_excel(os.path.join(data_path, "products.xlsx")).head(10)

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

dist_min = _read_dist(os.path.join(data_path, "distances - dakika.xlsx"), "duration_min")

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

c  = dist_min
e  = dict(zip(P, [ready_to_min(v) for v in products['ready_time']]))
sl = dict(zip(P, products['load_time']))
su = dict(zip(P, products['unload_time']))
q_product = dict(zip(P, products['area_m2']))
o  = dict(zip(P, products['origin']))
d  = dict(zip(P, products['destination']))

M = 10000.0
epsilon = 0.1
U = len(Nw)

print("="*80)
print("VERİ YÜKLENDİ")
print("="*80)
print(f"|N|={len(N)}, |Nw|={len(Nw)}, |K|={len(K)}, |R|={len(R)}, |P|={len(P)}")
print(f"M={M}, U={U}, ε={epsilon}")
print("="*80 + "\n")

m = gp.Model("InternalLogistics")

# KARAR DEĞİŞKENLERİ (Tablo 2)
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

# AMAÇ FONKSİYONLARI (1) ve (2)
obj1 = quicksum(w[p] for p in P)
obj2 = quicksum(c.get((i, j), 0.0) * x[i, j, k, r] 
                for i in N for j in N for k in K for r in R if (i, j, k, r) in x)

m.setObjective(obj2, GRB.MINIMIZE)
m.addConstr(obj1 <= EPS_WAIT, name="eps_wait")

# KISIT (3): Σ_j∈Nw x_hjkr = Σ_j∈Nw x_jhkr, ∀k∈K, r∈R
for k in K:
    for r in R:
        out_h = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        in_h  = quicksum(x[(j, 'h', k, r)] for j in Nw if (j, 'h', k, r) in x)
        m.addConstr(out_h == in_h, name=f"c3[{k},{r}]")

# KISIT (4): Σ_j∈Nw x_hjkr ≤ 1, ∀k∈K, r∈R
for k in K:
    for r in R:
        m.addConstr(
            quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x) <= 1,
            name=f"c4[{k},{r}]")

# KISIT (5): Σ_i∈N Σ_j∈N x_ijkr ≤ Σ_p∈P f_pkr, ∀k∈K, r∈R
for k in K:
    for r in R:
        lhs = quicksum(x[(i, j, k, r)] for i in N for j in N if (i, j, k, r) in x)
        rhs = quicksum(f[p, k, r] for p in P)
        m.addConstr(lhs <= rhs, name=f"c5[{k},{r}]")

# KISIT (6): Σ_i∈N x_ijkr ≤ Σ_(p∈P: op=j ∨ dp=j) f_pkr, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            lhs = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            rhs = quicksum(f[p, k, r] for p in P if (o[p] == j or d[p] == j))
            m.addConstr(lhs <= rhs, name=f"c6[{j},{k},{r}]")

# KISIT (7): Σ_(i∈N, i≠j) x_ijkr = Σ_(i∈N, i≠j) x_jikr, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            inflow  = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[(j, i, k, r)] for i in N if i != j and (j, i, k, r) in x)
            m.addConstr(inflow == outflow, name=f"c7[{j},{k},{r}]")

# KISIT (8): Σ_(i∈N, i≠j) x_ijkr ≤ 1, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(
                quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x) <= 1,
                name=f"c8[{j},{k},{r}]")

# KISIT (9): Σ_k∈K Σ_r∈R f_pkr = 1, ∀p∈P
for p in P:
    m.addConstr(quicksum(f[p, k, r] for k in K for r in R) == 1, name=f"c9[{p}]")

# KISIT (10): Σ_(i∈N, i≠op) x_i,op,kr ≥ f_pkr, ∀p∈P, k∈K, r∈R
for p in P:
    op = o[p]
    if op in N:
        for k in K:
            for r in R:
                lhs = quicksum(x[(i, op, k, r)] for i in N if i != op and (i, op, k, r) in x)
                m.addConstr(lhs >= f[p, k, r], name=f"c10[{p},{k},{r}]")

# KISIT (11): Σ_(i∈N, i≠dp) x_i,dp,kr ≥ f_pkr, ∀p∈P, k∈K, r∈R
for p in P:
    dp = d[p]
    if dp in N:
        for k in K:
            for r in R:
                lhs = quicksum(x[(i, dp, k, r)] for i in N if i != dp and (i, dp, k, r) in x)
                m.addConstr(lhs >= f[p, k, r], name=f"c11[{p},{k},{r}]")

# KISIT (12): td_hk1 = 0, ∀k∈K
for k in K:
    m.addConstr(td['h', k, 1] == 0, name=f"c12[{k}]")

# KISIT (13): td_hkr ≥ ta_hk|r-1|, ∀k∈K, r∈R\{1}
for k in K:
    for r in R[1:]:
        m.addConstr(td['h', k, r] >= ta['h', k, r-1] + epsilon, name=f"c13[{k},{r}]")

# KISIT (14): ta_jkr ≥ td_ikr + cij*x_ijkr - M(1-x_ijkr), ∀i,j∈N, i≠j, k∈K, r∈R
for i in N:
    for j in N:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    cij = c.get((i, j), 0.0)
                    m.addConstr(
                        ta[j, k, r] >= td[i, k, r] + cij * x[(i, j, k, r)] - M * (1 - x[(i, j, k, r)]),
                        name=f"c14[{i},{j},{k},{r}]")

# KISIT (15): ts_jkr ≥ ta_jkr + Σ_(p∈P: dp=j) su_p * f_pkr, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            unload = quicksum(su[p] * f[p, k, r] for p in P if d[p] == j)
            m.addConstr(ts[j, k, r] >= ta[j, k, r] + unload, name=f"c15[{j},{k},{r}]")

# KISIT (16): ts_op,kr ≥ ep * f_pkr, ∀p∈P, k∈K, r∈R
for p in P:
    op = o[p]
    if op in Nw:
        ep = e[p]
        for k in K:
            for r in R:
                m.addConstr(ts[op, k, r] >= ep * f[p, k, r], name=f"c16[{p},{k},{r}]")

# KISIT (17): td_jkr ≥ ts_jkr + Σ_(p∈P: op=j) sl_p * f_pkr, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            load = quicksum(sl[p] * f[p, k, r] for p in P if o[p] == j)
            m.addConstr(td[j, k, r] >= ts[j, k, r] + load, name=f"c17[{j},{k},{r}]")

# KISIT (18): ta_dp,kr ≥ td_op,kr - M(1-f_pkr), ∀p∈P, k∈K, r∈R
for p in P:
    op, dp = o[p], d[p]
    if (op in N) and (dp in N):
        for k in K:
            for r in R:
                m.addConstr(ta[dp, k, r] >= td[op, k, r] - M * (1 - f[p, k, r]),
                           name=f"c18[{p},{k},{r}]")

# KISIT (19): wp ≥ ta_dp,kr - ep - M(1-f_pkr), ∀p∈P, k∈K, r∈R
for p in P:
    dp = d[p]
    ep = e[p]
    for k in K:
        for r in R:
            m.addConstr(w[p] >= ta[dp, k, r] - ep - M * (1 - f[p, k, r]),
                       name=f"c19[{p},{k},{r}]")

# KISIT (20): Δjkr ≥ Σ_(p∈P: op=j) qp*f_pkr - Σ_(p∈P: dp=j) qp*f_pkr, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(q_product[p] * f[p, k, r] for p in P if o[p] == j)
            load_out = quicksum(q_product[p] * f[p, k, r] for p in P if d[p] == j)
            m.addConstr(delta[j, k, r] >= load_in - load_out, name=f"c20[{j},{k},{r}]")

# KISIT (21): yjkr ≥ yikr + Δjkr - M(1-x_ijkr), ∀i,j∈Nw, k∈K, r∈R
for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M * (1 - x[(i, j, k, r)]),
                        name=f"c21[{i},{j},{k},{r}]")

# KISIT (22): yjkr ≤ yikr + Δjkr + M(1-x_ijkr), ∀i,j∈Nw, k∈K, r∈R
for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M * (1 - x[(i, j, k, r)]),
                        name=f"c22[{i},{j},{k},{r}]")

# KISIT (23): yjkr ≤ qk, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(y[j, k, r] <= q_vehicle[k], name=f"c23[{j},{k},{r}]")

# Depo yükü sıfır (ek kısıt)
for k in K:
    for r in R:
        m.addConstr(y['h', k, r] == 0, name=f"yhome[{k},{r}]")

# KISIT (24): Σ_j∈Nw x_hjkr ≥ Σ_j∈Nw x_hj,k|r+1|, ∀k∈K, r∈R\{|R|}
for k in K:
    for r in R[:-1]:
        lhs = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        rhs = quicksum(x[('h', j, k, r+1)] for j in Nw if ('h', j, k, r+1) in x)
        m.addConstr(lhs >= rhs, name=f"c24[{k},{r}]")

# KISIT (25): ujkr ≥ uikr + 1 - |Nw|(1-x_ijkr), ∀i,j∈Nw, i≠j, k∈K, r∈R
for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        u[j, k, r] >= u[i, k, r] + 1 - U * (1 - x[(i, j, k, r)]),
                        name=f"c25[{i},{j},{k},{r}]")

# KISIT (26): ujkr ≤ |Nw| Σ_(i∈N, i≠j) x_ijkr, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            m.addConstr(u[j, k, r] <= U * indeg, name=f"c26[{j},{k},{r}]")

# KISIT (27): ujkr ≥ Σ_(i∈N, i≠j) x_ijkr, ∀j∈Nw, k∈K, r∈R
for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            m.addConstr(u[j, k, r] >= indeg, name=f"c27[{j},{k},{r}]")

# KISIT (28): u_dp,kr ≥ u_op,kr + 1 - |Nw|(1-f_pkr), ∀k∈K, r∈R, p∈P
for k in K:
    for r in R:
        for p in P:
            op, dp = o[p], d[p]
            if (op in Nw) and (dp in Nw):
                m.addConstr(
                    u[dp, k, r] >= u[op, k, r] + 1 - U * (1 - f[p, k, r]),
                    name=f"c28[{p},{k},{r}]")

timestamp  = datetime.now().strftime('%Y_%m_%d_%H_%M')
excel_path = os.path.join('results', f"result_of_run_{timestamp}.xlsx")
log_path   = os.path.join(desktop_dir, f"result_of_run_{timestamp}.txt")
os.makedirs('results', exist_ok=True)

m.setParam('TimeLimit', TIME_LIMIT)
m.setParam('MIPGap', MIP_GAP)
m.setParam('Threads', THREADS)
m.setParam('Presolve', 2)
m.setParam('LogFile', log_path)
m.update()

print("OPTİMİZASYON BAŞLIYOR...\n")
m.optimize()

if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    print("\n" + "="*80)
    print("ÇÖZÜM BULUNDU")
    print("="*80)
    
    total_wait = sum(w[p].X for p in P if w[p].X is not None)
    total_time = sum(c.get((i, j), 0.0) * x[(i, j, k, r)].X 
                     for i in N for j in N for k in K for r in R 
                     if (i, j, k, r) in x and x[(i, j, k, r)].X > 0.5)
    
    opt_results = pd.DataFrame([{
        'objective': 'min_total_time',
        'obj_value': m.objVal if m.SolCount > 0 else None,
        'best_bound': getattr(m, 'objbound', None),
        'mip_gap': getattr(m, 'mipgap', None),
        'runtime': m.Runtime,
        'total_time_minutes': round(total_time, 2),
        'total_wait_minutes': round(total_wait, 2),
        'epsilon_wait_upper': EPS_WAIT,
        'wait_slack_remaining': round(EPS_WAIT - total_wait, 2),
        '|N|': len(N), '|Nw|': len(Nw), '|K|': len(K), '|R|': len(R),
        'KxR': len(K) * len(R), 'U_(|Nw|)': U, 'M_single': M
    }])
    
    x_data = [{'var': 'x', 'i': i, 'j': j, 'k': k, 'r': r, 'val': round(var.X, 4)} 
              for (i, j, k, r), var in x.items() if var.X > 0.5]
    xdf = pd.DataFrame(x_data) if x_data else pd.DataFrame(columns=['var', 'i', 'j', 'k', 'r', 'val'])
    
    f_data = [{'var': 'f', 'p': p, 'k': k, 'r': r, 'val': round(f[p, k, r].X, 4)} 
              for p in P for k in K for r in R if f[p, k, r].X > 0.5]
    fdf = pd.DataFrame(f_data) if f_data else pd.DataFrame(columns=['var', 'p', 'k', 'r', 'val'])
    
    u_data = [{'var': 'u', 'j': j, 'k': k, 'r': r, 'u': int(u[j, k, r].X)} 
              for j in Nw for k in K for r in R if u[j, k, r].X > 0]
    udf = pd.DataFrame(u_data) if u_data else pd.DataFrame(columns=['var', 'j', 'k', 'r', 'u'])
    
    z_data = [{'var': 'z', 'k': k, 'r': r, 'z': 1 if any(x[(i, j, k, r)].X > 0.5 for (i, j, k, r_) in x.keys() if r_ == r) else 0} 
              for k in K for r in R]
    zdf = pd.DataFrame(z_data)
    
    wdf = pd.DataFrame([{'var': 'w', 'p': p, 'w_val': round(w[p].X, 4) if w[p].X else 0} for p in P])
    
    ta_data = [{'var': 'ta', 'node': j, 'k': k, 'r': r, 'time': round(ta[j, k, r].X, 4), 
                'stamp': minutes_to_hhmm(ta[j, k, r].X)} for j in N for k in K for r in R]
    tadf = pd.DataFrame(ta_data)
    
    td_data = [{'var': 'td', 'node': j, 'k': k, 'r': r, 'time': round(td[j, k, r].X, 4), 
                'stamp': minutes_to_hhmm(td[j, k, r].X)} for j in N for k in K for r in R]
    tddf = pd.DataFrame(td_data)
    
    ts_data = [{'var': 'ts', 'node': j, 'k': k, 'r': r, 'time': round(ts[j, k, r].X, 4), 
                'stamp': minutes_to_hhmm(ts[j, k, r].X)} for j in Nw for k in K for r in R]
    tsdf = pd.DataFrame(ts_data)
    
    ydf = pd.DataFrame([{'var': 'y', 'node': j, 'k': k, 'r': r, 'y_val': round(y[j, k, r].X, 4)} 
                        for j in N for k in K for r in R])
    
    deltadf = pd.DataFrame([{'var': 'delta', 'node': j, 'k': k, 'r': r, 'delta_val': round(delta[j, k, r].X, 4)} 
                            for j in Nw for k in K for r in R])
    
    used = xdf[xdf['val'] > 0.5].copy() if not xdf.empty else pd.DataFrame()
    
    if not used.empty:
        dep = used[used['i'] == 'h'][['k', 'r']].drop_duplicates()
        visit = (used[used['j'].isin(Nw)]
                 .merge(udf[['j', 'k', 'r', 'u']], on=['j', 'k', 'r'], how='left')
                 .merge(tadf[tadf['stamp'] != ''][['node', 'k', 'r', 'stamp']]
                        .rename(columns={'node': 'j', 'stamp': 'ta_stamp'}), on=['j', 'k', 'r'], how='left')
                 .merge(tddf[tddf['stamp'] != ''][['node', 'k', 'r', 'stamp']]
                        .rename(columns={'node': 'j', 'stamp': 'td_stamp'}), on=['j', 'k', 'r'], how='left')
                 .merge(ydf[['node', 'k', 'r', 'y_val']]
                        .rename(columns={'node': 'j', 'y_val': 'y_after'}), on=['j', 'k', 'r'], how='left'))
        visit = (visit[['k', 'r', 'j', 'u', 'ta_stamp', 'td_stamp', 'y_after']]
                 .drop_duplicates().sort_values(['k', 'r', 'u'], ignore_index=True))
        visit = visit.merge(dep, on=['k', 'r'], how='inner')
    else:
        visit = pd.DataFrame(columns=['k', 'r', 'j', 'u', 'ta_stamp', 'td_stamp', 'y_after'])
    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        opt_results.to_excel(writer, sheet_name='optimization_results', index=False)
        xdf.to_excel(writer, sheet_name='x_ijkr', index=False)
        fdf.to_excel(writer, sheet_name='f_pkr', index=False)
        udf.to_excel(writer, sheet_name='u_jkr', index=False)
        zdf.to_excel(writer, sheet_name='z_kr', index=False)
        wdf.to_excel(writer, sheet_name='w_p', index=False)
        tadf.to_excel(writer, sheet_name='ta', index=False)
        tddf.to_excel(writer, sheet_name='td', index=False)
        tsdf.to_excel(writer, sheet_name='ts', index=False)
        ydf.to_excel(writer, sheet_name='y_jkr', index=False)
        deltadf.to_excel(writer, sheet_name='delta_jkr', index=False)
        visit.to_excel(writer, sheet_name='itinerary', index=False)
    
    print(f"\n✓ Excel: {excel_path}")
    print(f"✓ Rota süresi: {total_time:.2f} dk")
    print(f"✓ Bekleme süresi: {total_wait:.2f} dk")

elif m.status == GRB.INFEASIBLE:
    print("\n" + "="*80)
    print("MODEL INFEASIBLE - IIS HESAPLANIYOR")
    print("="*80)
    m.computeIIS()
    iis_file = f"infeasible_{timestamp}.ilp"
    m.write(iis_file)
    print(f"\n✓ IIS dosyası: {iis_file}")
    print("\nÇELİŞEN KISITLAR:")
    for c in m.getConstrs():
        if c.IISConstr:
            print(f"  - {c.ConstrName}")
else:
    print(f"\nÇözüm bulunamadı. Status = {m.status}")