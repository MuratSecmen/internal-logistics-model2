import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import re, os
from datetime import datetime, time, timedelta

OBJECTIVE      = "distance"
START_AT_ZERO  = False
ENABLE_C2      = True 
BASE_DATE      = datetime(2025, 1, 1)
HORIZON_DAYS   = 30

TIME_LIMIT = 3600
MIP_GAP    = 0.03
THREADS    = 6

def ready_to_min(v):
    """Timestamp/'HH:MM'/'YYYY-MM-DD HH:MM'/time/sayı -> dakika (0..1439)"""
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

def minutes_to_stamp(total_min, base_dt=BASE_DATE):
    """Taban tarihten itibaren dakikayı ISO tarih-saat string'e çevirir."""
    if total_min is None:
        return ''
    dt = base_dt + timedelta(minutes=float(total_min))
    return dt.strftime("%Y-%m-%d %H:%M")

def tidy(var_values):
    """Gurobi varlist -> DataFrame"""
    rows=[]
    for v in var_values:
        parts = re.split(r"\[|,|]", v.varName)[:-1]
        val   = round(v.X,4) if v.X is not None else None
        rows.append(parts+[val])
    return pd.DataFrame(rows)

data_path = r"C:\Users\Asus\Desktop\Er\\"
desktop_dir = r"C:\Users\Asus\Desktop"

nodes     = pd.read_excel(os.path.join(data_path, "nodes.xlsx"))
vehicles  = pd.read_excel(os.path.join(data_path, "vehicles.xlsx"))
products  = pd.read_excel(os.path.join(data_path, "products.xlsx")).head(20)

def _read_dist(path, val_col):
    df = pd.read_excel(path, sheet_name=0)
    need = ['from_node','to_node', val_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{os.path.basename(path)} dosyasında eksik sütun(lar): {miss}")
    df['from_node'] = df['from_node'].astype(str).str.strip()
    df['to_node']   = df['to_node'].astype(str).str.strip()
    df = df.dropna(subset=['from_node','to_node', val_col])
    df = df[df['from_node'] != df['to_node']]
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    df = df.dropna(subset=[val_col])
    return {(r['from_node'], r['to_node']): float(r[val_col]) for _, r in df.iterrows()}

dist_min   = _read_dist(os.path.join(data_path, "distances - dakika.xlsx"), "duration_min")
dist_metre = _read_dist(os.path.join(data_path, "distances - metre.xlsx"),  "duration_metre")

# ============================================================================
# 2) SETLER / PARAMETRELER
# ============================================================================
nodes['node_id'] = nodes['node_id'].astype(str).str.strip()
N  = nodes['node_id'].dropna().drop_duplicates().tolist()
Nw = [n for n in N if n != 'h']

vehicles['vehicle_id'] = vehicles['vehicle_id'].astype(str).str.strip()
vehicles = vehicles.dropna(subset=['vehicle_id']).drop_duplicates('vehicle_id', keep='first')
K = vehicles['vehicle_id'].tolist()
q_vehicle = dict(zip(K, vehicles['capacity_m2']))

products['product_id']  = products['product_id'].astype(str).str.strip()
products['origin']      = products['origin'].astype(str).str.strip()
products['destination'] = products['destination'].astype(str).str.strip()
products = products.dropna(subset=['product_id']).drop_duplicates('product_id', keep='first')

P         = products['product_id'].tolist()
q_product = dict(zip(P, products['area_m2']))
orig      = dict(zip(P, products['origin']))
dest      = dict(zip(P, products['destination']))
sl        = dict(zip(P, products['load_time']))
su        = dict(zip(P, products['unload_time']))

ep_min_day = dict(zip(P, [ready_to_min(v) for v in products['ready_time']]))  # 0..1439
ep_min_abs = {p: ep_min_day[p] for p in P}

max_routes = min(len(P)//max(1,len(K)) + 1, 7)
R = [f"r{i}" for i in range(1, max_routes+1)]

U = int(max(1, len(Nw)))
T_HORIZON = float(HORIZON_DAYS*1440)
MAX_CAP   = float(max(q_vehicle.values()) if q_vehicle else 0.0)
M = max(T_HORIZON, MAX_CAP)
eps_route_gap = 1e-3

# ============================================================================
# 3) MODEL
# ============================================================================
m = gp.Model("InternalLogistics")

def arc_exists(i, j, k, r):
    return (i != j) and not (i == 'h' and j == 'h') and ((i, j) in dist_metre or (i, j) in dist_min)

arcs = [(i, j, k, r) for i in N for j in N for k in K for r in R if arc_exists(i, j, k, r)]

# Karar değişkenleri
x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
f = m.addVars(P, K, R, vtype=GRB.BINARY, name="f")
y = m.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="y") 
ta= m.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="ta")
td= m.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="td")
ts= m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, name="ts")
delta = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="delta")
u = m.addVars(Nw, K, R, vtype=GRB.INTEGER, lb=0, ub=U, name="u")
w = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="w")

def X(i, j, k, r):
    return x[(i, j, k, r)] if (i, j, k, r) in x else gp.LinExpr(0.0)

# ============================================================================
# 4) AMAÇ
# ============================================================================
if OBJECTIVE.lower() == "wait":
    m.setObjective(quicksum(w[p] for p in P), GRB.MINIMIZE)
else:
    m.setObjective(quicksum(dist_min.get((i,j), 0.0) * x[i,j,k,r] for (i,j,k,r) in arcs),
                   GRB.MINIMIZE)

# ============================================================================
# 5) KISITLAR
# ============================================================================
m.addConstr(
    quicksum(dist_metre.get((i,j), 0.0) * x[i,j,k,r] for (i,j,k,r) in arcs) >= 120,
    name="AUX_lb1_total_distance_ge_120"
)
m.addConstr(
    quicksum(dist_metre.get((j,i), 0.0) * X(j,i,k,r) for (i,j,k,r) in arcs) >= 70,
    name="AUX_lb2_total_distance_rev_ge_70"
)

for k in K:
    for r in R:
        out_h = quicksum(x['h', j, k, r] for j in Nw if ('h', j, k, r) in x)
        in_h  = quicksum(x[i, 'h', k, r] for i in Nw if (i, 'h', k, r) in x)
        m.addConstr(out_h == in_h, name=f"eq3_flow_home_{k}_{r}")

# eq_4: Depodan en fazla 1 çıkış
for k in K:
    for r in R:
        out_h = quicksum(x['h', j, k, r] for j in Nw if ('h', j, k, r) in x)
        m.addConstr(out_h <= 1, name=f"eq4_one_depart_{k}_{r}")

if ENABLE_C2:
    m.addConstrs((
        gp.quicksum(x[i, j, k, r] for i in N for j in N if (i, j, k, r) in x)
        <= gp.quicksum(f[p, k, r] for p in P)
        for k in K for r in R
    ), name="eq5_route_activation")

# eq_6: j’den çıkan akış, j’nin orig/dest olduğu atamalarla sınırlı
for j in Nw:
    for k in K:
        for r in R:
            lhs = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            rhs = quicksum(f[p, k, r] for p in P if (orig[p] == j or dest[p] == j))
            m.addConstr(lhs <= rhs, name=f"eq6_restrict_by_od_{j}_{k}_{r}")

# eq_7–eq_8: Akış korunumu + tek giriş
for j in Nw:
    for k in K:
        for r in R:
            inflow  = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[j, i, k, r] for i in N if i != j and (j, i, k, r) in x)
            m.addConstr(inflow == outflow, name=f"eq7_flow_keep_{j}_{k}_{r}")
            m.addConstr(inflow <= 1,       name=f"eq8_one_in_{j}_{k}_{r}")

# eq_9: Her ürün tam olarak bir araca-ve-rotaya atanır
for p in P:
    m.addConstr(quicksum(f[p, k, r] for k in K for r in R) == 1, name=f"eq9_assign_eq1_{p}")

# eq_10–eq_11: Atandıysa origin ve destination ziyaret
for p in P:
    for k in K:
        for r in R:
            org = orig[p]; de = dest[p]
            lhs_o = quicksum(x[i, org, k, r] for i in N if i != org and (i, org, k, r) in x)
            lhs_d = quicksum(x[i, de,  k, r] for i in N if i != de  and (i, de,  k, r) in x)
            m.addConstr(lhs_o >= f[p, k, r], name=f"eq10_visit_origin_p{p}_{k}_{r}")
            m.addConstr(lhs_d >= f[p, k, r], name=f"eq11_visit_dest_p{p}_{k}_{r}")

# eq_12: İlk turun depo çıkışı (= 00:00 veya 07:00)
for k in K:
    if START_AT_ZERO:
        m.addConstr(td['h', k, 'r1'] == 0,   name=f"eq12_home_depart_0000_{k}")
    else:
        m.addConstr(td['h', k, 'r1'] == 420, name=f"eq12_home_depart_0700_{k}")

# eq_13: Rotalar arası zaman tutarlılığı
for k in K:
    for idx in range(1, len(R)):
        r_prev, r_now = R[idx-1], R[idx]
        m.addConstr(td['h', k, r_now] >= ta['h', k, r_prev] + eps_route_gap,
                    name=f"eq13_next_route_after_prev_{k}_{r_now}")

# eq_14: Zaman tutarlılığı — dist_min
for (i, j), tmin in dist_min.items():
    if i == j: continue
    for k in K:
        for r in R:
            if (i, j, k, r) in x:
                m.addConstr(
                    ta[j, k, r] >= td[i, k, r] + tmin * x[i, j, k, r] - M * (1 - x[i, j, k, r]),
                    name=f"eq14_time_{i}_{j}_{k}_{r}"
                )

# ts ≥ ta
m.addConstrs((ts[j, k, r] >= ta[j, k, r] for j in Nw for k in K for r in R), name="TS_ge_TA")

# eq_15: Servis (boşaltma)
for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(
                ts[j, k, r] >= ta[j, k, r] + quicksum(su[p] * f[p, k, r] for p in P if dest[p] == j),
                name=f"eq15_service_unload_{j}_{k}_{r}"
            )

# eq_16: Ready time (origin)
for p in P:
    rt = ep_min_abs[p]
    hnode = orig[p]
    for k in K:
        for r in R:
            if hnode in Nw:
                m.addConstr(ts[hnode, k, r] >= rt * f[p, k, r], name=f"eq16_ready_{p}_{k}_{r}")
                m.addConstr(ts[hnode, k, r] >= rt - M * (1 - f[p, k, r]),
                            name=f"eq16b_ready_max_{p}_{k}_{r}")

# eq_17: Yükleme sonrası ayrılma
for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(
                td[j, k, r] >= ts[j, k, r] + quicksum(sl[p] * f[p, k, r] for p in P if orig[p] == j),
                name=f"eq17_depart_after_load_{j}_{k}_{r}"
            )

# eq_18: Pickup -> Delivery öncelik
for p in P:
    for k in K:
        for r in R:
            m.addConstr(
                ta[dest[p], k, r] >= td[orig[p], k, r] - M * (1 - f[p, k, r]),
                name=f"eq18_origin_before_dest_{p}_{k}_{r}"
            )

# eq_19: Bekleme alt sınırı
for p in P:
    rt = ep_min_abs[p]
    for k in K:
        for r in R:
            m.addConstr(
                w[p] >= ta[dest[p], k, r] - rt - M * (1 - f[p, k, r]),
                name=f"eq19_wait_lb_{p}_{k}_{r}"
            )

# eq_20: Net yük değişimi
for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(q_product[p] * f[p, k, r] for p in P if orig[p] == j)
            load_out = quicksum(q_product[p] * f[p, k, r] for p in P if dest[p] == j)
            m.addConstr(delta[j, k, r] >= load_in - load_out, name=f"eq20_delta_{j}_{k}_{r}")

# eq_21–eq_22: Yük evrimi
for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M * (1 - x[i, j, k, r]),
                        name=f"eq21_load_lb_{i}_{j}_{k}_{r}"
                    )
                    m.addConstr(
                        y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M * (1 - x[i, j, k, r]),
                        name=f"eq22_load_ub_{i}_{j}_{k}_{r}"
                    )

# eq_23: Kapasite
for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(y[j, k, r] <= q_vehicle[k], name=f"eq23_capacity_{j}_{k}_{r}")

# Y nonneg, depoda tur başı 0
m.addConstrs((y[j, k, r] >= 0.0 for j in N for k in K for r in R), name="Y_nonneg")
m.addConstrs((y['h', k, r] == 0.0 for k in K for r in R),          name="Y_home_zero")

# eq_24: Rota aktivasyon monotonisi (r -> r+1)
for k in K:
    for idx in range(len(R)-1):
        r_cur = R[idx]
        r_nxt = R[idx+1]
        lhs_cur = gp.quicksum(x['h', j, k, r_cur] for j in Nw if ('h', j, k, r_cur) in x)
        lhs_nxt = gp.quicksum(x['h', j, k, r_nxt] for j in Nw if ('h', j, k, r_nxt) in x)
        m.addConstr(lhs_cur >= lhs_nxt, name=f"eq24_route_monotone[{k},{r_cur}->{r_nxt}]")

# eq_25–eq_27: MTZ
for k in K:
    for r in R:
        for i in Nw:
            for j in Nw:
                if i == j: continue
                if (i, j, k, r) in x:
                    m.addConstr(
                        u[j, k, r] >= u[i, k, r] + 1 - U * (1 - x[i, j, k, r]),
                        name=f"eq25_MTZ[{i}->{j},{k},{r}]"
                    )

for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            m.addConstr(u[j, k, r] <= U * indeg, name=f"eq26_u_ub[{j},{k},{r}]")
            m.addConstr(u[j, k, r] >= indeg,     name=f"eq27_u_lb[{j},{k},{r}]")

# eq_28: Sıra öncelik (dest > orig)
for k in K:
    for r in R:
        for p in P:
            op = orig[p]; dp = dest[p]
            if (op in Nw) and (dp in Nw):
                m.addConstr(
                    u[dp, k, r] >= u[op, k, r] + 1 - U * (1 - f[p, k, r]),
                    name=f"eq28_seq_prec[{p},{k},{r}]"
                )

# ============================================================================
# 6) PARAMETRELER ve LOG
# ============================================================================
timestamp   = datetime.now().strftime('%Y_%m_%d_%H_%M')
excel_base  = f"result_of_run_{timestamp}"
excel_dir   = 'results'
os.makedirs(excel_dir, exist_ok=True)
excel_path  = os.path.join(excel_dir, f"{excel_base}.xlsx")
log_path    = os.path.join(desktop_dir, f"{excel_base}.txt")

m.setParam('TimeLimit', TIME_LIMIT)
m.setParam('MIPGap', MIP_GAP)
m.setParam('Threads', THREADS)
m.setParam('Presolve', 2)
m.setParam('MIPFocus', 1)
m.setParam('Heuristics', 0.1)
m.setParam('SoftMemLimit', 6)
m.setParam('NodefileStart', 0.5)
m.setParam('Cuts', 2)
m.setParam('Aggregate', 2)
m.setParam('Symmetry', 2)
m.setParam('LogFile', log_path)

m.update()
m.printStats()

try:
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write("=== BASE DATE INFO ===\n")
        logf.write(f"Base date (t=0): {BASE_DATE.strftime('%Y-%m-%d 00:00')}\n")
        logf.write(f"Horizon (days): {HORIZON_DAYS}\n")
        logf.write("======================\n\n")
except Exception:
    pass

m.optimize()

def append_log(text):
    try:
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(text + "\n")
    except Exception as e:
        print("Log yazım hatası:", e)

# ============================================================================
# 7) RAPOR
# ============================================================================
if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    xdf = tidy(x.values()); xdf.columns = ['var','i','j','k','r','val']
    fdf = tidy(f.values()); fdf.columns = ['var','p','k','r','val']
    udf = tidy(u.values()); udf.columns = ['var','j','k','r','u']
    wdf = tidy(w.values()); wdf.columns = ['var','p','w_val']

    tadf = tidy(ta.values()); tadf.columns=['var','node','k','r','time']
    tddf = tidy(td.values()); tddf.columns=['var','node','k','r','time']
    tsdf = tidy(ts.values()); tsdf.columns=['var','node','k','r','time']
    for df_ in (tadf, tddf, tsdf):
        df_['stamp'] = df_['time'].apply(lambda m_: minutes_to_stamp(m_, BASE_DATE))

    ydf   = tidy(y.values());    ydf.columns   = ['var','node','k','r','y_val']
    deldf = tidy(delta.values()); deldf.columns = ['var','node','k','r','delta_val']

    used = xdf[(xdf['val']>0.5)].copy()
    dep = (used[used['i']=='h'][['k','r']].drop_duplicates())

    visit = (used[used['j'].isin(Nw)]
             .merge(udf[['j','k','r','u']], on=['j','k','r'], how='left')
             .merge(tadf[['node','k','r','stamp']].rename(columns={'node':'j','stamp':'ta_stamp'}), on=['j','k','r'], how='left')
             .merge(tddf[['node','k','r','stamp']].rename(columns={'node':'j','stamp':'td_stamp'}), on=['j','k','r'], how='left')
             .merge(ydf[['node','k','r','y_val']].rename(columns={'node':'j','y_val':'y_after'}), on=['j','k','r'], how='left'))
    visit = (visit[['k','r','j','u','ta_stamp','td_stamp','y_after']]
             .drop_duplicates().sort_values(['k','r','u'], ignore_index=True))
    visit = visit.merge(dep, on=['k','r'], how='inner')

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as wr:
        pd.DataFrame([{
            'objective': OBJECTIVE,
            'obj_value': m.objVal if m.SolCount>0 else None,
            'best_bound': getattr(m, 'objbound', None),
            'mip_gap': getattr(m, 'mipgap', None),
            'runtime': m.Runtime,
            'C2_enabled': ENABLE_C2,
            'start_at_zero': START_AT_ZERO,
            'U_(|Nw|)': U,
            'M_single': M,
            'base_date': BASE_DATE.strftime('%Y-%m-%d'),
            'horizon_days': HORIZON_DAYS,
            'num_vehicles_|K|': len(K),
            'num_routes_|R|': len(R),
            'vehicle_route_pairs_KxR': len(K)*len(R)
        }]).to_excel(wr, sheet_name='optimization_results', index=False)

        xdf.to_excel(wr, sheet_name='x_ijkr', index=False)
        fdf.to_excel(wr, sheet_name='f_pkr', index=False)
        udf.to_excel(wr, sheet_name='u_jkr', index=False)
        wdf.to_excel(wr, sheet_name='w_p', index=False)
        tadf.to_excel(wr, sheet_name='ta', index=False)
        tddf.to_excel(wr, sheet_name='td', index=False)
        tsdf.to_excel(wr, sheet_name='ts', index=False)
        ydf.to_excel(wr, sheet_name='y_jkr', index=False)
        deldf.to_excel(wr, sheet_name='delta_jkr', index=False)
        visit.to_excel(wr, sheet_name='itinerary', index=False)

    append_log("\n=== RUN SUMMARY ===")
    append_log(f"Objective: {OBJECTIVE}")
    append_log(f"Status: {m.Status}")
    append_log(f"ObjValue: {getattr(m, 'objVal', None)}")
    append_log(f"BestBound: {getattr(m, 'objbound', None)}")
    append_log(f"MIPGap: {getattr(m, 'mipgap', None)}")
    append_log(f"Runtime (s): {getattr(m, 'Runtime', None)}")
    append_log(f"Excel file: {excel_path}")
    append_log(f"Base date: {BASE_DATE.strftime('%Y-%m-%d')}, Horizon days: {HORIZON_DAYS}")
    append_log(f"|K| = {len(K)} (vehicles), |R| = {len(R)} (routes), toplam kombinasyon: {len(K)*len(R)} (K×R)")

    print("Excel yazıldı:", excel_path)
    print("Log yazıldı:", log_path)

elif m.status == GRB.INFEASIBLE:
    try:
        m.computeIIS()
        m.write("infeasible.ilp")
    except Exception:
        pass
    try:
        append_log("\n=== RUN SUMMARY ===")
        append_log("Status: INFEASIBLE")
        append_log("IIS kaydedildi: infeasible.ilp")
    except Exception:
        pass
    print("Model infeasible. IIS hesaplanıyor…\nIIS kaydedildi: infeasible.ilp")

else:
    try:
        append_log("\n=== RUN SUMMARY ===")
        append_log(f"Status: {m.status}")
    except Exception:
        pass
    print(f"Çözüm bulunamadı. Status = {m.status}")