import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import re, os
from datetime import datetime, time, timedelta

START_AT_ZERO  = False
BASE_DATE      = datetime(2025, 1, 1)
HORIZON_DAYS   = 30

TIME_LIMIT = 43200
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
    if total_min is None: return ''
    dt = base_dt + timedelta(minutes=float(total_min))
    return dt.strftime("%Y-%m-%d %H:%M")

def tidy(var_values):
    rows=[]
    for v in var_values:
        parts = re.split(r"\[|,|]", v.varName)[:-1]
        val   = round(v.X,4) if v.X is not None else None
        rows.append(parts+[val])
    return pd.DataFrame(rows)

# ---------- Veri Okuma ----------
data_path  = r"C:\Users\Asus\Desktop\Er\\"
desktop_dir= r"C:\Users\Asus\Desktop"

nodes     = pd.read_excel(os.path.join(data_path, "nodes.xlsx"))
vehicles  = pd.read_excel(os.path.join(data_path, "vehicles.xlsx"))
products  = pd.read_excel(os.path.join(data_path, "products.xlsx")).head(50)

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

# ---------- Kümeler / Parametreler ----------
nodes['node_id'] = nodes['node_id'].astype(str).str.strip()
N  = nodes['node_id'].dropna().drop_duplicates().tolist()
Nw = [n for n in N if n != 'h']

vehicles['vehicle_id'] = vehicles['vehicle_id'].astype(str).str.strip()
vehicles = vehicles.dropna(subset=['vehicle_id']).drop_duplicates('vehicle_id', keep='first')
K = vehicles['vehicle_id'].tolist()
if len(K) < 3:
    raise ValueError("Bu kurgu için en az 3 araç gerekir (S1: 3 araç, S2: 2 araç).")
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

ep_min_day = dict(zip(P, [ready_to_min(v) for v in products['ready_time']]))
ep_min_abs = {p: ep_min_day[p] for p in P}

# Vardiya/rota takvimi
S1_ROUTES = [f"S1_r{i}" for i in range(1, 3+1)]
S2_ROUTES = [f"S2_r{i}" for i in range(1, 4+1)]
R = S1_ROUTES + S2_ROUTES

U = int(max(1, len(Nw)))
T_HORIZON = float(HORIZON_DAYS*1440)
MAX_CAP   = float(max(q_vehicle.values()) if q_vehicle else 0.0)
M = max(T_HORIZON, MAX_CAP)
eps_route_gap = 1e-3

# Vardiya pencereleri
S1_START = 7*60
S1_END   = 14*60 + 59
S2_START = 15*60
S2_END   = 23*60

def shift_window(r_name):
    if r_name in S1_ROUTES: return S1_START, S1_END
    if r_name in S2_ROUTES: return S2_START, S2_END
    return 0, int(T_HORIZON)

# ---------- Model ----------
m = gp.Model("InternalLogistics_LaTeXAligned")

def arc_exists(i, j, k, r):
    return (i != j) and not (i == 'h' and j == 'h') and ((i, j) in dist_metre or (i, j) in dist_min)

arcs = [(i, j, k, r) for i in N for j in N for k in K for r in R if arc_exists(i, j, k, r)]

# Değişkenler
x     = m.addVars(arcs, vtype=GRB.BINARY, name="x")
f     = m.addVars(P, K, R, vtype=GRB.BINARY, name="f")
y     = m.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="y")
ta    = m.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="ta")
td    = m.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="td")
ts    = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, name="ts")
delta = m.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="delta")
u     = m.addVars(Nw, K, R, vtype=GRB.INTEGER, lb=0, ub=U, name="u")
w     = m.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
z     = m.addVars(K, R, vtype=GRB.BINARY, name="z")

def X(i, j, k, r):
    return x[(i, j, k, r)] if (i, j, k, r) in x else gp.LinExpr(0.0)

# ---------- Amaç (Σ w_p) ve ε-kısıtı (TIME ≤ 8151) ----------
wait_expr = quicksum(w[p] for p in P)  # ∑ w_p
time_expr = quicksum(dist_min.get((i, j), 0.0) * x[i, j, k, r] for (i, j, k, r) in arcs)

m.setObjective(wait_expr, GRB.MINIMIZE)
m.addConstr(time_expr <= 8151, name="TIME_LIMIT_EPSILON")

# ---------- Kısıtlar ----------
# eq_3: Depoya giriş=çıkış (rota kapatma)
for k in K:
    for r in R:
        out_h = quicksum(x['h', j, k, r] for j in Nw if ('h', j, k, r) in x)
        in_h  = quicksum(x[i, 'h', k, r] for i in Nw if (i, 'h', k, r) in x)
        m.addConstr(out_h == in_h, name=f"eq3_flow_home[{k},{r}]")

# eq_4: Depodan en fazla 1 çıkış
for k in K:
    for r in R:
        out_h = quicksum(x['h', j, k, r] for j in Nw if ('h', j, k, r) in x)
        m.addConstr(out_h <= 1, name=f"eq4_one_depart[{k},{r}]")

# z aktivasyonu ve bağlayıcılar (eq_5' ve eq_5'')
for k in K:
    for r in R:
        dep_out = quicksum(x['h', j, k, r] for j in Nw if ('h', j, k, r) in x)
        m.addConstr(dep_out == z[k, r], name=f"route_open[{k},{r}]")  # eq_5'
        m.addConstr(quicksum(x[i, j, k, r] for i in N for j in N if (i, j, k, r) in x)
                    <= len(N) * z[k, r], name=f"route_edge_cap[{k},{r}]")  # eq_5''
        m.addConstr(quicksum(f[p, k, r] for p in P)
                    <= len(P) * z[k, r], name=f"route_assign_cap[{k},{r}]")  # eq_5''

# eq_6: İstasyon ziyareti yalnız ilgili ürün varsa (giriş/çıkış simetrik)
for j in Nw:
    for k in K:
        for r in R:
            rhs = quicksum(f[p, k, r] for p in P if (orig[p] == j or dest[p] == j))
            inflow  = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[j, i, k, r] for i in N if i != j and (j, i, k, r) in x)
            m.addConstr(inflow  <= rhs, name=f"eq6_in_restrict[{j},{k},{r}]")
            m.addConstr(outflow <= rhs, name=f"eq6_out_restrict[{j},{k},{r}]")

# eq_7–eq_8: Akış korunumu + istasyona en çok bir giriş
for j in Nw:
    for k in K:
        for r in R:
            inflow  = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[j, i, k, r] for i in N if i != j and (j, i, k, r) in x)
            m.addConstr(inflow == outflow, name=f"eq7_flow_keep[{j},{k},{r}]")
            m.addConstr(inflow <= 1,       name=f"eq8_one_in[{j},{k},{r}]")

# eq_9: Her ürün tam bir (k,r)’ye atanır
for p in P:
    m.addConstr(quicksum(f[p, k, r] for k in K for r in R) == 1, name=f"eq9_assign[{p}]")

# eq_10–eq_11: Atandıysa origin/destination ziyaret
for p in P:
    op, dp = orig[p], dest[p]
    for k in K:
        for r in R:
            if (op in N) and (dp in N):
                lhs_o = quicksum(x[i, op, k, r] for i in N if i != op and (i, op, k, r) in x)
                lhs_d = quicksum(x[i, dp, k, r] for i in N if i != dp and (i, dp, k, r) in x)
                m.addConstr(lhs_o >= f[p, k, r], name=f"eq10_visit_origin[{p},{k},{r}]")
                m.addConstr(lhs_d >= f[p, k, r], name=f"eq11_visit_dest[{p},{k},{r}]")

# eq_12: İlk turun depo çıkışı referansı
for k in K:
    if 'S1_r1' in R:
        if START_AT_ZERO:
            m.addConstr(td['h', k, 'S1_r1'] == 0,   name=f"eq12_home_depart_0000[{k}]")
        else:
            m.addConstr(td['h', k, 'S1_r1'] == 420, name=f"eq12_home_depart_0700[{k}]")  # 07:00

# eq_13: Rotalar arası zaman tutarlılığı
for k in K:
    for idx in range(1, len(R)):
        r_prev, r_now = R[idx-1], R[idx]
        m.addConstr(td['h', k, r_now] >= ta['h', k, r_prev] + eps_route_gap,
                    name=f"eq13_next_route_after_prev[{k},{r_prev}->{r_now}]")

# eq_14: Seyir zamanı (dakika) ile varış zamanı
for (i, j), tmin in dist_min.items():
    if i == j: continue
    for k in K:
        for r in R:
            if (i, j, k, r) in x:
                m.addConstr(
                    ta[j, k, r] >= td[i, k, r] + tmin * x[i, j, k, r] - M * (1 - x[i, j, k, r]),
                    name=f"eq14_time[{i}->{j},{k},{r}]"
                )

# ts ≥ ta (yardımcı)
m.addConstrs((ts[j, k, r] >= ta[j, k, r] for j in Nw for k in K for r in R), name="TS_ge_TA")

# eq_15: Boşaltma servis süresi
for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(
                ts[j, k, r] >= ta[j, k, r] + quicksum(su[p] * f[p, k, r] for p in P if dest[p] == j),
                name=f"eq15_service_unload[{j},{k},{r}]"
            )

# eq_16: Ready time (origin)
for p in P:
    rt = ep_min_abs[p]
    j0 = orig[p]
    if j0 in Nw:
        for k in K:
            for r in R:
                m.addConstr(ts[j0, k, r] >= rt * f[p, k, r], name=f"eq16_ready[{p},{k},{r}]")
                m.addConstr(ts[j0, k, r] >= rt - M * (1 - f[p, k, r]),
                            name=f"eq16b_ready_bigM[{p},{k},{r}]")

# eq_17: Yükleme sonrası ayrılma
for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(
                td[j, k, r] >= ts[j, k, r] + quicksum(sl[p] * f[p, k, r] for p in P if orig[p] == j),
                name=f"eq17_depart_after_load[{j},{k},{r}]"
            )

# eq_18: Pickup -> Delivery öncelik
for p in P:
    op, dp = orig[p], dest[p]
    for k in K:
        for r in R:
            if (op in N) and (dp in N):
                m.addConstr(
                    ta[dp, k, r] >= td[op, k, r] - M * (1 - f[p, k, r]),
                    name=f"eq18_origin_before_dest[{p},{k},{r}]"
                )

# eq_19: Bekleme alt sınırı
for p in P:
    rt = ep_min_abs[p]
    for k in K:
        for r in R:
            m.addConstr(
                w[p] >= ta[dest[p], k, r] - rt - M * (1 - f[p, k, r]),
                name=f"eq19_wait_lb[{p},{k},{r}]"
            )

# eq_20: Net yük değişimi (eşitlik — sıkılaştırma)
for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(q_product[p] * f[p, k, r] for p in P if orig[p] == j)
            load_out = quicksum(q_product[p] * f[p, k, r] for p in P if dest[p] == j)
            m.addConstr(delta[j, k, r] >= load_in - load_out, name=f"eq20_delta_eq[{j},{k},{r}]")

# eq_21–eq_22: Yük evrimi (Big-M lineerleştirmesi)
for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(
                        y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M * (1 - x[i, j, k, r]),
                        name=f"eq21_load_lb[{i}->{j},{k},{r}]"
                    )
                    m.addConstr(
                        y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M * (1 - x[i, j, k, r]),
                        name=f"eq22_load_ub[{i}->{j},{k},{r}]"
                    )

# eq_23: Kapasite
for j in Nw:
    for k in K:
        for r in R:
            m.addConstr(y[j, k, r] <= q_vehicle[k], name=f"eq23_capacity[{j},{k},{r}]")

# y alt sınır ve depoda 0
m.addConstrs((y[j, k, r] >= 0.0 for j in N for k in K for r in R), name="Y_nonneg")
m.addConstrs((y['h', k, r] == 0.0 for k in K for r in R),          name="Y_home_zero")

# eq_24: Rota aktivasyon monotonisi (z)
for k in K:
    for idx in range(len(R)-1):
        r_cur, r_nxt = R[idx], R[idx+1]
        m.addConstr(z[k, r_cur] >= z[k, r_nxt], name=f"eq24_z_monotone[{k},{r_cur}->{r_nxt}]")

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

# eq_28: Ürün bazlı sıra önceliği
for k in K:
    for r in R:
        for p in P:
            op, dp = orig[p], dest[p]
            if (op in Nw) and (dp in Nw):
                m.addConstr(
                    u[dp, k, r] >= u[op, k, r] + 1 - U * (1 - f[p, k, r]),
                    name=f"eq28_seq_prec[{p},{k},{r}]"
                )


bigM_t = T_HORIZON

# Depo çıkış/dönüş – yalnızca aktif rotalara uygulanır
for r in R:
    s_start, s_end = shift_window(r)
    for k in K:
        m.addConstr(td['h', k, r] >= s_start - bigM_t*(1 - z[k, r]),
                    name=f"shift_td_ge_start[{k},{r}]")
        m.addConstr(ta['h', k, r] <= s_end   + bigM_t*(1 - z[k, r]),
                    name=f"shift_ta_le_end[{k},{r}]")

# Rota içindeki tüm ta/ts/td değerlerini vardiya içine kilitle
ENFORCE_ALL_TIMES_IN_SHIFT = True
if ENFORCE_ALL_TIMES_IN_SHIFT:
    for r in R:
        s_start, s_end = shift_window(r)
        for k in K:
            for j in N:
                m.addConstr(ta[j, k, r] >= s_start - bigM_t*(1 - z[k, r]),
                            name=f"shift_ta_ge_start[{j},{k},{r}]")
                m.addConstr(ta[j, k, r] <= s_end   + bigM_t*(1 - z[k, r]),
                            name=f"shift_ta_le_end2[{j},{k},{r}]")
            for j in Nw:
                m.addConstr(ts[j, k, r] >= s_start - bigM_t*(1 - z[k, r]),
                            name=f"shift_ts_ge_start[{j},{k},{r}]")
                m.addConstr(ts[j, k, r] <= s_end   + bigM_t*(1 - z[k, r]),
                            name=f"shift_ts_le_end[{j},{k},{r}]")
            for j in N:
                m.addConstr(td[j, k, r] >= s_start - bigM_t*(1 - z[k, r]),
                            name=f"shift_td_ge_start2[{j},{k},{r}]")
                m.addConstr(td[j, k, r] <= s_end   + bigM_t*(1 - z[k, r]),
                            name=f"shift_td_le_end[{j},{k},{r}]")

S2_ALLOWED = set(K[:2])
for r in S2_ROUTES:
    for k in K:
        if k not in S2_ALLOWED:
            m.addConstr(quicksum(x['h', j, k, r] for j in Nw if ('h', j, k, r) in x) == 0,
                        name=f"S2_ban_depart[{k},{r}]")
            m.addConstr(quicksum(f[p, k, r] for p in P) == 0,
                        name=f"S2_ban_assign[{k},{r}]")
            m.addConstr(z[k, r] == 0, name=f"S2_ban_z[{k},{r}]")

# ---------- Parametreler ve Çözüm ----------
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
m.setParam('LogFile', log_path)

m.update(); m.printStats()
with open(log_path, 'a', encoding='utf-8') as logf:
    logf.write("=== BASE DATE INFO ===\n")
    logf.write(f"Base date (t=0): {BASE_DATE.strftime('%Y-%m-%d 00:00')}\n")
    logf.write(f"Horizon (days): {HORIZON_DAYS}\n")
    logf.write("======================\n\n")

m.optimize()

def append_log(text):
    try:
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(text + "\n")
    except Exception as e:
        print("Log yazım hatası:", e)

# ---------- Rapor ----------
if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    xdf = tidy(x.values()); xdf.columns = ['var','i','j','k','r','val']
    fdf = tidy(f.values()); fdf.columns = ['var','p','k','r','val']
    udf = tidy(u.values()); udf.columns = ['var','j','k','r','u']
    zdf = tidy(z.values()); zdf.columns = ['var','k','r','z']
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
            'objective': 'min_total_wait',
            'obj_value': m.objVal if m.SolCount>0 else None,
            'best_bound': getattr(m, 'objbound', None),
            'mip_gap': getattr(m, 'mipgap', None),
            'runtime': m.Runtime,
            '|N|': len(N), '|Nw|': len(Nw),
            '|K|': len(K), '|R|': len(R), 'KxR': len(K)*len(R),
            'U_(|Nw|)': U, 'M_single': M,
            'base_date': BASE_DATE.strftime('%Y-%m-%d'),
            'horizon_days': HORIZON_DAYS,
            'epsilon_time_upper': 8151
        }]).to_excel(wr, sheet_name='optimization_results', index=False)

        xdf.to_excel(wr, sheet_name='x_ijkr', index=False)
        fdf.to_excel(wr, sheet_name='f_pkr', index=False)
        udf.to_excel(wr, sheet_name='u_jkr', index=False)
        zdf.to_excel(wr, sheet_name='z_kr', index=False)
        wdf.to_excel(wr, sheet_name='w_p', index=False)
        tadf.to_excel(wr, sheet_name='ta', index=False)
        tddf.to_excel(wr, sheet_name='td', index=False)
        tsdf.to_excel(wr, sheet_name='ts', index=False)
        ydf.to_excel(wr, sheet_name='y_jkr', index=False)
        deldf.to_excel(wr, sheet_name='delta_jkr', index=False)
        visit.to_excel(wr, sheet_name='itinerary', index=False)

    append_log("\n=== RUN SUMMARY ===")
    append_log("Objective: min_total_wait")
    append_log(f"TIME ε upper bound: 8151")
    append_log(f"Status: {m.Status}")
    append_log(f"ObjValue: {getattr(m, 'objVal', None)}")
    append_log(f"BestBound: {getattr(m, 'objbound', None)}")
    append_log(f"MIPGap: {getattr(m, 'mipgap', None)}")
    append_log(f"Runtime (s): {getattr(m, 'Runtime', None)}")
    append_log(f"Excel file: {excel_path}")
    append_log(f"Base date: {BASE_DATE.strftime('%Y-%m-%d')}, Horizon days: {HORIZON_DAYS}")
    append_log(f"|K| = {len(K)}, |R| = {len(R)}, KxR = {len(K)*len(R)}")

    print("Excel yazıldı:", excel_path)
    print("Log yazıldı:", log_path)

elif m.status == GRB.INFEASIBLE:
    try:
        m.computeIIS(); m.write("infeasible.ilp")
    except Exception:
        pass
    print("Model infeasible. IIS kaydedildi: infeasible.ilp")
else:
    print(f"Çözüm bulunamadı. Status = {m.status}")