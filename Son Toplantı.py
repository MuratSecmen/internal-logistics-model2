import pandas as pd
from gurobipy import Model, GRB, quicksum
import re, os
import gurobipy as gp
from datetime import datetime, time

# ---------------------------- yardımcılar ----------------------------
def ready_to_min(v):
    """Timestamp/'HH:MM'/'YYYY-MM-DD HH:MM'/time/sayı -> dakika (int)"""
    if pd.isna(v): return 0
    if isinstance(v, (pd.Timestamp, datetime)): return int(v.hour)*60 + int(v.minute)
    if isinstance(v, time): return int(v.hour)*60 + int(v.minute)
    if isinstance(v, (int, float)) and not isinstance(v, bool): return int(v)
    s = str(v).strip()
    dt = pd.to_datetime(s, errors='coerce')
    if pd.notna(dt): return int(dt.hour)*60 + int(dt.minute)
    if ":" in s:
        pr = s.split(":")
        return int(pr[0])*60 + int(pr[1])
    return int(float(s))

def fmt_hhmm(total_min):
    total_min = int(round(total_min)) if total_min is not None else None
    if total_min is None: return ''
    if total_min < 0: return f"-{fmt_hhmm(-total_min)}"
    hh = (total_min // 60) % 24
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"

def tidy(var_values):
    rows=[]
    for v in var_values:
        parts = re.split(r"\[|,|]", v.varName)[:-1]
        val   = round(v.X,4) if v.X is not None else None
        rows.append(parts+[val])
    return pd.DataFrame(rows)

# ---------------------------- 1) veri ----------------------------
data_path = r"C:\Users\Asus\Desktop\Er\\"

nodes     = pd.read_excel(data_path + "nodes.xlsx")
vehicles  = pd.read_excel(data_path + "vehicles.xlsx")
products  = pd.read_excel(data_path + "products.xlsx").head(20)

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

dist_min   = _read_dist(data_path + "distances - dakika.xlsx", "duration_min")
dist_metre = _read_dist(data_path + "distances - metre.xlsx",  "duration_metre")

# ---------------------------- 2) setler/paramlar ----------------------------
nodes['node_id'] = nodes['node_id'].astype(str).str.strip()
N  = nodes['node_id'].dropna().drop_duplicates().tolist()
h = 'h'
Nw = [n for n in N if n != h]  # depo hariç

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
ep_min    = dict(zip(P, [ready_to_min(v) for v in products['ready_time']]))  # dakika
sl        = dict(zip(P, products['load_time']))
su        = dict(zip(P, products['unload_time']))

max_routes = min(len(P)//max(1,len(K)) + 1, 5)  # Rota sayısını sınırla
R = [f"r{i}" for i in range(1, max_routes+1)]

# Big-M ve eps
M_time = 900.0
M_load = 60.0
eps_route_gap = 1e-3

# ---------------------------- 3) model ----------------------------
model = Model("InternalLogistics")

# arc_exists'i dist_metre/ dist_min'e göre tanımla (sparse için)
def arc_exists(i, j, k, r):
    return (i != j) and not (i == h and j == h) and ((i, j) in dist_metre or (i, j) in dist_min)

# x[i,j,k,r] sparse tanımla
arcs = [(i, j, k, r) for i in N for j in N for k in K for r in R if arc_exists(i, j, k, r)]
x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

# rota-aktif: z[k,r]
z = model.addVars(K, R, vtype=GRB.BINARY, name="z")

# ürün atama
f   = model.addVars(P, K, R, vtype=GRB.BINARY, name="f")

# zamanlar/yük
y   = model.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="y")
ta  = model.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="ta")
td  = model.addVars(N,  K, R, vtype=GRB.CONTINUOUS, name="td")
ts  = model.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, name="ts")
delta = model.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="delta")

# MTZ sıralama: u[j,k,r] ∈ [0, Umax]
Umax = max(1, len(Nw))
u = model.addVars(Nw, K, R, vtype=GRB.INTEGER, lb=0, ub=Umax, name="u")

# bekleme değişkeni
w = model.addVars(P, vtype=GRB.CONTINUOUS, lb=0.0, name="w")

# ---------------------------- 4) amaç (MESAFE, metre) ----------------------------
obj = quicksum(dist_metre.get((i,j), 0.0) * x[i,j,k,r] for (i,j,k,r) in arcs)
model.setObjective(obj, GRB.MINIMIZE)

# ---------------------------- 5) yardımcı ----------------------------
def X(i, j, k, r):
    return x.get((i, j, k, r), 0.0)

# örnek iki alt/üst sınır kısıtı
model.addConstr(quicksum(dist_metre.get((i,j), 0.0) * x[i,j,k,r] for (i,j,k,r) in arcs) >= 120)
model.addConstr(quicksum(dist_metre.get((j,i), 0.0) * X(j,i,k,r) for (i,j,k,r) in arcs) >= 70)

# ---------------------------- 6) kısıtlar ----------------------------
# C3: Depo akış eşitliği
for k in K:
    for r in R:
        out_h = quicksum(x[h, j, k, r] for j in Nw if (h, j, k, r) in x)
        in_h  = quicksum(x[i, h, k, r] for i in Nw if (i, h, k, r) in x)
        model.addConstr(out_h == in_h, name=f"C3_flow_home_{k}_{r}")

# C4: Depodan en fazla 1 çıkış
for k in K:
    for r in R:
        out_h = quicksum(x[h, j, k, r] for j in Nw if (h, j, k, r) in x)
        model.addConstr(out_h <= 1, name=f"C4_one_depart_{k}_{r}")

# C5: Depodan çıkış atanan ürün sayısını aşamaz
model.addConstrs(
    (gp.quicksum(x[i, j, k, r] for i in N if (i, j, k, r) in x)
     <= gp.quicksum(f[p, k, r] for p in P)
     for j in Nw for k in K for r in R),
    name="C5_leave_only_if_assigned"
)

# C6: j’deki akış, j’nin orig/dest olduğu atamalarla sınırlı
for j in Nw:
    for k in K:
        for r in R:
            inflow_at_j = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            rhs = quicksum(f[p, k, r] for p in P if orig[p] == j or dest[p] == j)
            model.addConstr(inflow_at_j <= rhs, name=f"C6_restrict_by_od_{j}_{k}_{r}")

# C7–C8: Akış korunumu + tek giriş
for j in Nw:
    for k in K:
        for r in R:
            inflow  = quicksum(x[i, j, k, r] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[j, i, k, r] for i in N if i != j and (j, i, k, r) in x)
            model.addConstr(inflow == outflow, name=f"C7_flow_keep_{j}_{k}_{r}")
            model.addConstr(inflow <= 1,       name=f"C8_one_in_{j}_{k}_{r}")

# C9: Her ürün tam olarak bir araca-ve-rotaya atanır
for p in P:
    model.addConstr(quicksum(f[p, k, r] for k in K for r in R) == 1, name=f"C9_assign_eq1_{p}")

# C10–C11: Atandıysa origin ve destination ziyaret
for p in P:
    for k in K:
        for r in R:
            org = orig[p]; de = dest[p]
            lhs_o = quicksum(x[i, org, k, r] for i in N if i != org and (i, org, k, r) in x)
            lhs_d = quicksum(x[i, de,  k, r] for i in N if i != de  and (i, de,  k, r) in x)
            model.addConstr(lhs_o >= f[p, k, r], name=f"C10_visit_origin_p{p}_{k}_{r}")
            model.addConstr(lhs_d >= f[p, k, r], name=f"C11_visit_dest_p{p}_{k}_{r}")

# C12: İlk turun depo çıkış zamanı = 07:00 (420 dk)
for k in K:
    model.addConstr(td[h, k, 'r1'] == 420, name=f"C12_home_depart_0700_{k}")

# C13: Rotalar arası zaman tutarlılığı (r>1)
for k in K:
    for idx in range(1, len(R)):
        r_prev, r_now = R[idx-1], R[idx]
        model.addConstr(td[h, k, r_now] >= ta[h, k, r_prev] + eps_route_gap,
                        name=f"C13_next_route_after_prev_{k}_{r_now}")

# C14: Zaman tutarlılığı — dist_min (dakika)
for (i, j), tmin in dist_min.items():
    if i == j:
        continue
    for k in K:
        for r in R:
            if (i, j, k, r) in x:
                model.addConstr(
                    ta[j, k, r] >= td[i, k, r] + tmin * x[i, j, k, r] - M_time * (1 - x[i, j, k, r]),
                    name=f"C14_time_{i}_{j}_{k}_{r}"
                )

# C15: Servis (boşaltma)
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(
                ts[j, k, r] >= ta[j, k, r] + quicksum(su[p] * f[p, k, r] for p in P if dest[p] == j),
                name=f"C15_service_unload_{j}_{k}_{r}"
            )

# C16: Ürün hazır edilmeden servis (origin) başlamaz
for p in P:
    rt = ep_min[p]; hnode = orig[p]
    for k in K:
        for r in R:
            if hnode in Nw:
                model.addConstr(ts[hnode, k, r] >= rt * f[p, k, r], name=f"C16_ready_{p}_{k}_{r}")

# C17: Yükleme sonrası ayrılma
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(
                td[j, k, r] >= ts[j, k, r] + quicksum(sl[p] * f[p, k, r] for p in P if orig[p] == j),
                name=f"C17_depart_after_load_{j}_{k}_{r}"
            )

# C18: Önce yüklenir sonra teslim edilir
for p in P:
    for k in K:
        for r in R:
            model.addConstr(
                ta[dest[p], k, r] >= td[orig[p], k, r] - M_time * (1 - f[p, k, r]),
                name=f"C18_origin_before_dest_{p}_{k}_{r}"
            )

# C19: Depo zaman ilişkisi
for k in K:
    for r in R:
        model.addConstr(ta[h, k, r] >= td[h, k, r], name=f"C19_home_time_{k}_{r}")

# C20: Bekleme alt sınırı (delivery - ready)
for p in P:
    rt = ep_min[p]
    for k in K:
        for r in R:
            model.addConstr(
                w[p] >= ta[dest[p], k, r] - rt - M_time * (1 - f[p, k, r]),
                name=f"C20_wait_lb_{p}_{k}_{r}"
            )

# C22: Net yük değişimi (delta)
for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(q_product[p] * f[p, k, r] for p in P if orig[p] == j)
            load_out = quicksum(q_product[p] * f[p, k, r] for p in P if dest[p] == j)
            model.addConstr(delta[j, k, r] >= load_in - load_out, name=f"C22_delta_{j}_{k}_{r}")

# C23–C24: Yük evrimi alt/üst sınırları
for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    model.addConstr(
                        y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M_load * (1 - x[i, j, k, r]),
                        name=f"C23_load_lb_{i}_{j}_{k}_{r}"
                    )
                    model.addConstr(
                        y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M_load * (1 - x[i, j, k, r]),
                        name=f"C24_load_ub_{i}_{j}_{k}_{r}"
                    )

# C25: Kapasite
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(y[j, k, r] <= q_vehicle[k], name=f"C25_capacity_{j}_{k}_{r}")

h = 'h'
# C26: z[k,r] = sum_{j∈Nw} x[h,j,k,r]
model.addConstrs(
    ((z[k, r] == gp.quicksum(v for v in x.select(h, '*', k, r))) for k in K for r in R),
    name="C26_z_link"
)

# C27: z_{k,r} ≥ z_{k,r+1} (r = 1..|R|-1)
for k in K:
    for idx in range(len(R) - 1):
        r_cur = R[idx]
        r_nxt = R[idx + 1]
        model.addConstr(
            z[k, r_cur] >= z[k, r_nxt],
            name=f"C27_z_monotone[{k},{r_cur}->{r_nxt}]"
        )

# C28: u_{jkr} ≥ u_{ikr} + 1 − Umax (1 − x_{i j k r}),  ∀ i≠j ∈ Nw
for k in K:
    for r in R:
        for i in Nw:
            for j in Nw:
                if i == j:
                    continue
                model.addConstr(
                    u[j, k, r] >= u[i, k, r] + 1 - Umax * (1 - x.get((i, j, k, r), 0.0)),
                    name=f"C28_MTZ[{i}->{j},{k},{r}]"
                )

# C29: u[j,k,r] ≤ Umax * Σ_{i∈N, i≠j} x[i,j,k,r]
for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(
                x[i, j, k, r]
                for i in N
                if i != j and (i, j, k, r) in x
            )
            model.addConstr(u[j, k, r] <= Umax * indeg, name=f"C29_u_ub[{j},{k},{r}]")

# C30: u[j,k,r] ≥ Σ_{i∈N, i≠j} x[i,j,k,r]
for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(
                x[i, j, k, r]
                for i in N
                if i != j and (i, j, k, r) in x
            )
            model.addConstr(u[j, k, r] >= indeg, name=f"C30_u_lb[{j},{k},{r}]")

# C31: u[dp,k,r] ≥ u[op,k,r] + 1 − Umax * (1 − f[p,k,r])
for k in K:
    for r in R:
        for p in P:
            op = orig[p]
            dp = dest[p]
            model.addConstr(
                u[dp, k, r] >= u[op, k, r] + 1 - Umax * (1 - f[p, k, r]),
                name=f"C31_seq_prec[{p},{k},{r}]"
            )

# ---------------------------- 7) parametreler ----------------------------
model.setParam('TimeLimit', 3600)
model.setParam('MIPGap', 0.03)
model.setParam('LogFile', 'gurobi_log.txt')
model.setParam('Presolve', 2)
model.setParam('MIPFocus', 1)
model.setParam('Heuristics', 0.1)
model.setParam('SoftMemLimit', 6)
model.setParam('NodefileStart', 0.5)
model.setParam('Threads', 6)

model.update()
model.printStats()
model.optimize()

# ---------------------------- 9) çıktı/rapor (görsel YOK) ----------------------------
if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    xdf = tidy(x.values()); xdf.columns = ['var','i','j','k','r','val']
    fdf = tidy(f.values()); fdf.columns = ['var','p','k','r','val']
    zdf = tidy(z.values()); zdf.columns = ['var','k','r','val']
    udf = tidy(u.values()); udf.columns = ['var','j','k','r','u']
    wdf = tidy(w.values()); wdf.columns = ['var','p','w_val']

    tadf = tidy(ta.values()); tadf.columns=['var','node','k','r','time']; tadf['hhmm']=tadf['time'].apply(fmt_hhmm)
    tddf = tidy(td.values()); tddf.columns=['var','node','k','r','time']; tddf['hhmm']=tddf['time'].apply(fmt_hhmm)
    tsdf = tidy(ts.values()); tsdf.columns=['var','node','k','r','time']; tsdf['hhmm']=tsdf['time'].apply(fmt_hhmm)

    ydf   = tidy(y.values());    ydf.columns   = ['var','node','k','r','y_val']
    deldf = tidy(delta.values()); deldf.columns = ['var','node','k','r','delta_val']

    used = xdf[(xdf['val']>0.5)].copy()
    visit = (used[used['j'].isin(Nw)]
             .merge(udf[['j','k','r','u']], on=['j','k','r'], how='left')
             .merge(tadf[['node','k','r','hhmm']].rename(columns={'node':'j','hhmm':'ta_hhmm'}), on=['j','k','r'], how='left')
             .merge(tddf[['node','k','r','hhmm']].rename(columns={'node':'j','hhmm':'td_hhmm'}), on=['j','k','r'], how='left')
             .merge(ydf[['node','k','r','y_val']].rename(columns={'node':'j','y_val':'y_after'}), on=['j','k','r'], how='left'))
    visit = visit[['k','r','j','u','ta_hhmm','td_hhmm','y_after']].drop_duplicates().sort_values(['k','r','u'], ignore_index=True)

    outdir = 'results'
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, f"result_of_run_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.xlsx")
    with pd.ExcelWriter(fn, engine='xlsxwriter') as wr:
        pd.DataFrame([{
            'obj_value': model.objVal if model.SolCount>0 else None,
            'best_bound': getattr(model, 'objbound', None),
            'mip_gap': getattr(model, 'mipgap', None),
            'runtime': model.Runtime
        }]).to_excel(wr, sheet_name='optimization_results', index=False)

        xdf.to_excel(wr, sheet_name='x_ijkr', index=False)
        fdf.to_excel(wr, sheet_name='f_pkr', index=False)
        zdf.to_excel(wr, sheet_name='z_kr', index=False)
        udf.to_excel(wr, sheet_name='u_jkr', index=False)
        wdf.to_excel(wr, sheet_name='w_p', index=False)
        tadf.to_excel(wr, sheet_name='ta', index=False)
        tddf.to_excel(wr, sheet_name='td', index=False)
        tsdf.to_excel(wr, sheet_name='ts', index=False)
        ydf.to_excel(wr, sheet_name='y_jkr', index=False)
        deldf.to_excel(wr, sheet_name='delta_jkr', index=False)
        visit.to_excel(wr, sheet_name='itinerary', index=False)

    print("Excel yazıldı:", fn)

elif model.status == GRB.INFEASIBLE:
    print("Model infeasible. IIS hesaplanıyor…")
    model.computeIIS()
    model.write("infeasible.ilp")
    print("IIS kaydedildi: infeasible.ilp")
else:
    print(f"Çözüm bulunamadı. Status = {model.status}")
