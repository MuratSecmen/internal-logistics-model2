import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import re, os
from datetime import datetime, time, timedelta
import sys

# =====================================================================
# TERMİNAL ÇIKTISINI DOSYAYA KAYDET
# =====================================================================
class TeeOutput:
    """Hem ekrana hem dosyaya yazdırır"""
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

print(f"✅ Terminal çıktısı kaydediliyor: {terminal_log_path}\n")

TIME_LIMIT = 3600
MIP_GAP    = 0.03
THREADS    = 6
EPS_WAIT = 150

# =====================================================================
# YARDIMCI FONKSİYONLAR
# =====================================================================
def ready_to_min(v):
    """Zaman değerini dakikaya çevirir"""
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
    """Dakikayı HH:MM formatına çevirir"""
    if pd.isna(minutes) or minutes is None:
        return ''
    minutes = float(minutes)
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

# =====================================================================
# VERİ YÜKLEME
# =====================================================================
data_path   = r"C:\Users\Asus\Desktop\Er\\"
desktop_dir = r"C:\Users\Asus\Desktop"

nodes    = pd.read_excel(os.path.join(data_path, "nodes.xlsx"))
vehicles = pd.read_excel(os.path.join(data_path, "vehicles.xlsx"))
products = pd.read_excel(os.path.join(data_path, "products.xlsx")).head(30)

def _read_dist(path, val_col):
    """Mesafe/süre matrisini oku"""
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

# =====================================================================
# VERİ İŞLEME
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

c  = dist_min
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
N_w_count = len(Nw)


M_16 = T_max - e_min + C_max
M_20 = T_max - e_min
M_22 = T_max
M_24 = Q_max
M_25 = Q_max

epsilon = 0.1
U = len(Nw)

print("\n" + "="*80)
print("TIGHT BIG-M DEĞERLERİ (Kısıt Numaraları İle)")
print("="*80)
print(f"M_16 (c16: Zaman)      = {M_16:.1f} dk")
print(f"M_20 (c20: Teslimat)   = {M_20:.1f} dk")
print(f"M_22 (c22: Bekleme)    = {M_22:.1f} dk")
print(f"M_24 (c24: Kap. Alt)   = {M_24:.0f} m²")
print(f"M_25 (c25: Kap. Üst)   = {M_25:.0f} m²")
print(f"U (MTZ) = {U} = |Nw| = {N_w_count}")
print("="*80 + "\n")
# ============================================================================

print("="*80)
print("VERİ YÜKLENDİ")
print("="*80)
print(f"|N|={len(N)}, |Nw|={len(Nw)}, |K|={len(K)}, |R|={len(R)}, |P|={len(P)}")
print(f"TIGHT M VALUES:")
print(f"  M_16={M_16:.1f}, M_20={M_20:.1f}, M_22={M_22:.1f}, M_24={M_24:.0f}, M_25={M_25:.0f}")
print(f"  U={U}, ε={epsilon}, EPS_WAIT={EPS_WAIT}")
print("="*80 + "\n")

# =====================================================================
# MODEL OLUŞTURMA
# =====================================================================
m = gp.Model("InternalLogistics_TightM_Optimized")

# =====================================================================
# KARAR DEĞİŞKENLERİ
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
# AMAÇ FONKSİYONU
# =====================================================================
obj1 = quicksum(ta['h', k, MAX_ROUTES] for k in K)
obj2 = quicksum(w[p] for p in P)
m.addConstr(obj2 <= EPS_WAIT, name="eps_wait_constraint")
m.setObjective(obj1 + 0.001*obj2, GRB.MINIMIZE)

print("="*80)
print("MODEL: TOPLAM VARIŞ ZAMANLARI BİRİNCİL AMAÇ")
print("="*80)
print(f"Birincil Amaç: Σ_k ta_hkr (Toplam Varış Zamanları)")
print(f"İkincil Amaç:  Σ_p w_p ≤ {EPS_WAIT} dakika")
print("="*80 + "\n")

# =====================================================================
# KISITLAR (4-12): Rota ve atama kısıtları
# =====================================================================
for k in K:
    for r in R:
        out_h = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        in_h  = quicksum(x[(j, 'h', k, r)] for j in Nw if (j, 'h', k, r) in x)
        m.addConstr(out_h == in_h, name=f"c4[{k},{r}]")
        m.addConstr(out_h <= 1, name=f"c5[{k},{r}]")
        lhs = quicksum(x[(i, j, k, r)] for i in N for j in N if (i, j, k, r) in x)
        rhs = quicksum(f[p, k, r] for p in P)
        m.addConstr(lhs <= (2*len(P)+1)*rhs, name=f"c6[{k},{r}]")

for j in Nw:
    for k in K:
        for r in R:
            lhs = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            rhs = quicksum(f[p, k, r] for p in P if (o[p] == j or d[p] == j))
            m.addConstr(lhs <= rhs, name=f"c7[{j},{k},{r}]")
            inflow  = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[(j, i, k, r)] for i in N if i != j and (j, i, k, r) in x)
            m.addConstr(inflow == outflow, name=f"c8[{j},{k},{r}]")
            m.addConstr(inflow <= 1, name=f"c9[{j},{k},{r}]")

for p in P:
    m.addConstr(quicksum(f[p, k, r] for k in K for r in R) == 1, name=f"c10[{p}]")
    op = o[p]
    if op in N:
        for k in K:
            for r in R:
                lhs = quicksum(x[(i, op, k, r)] for i in N if i != op and (i, op, k, r) in x)
                m.addConstr(lhs >= f[p, k, r], name=f"c11[{p},{k},{r}]")
    dp = d[p]
    if dp in N:
        for k in K:
            for r in R:
                lhs = quicksum(x[(i, dp, k, r)] for i in N if i != dp and (i, dp, k, r) in x)
                m.addConstr(lhs >= f[p, k, r], name=f"c12[{p},{k},{r}]")

# =====================================================================
# KISITLAR (13-15): Zaman sıralaması
# =====================================================================
for k in K:
    m.addConstr(td['h', k, 1] == 420, name=f"c13[{k}]")
    for r in R[1:]:
        m.addConstr(td['h', k, r] >= ta['h', k, r-1] + epsilon, name=f"c14[{k},{r}]")
        m.addConstr(ta['h', k, r] >= ta['h', k, r-1], name=f"c15[{k},{r}]")

# =====================================================================
# KISITLAR (16-22): Zaman penceresi - TIGHT M KULLANILIYOR! 🔥
# =====================================================================
for i in N:
    for j in N:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    cij = c.get((i, j), 0.0)
                    m.addConstr(ta[j, k, r] >= td[i, k, r] + cij * x[(i, j, k, r)] - M_16 * (1 - x[(i, j, k, r)]),
                               name=f"c16[{i},{j},{k},{r}]")

for j in Nw:
    for k in K:
        for r in R:
            unload = quicksum(su[p] * f[p, k, r] for p in P if d[p] == j)
            m.addConstr(ts[j, k, r] >= ta[j, k, r] + unload, name=f"c17[{j},{k},{r}]")
            load = quicksum(sl[p] * f[p, k, r] for p in P if o[p] == j)
            m.addConstr(td[j, k, r] >= ts[j, k, r] + load, name=f"c19[{j},{k},{r}]")

for p in P:
    op = o[p]
    if op in Nw:
        ep = e[p]
        for k in K:
            for r in R:
                m.addConstr(ts[op, k, r] >= ep * f[p, k, r], name=f"c18[{p},{k},{r}]")
    op, dp = o[p], d[p]
    if (op in N) and (dp in N):
        for k in K:
            for r in R:
                m.addConstr(ta[dp, k, r] >= td[op, k, r] - M_20 * (1 - f[p, k, r]), name=f"c20[{p},{k},{r}]")
                
# Depot varış-ayrılış kısıtı (c21)
for k in K:
    for r in R:
        m.addConstr(ta['h', k, r] >= td['h', k, r], name=f"c21[{k},{r}]")

for p in P:
    dp = d[p]
    ep = e[p]
    for k in K:
        for r in R:
            m.addConstr(w[p] >= ta[dp, k, r] - ep - M_22 * (1 - f[p, k, r]), name=f"c22[{p},{k},{r}]")

# =====================================================================
# KISITLAR (23-27): Kapasite
# =====================================================================
for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(q_product[p] * f[p, k, r] for p in P if o[p] == j)
            load_out = quicksum(q_product[p] * f[p, k, r] for p in P if d[p] == j)
            m.addConstr(delta[j, k, r] >= load_in - load_out, name=f"c23[{j},{k},{r}]")
            m.addConstr(y[j, k, r] <= q_vehicle[k], name=f"c26[{j},{k},{r}]")

for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M_24 * (1 - x[(i, j, k, r)]), name=f"c24[{i},{j},{k},{r}]")
                    m.addConstr(y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M_25 * (1 - x[(i, j, k, r)]), name=f"c25[{i},{j},{k},{r}]")

for k in K:
    for r in R:
        m.addConstr(y['h', k, r] == 0, name=f"c27[{k},{r}]")

# =====================================================================
# KISIT (28): Rota sıralaması
# =====================================================================
for k in K:
    for r in R[:-1]:
        lhs = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        rhs = quicksum(x[('h', j, k, r+1)] for j in Nw if ('h', j, k, r+1) in x)
        m.addConstr(lhs >= rhs, name=f"c28[{k},{r}]")

# =====================================================================
# KISITLAR (29-32): Alt tur eliminasyonu (U zaten optimal)
# =====================================================================
for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(u[j, k, r] >= u[i, k, r] + 1 - U * (1 - x[(i, j, k, r)]),
                               name=f"c29[{i},{j},{k},{r}]")

for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            m.addConstr(u[j, k, r] <= U * indeg, name=f"c30[{j},{k},{r}]")
            m.addConstr(u[j, k, r] >= indeg, name=f"c31[{j},{k},{r}]")

for k in K:
    for r in R:
        for p in P:
            op, dp = o[p], d[p]
            if (op in Nw) and (dp in Nw):
                m.addConstr(u[dp, k, r] >= u[op, k, r] + 1 - U * (1 - f[p, k, r]),
                           name=f"c32[{p},{k},{r}]")

# =====================================================================
# MODEL PARAMETRELERİ VE OPTİMİZASYON
# =====================================================================
timestamp  = datetime.now().strftime('%Y_%m_%d_%H_%M')
excel_path = os.path.join('results', f"result_tightM_{timestamp}.xlsx")
log_path   = os.path.join(desktop_dir, f"result_tightM_{timestamp}.txt")
os.makedirs('results', exist_ok=True)

m.setParam('TimeLimit', TIME_LIMIT)
m.setParam('MIPGap', MIP_GAP)
m.setParam('Threads', THREADS)
m.setParam('Presolve', 2)
m.setParam('LogFile', log_path)
m.update()

print("OPTİMİZASYON BAŞLIYOR...\n")
m.optimize()

# =====================================================================
# SONUÇLARI İŞLE VE KAYDET
# =====================================================================
if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    print("\n" + "="*80)
    print("ÇÖZÜM BULUNDU")
    print("="*80)
    
    if m.status == GRB.TIME_LIMIT:
        print(f"\n Zaman limiti aşıldı ({TIME_LIMIT}s)")
        print(f"En iyi bulunan çözüm: {m.objVal if m.SolCount > 0 else 'YOK'}")
    
    total_wait = sum(w[p].X for p in P if w[p].X is not None)
    total_arrival_times = sum(ta['h', k, r].X for k in K for r in R if ta['h', k, r].X is not None)
    wait_slack = EPS_WAIT - total_wait
    
    opt_results = pd.DataFrame([{
        'model': 'tight_M_optimized',
        'objective': 'min_total_arrival',
        'obj_value': m.objVal if m.SolCount > 0 else None,
        'best_bound': getattr(m, 'objBound', None),
        'mip_gap': getattr(m, 'MIPGap', None),
        'runtime': m.Runtime,
        'status': m.status,
        'total_wait_minutes': round(total_wait, 2),
        'total_arrival_times': round(total_arrival_times, 2),
        'epsilon_wait_upper': EPS_WAIT,
        'wait_slack_remaining': round(wait_slack, 2),
        '|N|': len(N), 
        '|Nw|': len(Nw), 
        '|K|': len(K), 
        '|R|': len(R), 
        'KxR': len(K) * len(R),
        'U_(|Nw|)': U,
        'M_16': M_16,
        'M_20': M_20,
        'M_22': M_22,
        'M_24': M_24,
        'M_25': M_25
    }])
    
    # Değişkenleri detaylı kaydet
    x_data = [{'var': 'x', 'i': i, 'j': j, 'k': k, 'r': r, 'val': round(var.X, 4)} 
              for (i, j, k, r), var in x.items() if var.X > 0.5]
    xdf = pd.DataFrame(x_data) if x_data else pd.DataFrame(columns=['var', 'i', 'j', 'k', 'r', 'val'])
    
    f_data = [{'var': 'f', 'p': p, 'k': k, 'r': r, 'val': round(f[p, k, r].X, 4)} 
              for p in P for k in K for r in R if f[p, k, r].X > 0.5]
    fdf = pd.DataFrame(f_data) if f_data else pd.DataFrame(columns=['var', 'p', 'k', 'r', 'val'])
    
    # MTZ u değişkenleri
    u_data = [{'var': 'u', 'j': j, 'k': k, 'r': r, 'u': int(u[j, k, r].X)} 
              for j in Nw for k in K for r in R if u[j, k, r].X > 0.5]
    udf = pd.DataFrame(u_data) if u_data else pd.DataFrame(columns=['var', 'j', 'k', 'r', 'u'])
    
    # Rota kullanımı (z_kr)
    z_data = []
    for k in K:
        for r in R:
            used = 1 if any(x[('h', j, k, r)].X > 0.5 for j in Nw if ('h', j, k, r) in x) else 0
            z_data.append({'var': 'z', 'k': k, 'r': r, 'z': used})
    zdf = pd.DataFrame(z_data)
    
    wdf = pd.DataFrame([{'var': 'w', 'p': p, 'w_val': round(w[p].X, 4) if w[p].X else 0} for p in P])
    
    # Zaman değişkenleri
    ta_data = []
    for j in N:
        for k in K:
            for r in R:
                time_val = round(ta[j, k, r].X, 4) if ta[j, k, r].X else 0
                ta_data.append({
                    'var': 'ta', 
                    'node': j, 
                    'k': k, 
                    'r': r, 
                    'time': time_val,
                    'stamp': minutes_to_hhmm(time_val)
                })
    tadf = pd.DataFrame(ta_data)
    
    td_data = []
    for j in N:
        for k in K:
            for r in R:
                time_val = round(td[j, k, r].X, 4) if td[j, k, r].X else 0
                td_data.append({
                    'var': 'td', 
                    'node': j, 
                    'k': k, 
                    'r': r, 
                    'time': time_val,
                    'stamp': minutes_to_hhmm(time_val)
                })
    tddf = pd.DataFrame(td_data)
    
    ts_data = []
    for j in Nw:
        for k in K:
            for r in R:
                time_val = round(ts[j, k, r].X, 4) if ts[j, k, r].X else 0
                ts_data.append({
                    'var': 'ts', 
                    'node': j, 
                    'k': k, 
                    'r': r, 
                    'time': time_val,
                    'stamp': minutes_to_hhmm(time_val)
                })
    tsdf = pd.DataFrame(ts_data)
    
    # Yük değişkenleri
    y_data = []
    for j in N:
        for k in K:
            for r in R:
                y_val = round(y[j, k, r].X, 4) if y[j, k, r].X else 0
                if y_val > 0.01:  # Sadece anlamlı değerleri kaydet
                    y_data.append({'var': 'y', 'node': j, 'k': k, 'r': r, 'y_val': y_val})
    ydf = pd.DataFrame(y_data) if y_data else pd.DataFrame(columns=['var', 'node', 'k', 'r', 'y_val'])
    
    # Delta değişkenleri
    delta_data = []
    for j in Nw:
        for k in K:
            for r in R:
                delta_val = round(delta[j, k, r].X, 4) if delta[j, k, r].X else 0
                if abs(delta_val) > 0.01:  # Sadece anlamlı değerleri kaydet
                    delta_data.append({'var': 'delta', 'node': j, 'k': k, 'r': r, 'delta_val': delta_val})
    deltadf = pd.DataFrame(delta_data) if delta_data else pd.DataFrame(columns=['var', 'node', 'k', 'r', 'delta_val'])
    
    # Itinerary - Rota özeti
    itinerary_data = []
    for k in K:
        for r in R:
            # Bu rotada ziyaret edilen düğümleri bul
            visited = [(j, int(u[j, k, r].X)) for j in Nw if u[j, k, r].X > 0.5]
            visited.sort(key=lambda x: x[1])  # Sıraya göre sırala
            
            for node, seq in visited:
                ta_val = ta[node, k, r].X if ta[node, k, r].X else 0
                td_val = td[node, k, r].X if td[node, k, r].X else 0
                y_val = y[node, k, r].X if y[node, k, r].X else None
                
                itinerary_data.append({
                    'k': k,
                    'r': r,
                    'j': node,
                    'u': seq,
                    'ta_stamp': minutes_to_hhmm(ta_val),
                    'td_stamp': minutes_to_hhmm(td_val),
                    'y_after': y_val if y_val and y_val > 0.01 else None
                })
    
    itinerary_df = pd.DataFrame(itinerary_data) if itinerary_data else pd.DataFrame(
        columns=['k', 'r', 'j', 'u', 'ta_stamp', 'td_stamp', 'y_after'])
    
    try:
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
            itinerary_df.to_excel(writer, sheet_name='itinerary', index=False)
        
        print(f"\n Excel dosyası kaydedildi: {excel_path}")
        print(f"   12 sekme oluşturuldu:")
        print(f"      1. optimization_results")
        print(f"      2. x_ijkr")
        print(f"      3. f_pkr")
        print(f"      4. u_jkr")
        print(f"      5. z_kr")
        print(f"      6. w_p")
        print(f"      7. ta (zaman damgalı)")
        print(f"      8. td (zaman damgalı)")
        print(f"      9. ts (zaman damgalı)")
        print(f"     10. y_jkr")
        print(f"     11. delta_jkr")
        print(f"     12. itinerary")
    except Exception as e:
        print(f"\n❌ Excel yazma hatası: {e}")
    
    print(f"✅ Log dosyası: {log_path}")
    print(f"\n{'='*80}")
    print(f"TIGHT M İLE SONUÇLAR:")
    print(f"   Toplam varış: {total_arrival_times:.2f} dk")
    print(f"   Toplam bekleme: {total_wait:.2f} dk (üst sınır: {EPS_WAIT})")
    print(f"   Çözüm süresi: {m.Runtime:.2f} sn")
    print(f"   MIP Gap: {m.MIPGap*100:.2f}%")
    print(f"{'='*80}\n")

elif m.status == GRB.INFEASIBLE:
    print("\n" + "="*80)
    print(" MODEL INFEASIBLE")
    print("="*80)
    m.computeIIS()
    iis_file = f"infeasible_tightM_{timestamp}.ilp"
    m.write(iis_file)
    print(f"\n IIS dosyası: {iis_file}")

else:
    print(f"\n Çözüm bulunamadı. Status = {m.status}")

print("\n" + "="*80)
print("PROGRAM TAMAMLANDI")
print(f"Terminal çıktısı: {terminal_log_path}")
print("="*80)

sys.stdout = original_stdout
terminal_log_file.close()
