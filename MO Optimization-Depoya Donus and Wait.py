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
            f.flush()  # Anında yazsın
    
    def flush(self):
        for f in self.files:
            f.flush()

# Terminal log dosyası oluştur
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
terminal_log_path = os.path.join(r"C:\Users\Asus\Desktop", f"terminal_output_{timestamp}.txt")
terminal_log_file = open(terminal_log_path, 'w', encoding='utf-8')

# stdout'u yönlendir (hem ekrana hem dosyaya yazacak)
original_stdout = sys.stdout
sys.stdout = TeeOutput(original_stdout, terminal_log_file)

print(f" Terminal çıktısı kaydediliyor: {terminal_log_path}\n")


TIME_LIMIT = 1200
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

# Veri dosyalarını yükle
nodes    = pd.read_excel(os.path.join(data_path, "nodes.xlsx"))
vehicles = pd.read_excel(os.path.join(data_path, "vehicles.xlsx"))
products = pd.read_excel(os.path.join(data_path, "products.xlsx")).head(10)

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
# Nodes
nodes['node_id'] = nodes['node_id'].astype(str).str.strip()
N  = nodes['node_id'].dropna().drop_duplicates().tolist()
Nw = [n for n in N if n != 'h']

# Vehicles
vehicles['vehicle_id'] = vehicles['vehicle_id'].astype(str).str.strip()
vehicles = vehicles.dropna(subset=['vehicle_id']).drop_duplicates('vehicle_id', keep='first')
K = vehicles['vehicle_id'].tolist()
q_vehicle = dict(zip(K, vehicles['capacity_m2']))

MAX_ROUTES = int(vehicles['max_routes'].max()) if 'max_routes' in vehicles.columns else 5
R = list(range(1, MAX_ROUTES + 1))

# Products
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

# Model parametreleri
M = 10000.0
epsilon = 0.1
U = len(Nw)
TIME_THRESHOLD = 5000

print("="*80)
print("VERİ YÜKLENDİ")
print("="*80)
print(f"|N|={len(N)}, |Nw|={len(Nw)}, |K|={len(K)}, |R|={len(R)}, |P|={len(P)}")
print(f"M={M}, U={U}, ε={epsilon}, EPS_WAIT={EPS_WAIT}, TIME_THRESHOLD={TIME_THRESHOLD}")
print("="*80 + "\n")

# =====================================================================
# MODEL OLUŞTURMA
# =====================================================================
m = gp.Model("InternalLogistics_MinArrival_Primary")

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
# BİRİNCİL AMAÇ: Toplam varış zamanlarını minimize et
obj1 = quicksum(ta['h', k, MAX_ROUTES] for k in K)

# İKİNCİL AMAÇ (EPSILON-KISIT): Toplam bekleme süresi üst sınıra tabi
obj2 = quicksum(w[p] for p in P)
m.addConstr(obj2 <= EPS_WAIT, name="eps_wait_constraint")
m.setObjective(obj1 + 0.001*obj2, GRB.MINIMIZE)

print("="*80)
print("MODEL 2: TOPLAM VARIŞ ZAMANLARI BİRİNCİL AMAÇ")
print("="*80)
print(f"Birincil Amaç (minimize): Σ_k Σ_r ta_hkr (Toplam Varış Zamanları)")
print(f"İkincil Amaç (kısıt):     Σ_p w_p ≤ {EPS_WAIT} dakika")
print(f"Eklenen Kısıt (13*):      ta_hkr ≥ ta_hk(r-1)")
print("="*80 + "\n")

# =====================================================================
# KISITLAR (3-11): Rota ve atama kısıtları
# =====================================================================
for k in K:
    for r in R:
        out_h = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        in_h  = quicksum(x[(j, 'h', k, r)] for j in Nw if (j, 'h', k, r) in x)
        m.addConstr(out_h == in_h, name=f"c3[{k},{r}]")
        m.addConstr(out_h <= 1, name=f"c4[{k},{r}]")
        lhs = quicksum(x[(i, j, k, r)] for i in N for j in N if (i, j, k, r) in x)
        rhs = quicksum(f[p, k, r] for p in P)
        m.addConstr(lhs <= (2*len(P)+1)*rhs, name=f"c5[{k},{r}]")

for j in Nw:
    for k in K:
        for r in R:
            lhs = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            rhs = quicksum(f[p, k, r] for p in P if (o[p] == j or d[p] == j))
            m.addConstr(lhs <= rhs, name=f"c6[{j},{k},{r}]")
            inflow  = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            outflow = quicksum(x[(j, i, k, r)] for i in N if i != j and (j, i, k, r) in x)
            m.addConstr(inflow == outflow, name=f"c7[{j},{k},{r}]")
            m.addConstr(inflow <= 1, name=f"c8[{j},{k},{r}]")

for p in P:
    m.addConstr(quicksum(f[p, k, r] for k in K for r in R) == 1, name=f"c9[{p}]")
    op = o[p]
    if op in N:
        for k in K:
            for r in R:
                lhs = quicksum(x[(i, op, k, r)] for i in N if i != op and (i, op, k, r) in x)
                m.addConstr(lhs >= f[p, k, r], name=f"c10[{p},{k},{r}]")
    dp = d[p]
    if dp in N:
        for k in K:
            for r in R:
                lhs = quicksum(x[(i, dp, k, r)] for i in N if i != dp and (i, dp, k, r) in x)
                m.addConstr(lhs >= f[p, k, r], name=f"c11[{p},{k},{r}]")

# =====================================================================
# KISITLAR (12-13*): Zaman sıralaması
# =====================================================================
for k in K:
    m.addConstr(td['h', k, 1] == 0, name=f"c12[{k}]")
    for r in R[1:]:
        # KISIT (13): Ayrılış zamanı önceki varıştan sonra olmalı
        m.addConstr(td['h', k, r] >= ta['h', k, r-1] + epsilon, name=f"c13[{k},{r}]")
        # KISIT (13*): Varış zamanları monoton artan
        m.addConstr(ta['h', k, r] >= ta['h', k, r-1], name=f"c13_star[{k},{r}]")

# =====================================================================
# KISITLAR (14-19): Zaman penceresi ve servis süreleri
# =====================================================================
for i in N:
    for j in N:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    cij = c.get((i, j), 0.0)
                    m.addConstr(ta[j, k, r] >= td[i, k, r] + cij * x[(i, j, k, r)] - M * (1 - x[(i, j, k, r)]),
                               name=f"c14[{i},{j},{k},{r}]")

for j in Nw:
    for k in K:
        for r in R:
            unload = quicksum(su[p] * f[p, k, r] for p in P if d[p] == j)
            m.addConstr(ts[j, k, r] >= ta[j, k, r] + unload, name=f"c15[{j},{k},{r}]")
            load = quicksum(sl[p] * f[p, k, r] for p in P if o[p] == j)
            m.addConstr(td[j, k, r] >= ts[j, k, r] + load, name=f"c17[{j},{k},{r}]")

for p in P:
    op = o[p]
    if op in Nw:
        ep = e[p]
        for k in K:
            for r in R:
                m.addConstr(ts[op, k, r] >= ep * f[p, k, r], name=f"c16[{p},{k},{r}]")
    op, dp = o[p], d[p]
    if (op in N) and (dp in N):
        for k in K:
            for r in R:
                m.addConstr(ta[dp, k, r] >= td[op, k, r] - M * (1 - f[p, k, r]),
                           name=f"c18[{p},{k},{r}]")
    ep = e[p]
    for k in K:
        for r in R:
            m.addConstr(w[p] >= ta[dp, k, r] - ep - M * (1 - f[p, k, r]),
                       name=f"c19[{p},{k},{r}]")

# =====================================================================
# KISITLAR (20-23): Kapasite kısıtları
# =====================================================================
for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(q_product[p] * f[p, k, r] for p in P if o[p] == j)
            load_out = quicksum(q_product[p] * f[p, k, r] for p in P if d[p] == j)
            m.addConstr(delta[j, k, r] >= load_in - load_out, name=f"c20[{j},{k},{r}]")
            m.addConstr(y[j, k, r] <= q_vehicle[k], name=f"c23[{j},{k},{r}]")

for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M * (1 - x[(i, j, k, r)]),
                               name=f"c21[{i},{j},{k},{r}]")
                    m.addConstr(y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M * (1 - x[(i, j, k, r)]),
                               name=f"c22[{i},{j},{k},{r}]")

for k in K:
    for r in R:
        m.addConstr(y['h', k, r] == 0, name=f"yhome[{k},{r}]")

# =====================================================================
# KISIT (24): Rota sıralaması
# =====================================================================
for k in K:
    for r in R[:-1]:
        lhs = quicksum(x[('h', j, k, r)] for j in Nw if ('h', j, k, r) in x)
        rhs = quicksum(x[('h', j, k, r+1)] for j in Nw if ('h', j, k, r+1) in x)
        m.addConstr(lhs >= rhs, name=f"c24[{k},{r}]")

# =====================================================================
# KISITLAR (25-28): Alt tur eliminasyonu
# =====================================================================
for i in Nw:
    for j in Nw:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    m.addConstr(u[j, k, r] >= u[i, k, r] + 1 - U * (1 - x[(i, j, k, r)]),
                               name=f"c25[{i},{j},{k},{r}]")

for j in Nw:
    for k in K:
        for r in R:
            indeg = quicksum(x[(i, j, k, r)] for i in N if i != j and (i, j, k, r) in x)
            m.addConstr(u[j, k, r] <= U * indeg, name=f"c26[{j},{k},{r}]")
            m.addConstr(u[j, k, r] >= indeg, name=f"c27[{j},{k},{r}]")

for k in K:
    for r in R:
        for p in P:
            op, dp = o[p], d[p]
            if (op in Nw) and (dp in Nw):
                m.addConstr(u[dp, k, r] >= u[op, k, r] + 1 - U * (1 - f[p, k, r]),
                           name=f"c28[{p},{k},{r}]")

"""
   # Parçayı V3'e, Rota 1'e ata
    m.addConstr(f["P128", 'V2', 1] == 1, name="scn1_f_P128_V2_1")
    m.addConstr(f["P131", 'V2', 1] == 1, name="scn1_f_P131_V2_1")
    m.addConstr(f["P247", 'V2', 1] == 1, name="scn1_f_P247_V2_1")
    m.addConstr(f["P11", 'V1', 1] == 1, name="scn1_f_P17_V1_1")

    m.addConstr(f["P17", 'V3', 1] == 1, name="scn1_f_P17_V1_1")
    m.addConstr(f["P113", 'V3', 1] == 1, name="scn1_f_P113_V1_1")
    m.addConstr(f["P115", 'V3', 1] == 1, name="scn1_f_P4_V1_1")
    m.addConstr(f["P141", 'V3', 1] == 1, name="scn1_f_P141_V1_1")
    m.addConstr(f["P18", 'V3', 1] == 1, name="scn1_f_P18_V1_1")
    m.addConstr(f['P236', 'V3', 1] == 1, name="scn1_f_P236_V1_1")
 """
# =====================================================================
# MODEL PARAMETRELERİ VE OPTİMİZASYON
# =====================================================================
timestamp  = datetime.now().strftime('%Y_%m_%d_%H_%M')
excel_path = os.path.join('results', f"result_arrival_primary_{timestamp}.xlsx")
log_path   = os.path.join(desktop_dir, f"result_arrival_primary_{timestamp}.txt")
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
    print("ÇÖZÜM BULUNDU - MODEL 2 (TOPLAM VARIŞ ZAMANLARI BİRİNCİL)")
    print("="*80)
    
    # Timeout kontrolü
    if m.status == GRB.TIME_LIMIT:
        print(f"\n UYARI: Zaman limiti aşıldı ({TIME_LIMIT}s)")
        print(f"En iyi bulunan çözüm: {m.objVal if m.SolCount > 0 else 'YOK'}")
    
    total_wait = sum(w[p].X for p in P if w[p].X is not None)
    total_arrival_times = sum(ta['h', k, r].X for k in K for r in R if ta['h', k, r].X is not None)
    
    # Optimizasyon sonuçları
    opt_results = pd.DataFrame([{
        'model': 'arrival_primary',
        'objective': 'min_total_arrival',
        'obj_value': m.objVal if m.SolCount > 0 else None,
        'best_bound': getattr(m, 'objBound', None),
        'mip_gap': getattr(m, 'MIPGap', None),
        'runtime': m.Runtime,
        'status': m.status,
        'total_wait_minutes': round(total_wait, 2),
        'total_arrival_times': round(total_arrival_times, 2),
        'epsilon_wait_upper': EPS_WAIT,
        'wait_slack_remaining': round(EPS_WAIT - total_wait, 2),
        '|N|': len(N), '|Nw|': len(Nw), '|K|': len(K), '|R|': len(R),
        'KxR': len(K) * len(R), 'U_(|Nw|)': U, 'M_single': M
    }])
    
    # x değişkenleri
    x_data = [{'var': 'x', 'i': i, 'j': j, 'k': k, 'r': r, 'val': round(var.X, 4)} 
              for (i, j, k, r), var in x.items() if var.X > 0.5]
    xdf = pd.DataFrame(x_data) if x_data else pd.DataFrame(columns=['var', 'i', 'j', 'k', 'r', 'val'])
    
    # f değişkenleri
    f_data = [{'var': 'f', 'p': p, 'k': k, 'r': r, 'val': round(f[p, k, r].X, 4)} 
              for p in P for k in K for r in R if f[p, k, r].X > 0.5]
    fdf = pd.DataFrame(f_data) if f_data else pd.DataFrame(columns=['var', 'p', 'k', 'r', 'val'])
    
    # u değişkenleri
    u_data = [{'var': 'u', 'j': j, 'k': k, 'r': r, 'u': int(u[j, k, r].X)} 
              for j in Nw for k in K for r in R if u[j, k, r].X > 0]
    udf = pd.DataFrame(u_data) if u_data else pd.DataFrame(columns=['var', 'j', 'k', 'r', 'u'])
    
    # DÜZELTME: z değişkenleri (rota kullanımı)
    z_data = []
    for k in K:
        for r in R:
            # Belirli k ve r için herhangi bir rota kullanılıyor mu?
            route_used = any(
                var.X > 0.5 
                for (i_, j_, k_, r_), var in x.items() 
                if k_ == k and r_ == r
            )
            z_data.append({
                'var': 'z', 
                'k': k, 
                'r': r, 
                'z': 1 if route_used else 0
            })
    zdf = pd.DataFrame(z_data)
    
    # w değişkenleri (bekleme süreleri)
    wdf = pd.DataFrame([{'var': 'w', 'p': p, 'w_val': round(w[p].X, 4) if w[p].X else 0} for p in P])
    
    # ta değişkenleri (varış zamanları)
    ta_data = []
    for j in N:
        for k in K:
            for r in R:
                time_val = ta[j, k, r].X if ta[j, k, r].X is not None else 0
                if time_val < TIME_THRESHOLD:
                    ta_data.append({
                        'var': 'ta',
                        'node': j,
                        'k': k,
                        'r': r,
                        'time': round(time_val, 4),
                        'stamp': minutes_to_hhmm(time_val)
                    })
    tadf = pd.DataFrame(ta_data) if ta_data else pd.DataFrame(columns=['var', 'node', 'k', 'r', 'time', 'stamp'])
    
    # td değişkenleri (ayrılış zamanları)
    td_data = []
    for j in N:
        for k in K:
            for r in R:
                time_val = td[j, k, r].X if td[j, k, r].X is not None else 0
                if time_val < TIME_THRESHOLD:
                    td_data.append({
                        'var': 'td',
                        'node': j,
                        'k': k,
                        'r': r,
                        'time': round(time_val, 4),
                        'stamp': minutes_to_hhmm(time_val)
                    })
    tddf = pd.DataFrame(td_data) if td_data else pd.DataFrame(columns=['var', 'node', 'k', 'r', 'time', 'stamp'])
    
    # ts değişkenleri (servis başlangıç zamanları)
    ts_data = []
    for j in Nw:
        for k in K:
            for r in R:
                time_val = ts[j, k, r].X if ts[j, k, r].X is not None else 0
                if time_val < TIME_THRESHOLD:
                    ts_data.append({
                        'var': 'ts',
                        'node': j,
                        'k': k,
                        'r': r,
                        'time': round(time_val, 4),
                        'stamp': minutes_to_hhmm(time_val)
                    })
    tsdf = pd.DataFrame(ts_data) if ts_data else pd.DataFrame(columns=['var', 'node', 'k', 'r', 'time', 'stamp'])
    
    # y değişkenleri (yük miktarları)
    y_data = []
    for j in N:
        for k in K:
            for r in R:
                y_val = y[j, k, r].X if y[j, k, r].X is not None else 0
                if y_val > 0.01 or j == 'h':
                    y_data.append({
                        'var': 'y',
                        'node': j,
                        'k': k,
                        'r': r,
                        'y_val': round(y_val, 4)
                    })
    ydf = pd.DataFrame(y_data) if y_data else pd.DataFrame(columns=['var', 'node', 'k', 'r', 'y_val'])
    
    # delta değişkenleri (yük değişimleri)
    delta_data = []
    for j in Nw:
        for k in K:
            for r in R:
                delta_val = delta[j, k, r].X if delta[j, k, r].X is not None else 0
                if abs(delta_val) > 0.01:
                    delta_data.append({
                        'var': 'delta',
                        'node': j,
                        'k': k,
                        'r': r,
                        'delta_val': round(delta_val, 4)
                    })
    deltadf = pd.DataFrame(delta_data) if delta_data else pd.DataFrame(columns=['var', 'node', 'k', 'r', 'delta_val'])
    
    # Rota planı oluştur
    used = xdf[xdf['val'] > 0.5].copy() if not xdf.empty else pd.DataFrame()
    
    if not used.empty and not udf.empty and not tadf.empty and not tddf.empty:
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
        print("\n UYARI: Rota bilgisi bulunamadı, itinerary boş.")
    
    # Excel'e kaydet
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
            visit.to_excel(writer, sheet_name='itinerary', index=False)
        
        print(f"\n Excel dosyası kaydedildi: {excel_path}")
    except Exception as e:
        print(f"\n Excel yazma hatası: {e}")
        print(f"Alternatif CSV dosyaları oluşturuluyor...")
        csv_path = excel_path.replace('.xlsx', '_results.csv')
        opt_results.to_csv(csv_path, index=False)
        print(f" CSV dosyası kaydedildi: {csv_path}")
    
    print(f" Log dosyası: {log_path}")
    print(f"\n{'='*80}")
    print(f" Toplam varış zamanları (minimize): {total_arrival_times:.2f} dk")
    print(f" Toplam bekleme süresi (kısıtlı): {total_wait:.2f} dk (üst sınır: {EPS_WAIT})")
    print(f" Kalan bekleme payı: {EPS_WAIT - total_wait:.2f} dk")
    print(f"{'='*80}\n")

elif m.status == GRB.INFEASIBLE:
    print("\n" + "="*80)
    print("MODEL INFEASIBLE - IIS HESAPLANIYOR")
    print("="*80)
    m.computeIIS()
    iis_file = f"infeasible_arrival_primary_{timestamp}.ilp"
    m.write(iis_file)
    print(f"\n IIS dosyası kaydedildi: {iis_file}")
    print("\nÇELİŞEN KISITLAR:")
    for c in m.getConstrs():
        if c.IISConstr:
            print(f"  - {c.ConstrName}")
else:
    print(f"\n Çözüm bulunamadı. Gurobi Status = {m.status}")
    print("Status açıklaması:")
    status_dict = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",  
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT"
    }
    print(f"  {status_dict.get(m.status, 'UNKNOWN')}")

# =====================================================================
# TERMİNAL LOG DOSYASINI KAPAT
# =====================================================================
print("\n" + "="*80)
print("PROGRAM TAMAMLANDI")
print(f"Terminal çıktısı kaydedildi: {terminal_log_path}")
print("="*80)

# Dosyayı kapat ve stdout'u eski haline getir
sys.stdout = original_stdout
terminal_log_file.close()
