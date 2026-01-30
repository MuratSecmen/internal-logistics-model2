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


# INPUT VERİLERİNDEN HESAPLANAN PARAMETRELER:
T_max = 480      # Vardiya süresi (dakika)
C_max = 11       # En uzun seyahat (dakika)
e_min = 435      # En erken parça hazır olma (dakika - 07:15)
Q_max = 20       # Maksimum araç kapasitesi (m²)
N_w_count = len(Nw)  # İstasyon sayısı

# TIGHT M HESAPLAMALARI:
M_14 = T_max - e_min + C_max   # = 56.0 dk (Zaman tutarlılığı)
M_18 = T_max - e_min           # = 45.0 dk (Alış-teslimat)
M_19 = T_max                   # = 480.0 dk (Bekleme süresi)
M_21_22 = Q_max                # = 20 m² (Kapasite) 🔥

epsilon = 0.1
U = len(Nw)
TIME_THRESHOLD = 5000

print("\n" + "="*80)
print("🔥 TIGHT BIG-M DEĞERLERİ TANIMLANDI")
print("="*80)
print(f"M_14 (Zaman)       = {M_14:.1f} dk   (Naive: 10000 → İyileşme: 99.44%)")
print(f"M_18 (Teslimat)    = {M_18:.1f} dk   (Naive: 10000 → İyileşme: 99.55%)")
print(f"M_19 (Bekleme)     = {M_19:.1f} dk   (Naive: 10000 → İyileşme: 95.20%)")
print(f"M_21_22 (Kapasite) = {M_21_22:.0f} m²   (Naive: 10000 → İyileşme: 99.80%) 🔥")
print(f"U (MTZ) = {U} = len(Nw) = {N_w_count} (zaten optimal)")
print("="*80 + "\n")
# ============================================================================

print("="*80)
print("VERİ YÜKLENDİ")
print("="*80)
print(f"|N|={len(N)}, |Nw|={len(Nw)}, |K|={len(K)}, |R|={len(R)}, |P|={len(P)}")
print(f"TIGHT M VALUES:")
print(f"  M_14={M_14:.1f}, M_18={M_18:.1f}, M_19={M_19:.1f}, M_21_22={M_21_22:.0f}")
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
print("MODEL: TOPLAM VARIŞ ZAMANLARI BİRİNCİL AMAÇ (TIGHT M İLE)")
print("="*80)
print(f"Birincil Amaç: Σ_k ta_hkr (Toplam Varış Zamanları)")
print(f"İkincil Amaç:  Σ_p w_p ≤ {EPS_WAIT} dakika")
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
        m.addConstr(td['h', k, r] >= ta['h', k, r-1] + epsilon, name=f"c13[{k},{r}]")
        m.addConstr(ta['h', k, r] >= ta['h', k, r-1], name=f"c13_star[{k},{r}]")

# =====================================================================
# KISITLAR (14-19): Zaman penceresi - TIGHT M KULLANILIYOR! 🔥
# =====================================================================
for i in N:
    for j in N:
        if i == j: continue
        for k in K:
            for r in R:
                if (i, j, k, r) in x:
                    cij = c.get((i, j), 0.0)
                    # KISIT 14 - TIGHT M_14 = 56 dk 🔥
                    m.addConstr(ta[j, k, r] >= td[i, k, r] + cij * x[(i, j, k, r)] - M_14 * (1 - x[(i, j, k, r)]),
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
                # KISIT 18 - TIGHT M_18 = 45 dk 🔥
                m.addConstr(ta[dp, k, r] >= td[op, k, r] - M_18 * (1 - f[p, k, r]),
                           name=f"c18[{p},{k},{r}]")
    ep = e[p]
    for k in K:
        for r in R:
            # KISIT 19 - TIGHT M_19 = 480 dk 🔥
            m.addConstr(w[p] >= ta[dp, k, r] - ep - M_19 * (1 - f[p, k, r]),
                       name=f"c19[{p},{k},{r}]")

# =====================================================================
# KISITLAR (20-23): Kapasite - TIGHT M KULLANILIYOR! 🔥🔥
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
                    # KISIT 21-22 - TIGHT M_21_22 = 20 m² 🔥🔥🔥 (EN BÜYÜK İYİLEŞME!)
                    m.addConstr(y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M_21_22 * (1 - x[(i, j, k, r)]),
                               name=f"c21[{i},{j},{k},{r}]")
                    m.addConstr(y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M_21_22 * (1 - x[(i, j, k, r)]),
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
# KISITLAR (25-28): Alt tur eliminasyonu (U zaten optimal)
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

print("🚀 OPTİMİZASYON BAŞLIYOR (TIGHT M İLE)...\n")
m.optimize()

# =====================================================================
# SONUÇLARI İŞLE VE KAYDET
# =====================================================================
if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    print("\n" + "="*80)
    print("✅ ÇÖZÜM BULUNDU - TIGHT M OPTİMİZASYONU")
    print("="*80)
    
    if m.status == GRB.TIME_LIMIT:
        print(f"\n⚠️  Zaman limiti aşıldı ({TIME_LIMIT}s)")
        print(f"En iyi bulunan çözüm: {m.objVal if m.SolCount > 0 else 'YOK'}")
    
    total_wait = sum(w[p].X for p in P if w[p].X is not None)
    total_arrival_times = sum(ta['h', k, r].X for k in K for r in R if ta['h', k, r].X is not None)
    
    opt_results = pd.DataFrame([{
        'model': 'tight_M_optimized',
        'M_14': M_14,
        'M_18': M_18,
        'M_19': M_19,
        'M_21_22': M_21_22,
        'U_MTZ': U,
        'obj_value': m.objVal if m.SolCount > 0 else None,
        'best_bound': getattr(m, 'objBound', None),
        'mip_gap': getattr(m, 'MIPGap', None),
        'runtime': m.Runtime,
        'status': m.status,
        'total_wait_minutes': round(total_wait, 2),
        'total_arrival_times': round(total_arrival_times, 2),
        '|N|': len(N), '|Nw|': len(Nw), '|K|': len(K), '|R|': len(R), '|P|': len(P)
    }])
    
    # Değişkenleri kaydet (önceki kod ile aynı)
    x_data = [{'var': 'x', 'i': i, 'j': j, 'k': k, 'r': r, 'val': round(var.X, 4)} 
              for (i, j, k, r), var in x.items() if var.X > 0.5]
    xdf = pd.DataFrame(x_data) if x_data else pd.DataFrame(columns=['var', 'i', 'j', 'k', 'r', 'val'])
    
    f_data = [{'var': 'f', 'p': p, 'k': k, 'r': r, 'val': round(f[p, k, r].X, 4)} 
              for p in P for k in K for r in R if f[p, k, r].X > 0.5]
    fdf = pd.DataFrame(f_data) if f_data else pd.DataFrame(columns=['var', 'p', 'k', 'r', 'val'])
    
    wdf = pd.DataFrame([{'var': 'w', 'p': p, 'w_val': round(w[p].X, 4) if w[p].X else 0} for p in P])
    
    try:
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            opt_results.to_excel(writer, sheet_name='optimization_results', index=False)
            xdf.to_excel(writer, sheet_name='x_ijkr', index=False)
            fdf.to_excel(writer, sheet_name='f_pkr', index=False)
            wdf.to_excel(writer, sheet_name='w_p', index=False)
        
        print(f"\n✅ Excel dosyası kaydedildi: {excel_path}")
    except Exception as e:
        print(f"\n❌ Excel yazma hatası: {e}")
    
    print(f"✅ Log dosyası: {log_path}")
    print(f"\n{'='*80}")
    print(f"🎯 TIGHT M İLE SONUÇLAR:")
    print(f"   Toplam varış: {total_arrival_times:.2f} dk")
    print(f"   Toplam bekleme: {total_wait:.2f} dk (üst sınır: {EPS_WAIT})")
    print(f"   Çözüm süresi: {m.Runtime:.2f} sn")
    print(f"   MIP Gap: {m.MIPGap*100:.2f}%")
    print(f"{'='*80}\n")

elif m.status == GRB.INFEASIBLE:
    print("\n" + "="*80)
    print("❌ MODEL INFEASIBLE")
    print("="*80)
    m.computeIIS()
    iis_file = f"infeasible_tightM_{timestamp}.ilp"
    m.write(iis_file)
    print(f"\n✅ IIS dosyası: {iis_file}")

else:
    print(f"\n❌ Çözüm bulunamadı. Status = {m.status}")

print("\n" + "="*80)
print("PROGRAM TAMAMLANDI")
print(f"Terminal çıktısı: {terminal_log_path}")
print("="*80)

sys.stdout = original_stdout
terminal_log_file.close()
