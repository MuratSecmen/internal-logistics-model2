import pandas as pd
from gurobipy import Model, GRB, quicksum

# 1) Veri yükleme
data_path = r"C:\Users\Asus\Desktop\Er\\"
nodes     = pd.read_excel(data_path + "nodes.xlsx")
vehicles  = pd.read_excel(data_path + "vehicles.xlsx")
products  = pd.read_excel(data_path + "products.xlsx")
distances = pd.read_excel(data_path + "distances.xlsx")

# 2) Setler ve parametreler
N    = nodes['node_id'].astype(str).tolist()
Nw   = [n for n in N if n != 'h']
K    = vehicles['vehicle_id'].tolist()
q    = dict(zip(K, vehicles['capacity_m2']))
P    = products['product_id'].tolist()
orig = dict(zip(P, products['origin'].astype(str)))
dest = dict(zip(P, products['destination'].astype(str)))
ep   = dict(zip(P, products['ready_time']))
sl   = dict(zip(P, products['load_time']))
su   = dict(zip(P, products['unload_time']))
dist = {(str(r['from_node']), str(r['to_node'])): r['duration_min']
        for _, r in distances.iterrows()}
R    = ['r1','r2','r3']

# 3) Big M’ler
M_time = 900
M_load = 60
eps    = 1e-3

# 4) Model oluştur
model = Model("InternalLogistics")

# 5) Karar değişkenleri
x   = model.addVars(N, N, K, R, vtype=GRB.BINARY,    name="x")
f   = model.addVars(P, K, R, vtype=GRB.BINARY,    name="f")
w   = model.addVars(P, vtype=GRB.CONTINUOUS, name="w")
y   = model.addVars(N, K, R, vtype=GRB.CONTINUOUS, name="y")
ta  = model.addVars(N, K, R, vtype=GRB.CONTINUOUS, name="ta")
td  = model.addVars(N, K, R, vtype=GRB.CONTINUOUS, name="td")
ts  = model.addVars(N, K, R, vtype=GRB.CONTINUOUS, name="ts")

# 6) Amaç: Toplam bekleme süresi minimize
model.setObjective(quicksum(w[p] for p in P), GRB.MINIMIZE)

# 7) Kısıtlar
# 7.1 Araç depot'tan çıkarsa geri döner ve en fazla 1 çıkış olur
for k in K:
    for r in R:
        model.addConstr(quicksum(x['h',j,k,r] for j in Nw) == quicksum(x[i,'h',k,r] for i in Nw))
        model.addConstr(quicksum(x['h',j,k,r] for j in Nw) <= 1)

# 7.2 Akış koruma + tek giriş
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(quicksum(x[i,j,k,r] for i in N if i!=j) == quicksum(x[j,i,k,r] for i in N if i!=j))
            model.addConstr(quicksum(x[i,j,k,r] for i in N if i!=j) <= 1)

# 7.3 Ürün atanmışsa origin/destination’u ziyaret et
for p in P:
    for k in K:
        for r in R:
            model.addConstr(quicksum(x[i, orig[p], k, r] for i in N if i!=orig[p]) >= f[p,k,r])
            model.addConstr(quicksum(x[i, dest[p], k, r] for i in N if i!=dest[p]) >= f[p,k,r])

# 7.4 Her ürün bir araca ve rotaya atanır
for p in P:
    model.addConstr(quicksum(f[p,k,r] for k in K for r in R) == 1)

# 7.5 Başlangıç zamanı
for k in K:
    model.addConstr(td['h',k,'r1'] == 0)

# 7.6 Rotalar arası zaman tutarlılığı
for k in K:
    for idx in range(1, len(R)):
        model.addConstr(td['h',k,R[idx]] >= ta['h',k,R[idx-1]])

# 7.7 Zaman tutarlılığı (i->j hareketi)
for i in N:
    for j in N:
        if i!=j and (i,j) in dist:
            for k in K:
                for r in R:
                    model.addConstr(ta[j,k,r] >= td[i,k,r] + dist[(i,j)] - M_time*(1-x[i,j,k,r]))
                    model.addConstr(ta[j,k,r] <= td[i,k,r] + dist[(i,j)] + M_time*(1-x[i,j,k,r]))

# 7.8 Servis ve çıkış zamanları
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(ts[j,k,r] >= ta[j,k,r] + quicksum(su[p]*f[p,k,r] for p in P if dest[p]==j))
            model.addConstr(td[j,k,r] >= ts[j,k,r] + quicksum(sl[p]*f[p,k,r] for p in P if orig[p]==j))

# 7.9 Ürün hazır olmadan servis başlamaz
for p in P:
    rt = int(ep[p].split(":")[0])*60 + int(ep[p].split(":")[1])
    h  = orig[p]
    for k in K:
        for r in R:
            model.addConstr(ts[h,k,r] >= rt * f[p,k,r])

# 7.10 Ürün önce yüklenir sonra teslim edilir (strict)
for p in P:
    for k in K:
        for r in R:
            model.addConstr(ta[dest[p],k,r] >= td[orig[p],k,r] - eps - M_time*(1-f[p,k,r]))

# 7.11 Ürün bekleme süresi
for p in P:
    rt = int(ep[p].split(":")[0])*60 + int(ep[p].split(":")[1])
    for k in K:
        for r in R:
            model.addConstr(w[p] >= ta[dest[p],k,r] + su[p]*f[p,k,r] - rt - M_time*(1-f[p,k,r]))

# 7.12 Yük değişimi ve evrimi (Big M_load) + kapasite limiti
for j in Nw:
    for k in K:
        for r in R:
            load_in  = quicksum(f[p,k,r] for p in P if orig[p]==j)
            load_out = quicksum(f[p,k,r] for p in P if dest[p]==j)
            model.addConstr(y[j,k,r] <= load_in - load_out + M_load)
            model.addConstr(y[j,k,r] >= load_in - load_out - M_load)
            model.addConstr(y[j,k,r] <= q[k])

# 8) Gurobi parametreleri ve optimize et
model.setParam('TimeLimit', 86400)
model.setParam('MIPGap', 0.05)
model.setParam('LogFile', 'gurobi_log.txt')
model.setParam('Presolve', 2)
# model.setParam('NoRelHeurTime', 180)

model.optimize()

# 9) Sonuç veya IIS analizi
if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    for p in P:
        for k in K:
            for r in R:
                if f[p,k,r].X > 0.5:
                    print(f"Ürün {p}: Araç {k}, Rota {r}, Bekleme: {w[p].X:.1f} dk")
elif model.status == GRB.INFEASIBLE:
    print("Model infeasible! IIS raporu oluşturuluyor...")
    model.computeIIS()
    model.write(data_path + "model.ilp")
    print("IIS raporu kaydedildi:", data_path + "model.ilp")
else:
    print(f"Çözüm bulunamadı. Status = {model.status}")
