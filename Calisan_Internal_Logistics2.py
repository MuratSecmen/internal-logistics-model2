import pandas as pd
from gurobipy import Model, GRB, quicksum
import re
import os
from datetime import datetime

def format_minutes_with_day(total_minutes: int) -> str:
    days = total_minutes // 1440
    rem  = total_minutes % 1440
    hh   = rem // 60
    mm   = rem % 60
    hhmm = f"{hh:02d}{mm:02d}"
    return f"{days}+{hhmm}" if days > 0 else hhmm

def model_organize_results(var_values):
    rows = []
    for v in var_values:
        current_var = re.split(r"\[|,|]", v.varName)[:-1]
        current_var.append(round(v.X, 4))
        rows.append(current_var)
    return pd.DataFrame(rows)

# 1) Veri yükleme
data_path = r"C:\Users\Asus\Desktop\Er\\"
# data_path = "inputs/"
nodes     = pd.read_excel(data_path + "nodes.xlsx")
vehicles  = pd.read_excel(data_path + "vehicles.xlsx")
products  = pd.read_excel(data_path + "products.xlsx")
products = products.head(100)
distances = pd.read_excel(data_path + "distances.xlsx")

# 2) Setler ve parametreler
N    = nodes['node_id'].astype(str).tolist()
Nw   = [n for n in N if n != 'h']
K    = vehicles['vehicle_id'].tolist()
# q    = dict(zip(K, vehicles['capacity_m2']))
q_vehicle = dict(zip(K, vehicles['capacity_m2']))      
P    = products['product_id'].tolist()
q_product = dict(zip(P, products['area_m2']))
orig = dict(zip(P, products['origin'].astype(str)))
dest = dict(zip(P, products['destination'].astype(str)))
ep   = dict(zip(P, products['ready_time']))
sl   = dict(zip(P, products['load_time']))
su   = dict(zip(P, products['unload_time']))
dist = {(str(r['from_node']), str(r['to_node'])): r['duration_min']
        for _, r in distances.iterrows()}
# R    = ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10']
# R    = ['r1','r2']
R    = ['r1','r2','r3']
# R    = ['r1','r2','r3','r4','r5','r6','r7','r8']
# R      = ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20']

# 3) Big M’ler
M_time = 900
M_load = 60
eps_wait_time = 0

# 4) Model oluştur
model = Model("InternalLogistics")

# 5) Karar değişkenleri
# x   = model.addVars(N, N, K, R, vtype=GRB.BINARY,    name="x")
x = model.addVars(
    ((i, j, k, r) for i in N for j in N if not (i == "h" and j == "h") for k in K for r in R),
    vtype=GRB.BINARY,
    name="x"
)
f   = model.addVars(P, K, R, vtype=GRB.BINARY,    name="f")
w   = model.addVars(P, vtype=GRB.CONTINUOUS, name="w")
y   = model.addVars(N, K, R, vtype=GRB.CONTINUOUS, name="y")
ta  = model.addVars(N, K, R, vtype=GRB.CONTINUOUS, name="ta")
td  = model.addVars(N, K, R, vtype=GRB.CONTINUOUS, name="td")
ts  = model.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, name="ts")
delta = model.addVars(Nw, K, R, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="delta")

# 6) Amaç: Toplam bekleme süresi minimize
model.setObjective(quicksum(w[p] for p in P), GRB.MINIMIZE)

# Kısıtlar
# C3 - C4: Araç depot'tan çıkarsa geri döner ve en fazla 1 çıkış olur
for k in K:
    for r in R:
        model.addConstr(quicksum(x['h',j,k,r] for j in Nw) == quicksum(x[i,'h',k,r] for i in Nw))
        # model.addConstr(quicksum(x['h',j,k,r] for j in Nw) == 1)
        model.addConstr(quicksum(x['h',j,k,r] for j in Nw) <= 1)

# C5: Her araç ve rota için, depodan (home) yapılan çıkış sayısı, o araca ve rotaya atanan ürün sayısını aşamaz.
for k in K:
    for r in R:
        model.addConstr(
            quicksum(x["h", j, k, r] for j in Nw) <= quicksum(f[p, k, r] for p in P),
            name=f"C5_BaseDeparture_k{k}_r{r}"
        )

# C6: Her müşteri, araç ve rota için: müşteriden çıkan akış, müşterinin origin veya destination olduğu atanan ürünlerle sınırlıdır.
# Assuming Nw is defined as customer nodes (e.g., Nw = [node for node in N if node != 'h'])
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(
                quicksum(x[i, j, k, r] for i in N if i != j) <= quicksum(f[p, k, r] for p in P if orig[p] == j or dest[p] == j),
                name=f"C6_j{j}_k{k}_r{r}"
            )

# C7 - C8: Akış koruma + tek giriş
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(quicksum(x[i,j,k,r] for i in N if i!=j) <= quicksum(x[j,i,k,r] for i in N if i!=j))
            model.addConstr(quicksum(x[i,j,k,r] for i in N if i!=j) <= 1)

# C9: Her ürün en fazla bir araca ve rotaya atanır
for p in P:
    model.addConstr(quicksum(f[p,k,r] for k in K for r in R) <= 1)

# C10 - C11: Ürün atanmışsa origin/destination’u ziyaret et
for p in P:
    for k in K:
        for r in R:
            model.addConstr(quicksum(x[i, orig[p], k, r] for i in N if i!=orig[p]) >= f[p,k,r])
            model.addConstr(quicksum(x[i, dest[p], k, r] for i in N if i!=dest[p]) >= f[p,k,r])

# C12: Başlangıç zamanı
for k in K:
    # model.addConstr(td['h',k,'r1'] == 0)
    model.addConstr(td['h',k,'r1'] == 420)

# C13: Rotalar arası zaman tutarlılığı
for k in K:
    for idx in range(1, len(R)):
        model.addConstr(td['h',k,R[idx]] >= ta['h',k,R[idx-1]])

# C14: Zaman tutarlılığı (i->j hareketi)
for i in N:
    for j in N:
        if i!=j and (i,j) in dist:
            for k in K:
                for r in R:
                    model.addConstr(ta[j,k,r] >= td[i,k,r] + dist[(i,j)]*x[i,j,k,r] - M_time*(1-x[i,j,k,r]))
                    # model.addConstr(ta[j,k,r] <= td[i,k,r] + dist[(i,j)]*x[i,j,k,r] + M_time*(1-x[i,j,k,r]))

# C15 Servis (Yük Boşaltma) zamanları
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(ts[j,k,r] >= ta[j,k,r] + quicksum(su[p]*f[p,k,r] for p in P if dest[p]==j))            

# C16 Servis zamanları (Ürün hazır edilmeden servis başlayamaz)
for p in P:
    rt = int(ep[p].split(":")[0])*60 + int(ep[p].split(":")[1])
    h  = orig[p]
    for k in K:
        for r in R:
            model.addConstr(ts[h,k,r] >= rt * f[p,k,r])

# C17 Çıkış (Yükleme sonrası istasyondan ayrılma) zamanları
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(td[j,k,r] >= ts[j,k,r] + quicksum(sl[p]*f[p,k,r] for p in P if orig[p]==j))

# C18: Ürün önce yüklenir sonra teslim edilir (strict)
for p in P:
    for k in K:
        for r in R:
            model.addConstr(ta[dest[p],k,r] >= td[orig[p],k,r] - M_time*(1-f[p,k,r]))

# C19: Her ürünün kaynak düğüm varış zamanı, çıkış düğüm çıkış zamanından büyük veya eşit olacak
home_node = "h"
for k in K:
    for r in R:
        model.addConstr(
            ta[home_node, k, r] >= td[home_node, k, r] + eps_wait_time,
            name=f"C19_home_k{k}_r{r}"
        )

# C20: Ürün bekleme süresi
for p in P:
    rt = int(ep[p].split(":")[0])*60 + int(ep[p].split(":")[1])
    for k in K:
        for r in R:
            model.addConstr(w[p] >= ta[dest[p],k,r] - rt - M_time*(1-f[p,k,r]))
            # model.addConstr(w[p] >= ta[dest[p],k,r] + su[p]*f[p,k,r] - rt - M_time*(1-f[p,k,r]))

# C21: Alınmayan ürünlere penaltı
# for p in P:
    # model.addConstr(w[p] >= 9999 * (1- quicksum(f[p,k,r] for k in K for r in R)))

W = 9999
for p in P:
    model.addConstr(w[p] >= W * (1 - quicksum(f[p,k,r] for k in K for r in R)))


# C22: Yük değişimi ve evrimi (Big M_load) + kapasite limiti
for j in Nw:
    for k in K:
        for r in R:
            # load_in  = quicksum(f[p,k,r] for p in P if orig[p]==j)
            # load_out = quicksum(f[p,k,r] for p in P if dest[p]==j)
            load_in  = quicksum(q_product[p] * f[p,k,r] for p in P if orig[p]==j)
            load_out = quicksum(q_product[p] * f[p,k,r] for p in P if dest[p]==j)
            model.addConstr(delta[j,k,r] >= load_in - load_out)
            for i in Nw:
                model.addConstr(y[j,k,r] <=  y[i,k,r] + delta[j,k,r] + M_load*(1-x[i,j,k,r]))
                model.addConstr(y[j,k,r] >=  y[i,k,r] + delta[j,k,r] - M_load*(1-x[i,j,k,r]))
                # model.addConstr(y[j,k,r] <= q[k])
                model.addConstr(y[j,k,r] <= q_vehicle[k])

# C23 and C24: Araç yük güncelleme
for i in Nw:
    for j in Nw:
        for k in K:
            for r in R:
                model.addConstr(y[j, k, r] >= y[i, k, r] + delta[j, k, r] - M_load * (1 - x[i, j, k, r]), name=f"C23_{i}_{j}_{k}_{r}")
                model.addConstr(y[j, k, r] <= y[i, k, r] + delta[j, k, r] + M_load * (1 - x[i, j, k, r]), name=f"C24_{i}_{j}_{k}_{r}")

# C25: Aracın yükü kapasitesini aşamaz
for j in Nw:
    for k in K:
        for r in R:
            model.addConstr(y[j, k, r] <= q_vehicle[k], name=f"C25_{j}_{k}_{r}")                

# 8) Gurobi parametreleri ve optimize et
model.setParam('TimeLimit', 600)
model.setParam('MIPGap', 0.03)
model.setParam('LogFile', 'gurobi_log.txt')
# model.setParam('Presolve', 2)
# model.setParam('NoRelHeurTime', 360)
# model.setParam('MIPFocus', 1)
model.setParam('MIPFocus', 3)
model.setParam('Cuts', 3)

model.optimize()

# 9) Sonuç veya IIS analizi
if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
    for p in P:
        for k in K:
            for r in R:
                if f[p,k,r].X > 0.5:
                    print(f"Ürün {p}: Araç {k}, Rota {r}, Bekleme: {w[p].X:.1f} dk")

    xijkr_results_df = model_organize_results(x.values())
    xijkr_results_df.columns = ['var_name', 'from_node', 'to_node', 'vehicle', 'route', 'value']
    print("x_ijkr results are extracted...")

    f_pkr_results_df = model_organize_results(f.values())
    f_pkr_results_df.columns = ['var_name', 'product', 'vehicle', 'route', 'value']
    print("f_pkr results are extracted...")

    w_p_results_df = model_organize_results(w.values())
    w_p_results_df.columns = ['var_name', 'product', 'value']
    print("w_p results are extracted...")

    y_jkr_results_df = model_organize_results(y.values())
    y_jkr_results_df.columns = ['var_name', 'node', 'vehicle', 'route', 'value']
    print("y_jkr results are extracted...")

    ta_jkr_results_df = model_organize_results(ta.values())
    ta_jkr_results_df.columns = ['var_name', 'node', 'vehicle', 'route', 'value']
    ta_jkr_results_df['time_fmt'] = ta_jkr_results_df['value'].astype(int).apply(format_minutes_with_day)
    print("ta_jkr results are extracted...")

    td_jkr_results_df = model_organize_results(td.values())
    td_jkr_results_df.columns = ['var_name', 'node', 'vehicle', 'route', 'value']
    td_jkr_results_df['time_fmt'] = td_jkr_results_df['value'].astype(int).apply(format_minutes_with_day)
    print("td_jkr results are extracted...")

    ts_jkr_results_df = model_organize_results(ts.values())
    ts_jkr_results_df.columns = ['var_name', 'node', 'vehicle', 'route', 'value']
    ts_jkr_results_df['time_fmt'] = ts_jkr_results_df['value'].astype(int).apply(format_minutes_with_day)
    print("ts_jkr results are extracted...")

    delta_jkr_results_df = model_organize_results(delta.values())
    delta_jkr_results_df.columns = ['var_name', 'product', 'vehicle', 'route', 'value']
    print("delta_jkr results are extracted...")

    optimization_results_df = pd.DataFrame(
        columns=['model_obj_value', 'model_obj_bound', 'gap', 'gurobi_time'])

    optimization_results_df.loc[len(optimization_results_df.index)] = [model.objval, model.objbound, model.mipgap, model.runtime]

    output_dir = 'results'
    #writer_file_name = os.path.join('outputs', "result_of_run_{}.xlsx".format(str(datetime.now().strftime('%Y_%m_%d_%H_%M'))))
    writer_file_name = os.path.join(output_dir, "result_of_run_{}.xlsx".format(str(datetime.now().strftime('%Y_%m_%d_%H_%M'))))

    print("start printing results to excel file.")

    writer = pd.ExcelWriter(writer_file_name)
    optimization_results_df.to_excel(writer, sheet_name='optimization_results')
    xijkr_results_df.to_excel(writer, sheet_name='xijkr_results')
    f_pkr_results_df.to_excel(writer, sheet_name='f_pkr_results')
    w_p_results_df.to_excel(writer, sheet_name='w_p_results')
    y_jkr_results_df.to_excel(writer, sheet_name='y_jkr_results')
    ta_jkr_results_df.to_excel(writer, sheet_name='ta_jkr_results')
    td_jkr_results_df.to_excel(writer, sheet_name='td_jkr_results')
    ts_jkr_results_df.to_excel(writer, sheet_name='ts_jkr_results')
    delta_jkr_results_df.to_excel(writer, sheet_name='delta_jkr_results')
    writer.close()

    print("All results are printed.")

elif model.status == GRB.INFEASIBLE:
    print("Model infeasible. Computing IIS...")
    model.computeIIS()
    model.write("infeasible.ilp")
    print("Go check infeasible_model.ilp file")

    # model.write(data_path + "model.ilp")

    print("IIS raporu kaydedildi:", data_path + "model.ilp")
else:
    print(f"Çözüm bulunamadı. Status = {model.status}")