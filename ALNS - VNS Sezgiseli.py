import pandas as pd
import random
import numpy as np
from datetime import datetime
import os

# Veri yükleme (Gurobi modelinizden alınmıştır)
data_path = r"C:\Users\Asus\Desktop\Sezgisel\\"
nodes = pd.read_excel(data_path + "nodes.xlsx")
vehicles = pd.read_excel(data_path + "vehicles.xlsx")
products = pd.read_excel(data_path + "products.xlsx")
distances = pd.read_excel(data_path + "distances.xlsx")

# Setler ve parametreler
N = nodes['node_id'].astype(str).tolist()
Nw = [n for n in N if n != 'h']
K = vehicles['vehicle_id'].tolist()[:2]  # Araç sayısını 2'ye sınırlıyoruz
q_vehicle = dict(zip(K, vehicles['capacity_m2'][:2]))
P = products['product_id'].tolist()
q_product = dict(zip(P, products['area_m2']))
orig = dict(zip(P, products['origin'].astype(str)))
dest = dict(zip(P, products['destination'].astype(str)))
ep = dict(zip(P, products['ready_time']))
sl = dict(zip(P, products['load_time']))
su = dict(zip(P, products['unload_time']))
dist = {(str(r['from_node']), str(r['to_node'])): r['duration_min'] for _, r in distances.iterrows()}
R = [f'r{i}' for i in range(1, 21)]  # Tur sayısını artırdık (20'ye çıkardık)
M_time = 900
M_load = 60
W = 9999

# Yardımcı fonksiyon: Zamanı dakikaya çevirme
def time_to_minutes(time_str):
    if isinstance(time_str, str):
        h, m = map(int, time_str.split(":"))
        return h * 60 + m
    return time_str

# Zaman formatlama fonksiyonu
def format_minutes_with_day(minutes):
    days = minutes // (24 * 60)
    remaining_minutes = minutes % (24 * 60)
    hours = remaining_minutes // 60
    mins = remaining_minutes % 60
    return f"Day {days + 1} {hours:02d}:{mins:02d}"

# Çözüm yapısı
class Solution:
    def __init__(self):
        self.assignments = {}  # ürün: (k, r)
        self.routes = {}  # (k, r): [düğüm listesi]
        self.loads = {}  # (j, k, r): yük miktarı
        self.ta = {}  # arrival time
        self.td = {}  # departure time
        self.ts = {}  # service start time

# Başlangıç çözümü
def create_initial_solution():
    solution = Solution()
    solution.routes = {(k, r): ['h'] for k in K for r in R}
    solution.loads = {(j, k, r): 0 for j in N for k in K for r in R}
    solution.ta = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    solution.td = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    solution.ts = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    
    for p in P:
        assigned = False
        for k in K:
            for r in R:
                if check_constraints(solution, p, k, r):
                    solution.assignments[p] = (k, r)
                    route = solution.routes[(k, r)]
                    if orig[p] not in route:
                        route.insert(-1, orig[p])
                    if dest[p] not in route:
                        route.insert(-1, dest[p])
                    update_times_and_loads(solution, p, k, r)
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            print(f"Failed to assign product {p} in initial solution")
    print(f"Initial assignments: {len(solution.assignments)} / {len(P)} products assigned")
    return solution

# Kısıt kontrolü (sadece kapasite, zaman bekleme ile hallediliyor)
def check_constraints(solution, p, k, r):
    current_load = sum(q_product[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r))
    if current_load + q_product[p] > q_vehicle[k]:
        return False
    return True

# Zaman ve yük güncelleme (pickup'ta bekleme eklendi)
def update_times_and_loads(solution, p, k, r):
    route = solution.routes[(k, r)]
    current_time = 420 if r == 'r1' else solution.td.get(('h', k, R[R.index(r)-1]), 420)
    current_load = 0
    
    for i in range(1, len(route)):
        prev_node = route[i-1]
        curr_node = route[i]
        travel_time = dist.get((prev_node, curr_node), M_time)
        current_time += travel_time
        solution.ta[(curr_node, k, r)] = current_time
        
        # Pickup ise bekleme (max rt)
        if any(orig[p2] == curr_node for p2 in solution.assignments if solution.assignments[p2] == (k, r)):
            max_rt = max(time_to_minutes(ep[p2]) for p2 in solution.assignments if solution.assignments[p2] == (k, r) and orig[p2] == curr_node)
            current_time = max(current_time, max_rt)
        solution.ts[(curr_node, k, r)] = current_time
        
        load_in = sum(q_product[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r) and orig[p2] == curr_node)
        load_out = sum(q_product[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r) and dest[p2] == curr_node)
        current_load += load_in - load_out
        solution.loads[(curr_node, k, r)] = current_load
        
        service_time = 0
        if load_in > 0:
            service_time += sum(sl[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r) and orig[p2] == curr_node)
        if load_out > 0:
            service_time += sum(su[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r) and dest[p2] == curr_node)
        current_time += service_time
        solution.td[(curr_node, k, r)] = current_time

# Amaç fonksiyonu
def calculate_objective(solution):
    total_waiting_time = 0
    for p in P:
        if p in solution.assignments:
            k, r = solution.assignments[p]
            rt = time_to_minutes(ep[p])
            arrival_time = solution.ta.get((dest[p], k, r), M_time)
            waiting_time = max(0, arrival_time - rt)
            total_waiting_time += waiting_time
        else:
            total_waiting_time += W
    return total_waiting_time

# Yok Etme Operatörleri
def random_removal(solution, num_remove):
    removed = random.sample(list(solution.assignments.keys()), min(num_remove, len(solution.assignments)))
    new_solution = Solution()
    new_solution.assignments = {p: solution.assignments[p] for p in solution.assignments if p not in removed}
    new_solution.routes = {(k, r): ['h'] for k in K for r in R}
    new_solution.loads = {(j, k, r): 0 for j in N for k in K for r in R}
    new_solution.ta = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    new_solution.td = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    new_solution.ts = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    
    for p in new_solution.assignments:
        k, r = new_solution.assignments[p]
        route = new_solution.routes[(k, r)]
        if orig[p] not in route:
            route.insert(-1, orig[p])
        if dest[p] not in route:
            route.insert(-1, dest[p])
        update_times_and_loads(new_solution, p, k, r)
    
    return new_solution, removed

def worst_waiting_time_removal(solution, num_remove):
    waiting_times = [(p, max(0, solution.ta.get((dest[p], *solution.assignments[p]), M_time) - time_to_minutes(ep[p])))
                     for p in solution.assignments]
    waiting_times.sort(key=lambda x: x[1], reverse=True)
    removed = [p for p, _ in waiting_times[:min(num_remove, len(waiting_times))]]
    new_solution = Solution()
    new_solution.assignments = {p: solution.assignments[p] for p in solution.assignments if p not in removed}
    new_solution.routes = {(k, r): ['h'] for k in K for r in R}
    new_solution.loads = {(j, k, r): 0 for j in N for k in K for r in R}
    new_solution.ta = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    new_solution.td = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    new_solution.ts = {(j, k, r): 420 if r == 'r1' else 0 for j in N for k in K for r in R}
    
    for p in new_solution.assignments:
        k, r = new_solution.assignments[p]
        route = new_solution.routes[(k, r)]
        if orig[p] not in route:
            route.insert(-1, orig[p])
        if dest[p] not in route:
            route.insert(-1, dest[p])
        update_times_and_loads(new_solution, p, k, r)
    
    return new_solution, removed

# Onarma Operatörü
def greedy_insert(solution, removed):
    new_solution = solution
    for p in removed:
        best_k, best_r, best_cost = None, None, float('inf')
        for k in K:
            for r in R:
                if check_constraints(new_solution, p, k, r):
                    temp_solution = Solution()
                    temp_solution.assignments = new_solution.assignments.copy()
                    temp_solution.assignments[p] = (k, r)
                    temp_solution.routes = {(k2, r2): new_solution.routes[(k2, r2)].copy() for k2, r2 in new_solution.routes}
                    temp_solution.loads = new_solution.loads.copy()
                    temp_solution.ta = new_solution.ta.copy()
                    temp_solution.td = new_solution.td.copy()
                    temp_solution.ts = new_solution.ts.copy()
                    
                    route = temp_solution.routes[(k, r)]
                    if orig[p] not in route:
                        route.insert(-1, orig[p])
                    if dest[p] not in route:
                        route.insert(-1, dest[p])
                    update_times_and_loads(temp_solution, p, k, r)
                    
                    cost = calculate_objective(temp_solution)
                    if cost < best_cost:
                        best_k, best_r, best_cost = k, r, cost
                        new_solution = temp_solution
        if best_k is None:
            print(f"Failed to insert product {p}")
    return new_solution

# VNS: Yerel Arama
def vns_local_search(solution, max_iterations=100):
    current_solution = solution
    current_cost = calculate_objective(current_solution)
    
    # Komşuluk yapıları
    def swap_nodes(solution):
        new_solution = Solution()
        new_solution.assignments = solution.assignments.copy()
        new_solution.routes = {(k, r): route.copy() for (k, r), route in solution.routes.items()}
        new_solution.loads = solution.loads.copy()
        new_solution.ta = solution.ta.copy()
        new_solution.td = solution.td.copy()
        new_solution.ts = solution.ts.copy()
        
        k, r = random.choice(list(new_solution.routes.keys()))
        route = new_solution.routes[(k, r)]
        if len(route) > 3:  # Depo hariç en az 2 düğüm olmalı
            i, j = random.sample(range(1, len(route)-1), 2)
            route[i], route[j] = route[j], route[i]
            update_times_and_loads_for_route(new_solution, k, r)
        return new_solution
    
    def move_product(solution):
        new_solution = Solution()
        new_solution.assignments = solution.assignments.copy()
        new_solution.routes = {(k, r): route.copy() for (k, r), route in solution.routes.items()}
        new_solution.loads = solution.loads.copy()
        new_solution.ta = solution.ta.copy()
        new_solution.td = solution.td.copy()
        new_solution.ts = solution.ts.copy()
        
        if not new_solution.assignments:
            return new_solution
        
        p = random.choice(list(new_solution.assignments.keys()))
        old_k, old_r = new_solution.assignments[p]
        new_k, new_r = random.choice([(k, r) for k in K for r in R if (k, r) != (old_k, old_r)])
        
        if check_constraints(new_solution, p, new_k, new_r):
            new_solution.assignments[p] = (new_k, new_r)
            old_route = new_solution.routes[(old_k, old_r)]
            if orig[p] in old_route:
                old_route.remove(orig[p])
            if dest[p] in old_route:
                old_route.remove(dest[p])
            new_route = new_solution.routes[(new_k, new_r)]
            if orig[p] not in new_route:
                new_route.insert(-1, orig[p])
            if dest[p] not in new_route:
                new_route.insert(-1, dest[p])
            update_times_and_loads_for_route(new_solution, new_k, new_r)
            update_times_and_loads_for_route(new_solution, old_k, old_r)
        return new_solution
    
    def update_times_and_loads_for_route(solution, k, r):
        route = solution.routes[(k, r)]
        current_time = 420 if r == 'r1' else solution.td.get(('h', k, R[R.index(r)-1] if r in R and R.index(r) > 0 else 'r1'), 420)
        current_load = 0
        
        for i in range(1, len(route)):
            prev_node = route[i-1]
            curr_node = route[i]
            travel_time = dist.get((prev_node, curr_node), M_time)
            current_time += travel_time
            solution.ta[(curr_node, k, r)] = current_time
            
            # Pickup ise bekleme
            if any(orig[p2] == curr_node for p2 in solution.assignments if solution.assignments[p2] == (k, r)):
                max_rt = max(time_to_minutes(ep[p2]) for p2 in solution.assignments if solution.assignments[p2] == (k, r) and orig[p2] == curr_node)
                current_time = max(current_time, max_rt)
            solution.ts[(curr_node, k, r)] = current_time
            
            load_in = sum(q_product[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r) and orig[p2] == curr_node)
            load_out = sum(q_product[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r) and dest[p2] == curr_node)
            current_load += load_in - load_out
            solution.loads[(curr_node, k, r)] = current_load
            
            service_time = 0
            if load_in > 0:
                service_time += sum(sl[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r) and orig[p2] == curr_node)
            if load_out > 0:
                service_time += sum(su[p2] for p2 in solution.assignments if solution.assignments[p2] == (k, r) and dest[p2] == curr_node)
            current_time += service_time
            solution.td[(curr_node, k, r)] = current_time
    
    neighborhoods = [swap_nodes, move_product]
    
    for _ in range(max_iterations):
        for neighborhood in neighborhoods:
            new_solution = neighborhood(current_solution)
            new_cost = calculate_objective(new_solution)
            if new_cost < current_cost:
                current_solution = new_solution
                current_cost = new_cost
                break
    
    return current_solution, current_cost

# ALNS + VNS Hibrit Algoritma
def alns_vns_hybrid(max_iterations=1000, num_remove=2, vns_iterations=50):
    current_solution = create_initial_solution()
    best_solution = current_solution
    best_cost = calculate_objective(best_solution)
    
    destroy_operators = [random_removal, worst_waiting_time_removal]
    repair_operators = [greedy_insert]
    weights = {op: 1.0 for op in destroy_operators + repair_operators}
    
    for i in range(max_iterations):
        destroy_op = random.choices(destroy_operators, weights=[weights[op] for op in destroy_operators])[0]
        repair_op = random.choices(repair_operators, weights=[weights[op] for op in repair_operators])[0]
        
        new_solution, removed = destroy_op(current_solution, num_remove)
        new_solution = repair_op(new_solution, removed)
        
        # VNS ile yerel iyileştirme
        new_solution, new_cost = vns_local_search(new_solution, vns_iterations)
        
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost
        current_solution = new_solution
        
        if new_cost < best_cost:
            weights[destroy_op] *= 1.1
            weights[repair_op] *= 1.1
        else:
            weights[destroy_op] *= 0.9
            weights[repair_op] *= 0.9
    
    return best_solution, best_cost

# Sonuçları kaydetme
def save_results(solution, cost, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    writer_file_name = os.path.join(output_dir, f"alns_vns_result_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.xlsx")
    
    with pd.ExcelWriter(writer_file_name) as writer:
        # x_ijkr
        x_data = []
        for (k, r) in solution.routes:
            route = solution.routes[(k, r)]
            for idx in range(len(route) - 1):
                from_node = route[idx]
                to_node = route[idx + 1]
                x_data.append(['x', from_node, to_node, k, r, 1])
        xijkr_results_df = pd.DataFrame(x_data, columns=['var_name', 'from_node', 'to_node', 'vehicle', 'route', 'value'])
        xijkr_results_df.to_excel(writer, sheet_name='x_ijkr', index=False)
        print("x_ijkr results are extracted...")

        # f_pkr
        f_data = []
        for p in solution.assignments:
            k, r = solution.assignments[p]
            f_data.append(['f', p, k, r, 1])
        f_pkr_results_df = pd.DataFrame(f_data, columns=['var_name', 'product', 'vehicle', 'route', 'value'])
        f_pkr_results_df.to_excel(writer, sheet_name='f_pkr', index=False)
        print("f_pkr results are extracted...")

        # w_p
        w_data = []
        for p in solution.assignments:
            k, r = solution.assignments[p]
            rt = time_to_minutes(ep[p])
            arrival_time = solution.ta.get((dest[p], k, r), M_time)
            waiting_time = max(0, arrival_time - rt)
            w_data.append(['w', p, waiting_time])
        w_p_results_df = pd.DataFrame(w_data, columns=['var_name', 'product', 'value'])
        w_p_results_df.to_excel(writer, sheet_name='w_p', index=False)
        print("w_p results are extracted...")

        # y_jkr
        y_data = []
        for (k, r) in solution.routes:
            visited = set(solution.routes[(k, r)]) - {'h'}
            for j in visited:
                y_data.append(['y', j, k, r, 1])
        y_jkr_results_df = pd.DataFrame(y_data, columns=['var_name', 'node', 'vehicle', 'route', 'value'])
        y_jkr_results_df.to_excel(writer, sheet_name='y_jkr', index=False)
        print("y_jkr results are extracted...")

        # ta_jkr
        ta_data = []
        for key, value in solution.ta.items():
            j, k, r = key
            ta_data.append(['ta', j, k, r, value])
        ta_jkr_results_df = pd.DataFrame(ta_data, columns=['var_name', 'node', 'vehicle', 'route', 'value'])
        ta_jkr_results_df['time_fmt'] = ta_jkr_results_df['value'].astype(int).apply(format_minutes_with_day)
        ta_jkr_results_df.to_excel(writer, sheet_name='ta_jkr', index=False)
        print("ta_jkr results are extracted...")

        # td_jkr
        td_data = []
        for key, value in solution.td.items():
            j, k, r = key
            td_data.append(['td', j, k, r, value])
        td_jkr_results_df = pd.DataFrame(td_data, columns=['var_name', 'node', 'vehicle', 'route', 'value'])
        td_jkr_results_df['time_fmt'] = td_jkr_results_df['value'].astype(int).apply(format_minutes_with_day)
        td_jkr_results_df.to_excel(writer, sheet_name='td_jkr', index=False)
        print("td_jkr results are extracted...")

        # ts_jkr
        ts_data = []
        for key, value in solution.ts.items():
            j, k, r = key
            ts_data.append(['ts', j, k, r, value])
        ts_jkr_results_df = pd.DataFrame(ts_data, columns=['var_name', 'node', 'vehicle', 'route', 'value'])
        ts_jkr_results_df['time_fmt'] = ts_jkr_results_df['value'].astype(int).apply(format_minutes_with_day)
        ts_jkr_results_df.to_excel(writer, sheet_name='ts_jkr', index=False)
        print("ts_jkr results are extracted...")

        # delta_jkr (assuming delta_pkr)
        delta_data = []
        for p in solution.assignments:
            k, r = solution.assignments[p]
            delta_data.append(['delta', p, k, r, 1])
        delta_jkr_results_df = pd.DataFrame(delta_data, columns=['var_name', 'product', 'vehicle', 'route', 'value'])
        delta_jkr_results_df.to_excel(writer, sheet_name='delta_jkr', index=False)
        print("delta_jkr results are extracted...")
    
    print(f"Toplam bekleme süresi: {cost:.1f} dakika")
    print(f"Sonuçlar {writer_file_name} dosyasına kaydedildi.")

# Hibrit ALNS + VNS’yi çalıştır
best_solution, best_cost = alns_vns_hybrid(max_iterations=1000, num_remove=2, vns_iterations=50)
save_results(best_solution, best_cost)