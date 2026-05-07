# Internal Composite Part Logistics Optimization
## BO-PDPRD-STW: Bi-Objective Pickup and Delivery Problem with Release Dates and Shift-based Time Windows

A mixed-integer linear programming (MILP) model and metaheuristic solution framework for internal composite part logistics in an aerospace manufacturing facility.

---

## Problem Definition

**BO-PDPRD-STW** is a novel bi-objective vehicle routing problem formulated for intra-facility logistics of semi-finished and finished composite parts between origin–destination pairs under:

- Pickup-before-delivery precedence constraints
- Part-specific release dates (ready times)
- Shift-based time windows with scheduled breaks
- Dead zone (14:45–15:15): no service allowed during shift handover
- Heterogeneous vehicle fleet with capacity constraints

**Fleet:**
| Shift | Time Window | Vehicles | Capacity |
|-------|-------------|----------|----------|
| Shift 1 | 07:00–15:00 | 3 ring vehicles | 20 m² each |
| Shift 2 | 15:00–23:00 | 2 electric vehicles | 15 m² each |

**Objective Functions:**
- **f₁ (primary):** Total vehicle return-to-depot time — `min Σₖ taₕₖ|R| − |K|·T_start`
- **f₂ (secondary):** Total part waiting time — `min Σₚ wₚ`, where `wₚ = ta_{d_p} − eₚ`

**Multi-objective method:** AUGMECON2 (augmented ε-constraint) for exact Pareto front on small/medium instances; NSGA-II metaheuristic for larger instances.

**Primary benchmark:** Daşdemir, Öztürk & Tezcaner Öztürk (2022, OMEGA).

---

## Technology Stack

- Python 3.10+
- Gurobi Optimizer 11.x (MILP solver)
- pymoo (NSGA-II metaheuristic framework)
- Pandas, openpyxl (data processing and output)

---

## Repository Structure

```
internal-logistics-model2/
├── inputs/
│   ├── nodes.xlsx
│   ├── vehicles.xlsx
│   ├── distances - dakika.xlsx
│   ├── products_4part.xlsx          ← test instances (3 sheets each)
│   ├── products_5part.xlsx
│   ├── products_10part.xlsx
│   └── heuristic_results/
├── results/
│   └── exact_solution/
│       ├── case1/                   ← all-distinct nodes (baseline)
│       ├── case2/                   ← LOC-C shared nodes (consolidation)
│       └── case3/                   ← shared pickup, RT-max stress test
├── MO Optimization-Depoya Donus and Wait - BIG M renewal.py   ← MILP (Gurobi)
├── solomon_i1_pdp.py                ← H1 Solomon I1-PDP construction heuristic
└── README.md
```

**Three canonical test cases (locked and verified):**
| Case | Structure | f₁ | f₂ |
|------|-----------|----|----|
| Case 1 | All-distinct nodes — baseline | 77 | 91 |
| Case 2 | LOC-C karma nodes — consolidation | 67 | 85 |
| Case 3 | Shared pickup, RT-max stress test | 85 | 130 |

---

## Mathematical Model

### Decision Variables

| Variable | Type | Definition |
|----------|------|------------|
| xᵢⱼₖᵣ | Binary | 1 if vehicle k (route r) traverses arc i→j |
| fₚₖᵣ | Binary | 1 if part p is assigned to vehicle k, route r |
| wₚ | Continuous ≥ 0 | Waiting time of part p (minutes) |
| taⱼₖᵣ, tdⱼₖᵣ, tsⱼₖᵣ | Continuous ≥ 0 | Arrival / departure / service-start times |
| yⱼₖᵣ | Continuous ≥ 0 | Load of vehicle k upon leaving node j (m²) |
| Δⱼₖᵣ | Continuous ∈ ℝ | Net load change at node j |
| uⱼₖᵣ | Integer ∈ {0,…,U} | MTZ visit sequence number |

### Constraint Groups

| Group | Constraints | Description |
|-------|-------------|-------------|
| Route structure | C4–C9 | Flow conservation, route closure, activation |
| Assignment | C10–C12 | Each part assigned exactly once; pickup/delivery visits enforced |
| Timing | C13–C22 | Shift start, route sequencing, travel, service, release date, precedence, waiting time |
| Capacity | C23–C27 | Load flow, vehicle capacity, depot load |
| Route ordering | C28 | Monotonic route activation |
| Subtour elimination | C29–C32 | Miller–Tucker–Zemlin (MTZ) formulation |

---

## Big-M Parameters

Tight Big-M values are derived analytically from problem-specific data to improve LP relaxation quality and solver performance (see Camm et al., 1990).

**Base parameters:**
```
T_start = 420 min  (07:00 absolute clock time)
T_max   = 480 min  (8-hour shift duration)
C_max   =  11 min  (longest travel time between any two nodes)
e_min   = 435 min  (earliest part ready time = 07:15)
Q_max   =  20 m²   (maximum vehicle capacity)
```

### Tight Big-M (Production — Recommended)

| Constraint | Parameter | Formula | Value |
|------------|-----------|---------|-------|
| C16 — time consistency | M₁₆ | T_max + C_max | **491 min** |
| C20 — pickup-delivery precedence | M₂₀ | T_max | **480 min** |
| C22 — waiting time definition | M₂₂ | (T_start + T_max) − e_min | **465 min** |
| C24 — load flow lower bound | M₂₄ | Q_max | **20 m²** |
| C25 — load flow upper bound | M₂₅ | Q_max | **20 m²** |
| C29 — MTZ subtour | U | \|Nw\| | **≤ 20** |

> ⚠️ **Critical note on M₂₂:** The correct derivation requires absolute clock-time arithmetic.
> Maximum feasible waiting time = `(T_start + T_max) − e_min = 900 − 435 = 465 min`.
> A common error is using `T_max − e_min = 45`, which mixes relative duration with absolute
> clock time and renders the model artificially infeasible on valid instances.

### Naive Big-M (Development / Debugging Only)

```python
M = 9999  # uniform — use only for feasibility cross-checking, never for Pareto runs
```

> ⚠️ Running AUGMECON2 with `EPS_WAIT = 9999` silently converts the method into a
> single weighted-sum point. All Pareto front claims made under this setting are
> methodologically invalid.

---

## AUGMECON2 Implementation

The model uses the augmented ε-constraint method (Mavrotas & Florios, 2013) to guarantee strictly Pareto-optimal solutions.

**Procedure:**
1. Solve `min f₁` (f₂ free) → obtain `f2_max`
2. Solve `min f₂` (f₁ free) → obtain `f2_min`
3. Generate ε grid: `ε ∈ [f2_min, f2_max]`, minimum **5 points** required
4. For each ε: minimize `f₁ + δ·f₂` subject to `Σwₚ ≤ ε`, where `δ = 10⁻⁴`

```python
N_POINTS  = 10       # minimum 5 for valid Pareto coverage
DELTA_EPS = 1e-4     # AUGMECON2 augmentation coefficient

eps_grid = np.linspace(f2_max, f2_min, N_POINTS)
for eps in eps_grid:
    m.remove(m.getConstrByName("c3"))
    m.addConstr(quicksum(w[p] for p in P) <= eps, name="c3")
    m.setObjective(obj1 + DELTA_EPS * obj2, GRB.MINIMIZE)
    m.optimize()
    # collect Pareto point...
```

**Multi-objective quality metrics:** Hypervolume (HV), IGD, C-metric — normalized to [0,1] using the Pareto front reference point.

---

## Solution Methods

### H1 — Solomon I1-PDP Construction Heuristic

Route construction heuristic adapted from Solomon (1987) for the pickup-delivery setting.

- **Category:** Insertion heuristic (Bräysy & Gendreau, 2005a)
- **Parameters:** α₁ = 1.0, α₂ = 0 (push-forward disabled per advisor instruction)
- **Insertion cases:** A (both nodes new), B (delivery exists), C (pickup exists), D (both exist / zero detour)
- **Tie-breaking:** max c₂ → min c₁ → smallest part ID

### H2 — Priority Dispatch (21-Case)

State-driven sequential construction heuristic (not Solomon-based). Handles en-route consolidation (LOC-C/D/E cases). Implementation ongoing.

### NSGA-II Metaheuristic

Two-layer chromosome encoding (Bean 1994 / BRKGA paradigm):
- **L1:** Part-to-tour assignment
- **L2:** Node visit sequence within each tour (random-key encoding)

| Operator | L1 | L2 |
|----------|----|----|
| Crossover | Uniform | SBX (η_c = 15) |
| Mutation | Random Reset + Shift-Aware | Polynomial (η_m = 20) |

Four-phase repair operator (in order): Precedence → Dead Zone → Release Date → Capacity.

---

## Gurobi Parameters

```python
m.setParam('TimeLimit',     21600)   # 6 hours
m.setParam('MIPGap',        0.01)    # 1% optimality gap
m.setParam('Threads',       6)       # i7-12650H: 6P cores
m.setParam('NodefileStart', 2)       # RAM safeguard (8 GB machine)
m.setParam('Presolve',      2)
```

**Hardware:** Intel Core i7-12650H 2.30 GHz (10-core: 6P+4E), 8 GB RAM, NVIDIA RTX 3050 Ti 4 GB VRAM, 477 GB NVMe SSD, Windows 64-bit.

> RAM is the critical bottleneck for large instances. `NodefileStart=2` writes branch-and-bound nodes to disk when RAM usage exceeds 2 GB.

---

## Data Format

```
products_Xpart.xlsx — three sheets per file (Case 1 / Case 2 / Case 3):

product_id | origin | destination | ready_time | load_time | unload_time | area_m2
P1         | N2     | N5          | 435        | 2         | 3           | 1.5
```

> `ready_time` is stored as **absolute clock minutes** (e.g., 435 = 07:15).
> Do not convert to shift-relative minutes; the MILP uses absolute time throughout.

---

## Troubleshooting

### Infeasible Model
1. Check IIS report: `results/infeasible_[timestamp].ilp`
2. Verify `ready_time` values are within `[T_start, T_start + T_max]`
3. Check `Σ area_m² ≤ vehicle capacity` for each candidate route
4. Confirm M₂₂ = 465 (not 45) — see Big-M note above

### Slow Convergence
1. Switch from naive M=9999 to tight Big-M
2. Set `MIPFocus=1` (prioritise feasible solutions) or `MIPFocus=3` (bound improvement)
3. Reduce instance size for debugging: `products.head(4)`

### AUGMECON2 Produces Single Point
- **Symptom:** All ε iterations return identical f₁, f₂ values
- **Cause:** `EPS_WAIT` set too large (e.g. 9999) — ε-constraint never binds
- **Fix:** Compute `f2_max` and `f2_min` first, then set grid within `[f2_min, f2_max]`

---

## Key References

| Reference | Contribution |
|-----------|-------------|
| Miller, Tucker & Zemlin (1960) | MTZ subtour elimination formulation |
| Solomon (1987) | I1 insertion heuristic for VRPTW |
| Camm et al. (1990) | Analytical tightening of Big-M coefficients |
| Desrochers & Laporte (1991) | Improved MTZ inequalities |
| Savelsbergh & Sol (1995) | General pickup-and-delivery problem framework |
| Deb et al. (2002) | NSGA-II algorithm |
| Bean (1994) | Random-key genetic algorithms (RKGA/BRKGA) |
| Bräysy & Gendreau (2005a) | Heuristic taxonomy for VRPTW |
| Mavrotas & Florios (2013) | AUGMECON2 method |
| **Daşdemir, Öztürk & Tezcaner Öztürk (2022, OMEGA)** | **Primary methodological benchmark** |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.1.0 | Feb 2026 | Tight Big-M + Unified Big-M dual implementation |
| v2.2.0 | May 2026 | M₂₂ corrected (45 → 465); AUGMECON2 ε-loop added; C23 fixed (≥ → =); print labels corrected |

**Model complexity:** O(\|N\|² · \|K\| · \|R\| · \|P\|)

**License:** Academic use only. Requires a valid Gurobi Academic License.
