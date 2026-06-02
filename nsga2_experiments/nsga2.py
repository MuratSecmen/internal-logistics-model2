"""
nsga2.py — NSGA-2 for the Internal Logistics PD-VRP.

Chromosome Encoding
===================
Her Individual bir gene dict taşır:

    gene: dict[(k, r), list[str]]

    Anahtarlar : inst.KR_pairs'teki tüm geçerli (k, r) çiftleri.
    Değerler   : o araç-rota slotuna atanan ürünlerin SIRALI listesi.
                 Decode sırasında Solomon bu sırayı kullanır.

Gene iki şeyi aynı anda kodlar:
    1. Araç-rota ataması  — hangi ürün hangi (k, r)'ye gidiyor.
    2. Insert sırası      — Solomon'un o-d çiftlerini hangi sırada işleyeceği.

Decoder
=======
Gene verildiğinde decoder şunu yapar:

    Her (k, r) için (araç sırasıyla):
        gene[(k, r)] listesindeki her p için:
            best_insert_into_route(p, route_(k,r), ...)  çağrılır.
            Solomon sadece geometrik pozisyonu (hangi iki düğüm arasına)
            belirler — araç ve insert sırası kromozomdan geliyor.

Eğer herhangi bir ürün için c1=inf dönerse birey cezalandırılır:
    f1 = f2 = PENALTY

Objectives
==========
    f1 = fleet.total_route_duration()   (minimize)
    f2 = fleet.total_wait()             (minimize)
    Not: f2, solomon.py'de t^a_{dp} + s^u_p - e_p olarak hesaplanmalıdır
         (Document-4 Constraint 21). Bu sorumluluk solomon.py'dedir.

NSGA-2 Components
=================
    non_dominated_sort   — her bireye Pareto rank atar.
    crowding_distance    — aynı fronttaki bireylere crowding distance atar.
    tournament_select    — (rank ↑, crowding ↓) binary tournament.

    ox_crossover         — araç bazlı Order Crossover (OX).
    swap_mutation        — rota içi iki ürünün yerini değiştirir.
    inversion_mutation   — rota içi segment ters çevirme.
    transfer_mutation    — ürünü farklı (k, r)'ye taşır.

    nsga2                — ana döngü.

"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

from solomon import (
    FleetState,
    best_insert_into_route,
    apply_insertion,
)

try:
    from solomon import construct, _STOCK_BIASES
    _SOLOMON_AVAILABLE = True
except ImportError:
    construct = None
    _SOLOMON_AVAILABLE = False
    _STOCK_BIASES = [
        ("route_duration", {"alpha_1": 0.0001, "alpha_2": 1.0,   "alpha_3": 0.01  }),
        ("wait_time",      {"alpha_1": 0.0001, "alpha_2": 0.01,  "alpha_3": 1.0   }),
        ("balanced",       {"alpha_1": 0.0001, "alpha_2": 0.5,   "alpha_3": 0.5   }),
        ("distance",       {"alpha_1": 1.0,    "alpha_2": 0.0001,  "alpha_3": 0.0001  }),
        ("dist_wait",      {"alpha_1": 0.5,    "alpha_2": 0.0001,  "alpha_3": 0.5   }),
        ("dist_duration",  {"alpha_1": 0.5,    "alpha_2": 0.5,   "alpha_3": 0.0001  }),
    ]

log = logging.getLogger("internal_logistics.nsga2")

PENALTY = 1e9
EPS     = 1e-9


# ============================================================
# Yardımcı: FleetState → gene çevirici
# ============================================================

def gene_from_fleet(fleet: FleetState) -> dict:
    gene = {kr: [] for kr in fleet.inst.KR_pairs}
    for kr, route in fleet.routes.items():
        inst    = fleet.inst
        ordered = []
        seen    = set()
        for node in route.nodes[1:-1]:
            for p in route.parts:
                if p not in seen and inst.o[p] == node:
                    ordered.append(p)
                    seen.add(p)
        gene[kr] = ordered
    return gene


# ============================================================
# Individual
# ============================================================

@dataclass
class Individual:
    """Tek bir NSGA-2 çözümü."""

    gene: dict
    f1: float = PENALTY
    f2: float = PENALTY
    rank: int = 0
    crowding: float = 0.0
    fleet: object = field(default=None, repr=False)

    def dominates(self, other: "Individual") -> bool:
        return (
            self.f1 <= other.f1 + EPS and
            self.f2 <= other.f2 + EPS and
            (self.f1 < other.f1 - EPS or self.f2 < other.f2 - EPS)
        )

    def is_feasible(self) -> bool:
        return self.f1 < PENALTY - EPS


def _clone(ind: Individual) -> Individual:
    return Individual(gene={kr: list(lst) for kr, lst in ind.gene.items()})


# ============================================================
# Chromosome yardımcıları
# ============================================================

def random_gene(inst, rng: random.Random) -> dict:
    products = list(inst.P)
    rng.shuffle(products)
    kr_pairs = list(inst.KR_pairs)
    gene     = {kr: [] for kr in kr_pairs}
    for i, p in enumerate(products):
        gene[kr_pairs[i % len(kr_pairs)]].append(p)
    for kr in gene:
        rng.shuffle(gene[kr])
    return gene


def gene_is_valid(gene: dict, inst) -> bool:
    """Her ürünün tam olarak bir kez göründüğünü doğrula."""
    all_p = [p for lst in gene.values() for p in lst]
    return set(all_p) == set(inst.P) and len(all_p) == len(inst.P)


# ============================================================
# Decoder
# ============================================================

def decode(ind: Individual, inst, cfg: dict) -> Individual:
    fleet = FleetState.empty(inst)

    for k in inst.K:
        for r in inst.routes_of(k):
            route = fleet.routes[(k, r)]
            for p in ind.gene.get((k, r), []):
                baseline_dur  = fleet.total_route_duration()
                baseline_wait = fleet.total_wait()
                plan = best_insert_into_route(
                    p, route, fleet, cfg, baseline_dur, baseline_wait
                )
                if not math.isfinite(plan.c1):
                    ind.f1    = PENALTY
                    ind.f2    = PENALTY
                    ind.fleet = None
                    return ind
                apply_insertion(p, plan, fleet)

    ind.fleet = fleet
    ind.f1    = fleet.total_route_duration()
    ind.f2    = fleet.total_wait()
    return ind


# ============================================================
# Non-dominated Sort  (Deb et al., 2002 — Algorithm 1)
# ============================================================

def non_dominated_sort(pop: list) -> list:
    n           = len(pop)
    S           = [[] for _ in range(n)]
    n_dominated = [0]  * n
    fronts      = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pop[i].dominates(pop[j]):
                S[i].append(j)
            elif pop[j].dominates(pop[i]):
                n_dominated[i] += 1
        if n_dominated[i] == 0:
            pop[i].rank = 1
            fronts[0].append(i)

    cur = 0
    while fronts[cur]:
        nxt = []
        for i in fronts[cur]:
            for j in S[i]:
                n_dominated[j] -= 1
                if n_dominated[j] == 0:
                    pop[j].rank = cur + 2
                    nxt.append(j)
        cur += 1
        fronts.append(nxt)

    return [f for f in fronts if f]


# ============================================================
# Crowding Distance
# ============================================================

def crowding_distance(pop: list, front: list) -> None:
    for i in front:
        pop[i].crowding = 0.0

    for key in ("f1", "f2"):
        srt   = sorted(front, key=lambda i: getattr(pop[i], key))
        pop[srt[0]].crowding  = math.inf
        pop[srt[-1]].crowding = math.inf
        f_min = getattr(pop[srt[0]],  key)
        f_max = getattr(pop[srt[-1]], key)
        span  = (f_max - f_min) if abs(f_max - f_min) > EPS else 1.0
        for k in range(1, len(srt) - 1):
            prev = getattr(pop[srt[k - 1]], key)
            nxt  = getattr(pop[srt[k + 1]], key)
            pop[srt[k]].crowding += (nxt - prev) / span


# ============================================================
# Selection — Binary Tournament
# ============================================================

def tournament_select(pop: list, rng: random.Random) -> Individual:
    i, j = rng.sample(range(len(pop)), 2)
    a, b = pop[i], pop[j]
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    return a if a.crowding >= b.crowding else b


# ============================================================
# Crossover — Order Crossover (OX), araç bazlı
# ============================================================

def _ox_lists(lst_a: list, lst_b: list, rng: random.Random) -> list:
    n = len(lst_a)
    if n <= 1:
        return list(lst_a)
    lo, hi  = sorted(rng.sample(range(n), 2))
    child   = [None] * n
    child[lo: hi + 1] = lst_a[lo: hi + 1]
    seg_set = set(child[lo: hi + 1])
    fill    = [x for x in lst_b if x not in seg_set]
    j = 0
    for i in list(range(hi + 1, n)) + list(range(0, lo)):
        child[i] = fill[j]; j += 1
    return child


def ox_crossover(p1: Individual, p2: Individual,
                 inst, rng: random.Random) -> tuple:
    kr_pairs = list(inst.KR_pairs)
    n_kr     = len(kr_pairs)
    s        = rng.randint(0, n_kr - 2) if n_kr > 1 else 0

    def _build_child(donor_a: Individual, donor_b: Individual) -> Individual:
        gene_c     = {kr: [] for kr in kr_pairs}
        assigned   = set()
        slot_donor = {}

        for idx, kr in enumerate(kr_pairs):
            primary = donor_a if idx <= s else donor_b
            alt     = donor_b if idx <= s else donor_a
            slot_donor[kr] = (primary, alt)
            lst = [p for p in primary.gene.get(kr, []) if p not in assigned]
            assigned.update(lst)
            gene_c[kr] = lst

        for kr in kr_pairs:
            lst_a = gene_c[kr]
            _, alt_donor = slot_donor[kr]
            lst_b = [p for p in alt_donor.gene.get(kr, [])
                     if p in set(lst_a)]
            if len(lst_a) >= 2 and len(lst_b) == len(lst_a):
                gene_c[kr] = _ox_lists(lst_a, lst_b, rng)

        missing = [p for p in inst.P if p not in assigned]
        for p in missing:
            lightest = min(kr_pairs, key=lambda kr: len(gene_c[kr]))
            gene_c[lightest].append(p)
            assigned.add(p)

        return Individual(gene=gene_c)

    c1 = _build_child(p1, p2)
    c2 = _build_child(p2, p1)
    assert gene_is_valid(c1.gene, inst), "ox_crossover c1 invalid"
    assert gene_is_valid(c2.gene, inst), "ox_crossover c2 invalid"
    return c1, c2


# ============================================================
# Mutasyon operatörleri
# ============================================================

def swap_mutation(ind: Individual, rng: random.Random,
                  p: float = 0.15) -> Individual:
    if rng.random() > p:
        return ind
    ind      = _clone(ind)
    eligible = [kr for kr, lst in ind.gene.items() if len(lst) >= 2]
    if not eligible:
        return ind
    kr       = rng.choice(eligible)
    lst      = ind.gene[kr]
    i, j     = rng.sample(range(len(lst)), 2)
    lst[i], lst[j] = lst[j], lst[i]
    return ind


def inversion_mutation(ind: Individual, rng: random.Random,
                       p: float = 0.10) -> Individual:
    if rng.random() > p:
        return ind
    ind      = _clone(ind)
    eligible = [kr for kr, lst in ind.gene.items() if len(lst) >= 2]
    if not eligible:
        return ind
    kr       = rng.choice(eligible)
    lst      = ind.gene[kr]
    lo, hi   = sorted(rng.sample(range(len(lst)), 2))
    lst[lo: hi + 1] = lst[lo: hi + 1][::-1]
    return ind


def transfer_mutation(ind: Individual, inst, rng: random.Random,
                      p: float = 0.10) -> Individual:
    if rng.random() > p:
        return ind
    ind       = _clone(ind)
    non_empty = [kr for kr, lst in ind.gene.items() if lst]
    if not non_empty:
        return ind
    src_kr  = rng.choice(non_empty)
    tgt_kr  = rng.choice(list(inst.KR_pairs))
    if tgt_kr == src_kr:
        return ind
    src_lst = ind.gene[src_kr]
    p_idx   = rng.randint(0, len(src_lst) - 1)
    product = src_lst.pop(p_idx)
    tgt_lst = ind.gene[tgt_kr]
    ins_pos = rng.randint(0, len(tgt_lst))
    tgt_lst.insert(ins_pos, product)
    return ind


# ============================================================
# İlk popülasyon
# ============================================================

def _solomon_seeds(inst, cfg, n: int) -> list:
    if not _SOLOMON_AVAILABLE or construct is None:
        log.warning(
            "solomon.construct veya _STOCK_BIASES bulunamadı; "
            "Solomon tohumları atlanıyor, rastgele bireyler kullanılacak."
        )
        return []

    seeds = []
    for name, alphas in _STOCK_BIASES[:n]:
        trial_cfg = dict(cfg)
        trial_cfg.update(alphas)
        try:
            fleet, status, _ = construct(inst, trial_cfg)
            if status == "feasible":
                gene = gene_from_fleet(fleet)
                ind  = Individual(gene=gene)
                ind.fleet = fleet
                ind.f1    = fleet.total_route_duration()
                ind.f2    = fleet.total_wait()
                seeds.append(ind)
                log.info("Solomon seed [%s]: f1=%.2f  f2=%.2f",
                         name, ind.f1, ind.f2)
            else:
                log.warning("Solomon seed [%s]: infeasible, atlandı.", name)
        except Exception as exc:
            log.warning("Solomon seed [%s] başarısız: %s", name, exc)

    return seeds


def initial_population(inst, cfg, pop_size: int,
                        rng: random.Random,
                        n_solomon_seeds: int = 3) -> list:
    population = []

    for ind in _solomon_seeds(inst, cfg, n_solomon_seeds):
        population.append(ind)

    while len(population) < pop_size:
        gene = random_gene(inst, rng)
        ind  = Individual(gene=gene)
        decode(ind, inst, cfg)
        population.append(ind)

    log.info("Başlangıç pop: %d birey, %d feasible",
             pop_size, sum(1 for i in population if i.is_feasible()))
    return population


# ============================================================
# NSGA-2 Ana Döngü
# ============================================================

def nsga2(inst, cfg: dict) -> tuple:
    """
    NSGA-2 çalıştır. (pareto_front, history) döner.

    history formatı: [(nesil, n_feasible, best_f1, best_f2, |front_0|), ...]

    Yapılandırma (cfg["nsga2"] altında):
    ─────────────────────────────────────
    pop_size          int   varsayılan 100
    n_generations     int   varsayılan 200
    p_crossover       float varsayılan 0.90
    p_swap            float varsayılan 0.15
    p_inv             float varsayılan 0.10
    p_transfer        float varsayılan 0.10
    n_solomon_seeds   int   varsayılan 3
    seed              int   varsayılan 42
    """
    nc         = cfg.get("nsga2", {})
    pop_size   = int(nc.get("pop_size",        100))
    n_gen      = int(nc.get("n_generations",   200))
    p_cross    = float(nc.get("p_crossover",  0.90))
    p_swap     = float(nc.get("p_swap",       0.15))
    p_inv      = float(nc.get("p_inv",        0.10))
    p_transfer = float(nc.get("p_transfer",   0.10))
    n_seeds    = int(nc.get("n_solomon_seeds",   3))
    seed       = nc.get("seed", 42)

    rng = random.Random(seed)
    log.info("NSGA-2 BAŞLIYOR | pop=%d  gen=%d  seed=%d",
             pop_size, n_gen, seed)

    # ── 1. Başlangıç popülasyonu ──────────────────────────────
    population = initial_population(inst, cfg, pop_size, rng, n_seeds)
    fronts     = non_dominated_sort(population)
    for front in fronts:
        crowding_distance(population, front)

    history = []

    # ── 2. Ana döngü ─────────────────────────────────────────
    for gen in range(1, n_gen + 1):

        # ── 2a. Offspring üretimi ─────────────────────────────
        offspring = []
        while len(offspring) < pop_size:

            p1 = tournament_select(population, rng)
            p2 = tournament_select(population, rng)

            if rng.random() < p_cross:
                c1, c2 = ox_crossover(p1, p2, inst, rng)
            else:
                c1, c2 = _clone(p1), _clone(p2)

            for child in (c1, c2):
                child = swap_mutation(child,      rng, p_swap)
                child = inversion_mutation(child,  rng, p_inv)
                child = transfer_mutation(child, inst, rng, p_transfer)
                decode(child, inst, cfg)

                if len(offspring) < pop_size:
                    offspring.append(child)

        # ── 2b. Birleşik havuz (2N) ───────────────────────────
        combined = population + offspring

        # ── 2c. Non-dominated sort + crowding ─────────────────
        fronts = non_dominated_sort(combined)
        for front in fronts:
            crowding_distance(combined, front)

        # ── 2d. Sonraki nesil seçimi (elitism) ────────────────
        next_pop = []
        for front in fronts:
            if len(next_pop) + len(front) <= pop_size:
                next_pop.extend(combined[i] for i in front)
            else:
                remaining      = pop_size - len(next_pop)
                best_of_front  = sorted(
                    front,
                    key=lambda i: combined[i].crowding,
                    reverse=True,
                )[:remaining]
                next_pop.extend(combined[i] for i in best_of_front)
                break

        population = next_pop

        # ── 2e. Kayıt ─────────────────────────────────────────
        feasible   = [ind for ind in population if ind.is_feasible()]
        n_feas     = len(feasible)
        best_f1    = min((ind.f1 for ind in feasible), default=PENALTY)
        best_f2    = min((ind.f2 for ind in feasible), default=PENALTY)
        pareto_now = [ind for ind in population if ind.rank == 1
                      and ind.is_feasible()]
        history.append((gen, n_feas, best_f1, best_f2, len(pareto_now)))

        if gen % 10 == 0 or gen == 1:
            log.info(
                "Nesil %3d | uygun=%d/%d | f1=%.2f | f2=%.2f | |PF|=%d",
                gen, n_feas, pop_size, best_f1, best_f2, len(pareto_now)
            )

    # ── 3. Final Pareto front ─────────────────────────────────
    pareto = [ind for ind in population
              if ind.rank == 1 and ind.is_feasible()]
    log.info("NSGA-2 BİTTİ | |Pareto|=%d", len(pareto))
    return pareto, history
