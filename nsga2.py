"""
nsga2.py — NSGA-II meta-heuristic for BO-MR-PDPTW-STW
=======================================================

Giant Tour 2-layer chromosome (per Deb et al., 2002; Vidal et al., 2013):

    Layer 1 — perm    : permutation of 0…|P|-1
                        Encodes the *global service order* of products.
                        Products that appear earlier in perm are served earlier
                        within their assigned vehicle.

    Layer 2 — vehicle : integer in 0…|K|-1 for each product index
                        Maps each product to a specific vehicle.

Decoder pipeline (decode()):
    1. Group products by assigned vehicle (in perm order).
    2. Greedy capacity split → per-vehicle route list (respects max_routes[k]).
    3. All-pickups-first route simulation → timing + wait times.
    4. Compute f1 = route_duration, f2 = total_wait.

NSGA-II loop (nsga2()):
    init_population()
        → evaluate() → fast_non_dominated_sort() → crowding_distance()
        → [selection → crossover → mutation → evaluate → merge → select] × n_gen

Feasibility handling:
    Constraint violations (capacity overflow, time-window breach, max-routes
    exceeded) are tracked as a scalar ``violation``. Feasibility-first
    tournament selection (Deb et al., 2002, §III-B) is used: feasible
    individuals always dominate infeasible ones; among infeasible individuals,
    lower violation wins.

Solomon seed integration (optional):
    Pass ``solomon_seed`` (an Instance-compatible route dict) to
    ``init_population()``; the first individual in the initial population is
    built from the Solomon solution. This warm-starts the search with a
    high-quality feasible solution.

Integration with run_model.py:
    Accepts ``Instance`` from run_model.py unchanged.
    Outputs ``pareto_nsga2_<timestamp>.xlsx`` with sheets mirroring
    run_model.py's write_results() layout so verify.py can consume it.

Usage (standalone)::

    python nsga2.py
    python nsga2.py --inputs inputs --config inputs/config.xlsx
    python nsga2.py --output-dir results

Usage (from run_model.py) — set run_mode = nsga2 in config.xlsx::

    from nsga2 import nsga2, export_pareto
    pareto = nsga2(inst, cfg)
    export_pareto(pareto, inst, output_path)

References
----------
Deb K., Pratap A., Agarwal S., Meyarivan T. (2002).
    A fast and elitist multiobjective genetic algorithm: NSGA-II.
    IEEE Trans. Evol. Comput., 6(2), 182-197.

Vidal T., Crainic T. G., Gendreau M., Prins C. (2013).
    A hybrid genetic algorithm with adaptive diversity management for a
    large class of vehicle routing problems with time windows.
    Comput. Oper. Res., 40(1), 475-489.

Goldberg D. E., Lingle R. (1985).
    Alleles, loci, and the traveling salesman problem.
    Proc. 1st ICGA, 154-159.  [Order Crossover — OX]
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-II hyper-parameter defaults (overridable via config.xlsx)
# ─────────────────────────────────────────────────────────────────────────────

NSGA2_DEFAULTS: dict = {
    # Population and generation budget
    "nsga2_pop_size":          100,     # N — population size (must be even)
    "nsga2_n_gen":             300,     # number of generations
    # Crossover / mutation rates
    "nsga2_p_crossover":       0.9,     # probability of applying OX crossover
    "nsga2_p_swap_mut":        0.02,    # per-gene probability of perm swap
    "nsga2_p_vehicle_mut":     0.05,    # per-gene probability of vehicle reassignment
    # Feasibility penalty weight (multiplied by violation magnitude in
    # the crowded-comparison operator; not added to objectives directly)
    "nsga2_violation_weight":  1.0,
    # Random seed (None → non-deterministic)
    "nsga2_seed":              42,
    # Fraction of initial population seeded from Solomon heuristic
    # (remainder is random).  0.0 → fully random; 1.0 → all from Solomon.
    "nsga2_solomon_fraction":  0.1,
    # Progress print interval (every N generations; 0 → silent)
    "nsga2_log_interval":      10,
    # Warm-start: if True and a heuristic solution is available in the
    # same run, it is injected as the first individual.
    "nsga2_warmstart":         True,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Individual / chromosome
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    """One solution (chromosome) in the NSGA-II population.

    The chromosome consists of two coupled layers of equal length |P|:

    perm[i]    : the i-th product (by global service priority) is inst.P[perm[i]].
    vehicle[i] : perm[i]-th product is transported by vehicle inst.K[vehicle[i]].

    Decoding always respects the pairing: product at position i in perm uses
    the vehicle at position i in vehicle.  Do NOT permute the two layers
    independently.
    """
    perm:    list   # list[int], length |P| — permutation of 0…|P|-1
    vehicle: list   # list[int], length |P| — vehicle index 0…|K|-1

    # ── Objective and NSGA-II bookkeeping ────────────────────────────────────
    f1: float = float("inf")     # route_duration (min)
    f2: float = float("inf")     # total_wait (min)
    rank: int = 0                # non-domination rank (1 = Pareto front)
    crowding_dist: float = 0.0   # crowding distance in objective space
    feasible: bool = False
    violation: float = 0.0       # weighted sum of constraint violations

    # ── Dominance ────────────────────────────────────────────────────────────
    def dominates(self, other: "Individual") -> bool:
        """Feasibility-first Pareto dominance (Deb et al., 2002, §III-B).

        Decision hierarchy:
            1. Feasible always dominates infeasible.
            2. Among infeasible: lower violation dominates.
            3. Among feasible: standard (f1, f2)-Pareto dominance.
        """
        if self.feasible and not other.feasible:
            return True
        if not self.feasible and other.feasible:
            return False
        if not self.feasible:                       # both infeasible
            return self.violation < other.violation
        # Both feasible — standard Pareto
        return (
            self.f1 <= other.f1 and self.f2 <= other.f2
            and (self.f1 < other.f1 or self.f2 < other.f2)
        )

    def clone(self) -> "Individual":
        return Individual(
            perm=self.perm[:],
            vehicle=self.vehicle[:],
            f1=self.f1, f2=self.f2,
            rank=self.rank,
            crowding_dist=self.crowding_dist,
            feasible=self.feasible,
            violation=self.violation,
        )

    def __repr__(self) -> str:
        feas = "✓" if self.feasible else f"✗(v={self.violation:.1f})"
        return (
            f"Ind(rank={self.rank}, "
            f"f1={self.f1:.2f}, f2={self.f2:.2f}, "
            f"{feas}, cd={self.crowding_dist:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Route simulation (decoder primitive)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_route(
    products: list,
    inst,
    start_time: float = 0.0,
) -> tuple:
    """Forward timing simulation for a single vehicle route.

    Visit sequence — all-pickups-first, deduplicated
    ------------------------------------------------
    Phase 1 (pickups):  unique origin nodes in ``products`` order.
    Phase 2 (deliveries): unique destination nodes in ``products`` order,
        SKIPPING nodes already visited in Phase 1 (they are handled there).

    At each visited node *j* (mimicking MIP constraints 17–19):
        ta_j = t after travel from previous node
        t += Σ s_unload[p]  for p delivered at j       (constraint 17)
        t  = max(t, max_p e_p)  for p picked up at j   (constraint 18)
        t += Σ s_load[p]    for p picked up at j        (constraint 19)

    Wait time for product p (constraint 22):
        w_p = ta[d_p] + s_unload_p − e_p    (≥ 0 by feasibility)

    Parameters
    ----------
    products   : list of product ids (elements of inst.P), in perm order.
    inst       : Instance object from run_model.py.
    start_time : shift-relative minute when this route departs the depot.

    Returns
    -------
    completion_time : float
        Absolute shift-minute when the vehicle returns to the depot.
    total_wait : float
        Σ w_p for all products on this route.
    feasible : bool
        False when completion_time > inst.T_max.
    """
    if not products:
        return start_time, 0.0, True

    # Index events per node
    pickup_at:   dict = defaultdict(list)   # node → [products loaded here]
    delivery_at: dict = defaultdict(list)   # node → [products unloaded here]
    for p in products:
        pickup_at[inst.o[p]].append(p)
        delivery_at[inst.d[p]].append(p)

    # Build visit sequence (all-pickups-first, no node repetition)
    seen: set = set()
    visit_seq: list = []
    for p in products:                      # Phase 1: pickups in perm order
        node = inst.o[p]
        if node not in seen:
            visit_seq.append(node)
            seen.add(node)
    for p in products:                      # Phase 2: deliveries in perm order
        node = inst.d[p]
        if node not in seen:
            visit_seq.append(node)
            seen.add(node)

    # Forward simulation
    t: float = float(start_time)
    prev: str = "h"
    ta_at_delivery: dict = {}               # p → arrival time at delivery node
    feasible: bool = True

    for node in visit_seq:
        t += inst.c.get((prev, node), 0.0)
        ta = t                              # arrival time before service

        # Unload products delivered at this node
        for p in delivery_at.get(node, []):
            t += inst.s_unload[p]
            ta_at_delivery[p] = ta          # ta[d_p, k, r] in MIP notation

        # Wait for ready-times; then load products picked up at this node
        prods_here = pickup_at.get(node, [])
        if prods_here:
            max_ready = max(inst.e[p] for p in prods_here)
            if t < max_ready:
                t = max_ready               # constraint 18: ts[j,k,r] >= e_p
            for p in prods_here:
                t += inst.s_load[p]

        prev = node

    # Return to depot
    t += inst.c.get((prev, "h"), 0.0)
    completion = t

    if completion > inst.T_max + 1e-6:
        feasible = False

    # Compute wait times: w_p = ta[d_p] + s_unload_p − e_p
    total_wait: float = 0.0
    for p in products:
        if p in ta_at_delivery:
            w_p = ta_at_delivery[p] + inst.s_unload[p] - inst.e[p]
            total_wait += max(0.0, w_p)
        else:
            # Delivery node not reached — shouldn't happen with correct routing,
            # but guard against edge cases in random chromosomes.
            total_wait += inst.T_max       # heavy penalty signal
            feasible = False

    return completion, total_wait, feasible


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Decoder: chromosome → (f1, f2, feasible, violation)
# ─────────────────────────────────────────────────────────────────────────────

def decode(ind: Individual, inst) -> Individual:
    """Decode an Individual into objective values and feasibility status.

    Modifies ind in-place and returns it for chaining.

    Decoding steps
    --------------
    1. Reconstruct the (product, vehicle) pairing from perm and vehicle layers.
    2. For each vehicle k:
        a. Collect products assigned to k in perm order.
        b. Greedy capacity split → list of routes (each route = list of products).
           New route is opened when next product would overflow q_k[k].
        c. Routes are sequential within a vehicle (constraint 14):
           route r+1 departs depot no earlier than route r's depot arrival.
        d. Simulate each route with simulate_route() to get timing + wait.
    3. f1 = Σ_k last_depot_arrival[k]
       f2 = Σ_p w_p
    4. Constraint violations:
        - capacity_overflow  (sum of overflows in m²)
        - max_routes_exceeded (extra routes beyond max_routes[k])
        - time_overrun        (minutes beyond T_max)
    """
    P = inst.P
    K = inst.K
    n_p = len(P)
    n_k = len(K)

    # Reconstruct (product_id, vehicle_id) from chromosome
    vehicle_products: dict = {ki: [] for ki in range(n_k)}
    for pos, prod_idx in enumerate(ind.perm):
        veh_idx = ind.vehicle[pos]
        vehicle_products[veh_idx].append(P[prod_idx])

    f1: float = 0.0
    f2: float = 0.0
    violation: float = 0.0

    for veh_idx, prods in vehicle_products.items():
        if not prods:
            continue
        k = K[veh_idx]
        cap_k = inst.q_k[k]
        max_r = inst.max_routes[k]

        # ── Greedy capacity split into routes ────────────────────────────────
        # Each route is a list of product ids.  A new route is opened when
        # the next product cannot fit in the current route's capacity.
        routes: list = []
        cur_route: list = []
        cur_load: float = 0.0

        for p in prods:
            q_p = inst.q_p[p]
            if cur_load + q_p <= cap_k + 1e-9:
                cur_route.append(p)
                cur_load += q_p
            else:
                if cur_route:
                    routes.append(cur_route)
                # Start new route; if the single product overflows capacity,
                # record violation but still assign it so decoding continues.
                if q_p > cap_k + 1e-9:
                    violation += (q_p - cap_k)
                cur_route = [p]
                cur_load = q_p

        if cur_route:
            routes.append(cur_route)

        # ── max_routes[k] check ──────────────────────────────────────────────
        if len(routes) > max_r:
            violation += (len(routes) - max_r) * 100.0   # heavy penalty unit
            # Truncate: drop surplus routes (their products become unrouted).
            # TODO: a repair operator that redistributes unrouted products to
            # other vehicles would be more sophisticated.
            routes = routes[:max_r]

        # ── Sequential route simulation ──────────────────────────────────────
        # Routes are chained: route r+1 starts at the depot arrival of route r.
        t_vehicle: float = 0.0     # departure time of the current route

        for route_prods in routes:
            completion, wait_sum, ok = simulate_route(
                route_prods, inst, start_time=t_vehicle
            )
            f2 += wait_sum
            t_vehicle = completion          # next route departs at this time
            if not ok:
                violation += max(0.0, completion - inst.T_max)

        f1 += t_vehicle                     # last-depot-arrival for this vehicle

    ind.f1 = f1
    ind.f2 = f2
    ind.violation = violation
    ind.feasible = (violation < 1e-6)
    return ind


# ─────────────────────────────────────────────────────────────────────────────
# 4.  NSGA-II core: fast non-dominated sort + crowding distance
# ─────────────────────────────────────────────────────────────────────────────

def fast_non_dominated_sort(pop: list) -> list:
    """Fast non-dominated sort (Deb et al., 2002, Algorithm 1).

    Parameters
    ----------
    pop : list of Individual (already evaluated — f1, f2, feasible, violation set).

    Returns
    -------
    fronts : list of lists
        fronts[0] = Pareto-optimal individuals (rank 1),
        fronts[1] = rank-2 individuals, etc.
    """
    n = len(pop)
    dominated_by: list = [0] * n           # |S_p| counter
    dominates_set: list = [[] for _ in range(n)]   # S_p
    fronts: list = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if pop[i].dominates(pop[j]):
                dominates_set[i].append(j)
                dominated_by[j] += 1
            elif pop[j].dominates(pop[i]):
                dominates_set[j].append(i)
                dominated_by[i] += 1

    for i in range(n):
        if dominated_by[i] == 0:
            pop[i].rank = 1
            fronts[0].append(pop[i])

    k = 0
    while fronts[k]:
        next_front: list = []
        for ind in fronts[k]:
            idx = pop.index(ind)
            for j in dominates_set[idx]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    pop[j].rank = k + 2
                    next_front.append(pop[j])
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]   # drop empty trailing front


def crowding_distance_assignment(front: list) -> None:
    """Assign crowding distances to individuals in a single front (in-place).

    Crowding distance is the perimeter of the cuboid in objective space formed
    by the nearest neighbours on each objective axis.  Boundary individuals
    receive distance = ∞.
    """
    n = len(front)
    if n == 0:
        return
    for ind in front:
        ind.crowding_dist = 0.0
    if n <= 2:
        for ind in front:
            ind.crowding_dist = float("inf")
        return

    for obj_fn in (lambda ind: ind.f1, lambda ind: ind.f2):
        sorted_front = sorted(front, key=obj_fn)
        sorted_front[0].crowding_dist = float("inf")
        sorted_front[-1].crowding_dist = float("inf")
        f_min = obj_fn(sorted_front[0])
        f_max = obj_fn(sorted_front[-1])
        if f_max - f_min < 1e-12:
            continue
        for i in range(1, n - 1):
            sorted_front[i].crowding_dist += (
                (obj_fn(sorted_front[i + 1]) - obj_fn(sorted_front[i - 1]))
                / (f_max - f_min)
            )


def crowded_comparison(a: Individual, b: Individual) -> int:
    """Crowded comparison operator (Deb et al., 2002).

    Returns -1 if a ≺_n b (a is preferred), +1 if b ≺_n a, 0 if tied.
    Preference: lower rank; ties broken by higher crowding distance.
    """
    if a.rank < b.rank:
        return -1
    if b.rank < a.rank:
        return 1
    if a.crowding_dist > b.crowding_dist:
        return -1
    if b.crowding_dist > a.crowding_dist:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Genetic operators
# ─────────────────────────────────────────────────────────────────────────────

def tournament_select(pop: list, k: int = 2) -> Individual:
    """Binary (k=2) tournament selection using the crowded-comparison operator."""
    candidates = random.sample(pop, k)
    best = candidates[0]
    for c in candidates[1:]:
        if crowded_comparison(c, best) < 0:
            best = c
    return best


def ox_crossover(p1: Individual, p2: Individual) -> tuple:
    """Order Crossover (OX) on the perm layer (Goldberg & Lingle, 1985).

    Two cut points are chosen at random; the segment between them is copied
    from parent 1 to child 1 (and vice versa); the remaining positions are
    filled in the order they appear in the other parent, skipping already-
    present values.  The vehicle layer is crossed uniformly.

    Returns
    -------
    (child1, child2) : two new Individual objects.
    """
    n = len(p1.perm)
    # ── Perm layer: OX ───────────────────────────────────────────────────────
    a, b = sorted(random.sample(range(n), 2))

    def _ox(par1_perm: list, par2_perm: list) -> list:
        child = [None] * n
        # Copy segment from par1
        child[a:b + 1] = par1_perm[a:b + 1]
        segment_set = set(par1_perm[a:b + 1])
        # Fill remaining positions in par2 order
        fill = [x for x in par2_perm if x not in segment_set]
        fill_iter = iter(fill)
        for i in list(range(0, a)) + list(range(b + 1, n)):
            child[i] = next(fill_iter)
        return child

    c1_perm = _ox(p1.perm, p2.perm)
    c2_perm = _ox(p2.perm, p1.perm)

    # ── Vehicle layer: uniform crossover ─────────────────────────────────────
    # Each gene is inherited from either parent with equal probability.
    # The vehicle choice is position-aligned (gene i corresponds to perm[i]).
    c1_vehicle = [p1.vehicle[i] if random.random() < 0.5 else p2.vehicle[i]
                  for i in range(n)]
    c2_vehicle = [p2.vehicle[i] if random.random() < 0.5 else p1.vehicle[i]
                  for i in range(n)]

    c1 = Individual(perm=c1_perm, vehicle=c1_vehicle)
    c2 = Individual(perm=c2_perm, vehicle=c2_vehicle)
    return c1, c2


def swap_mutate_perm(ind: Individual, p_swap: float) -> Individual:
    """Per-gene swap mutation on the perm layer.

    Each position i is selected with probability p_swap; if selected, its
    value is swapped with a randomly chosen other position j.  Applied in-
    place; returns ind for chaining.
    """
    n = len(ind.perm)
    for i in range(n):
        if random.random() < p_swap:
            j = random.randrange(n)
            ind.perm[i], ind.perm[j] = ind.perm[j], ind.perm[i]
    return ind


def reassign_mutate_vehicle(ind: Individual, p_vehicle: float,
                             n_vehicles: int) -> Individual:
    """Per-gene random reassignment mutation on the vehicle layer.

    Each position i is selected with probability p_vehicle; if selected, its
    vehicle index is replaced by a uniformly random vehicle index in
    0…n_vehicles-1.  Applied in-place; returns ind for chaining.
    """
    n = len(ind.vehicle)
    for i in range(n):
        if random.random() < p_vehicle:
            ind.vehicle[i] = random.randrange(n_vehicles)
    return ind


def mutate(ind: Individual, inst, p_swap: float, p_vehicle: float) -> Individual:
    """Combined mutation: swap on perm + reassignment on vehicle layer."""
    swap_mutate_perm(ind, p_swap)
    reassign_mutate_vehicle(ind, p_vehicle, n_vehicles=len(inst.K))
    return ind


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Population initialisation
# ─────────────────────────────────────────────────────────────────────────────

def random_individual(n_products: int, n_vehicles: int) -> Individual:
    """Generate one Individual with a uniformly random chromosome."""
    perm = list(range(n_products))
    random.shuffle(perm)
    vehicle = [random.randrange(n_vehicles) for _ in range(n_products)]
    return Individual(perm=perm, vehicle=vehicle)


def individual_from_solomon(solomon_sol, inst) -> Optional[Individual]:
    """Convert a Solomon heuristic solution into a chromosome.

    Parameters
    ----------
    solomon_sol : object with attribute ``f_assigned`` (dict p → (k, r)).
                  Produced by solomon.fleet_to_solution() in solomon.py.
    inst        : Instance object.

    Returns
    -------
    Individual or None if conversion fails.

    Notes
    -----
    The Solomon solution defines an explicit (vehicle, route) assignment per
    product and an implicit service order (via route_order).  We reconstruct:
        perm    : products sorted by (vehicle_idx, route_idx, position_in_route)
        vehicle : vehicle index per product

    TODO: integrate with solomon.py's Solution.route_order() to extract
          the within-route product ordering precisely.
    """
    if solomon_sol is None:
        return None
    try:
        P = inst.P
        K = inst.K
        n_p = len(P)
        n_k = len(K)
        k_idx = {k: i for i, k in enumerate(K)}
        p_idx = {p: i for i, p in enumerate(P)}

        f_assigned = solomon_sol.f_assigned   # p → (k, r)

        # Build (product_index, vehicle_idx, route) triples and sort
        triples = []
        for p, (k, r) in f_assigned.items():
            vi = k_idx.get(k, 0)
            triples.append((vi, r, p_idx[p]))

        triples.sort()  # (vehicle, route, product_index) → stable service order
        perm    = [t[2] for t in triples]
        vehicle = [t[0] for t in triples]

        # Products not in f_assigned (shouldn't happen in a feasible sol):
        assigned_p = {t[2] for t in triples}
        missing = [i for i in range(n_p) if i not in assigned_p]
        for i in missing:
            perm.append(i)
            vehicle.append(random.randrange(n_k))

        return Individual(perm=perm, vehicle=vehicle)
    except Exception as exc:
        print(f"[nsga2] Solomon-to-chromosome conversion failed: {exc}")
        return None


def init_population(
    inst,
    cfg: dict,
    pop_size: int,
    solomon_seed=None,
) -> list:
    """Initialise a population of ``pop_size`` individuals.

    Strategy:
        1. If a Solomon seed solution is provided and
           cfg['nsga2_solomon_fraction'] > 0, convert it to a chromosome
           and inject it (just once, as individual 0).
        2. Fill the remainder with random chromosomes.

    Parameters
    ----------
    inst         : Instance from run_model.py.
    cfg          : config dict (merged with NSGA2_DEFAULTS).
    pop_size     : target population size.
    solomon_seed : optional Solution object from solomon.py (or None).

    Returns
    -------
    List of unevaluated Individual objects (f1, f2 still at inf).
    """
    n_p = len(inst.P)
    n_k = len(inst.K)
    pop: list = []

    # Inject Solomon warm-start individual
    if solomon_seed is not None and cfg.get("nsga2_solomon_fraction", 0.1) > 0:
        seed_ind = individual_from_solomon(solomon_seed, inst)
        if seed_ind is not None:
            pop.append(seed_ind)
            print(f"[nsga2] Solomon warm-start individual injected.")

    # Fill remainder with random individuals
    while len(pop) < pop_size:
        pop.append(random_individual(n_p, n_k))

    return pop


# ─────────────────────────────────────────────────────────────────────────────
# 7.  NSGA-II main loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_population(pop: list, inst) -> None:
    """Decode and evaluate all individuals in pop (in-place)."""
    for ind in pop:
        decode(ind, inst)


def select_next_generation(combined: list, pop_size: int) -> list:
    """Select next generation from combined parent+offspring pool.

    Algorithm:
        1. Fast non-dominated sort → fronts F1, F2, ...
        2. Fill next generation with complete fronts until it would overflow.
        3. For the critical (overflow) front, compute crowding distances and
           select the least-crowded individuals to fill exactly pop_size slots.
    """
    fronts = fast_non_dominated_sort(combined)
    for front in fronts:
        crowding_distance_assignment(front)

    next_gen: list = []
    for front in fronts:
        if len(next_gen) + len(front) <= pop_size:
            next_gen.extend(front)
        else:
            # Partial front: sort by descending crowding distance
            remaining = pop_size - len(next_gen)
            sorted_front = sorted(
                front,
                key=lambda ind: ind.crowding_dist,
                reverse=True,
            )
            next_gen.extend(sorted_front[:remaining])
            break

    return next_gen


def nsga2(inst, cfg: dict, solomon_seed=None) -> list:
    """Run NSGA-II and return the final Pareto front (list of Individuals).

    Parameters
    ----------
    inst         : Instance from run_model.py.
    cfg          : config dict (run_model.py's load_config() output, merged
                   with NSGA2_DEFAULTS for any missing nsga2_* keys).
    solomon_seed : optional Solution object from solomon.py for warm-starting.

    Returns
    -------
    pareto : list of Individual
        Non-dominated individuals from the final population (rank-1 front).
    """
    # ── Merge NSGA-II defaults into cfg ──────────────────────────────────────
    for k, v in NSGA2_DEFAULTS.items():
        cfg.setdefault(k, v)

    pop_size   = int(cfg["nsga2_pop_size"])
    n_gen      = int(cfg["nsga2_n_gen"])
    p_cross    = float(cfg["nsga2_p_crossover"])
    p_swap     = float(cfg["nsga2_p_swap_mut"])
    p_vehicle  = float(cfg["nsga2_p_vehicle_mut"])
    log_every  = int(cfg["nsga2_log_interval"])
    seed       = cfg.get("nsga2_seed")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print(f"[nsga2] |P|={len(inst.P)}, |K|={len(inst.K)}, "
          f"pop={pop_size}, gen={n_gen}")
    print(f"[nsga2] p_cross={p_cross}, p_swap={p_swap}, p_vehicle={p_vehicle}")

    t_start = time.perf_counter()

    # ── Generation 0: initialise and evaluate ────────────────────────────────
    pop = init_population(inst, cfg, pop_size, solomon_seed=solomon_seed)
    evaluate_population(pop, inst)

    fronts = fast_non_dominated_sort(pop)
    for front in fronts:
        crowding_distance_assignment(front)

    _log_progress(0, pop, t_start, fronts)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for gen in range(1, n_gen + 1):

        # ── Offspring generation ─────────────────────────────────────────────
        offspring: list = []
        while len(offspring) < pop_size:
            parent1 = tournament_select(pop)
            parent2 = tournament_select(pop)

            if random.random() < p_cross:
                child1, child2 = ox_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()

            mutate(child1, inst, p_swap, p_vehicle)
            mutate(child2, inst, p_swap, p_vehicle)
            offspring.extend([child1, child2])

        offspring = offspring[:pop_size]    # trim to exact size if pop_size is odd

        # ── Evaluate offspring ───────────────────────────────────────────────
        evaluate_population(offspring, inst)

        # ── Elitist selection (parent ∪ offspring → next gen) ────────────────
        combined = pop + offspring
        pop = select_next_generation(combined, pop_size)

        # Re-assign ranks and crowding distances for the new population
        fronts = fast_non_dominated_sort(pop)
        for front in fronts:
            crowding_distance_assignment(front)

        if log_every > 0 and gen % log_every == 0:
            _log_progress(gen, pop, t_start, fronts)

    # ── Extract and return the final Pareto front ─────────────────────────────
    final_fronts = fast_non_dominated_sort(pop)
    pareto = final_fronts[0] if final_fronts else pop
    pareto = [ind for ind in pareto if ind.feasible]   # feasible only

    t_elapsed = time.perf_counter() - t_start
    print(
        f"[nsga2] DONE  {n_gen} gen  |Pareto|={len(pareto)}  "
        f"runtime={t_elapsed:.2f}s"
    )

    # Sort Pareto front by f1 for readability
    pareto.sort(key=lambda ind: ind.f1)
    return pareto


def _log_progress(gen: int, pop: list, t_start: float, fronts: list) -> None:
    """Print a one-line progress summary."""
    pareto = fronts[0] if fronts else []
    feasible = [ind for ind in pareto if ind.feasible]
    if feasible:
        best_f1 = min(ind.f1 for ind in feasible)
        best_f2 = min(ind.f2 for ind in feasible)
    else:
        best_f1 = best_f2 = float("nan")

    elapsed = time.perf_counter() - t_start
    n_feas = sum(1 for ind in pop if ind.feasible)
    print(
        f"[nsga2] gen={gen:4d}  |F1|={len(pareto):3d}  feas={n_feas:3d}/"
        f"{len(pop)}  "
        f"best_f1={best_f1:8.2f}  best_f2={best_f2:8.2f}  "
        f"t={elapsed:6.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Result export
# ─────────────────────────────────────────────────────────────────────────────

def export_pareto(
    pareto: list,
    inst,
    output_path: Path,
    cfg: Optional[dict] = None,
) -> None:
    """Write the Pareto front to an xlsx file.

    Sheet layout mirrors run_model.py's write_results() so that verify.py
    and downstream visualisation scripts can consume the output unchanged.

    Sheets
    ------
    pareto_summary  : one row per Pareto-optimal solution (f1, f2, rank …).
    assignment_f    : product → vehicle/route assignments for each solution
                      (solution index in column ``sol_idx``).
    config_used     : cfg snapshot for reproducibility.
    """
    if not pareto:
        print("[nsga2] export_pareto: empty Pareto front — nothing to write.")
        return

    # ── pareto_summary sheet ──────────────────────────────────────────────────
    summary_rows = []
    for idx, ind in enumerate(pareto):
        summary_rows.append({
            "sol_idx":           idx,
            "rank":              ind.rank,
            "f1_route_duration": round(ind.f1, 4),
            "f2_total_wait":     round(ind.f2, 4),
            "feasible":          ind.feasible,
            "violation":         round(ind.violation, 4),
            "crowding_dist":     round(ind.crowding_dist, 4),
        })

    # ── assignment_f sheet ────────────────────────────────────────────────────
    P = inst.P
    K = inst.K
    assign_rows = []
    for idx, ind in enumerate(pareto):
        for pos, prod_idx in enumerate(ind.perm):
            p = P[prod_idx]
            k = K[ind.vehicle[pos]]
            assign_rows.append({
                "sol_idx":     idx,
                "product":     p,
                "vehicle":     k,
                "origin":      inst.o[p],
                "destination": inst.d[p],
            })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="pareto_summary", index=False
        )
        pd.DataFrame(assign_rows).to_excel(
            writer, sheet_name="assignment_f", index=False
        )
        if cfg is not None:
            cfg_rows = [{"parameter": k, "value": str(v)}
                        for k, v in sorted(cfg.items())]
            pd.DataFrame(cfg_rows).to_excel(
                writer, sheet_name="config_used", index=False
            )

    print(f"[nsga2] Pareto front ({len(pareto)} solutions) → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Standalone entry point (mirrors run_model.py's CLI)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="NSGA-II runner for BO-MR-PDPTW-STW")
    parser.add_argument("--inputs", type=Path, default=Path("inputs"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Lazy import to avoid circular dependency when imported from run_model.py
    from run_model import load_config, load_instance, configure_logging
    from datetime import datetime

    config_path = args.config or (args.inputs / "config.xlsx")
    cfg = load_config(config_path)

    # Inject NSGA-II defaults for any missing nsga2_* keys
    for k, v in NSGA2_DEFAULTS.items():
        cfg.setdefault(k, v)

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_dir = out_dir / f"nsga2_{cfg['product_set_id']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / f"nsga2_{timestamp}.log"
    configure_logging(log_file=log_path, verbose=args.verbose)

    print(f"[nsga2] inputs  : {args.inputs}")
    print(f"[nsga2] config  : {config_path}")
    print(f"[nsga2] output  : {run_dir}")

    try:
        from run_model import load_instance, DataConsistencyError
        inst = load_instance(args.inputs, cfg)
    except Exception as exc:
        print(f"[nsga2] DATA ERROR: {exc}")
        return 2

    # Optional Solomon warm-start
    solomon_seed = None
    if cfg.get("nsga2_warmstart", True):
        try:
            from solomon import construct_with_multistart, fleet_to_solution
            fleet, status, _ = construct_with_multistart(inst, cfg)
            if status == "feasible":
                solomon_seed = fleet_to_solution(fleet)
                print("[nsga2] Solomon warm-start solution: feasible ✓")
            else:
                print("[nsga2] Solomon warm-start solution: infeasible — skipping.")
        except ImportError:
            print("[nsga2] solomon.py not found — starting without warm-start.")
        except Exception as exc:
            print(f"[nsga2] Solomon warm-start failed: {exc}")

    pareto = nsga2(inst, cfg, solomon_seed=solomon_seed)

    result_xlsx = run_dir / f"pareto_nsga2_{timestamp}.xlsx"
    export_pareto(pareto, inst, result_xlsx, cfg=cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
