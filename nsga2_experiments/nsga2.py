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

E�er herhangi bir ürün için c1=inf dönerse birey cezalandırılır:
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

Düzeltmeler (v2)
================
    FIX-1  gene_from_fleet_helper import kaldırıldı (fonksiyon bu dosyada
           gene_from_fleet adıyla tanımlı; solomon.py'den almaya gerek yok).
    FIX-2  _STOCK_BIASES ve construct, try/except ile import edilir;
           solomon.py'de yoksa built-in fallback devreye girer.
    FIX-3  gene_from_fleet: ürün, yalnızca origin düğümünde tespit edilir
           (önceki versiyonda "or inst.d[p] == node" ekliydi, gereksiz ve
           geçersiz rotada yanlış sıra üretebilirdi).
    FIX-4  random_gene: slot içi karışıklık için rng.shuffle eklendi.
    FIX-5  ox_crossover _build_child Phase-3: artık doğru alt_donor'ı
           kullanıyor. Önceki versiyonda tüm slotlar için donor_b'nin
           sırası baz alınıyordu; donor_b kaynaklı slotlarda bu OX'u
           no-op hale getiriyordu.
    FIX-6  tournament_select: rng.sample ile iki farklı birey seçilmesi
           garanti altına alındı (önceden aynı birey iki kez seçilebiliyordu).
    FIX-7  nsga2 offspring döngüsü: pop_size tek sayıysa pop_size+1 offspring
           üretme hatası giderildi (c2 eklenmeden önce boyut kontrolü yapılıyor).
    FIX-8  _solomon_seeds: construct=None ise graceful fallback.
    FIX-9  history: Pareto front büyüklüğü (|front_0|) 5. kolon olarak eklendi.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

# ── Zorunlu solomon importları ────────────────────────────────────────────────
from solomon import (
    FleetState,
    best_insert_into_route,
    apply_insertion,
    # FIX-1: gene_from_fleet_helper ÇIKARILDI.
    # Fonksiyon bu dosyada gene_from_fleet adıyla tanımlı;
    # solomon.py'den import etmeye gerek yok.
)

# ── Opsiyonel solomon importları (FIX-2) ──────────────────────────────────────
# _STOCK_BIASES ve construct solomon.py'de yoksa built-in fallback kullanılır.
try:
    from solomon import construct, _STOCK_BIASES
    _SOLOMON_AVAILABLE = True
except ImportError:
    construct = None
    _SOLOMON_AVAILABLE = False
    # Document-5'teki 6-bias ladder (fallback)
    _STOCK_BIASES = [
        ("route_duration", {"alpha": (0.0001, 1.0,  0.01 )}),
        ("wait_time",      {"alpha": (0.0001, 0.01, 1.0  )}),
        ("balanced",       {"alpha": (0.0001, 0.5,  0.5  )}),
        ("distance",       {"alpha": (1.0,    1e-4, 1e-4 )}),
        ("dist_wait",      {"alpha": (0.5,    1e-4, 0.5  )}),
        ("dist_duration",  {"alpha": (0.5,    0.5,  1e-4 )}),
    ]

log = logging.getLogger("internal_logistics.nsga2")

PENALTY = 1e9
EPS     = 1e-9


# ============================================================
# Yardımcı: FleetState → gene çevirici  (FIX-3)
# ============================================================

def gene_from_fleet(fleet: FleetState) -> dict:
    """
    Decode edilmiş bir FleetState'den gene üret (Solomon seed'leri için).

    FIX-3: Önceki versiyonda bir ürün, origin VEYA destination düğümü
    ilk karşılaşıldığında listeye ekleniyordu:
        if p not in seen and (inst.o[p] == node or inst.d[p] == node)
    Bu gereksiz: geçerli bir rotada destination her zaman origin'den
    sonra gelir; dolayısıyla destination'a ulaşıldığında ürün zaten
    'seen' içindedir. Bununla birlikte, geçersiz bir FleetState'de
    (origin'den önce destination ziyaret edilmişse) ürün yanlış
    pozisyonda listeye girebilirdi.
    Düzeltme: yalnızca origin düğümünde tespit yap.
    """
    gene = {kr: [] for kr in fleet.inst.KR_pairs}
    for kr, route in fleet.routes.items():
        inst    = fleet.inst
        ordered = []
        seen    = set()
        for node in route.nodes[1:-1]:          # depot hariç
            for p in route.parts:
                # FIX-3: sadece origin koşulu
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

    gene: dict                          # {(k, r): [p1, p2, ...]}
    f1: float = PENALTY                 # route_duration
    f2: float = PENALTY                 # total_wait
    rank: int = 0                       # Pareto rank (1 = en iyi front)
    crowding: float = 0.0               # crowding distance
    fleet: object = field(default=None, repr=False)  # decode sonrası FleetState

    # ----------------------------------------------------------
    def dominates(self, other: "Individual") -> bool:
        """self, other'ı Pareto-domine ediyorsa True (minimizasyon)."""
        return (
            self.f1 <= other.f1 + EPS and
            self.f2 <= other.f2 + EPS and
            (self.f1 < other.f1 - EPS or self.f2 < other.f2 - EPS)
        )

    def is_feasible(self) -> bool:
        return self.f1 < PENALTY - EPS


def _clone(ind: Individual) -> Individual:
    """Gene'i deep-copy eden hafif klonlayıcı."""
    return Individual(gene={kr: list(lst) for kr, lst in ind.gene.items()})


# ============================================================
# Chromosome yardımcıları
# ============================================================

def random_gene(inst, rng: random.Random) -> dict:
    """
    Tüm ürünleri karıştırıp round-robin ile (k,r) slotlarına dağıt.

    FIX-4: Önceki versiyonda slot içi sıra, round-robin atama sırasına
    göre deterministikti (ürün listesi shuffle edilse de slot içindeki
    göreli sıra korunuyordu). Şimdi her slot'un listesi de ayrıca
    shuffle ediliyor; bu, başlangıç popülasyonuna daha fazla çeşitlilik
    katar.
    """
    products = list(inst.P)
    rng.shuffle(products)
    kr_pairs = list(inst.KR_pairs)
    gene     = {kr: [] for kr in kr_pairs}
    for i, p in enumerate(products):
        gene[kr_pairs[i % len(kr_pairs)]].append(p)
    # FIX-4: slot içi shuffle
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
    """
    Gene'i FleetState'e dönüştür; f1, f2 ve fleet'i in-place güncelle.

    Her (k, r) slotu için gene listesindeki ürünler sırayla işlenir.
    best_insert_into_route yalnızca o-d çiftinin geometrik pozisyonunu
    belirler (araç ve insert sırası kromozomdan gelir).

    Herhangi bir ürün yerleştirilemezse → f1 = f2 = PENALTY.

    NOT: f2 hesabı (fleet.total_wait()) solomon.py'deki FleetState'e
    delege edilir. Document-4 Constraint-21'e göre bekleme süresi
    t^a_{dp} + s^u_p - e_p şeklinde hesaplanmalıdır; bu doğruluğun
    sorumluluğu solomon.py'dedir.
    """
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
    """
    Popülasyonu Pareto frontlarına ayır.

    Returns
    -------
    fronts : list[list[int]]
        fronts[0] = Pareto-optimal bireylerin indeksleri (rank 1),
        fronts[1] = rank 2, vs.

    Karmaşıklık: O(M · N²), M = hedef sayısı (2), N = popülasyon büyüklüğü.
    """
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
    """
    Bir fronttaki bireylere crowding distance ata (in-place).

    Her hedef ekseninde sıralayıp komşu değer farklarını normalize ederek
    toplar. Boundary bireyler sonsuz mesafe alır (her zaman hayatta kalır).
    """
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
# Selection — Binary Tournament  (FIX-6)
# ============================================================

def tournament_select(pop: list, rng: random.Random) -> Individual:
    """
    İkili turnuva seçimi.
    Kural: düşük rank kazanır; eşit rankta yüksek crowding distance kazanır.

    FIX-6: Önceki versiyonda rng.choice(pop) iki kez çağrılıyordu;
    aynı bireyin iki kez seçilmesi mümkündü (self-tournament).
    rng.sample(range(len(pop)), 2) ile iki farklı indeks garanti edilir.
    """
    i, j = rng.sample(range(len(pop)), 2)
    a, b = pop[i], pop[j]
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    return a if a.crowding >= b.crowding else b


# ============================================================
# Crossover — Order Crossover (OX), araç bazlı  (FIX-5)
# ============================================================

def _ox_lists(lst_a: list, lst_b: list, rng: random.Random) -> list:
    """
    Aynı eleman kümesini içeren iki liste üzerinde standart OX uygular.

    1. lst_a'dan rastgele [lo, hi] segmenti al.
    2. lst_b'nin sırasına göre kalan elemanları doldur.
    """
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
    """
    Araç-farkında OX Crossover → iki çocuk üretir.

    Strateji (3 aşama):
    ──────────────────
    Aşama 1 — Slot bölme:
        Rastgele bir s seçilir.
        Çocuk 1: slot 0..s → P1'den, slot s+1.. → P2'den.
        Çocuk 2: slot 0..s → P2'den, slot s+1.. → P1'den.

    Aşama 2 — Çakışma giderme:
        Her slot, önce kendi donor'ından ürünleri alır (henüz atanmayanlar).
        Eksik ürünler en hafif slota eklenir. Her ürün tam bir kez görünür.

    Aşama 3 — OX sıralama (FIX-5):
        Her slot için primary donor'ın sırası (lst_a) ile o slotun
        ALT donor'ının sırası (lst_b) arasında OX uygulanır.
        Önceki versiyonda tüm slotlar için donor_b kullanılıyordu;
        donor_b kaynaklı slotlarda lst_a == lst_b olduğundan OX
        işlevsiz kalıyordu. Şimdi slot_donor sözlüğü hangi slotun
        hangi donor'dan geldiğini takip eder ve alt_donor doğru seçilir.
    """
    kr_pairs = list(inst.KR_pairs)
    n_kr     = len(kr_pairs)
    s        = rng.randint(0, n_kr - 2) if n_kr > 1 else 0

    def _build_child(donor_a: Individual, donor_b: Individual) -> Individual:
        gene_c     = {kr: [] for kr in kr_pairs}
        assigned   = set()
        slot_donor = {}   # kr → (primary_donor, alt_donor)

        # ── Aşama 1 & 2: slot bölme + çakışma giderme ────────
        for idx, kr in enumerate(kr_pairs):
            primary = donor_a if idx <= s else donor_b
            alt     = donor_b if idx <= s else donor_a
            slot_donor[kr] = (primary, alt)             # kaydediliyor
            lst = [p for p in primary.gene.get(kr, []) if p not in assigned]
            assigned.update(lst)
            gene_c[kr] = lst

        # ── Aşama 3: OX sıralama (FIX-5) ─────────────────────
        for kr in kr_pairs:
            lst_a = gene_c[kr]
            _, alt_donor = slot_donor[kr]               # doğru alt donor
            lst_b = [p for p in alt_donor.gene.get(kr, [])
                     if p in set(lst_a)]
            if len(lst_a) >= 2 and len(lst_b) == len(lst_a):
                gene_c[kr] = _ox_lists(lst_a, lst_b, rng)

        # ── Güvenlik: eksik ürünleri en az dolu slota ekle ────
        missing = [p for p in inst.P if p not in assigned]
        for p in missing:
            lightest = min(kr_pairs, key=lambda kr: len(gene_c[kr]))
            gene_c[lightest].append(p)
            assigned.add(p)

        return Individual(gene=gene_c)

    return _build_child(p1, p2), _build_child(p2, p1)


# ============================================================
# Mutasyon operatörleri
# ============================================================

def swap_mutation(ind: Individual, rng: random.Random,
                  p: float = 0.15) -> Individual:
    """
    Rota-içi takas (intra-route swap).
    p olasılığıyla ≥2 ürün içeren rastgele bir (k,r) slotu seçilir;
    o slottaki iki rastgele pozisyon yer değiştirir.
    """
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
    """
    Rota-içi segment ters çevirme (intra-route inversion).
    p olasılığıyla ≥2 ürün içeren bir slot seçilir; rastgele [lo,hi]
    alt-segmenti tersine çevrilir.
    """
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
    """
    Rotalar-arası transfer (inter-route transfer).
    p olasılığıyla boş olmayan rastgele bir kaynak (k,r)'den bir ürün
    koparılır ve rastgele bir hedef (k',r')'nin rastgele pozisyonuna eklenir.
    """
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
    """
    Solomon multi-start'tan n farklı bias ile tohum bireyler üret.

    FIX-8: construct=None ise (solomon.py'de export yoksa) graceful
    fallback — boş liste döner, rastgele bireyler ile doldurulur.
    """
    # FIX-8
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
    """
    pop_size büyüklüğünde ilk popülasyon oluşturur ve decode eder.
    İlk n_solomon_seeds birey Solomon tohumlarından gelir.
    Geri kalanı rastgele gene'lerle doldurulur.
    """
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

                # FIX-7: pop_size tek sayıysa taşmayı önle
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

        # ── 2e. Kayıt (FIX-9: |front_0| eklendi) ─────────────
        feasible   = [ind for ind in population if ind.is_feasible()]
        n_feas     = len(feasible)
        best_f1    = min((ind.f1 for ind in feasible), default=PENALTY)
        best_f2    = min((ind.f2 for ind in feasible), default=PENALTY)
        pareto_now = [ind for ind in population if ind.rank == 1
                      and ind.is_feasible()]
        # FIX-9: 5. kolon = Pareto front büyüklüğü
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
