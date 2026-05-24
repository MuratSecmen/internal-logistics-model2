"""
frontier_metrics.py — Pareto frontier karşılaştırma metrikleri.

Desteklenen metrikler
=====================
HV   — Hypervolume indicator
       Referans noktasına göre Pareto frontunun kapladığı alan.
       Yüksek = daha iyi. 2-D için sweep-line algoritması (O(N log N)).

GD   — Generational Distance
       Yaklaşık fronttaki her noktanın gerçek (referans) fronta
       olan ortalama Öklid mesafesi. Düşük = daha iyi.

IGD  — Inverted Generational Distance
       Referans fronttaki her noktanın yaklaşık fronta olan
       ortalama Öklid mesafesi. GD'den kapsamlı; yayılımı da ölçer.
       Düşük = daha iyi.

SP   — Spacing (Schott, 1995)
       Ardışık noktalar arası mesafelerin standart sapması.
       Düşük = daha düzgün yayılım.

EI   — Epsilon-indicator (I_ε+, Zitzler 2003)
       A kümesinin B kümesini ε-domine etmesi için gereken
       minimum ε. Düşük = daha iyi yaklaşım.

Kullanım
--------
    from frontier_metrics import compute_all

    mip_front  = [(dur1, wait1), (dur2, wait2), ...]   # MIP Pareto noktaları
    nsga_front = [(dur1, wait1), (dur2, wait2), ...]   # NSGA-2 Pareto noktaları

    ref_point  = (max_dur * 1.1, max_wait * 1.1)       # HV referansı

    metrics = compute_all(
        approx  = nsga_front,   # değerlendirilen front
        ref_set = mip_front,    # referans (genellikle MIP)
        ref_point = ref_point,
    )
    # metrics = {"HV": ..., "GD": ..., "IGD": ..., "SP": ..., "EI": ...}

Not: Her iki boyut da minimize edildiği varsayılır.
"""

from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]


# ─────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────

def _euclidean(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _non_dominated_2d(points: list[Point]) -> list[Point]:
    """
    2-D minimize probleminde non-dominated noktaları döner.
    f1 artan sırada sıralar, f2 üzerinden filtreler.
    """
    if not points:
        return []
    srt = sorted(points, key=lambda p: (p[0], p[1]))
    nd = [srt[0]]
    min_f2 = srt[0][1]
    for p in srt[1:]:
        if p[1] < min_f2:
            nd.append(p)
            min_f2 = p[1]
    return nd


def _normalize(points: list[Point],
               ideal: Point, nadir: Point) -> list[Point]:
    """Her boyutu [0, 1]'e normalize et."""
    span0 = nadir[0] - ideal[0] or 1.0
    span1 = nadir[1] - ideal[1] or 1.0
    return [((p[0] - ideal[0]) / span0,
             (p[1] - ideal[1]) / span1) for p in points]


# ─────────────────────────────────────────────────────────────
# Hypervolume — 2-D sweep-line  (minimizasyon)
# ─────────────────────────────────────────────────────────────

def hypervolume_2d(front: list[Point], ref: Point) -> float:
    """
    Referans noktasına göre 2-D Pareto frontunun hypervolume'unu hesaplar.

    Algoritma: noktaları f1'e göre sırala; her dilimde
    (f1_i - f1_{i-1}) * (ref_f2 - f2_i) dikdörtgenlerini topla.

    Geçersiz durumlar (ref noktasını domine eden nokta yok) → 0 döner.
    """
    if not front:
        return 0.0

    # Referansı domine etmeyen noktaları filtrele
    valid = [p for p in front if p[0] < ref[0] and p[1] < ref[1]]
    if not valid:
        return 0.0

    # Non-dominated alt küme
    nd = _non_dominated_2d(valid)

    hv = 0.0
    prev_f1 = ref[0]
    # Sağdan sola (f1 büyükten küçüğe) tara
    for p in reversed(nd):
        width  = prev_f1 - p[0]
        height = ref[1]  - p[1]
        hv    += width * height
        prev_f1 = p[0]
    return hv


# ─────────────────────────────────────────────────────────────
# Generational Distance
# ─────────────────────────────────────────────────────────────

def generational_distance(approx: list[Point],
                           ref_set: list[Point]) -> float:
    """
    GD: yaklaşık fronttaki her noktanın ref_set'e olan
    minimum Öklid mesafelerinin ortalaması.
    """
    if not approx or not ref_set:
        return float("inf")
    total = sum(
        min(_euclidean(a, r) for r in ref_set)
        for a in approx
    )
    return total / len(approx)


# ─────────────────────────────────────────────────────────────
# Inverted Generational Distance
# ─────────────────────────────────────────────────────────────

def inverted_generational_distance(approx: list[Point],
                                    ref_set: list[Point]) -> float:
    """
    IGD: ref_set'teki her noktanın yaklaşık fronta olan
    minimum Öklid mesafelerinin ortalaması.
    """
    if not approx or not ref_set:
        return float("inf")
    total = sum(
        min(_euclidean(r, a) for a in approx)
        for r in ref_set
    )
    return total / len(ref_set)


# ─────────────────────────────────────────────────────────────
# Spacing (Schott 1995)
# ─────────────────────────────────────────────────────────────

def spacing(front: list[Point]) -> float:
    """
    SP: ardışık Pareto noktaları arası mesafelerin standart sapması.
    Tek nokta varsa 0 döner (yayılım tanımsız).
    """
    if len(front) <= 1:
        return 0.0

    nd = _non_dominated_2d(front)
    if len(nd) <= 1:
        return 0.0

    dists = [_euclidean(nd[i], nd[i + 1]) for i in range(len(nd) - 1)]
    mean_d = sum(dists) / len(dists)
    variance = sum((d - mean_d) ** 2 for d in dists) / len(dists)
    return math.sqrt(variance)


# ─────────────────────────────────────────────────────────────
# Epsilon-indicator (I_ε+, additive, Zitzler 2003)
# ─────────────────────────────────────────────────────────────

def epsilon_indicator(approx: list[Point],
                       ref_set: list[Point]) -> float:
    """
    I_ε+(approx, ref_set): approx'un ref_set'i ε-domine etmesi için
    gereken minimum additif ε.

    I_ε+ = max_{r ∈ ref} min_{a ∈ approx} max_i (a_i - r_i)

    Değer ne kadar küçükse (hatta negatifse) approx o kadar iyi.
    """
    if not approx or not ref_set:
        return float("inf")
    eps = -math.inf
    for r in ref_set:
        min_e = math.inf
        for a in approx:
            e = max(a[0] - r[0], a[1] - r[1])
            if e < min_e:
                min_e = e
        eps = max(eps, min_e)
    return eps


# ─────────────────────────────────────────────────────────────
# Tek seferde tüm metrikler
# ─────────────────────────────────────────────────────────────

def compute_all(approx: list[Point],
                ref_set: list[Point],
                ref_point: Point | None = None,
                normalize: bool = True) -> dict:
    """
    Tüm metrikleri hesapla ve sözlük olarak döner.

    Parametreler
    ------------
    approx     : değerlendirilen Pareto yaklaşımı (örn. NSGA-2 frontu)
    ref_set    : referans Pareto seti (örn. MIP frontu)
    ref_point  : HV için referans noktası; None ise otomatik seçilir
                 (nadir noktanın %10 üstü)
    normalize  : True ise GD/IGD/SP hesabında noktalar önce [0,1]'e
                 normalize edilir (ölçek bağımsız karşılaştırma)

    Dönüş
    -----
    dict: {HV, GD, IGD, SP, EI, n_approx, n_ref}
    """
    # Non-dominated alt küme al
    approx_nd  = _non_dominated_2d(approx)  if approx  else []
    ref_nd     = _non_dominated_2d(ref_set) if ref_set else []

    # Referans noktası (HV için)
    all_pts = approx_nd + ref_nd
    if ref_point is None:
        if all_pts:
            max_f1 = max(p[0] for p in all_pts)
            max_f2 = max(p[1] for p in all_pts)
            ref_point = (max_f1 * 1.10 + 1, max_f2 * 1.10 + 1)
        else:
            ref_point = (1.0, 1.0)

    # Normalizasyon
    if normalize and all_pts:
        ideal = (min(p[0] for p in all_pts), min(p[1] for p in all_pts))
        nadir = (max(p[0] for p in all_pts), max(p[1] for p in all_pts))
        approx_norm = _normalize(approx_nd, ideal, nadir)
        ref_norm    = _normalize(ref_nd,    ideal, nadir)
        ref_pt_norm = (
            (ref_point[0] - ideal[0]) / ((nadir[0] - ideal[0]) or 1.0),
            (ref_point[1] - ideal[1]) / ((nadir[1] - ideal[1]) or 1.0),
        )
    else:
        approx_norm = approx_nd
        ref_norm    = ref_nd
        ref_pt_norm = ref_point

    return {
        "HV":       hypervolume_2d(approx_norm, ref_pt_norm),
        "GD":       generational_distance(approx_norm, ref_norm),
        "IGD":      inverted_generational_distance(approx_norm, ref_norm),
        "SP":       spacing(approx_norm),
        "EI":       epsilon_indicator(approx_norm, ref_norm),
        "n_approx": len(approx_nd),
        "n_ref":    len(ref_nd),
    }
