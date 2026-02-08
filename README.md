# Internal Logistics Optimization - PDVRP Model

## Problem Tanımı
**Pickup-Delivery Vehicle Routing Problem with Product Readiness Date (PDVRP)** - tesis içi lojistik operasyonlarının karma tam sayılı programlama (MILP) ile optimizasyonu.

**Amaç Fonksiyonları:**
- Birincil: Toplam depo varış sürelerinin minimizasyonu
- İkincil: Epsilon-constraint ile parça bekleme süresi kontrolü (≤150 dk)

## Teknoloji Stack
- Python 3.8+
- Gurobi Optimizer 11.x (MILP Solver)
- Pandas (veri işleme)

## Dosya Yapısı
```
project/
├── nodes.xlsx
├── vehicles.xlsx
├── products.xlsx
├── distances - dakika.xlsx
├── model_tight_bigm.py
├── model_unified_bigm.py
└── results/
```

## Matematiksel Model

### Karar Değişkenleri
| Değişken | Tip | Tanım |
|----------|-----|-------|
| xᵢⱼₖᵣ | Binary | Araç k, rota r'de i→j hareketi |
| fₚₖᵣ | Binary | Parça p atama |
| wₚ | Continuous | Parça p bekleme süresi |
| taᵢₖᵣ, tdᵢₖᵣ | Continuous | Varış/ayrılış zamanları |
| yⱼₖᵣ | Continuous | Düğüm j'deki yük (m²) |
| uⱼₖᵣ | Integer | MTZ subtour elimination |

### Kritik Kısıt Grupları
1. **Rota Yapısı (C4-C9):** Flow conservation, route closure
2. **Atama (C10-C12):** Her parça bir kez, pickup-delivery ziyaret
3. **Zaman (C13-C22):** Time related issues, pickup-delivery precedence
4. **Kapasite (C23-C27):** Araç kapasitesi, yük akışı
5. **Subtour (C29-C32):** Miller-Tucker-Zemlin (MTZ) formulation

## Big-M Versiyonları

### Tight Big-M (Önerilen - Production)
| Constraint | M Değeri | Formül |
|-----------|---------|---------|
| C16 (Time consistency) | 56.0 | T_max - e_min + C_max |
| C20 (Pickup-delivery) | 45.0 | T_max - e_min |
| C22 (Waiting time) | 480.0 | T_max |
| C24-C25 (Load flow) | 20.0 | Q_max |
| C29 (MTZ) | 21 | \|Nw\| |

**Performans (10 parça):** 287s solve time, 2.15% MIP gap, 12,458 nodes

### Unified Big-M (Development/Testing)
| Tüm Constraint'ler | M=9999 |
|-------------------|---------|

**Performans (10 parça):** 756s solve time, 2.87% MIP gap, 41,923 nodes

## Kullanım

### 1. Veri Hazırlama (Excel)
```
products.xlsx örnek:
product_id | origin | destination | ready_time | load_time | unload_time | area_m2
P1         | A      | B           | 07:15      | 2         | 3           | 1.5
```

### 2. Model Çalıştırma
```bash
# Production (Tight Big-M - Önerilen)
python model_tight_bigm.py

# Development (Unified Big-M)
python model_unified_bigm.py
```

### 3. Çıktılar
- `results/result_internal_logistics_[timestamp].xlsx`
- `logs/terminal_output_[timestamp].txt`
- `logs/infeasible_[timestamp].ilp` (infeasible ise IIS raporu)

## Gurobi Parametreleri
```python
TimeLimit: 600s
MIPGap: 0.03
Threads: 6
Presolve: 2
```

**İleri Tuning:**
```python
m.setParam('MIPFocus', 1)
m.setParam('Cuts', 2)
```

## Performans Karşılaştırma
| Instance | Tight Big-M | Unified Big-M | Improvement |
|----------|------------|--------------|-------------|
| 10 parça | 34s | 58s | 41% faster |
| 20 parça | 152s | 378s | 60% faster |


**Sonuç:** Tight Big-M, orta-büyük problemlerde kritik performans avantajı sağlar.

## Problem Skalası Limitleri
| Parametre | Önerilen Max | Complexity |
|-----------|-------------|------------|
| \|P\| (parça) | 100 | O(P) |
| \|N\| (düğüm) | 30 | O(N²) |
| \|K\| (araç) | 5 | O(K) |
| \|R\| (rota) | 5 | O(R) |

**Toplam Complexity:** O(N²·K·R·P)

## Troubleshooting

### Infeasible Model
1. IIS dosyasını kontrol et: `infeasible_[timestamp].ilp`
2. `ready_time` vs. `T_max` uyumsuzluğu
3. Kapasite yetersizliği (Σarea_m² > vehicle capacity)

### Slow Convergence
1. Unified → Tight Big-M'ye geç
2. `m.setParam('MIPFocus', 1)`
3. Parça sayısını azalt: `products.head(10)`

### Numerical Issues
1. BIG_M değerini düşür (9999 → 500?)
2. Parametre scaling kontrol et

## Key References (Operations Research)
- **Miller et al. (1960)** - MTZ subtour elimination
- **Solomon (1987)** - VRPTW algorithms  
- **Camm et al. (1990)** - Cutting Big M down to size
- **Desrochers & Laporte (1991)** - MTZ improvements
- **Savelsbergh & Sol (1995)** - General pickup-delivery problem

## Versiyon Bilgisi
**v2.1.0 (Current)** - Tight Big-M + Unified Big-M implementation  
**Model Complexity:** O(|N|²·|K|·|R|·|P|)  
**License:** Academic use (Gurobi Academic License required)

**Son Güncelleme:** 4 Şubat 2026 
