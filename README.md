# Internal Logistics Optimization — PDVRP Model

## Problem Tanımı

**Pickup-Delivery Vehicle Routing Problem with Product Readiness Date (PDVRP)** — tesis içi lojistik operasyonlarının karma tam sayılı programlama (MILP) ve Solomon tabanlı sezgisel yöntemlerle optimizasyonu.

**Amaç (varsayılan):** Toplam parça bekleme süresinin (`wait`) minimizasyonu. `run_model.py`'deki `OBJECTIVE` değişkeni ile `time` (toplam süre) veya `distance` (toplam mesafe) olarak da çalıştırılabilir.

## Teknoloji Stack

- Python 3.8+
- Gurobi Optimizer (MILP Solver) — MIP modeli ve NSGA-II decoder karşılaştırmaları için
- Pandas / openpyxl (veri işleme)

## Dosya Yapısı

```
internal-logistics-model2/
├── run_model.py                    # Ana MILP modeli (Gurobi)
├── solomon_heuristic.py            # Solomon I1 yerleştirme sezgiseli (çok parçalı senaryolar)
├── inputs/
│   ├── vehicles.xlsx
│   ├── distances - dakika.xlsx     # OD seyahat süresi matrisi (dakika)
│   ├── distances - metre.xlsx      # OD mesafe matrisi (metre)
│   └── B320_Mesafe_Matrisleri.xlsx # Ham mesafe verisi (referans)
│
│   NOT: run_model.py ayrıca `inputs/nodes.xlsx` ve `inputs/products.xlsx`
│   dosyalarını da bekler — bu iki dosya bu repoda YOK, kendi veri
│   setinizle eklemeniz gerekiyor (bkz. "Kullanım" altında beklenen şema).
│
├── nsga2_experiments/               # MIP / Solomon / NSGA-II karşılaştırma deneyleri
│   ├── nsga2.py                     # NSGA-II algoritması (Solomon tabanlı decoder)
│   ├── frontier_metrics.py          # Pareto frontier metrikleri (HV, GD, IGD)
│   ├── task3_runner.py              # Deney: MIP vs Solomon (64 run)
│   ├── task4_runner.py              # Deney: MIP (aug. ε-constraint) vs NSGA-II (48 run)
│   ├── NSGA2_BilgiNotlari.xlsx
│   └── BilgiNotu{1,2,3}_*.docx      # Kromozom / Decoder / Offspring tasarım notları
│
├── parca nu genel/                  # Bağımsız veri hazırlama araçları (area_m2 tahmini)
│   ├── parca_yuzey_hacim_analizi.py # Geçmiş taşıma verisinden parça yüzey alanı/hacim çıkarımı
│   ├── romur_boyut_tahmini.py       # Boyutu bilinmeyen parçalar için kapasite-tabanlı tahmin
│   ├── 2025_parça detayları.xlsx    # Girdi
│   └── 2026_taşınan parça nu.s.xlsx # Girdi
│
├── requirements.txt
└── README.md
```

`results/` ve `logs/` klasörleri `run_model.py` çalıştırıldığında otomatik oluşturulur, git'e commit edilmez.

## Kullanım

### 1. Kurulum

```bash
pip install -r requirements.txt
```

> Gurobi lisansı gerektirir (akademik lisans: gurobi.com/academia).

### 2. Ana MILP modeli

`run_model.py`, varsayılan olarak `./inputs` klasöründen okur; farklı bir klasör için `--inputs` kullanın:

```bash
python run_model.py
python run_model.py --inputs /baska/bir/klasor
```

**Beklenen girdi dosyaları** (`inputs/` altında):

| Dosya | Zorunlu sütunlar |
|---|---|
| `nodes.xlsx` | `node_id` (depo `h` dahil) |
| `vehicles.xlsx` | `vehicle_id`, `capacity_m2` |
| `products.xlsx` | `product_id`, `origin`, `destination`, `ready_time`, `load_time`, `unload_time`, `area_m2` |
| `distances - dakika.xlsx` | `from_node`, `to_node`, `duration_min` |
| `distances - metre.xlsx` | `from_node`, `to_node`, `duration_metre` |

`nodes.xlsx` ve `products.xlsx` bu repoda **bulunmuyor** — kendi veri setinizle `inputs/` klasörüne eklemeniz gerekiyor. Eksik bir dosya varsa `run_model.py` hangi dosyanın nereye eklenmesi gerektiğini açıkça belirten bir hata verir.

### 3. Solomon sezgiseli

```bash
python solomon_heuristic.py --inputs inputs
```

`inputs/products_4part.xlsx`, `products_5part.xlsx`, `products_6part.xlsx`, `products_10part.xlsx` dosyalarını arar (bulunamayanları atlar); çıktıları `inputs/heuristic_results/` altına yazar.

### 4. NSGA-II / karşılaştırma deneyleri

`nsga2_experiments/` içindeki `task3_runner.py` ve `task4_runner.py`, MIP çözümünü Solomon sezgiseli ve NSGA-II ile karşılaştıran toplu deney betikleridir — her ikisi de kendi docstring'lerinde deney tasarımını (ürün sayısı/case/config ızgarası) detaylı anlatır.

### 5. `parça nu genel/` — veri hazırlama araçları

Ana modelden bağımsız, `products.xlsx`'in `area_m2` sütununu geçmiş taşıma verisinden tahmin etmek için kullanılan iki yardımcı script:

```bash
cd "parca nu genel"
python parca_yuzey_hacim_analizi.py   # 2025/2026 verisinden malzeme başına max yüzey alanı/hacim
python romur_boyut_tahmini.py         # boyutu bilinmeyen parçalar için kapasite-tabanlı tahmin (masalar.xlsx gerekir, repoda yok)
```

## Notlar

- `run_model.py`'nin çözüm süresi/gap gibi performans rakamları veri setine göre büyük ölçüde değişir; bu README artık uydurma/varsayımsal performans tabloları içermiyor — kendi veri setinizle çalıştırıp gerçek rakamları buraya siz ekleyin.
- Sadece tek bir MILP dosyası (`run_model.py`) var; ayrı "tight Big-M" / "unified Big-M" varyantları yok.

## Key References (Operations Research)

- **Miller et al. (1960)** — MTZ subtour elimination
- **Solomon (1987)** — VRPTW algorithms
- **Camm et al. (1990)** — Cutting Big M down to size
- **Desrochers & Laporte (1991)** — MTZ improvements
- **Savelsbergh & Sol (1995)** — General pickup-delivery problem

## License

Academic use (Gurobi Academic License required).
