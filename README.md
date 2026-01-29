# gurobi-InternalLogistics-model2
İlk Modelin devamı niteliğinde farklı performans metriği tanımlanmıştır.

Bu proje, bir tesis içindeki lojistik operasyonlarının optimizasyonunu gerçekleştiren karma tam sayılı programlama modelini içermektedir. Optimizasyonun temel amacı, ürünlerin toplam bekleme sürelerini minimize ederek iç lojistik verimliliğini artırmaktır.

## Kullanılan Kütüphaneler

* **Python 3.x**
* **Pandas** (veri yönetimi için)
* **Gurobi** (optimizasyon için)

## Klasör Yapısı

```
project/
├── nodes.xlsx
├── vehicles.xlsx
├── products.xlsx
├── distances.xlsx
├── main.py
└── optimization_results.xlsx (model sonuçları)
```

## Excel Girdi Dosyaları

* **nodes.xlsx**: Tüm düğüm noktalarının listesi.
* **vehicles.xlsx**: Araç ID'leri ve kapasiteleri.
* **products.xlsx**: Ürün ID'leri, başlangıç ve bitiş noktaları, hazır olma zamanları ve yükleme/boşaltma süreleri.
* **distances.xlsx**: Düğümler arası seyahat süreleri.

## Model Parametreleri

* **Setler:** Düğümler (`N`), araçlar (`K`), ürünler (`P`), rotalar (`R`).
* **Big M değerleri:**

  * `M_time`: Zaman kısıtları için büyük M (900 dk).
  * `M_load`: Yükleme kısıtları için büyük M (60 m²).

## Model Çıktıları

Optimizasyonun ardından her ürün için atanan araç, rota ve bekleme sürelerini içeren bir `optimization_results.xlsx` dosyası oluşturulur.

## Kullanım Talimatı

1. Excel dosyalarını belirtilen yapıda hazırlayın.
2. `main.py` scriptini çalıştırın:

```bash
python main.py
```

## Gurobi Ayarları

* Çözüm süresi limiti: 1200 saniye (20 dakika)
* Kabul edilebilir optimalite boşluğu (`MIPGap`): %5
* Log dosyası: `gurobi_log.txt`

## Sonuçlar

Çözüm bulunamazsa IIS raporu (`model.ilp`) otomatik olarak oluşturulur ve modeldeki tutarsızlıklar hakkında bilgi sağlar.


## Recent Updates

### 28 Ocak 2026 - Big-M Renewal
- Tight bounds implemented for all 7 constraints
- Constraint 21-22 (Capacity): 50,000x improvement (M: 999,999 → 20/10)
- Expected 60-80% solution time improvement
- Tested with 10 parts dataset

### Tight M Values
- C-14 (Time consistency): M = 449.9 min
- C-18 (Pickup-delivery): M = 442.9 min  
- C-19 (Waiting time): M = 480 min
- C-21-22 (Capacity flow): M = 20/10 m²  
- C-25 (MTZ subtour): M = 21
