import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# İlk olarak örnek Excel dosyası oluştur
def create_sample_excel():
    """Test verisi ile Excel dosyası oluştur"""
    
    # Örnek veri (H kolonu: Ürün Bekleme Zamanı, I kolonu: Araç Varış Zamanı)
    np.random.seed(42)
    n_solutions = 50
    
    # Pareto frontier oluştur (trade-off ilişkisi olan veriler)
    h_values = np.sort(np.random.uniform(100, 500, n_solutions))
    i_values = np.sort(np.random.uniform(200, 800, n_solutions))[::-1]
    
    # Trade-off etkisini artır
    i_values = 1000 - (h_values / h_values.max()) * 600
    
    # Hakim noktalar ekle (random noise)
    hakim_h = np.random.uniform(150, 480, 30)
    hakim_i = np.random.uniform(250, 750, 30)
    
    # Tüm verileri birleştir
    all_h = np.concatenate([h_values, hakim_h])
    all_i = np.concatenate([i_values, hakim_i])
    
    # DataFrame oluştur
    df = pd.DataFrame({
        'A': range(1, len(all_h) + 1),
        'B': np.random.randint(1, 10, len(all_h)),
        'C': np.random.choice(['Vehicle1', 'Vehicle2', 'Vehicle3'], len(all_h)),
        'D': np.random.uniform(0, 100, len(all_h)),
        'E': np.random.uniform(0, 100, len(all_h)),
        'F': np.random.uniform(0, 100, len(all_h)),
        'G': np.random.uniform(0, 100, len(all_h)),
        'H': all_h,  # Ürün Bekleme Zamanı
        'I': all_i   # Araç Varış Zamanı
    })
    
    # Excel'e kaydet
    output_file = 'birlestirilmis_sonuclar.xlsx'
    df.to_excel(output_file, sheet_name='optimization_results', index=False)
    print(f"✓ Test dosyası oluşturuldu: {output_file}")
    return output_file


# Ana Pareto frontier fonksiyonu
def plot_pareto_frontier(excel_file, sheet_name, x_col='I', y_col='H'):
    """
    Excel dosyasından Pareto frontier çizer
    """
    
    # Excel dosyasını oku
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # H ve I kolonlarını al
    x_data = pd.to_numeric(df[x_col], errors='coerce').values
    y_data = pd.to_numeric(df[y_col], errors='coerce').values
    
    # NaN değerleri kaldır
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    
    print(f"\n{'='*70}")
    print(f"PARETO FRONTIER ANALİZİ")
    print(f"{'='*70}")
    print(f"Dosya: {excel_file}")
    print(f"Sheet: {sheet_name}")
    print(f"Toplam veri noktası: {len(x_data)}")
    
    # Pareto optimal noktaları belirle
    pareto_mask = np.ones(len(x_data), dtype=bool)
    
    for i in range(len(x_data)):
        for j in range(len(x_data)):
            if i != j:
                # j, i'yi her iki hedefte de dominate ediyorsa
                if x_data[j] <= x_data[i] and y_data[j] <= y_data[i]:
                    if x_data[j] < x_data[i] or y_data[j] < y_data[i]:
                        pareto_mask[i] = False
                        break
    
    # Pareto ve non-Pareto noktalarını ayır
    pareto_x = x_data[pareto_mask]
    pareto_y = y_data[pareto_mask]
    non_pareto_x = x_data[~pareto_mask]
    non_pareto_y = y_data[~pareto_mask]
    
    # Pareto noktalarını sırala
    if len(pareto_x) > 0:
        sort_idx = np.argsort(pareto_x)
        pareto_x_sorted = pareto_x[sort_idx]
        pareto_y_sorted = pareto_y[sort_idx]
    
    # Grafik oluştur
    plt.figure(figsize=(14, 9), dpi=300)
    
    # Non-Pareto noktaları
    if len(non_pareto_x) > 0:
        plt.scatter(non_pareto_x, non_pareto_y, 
                   c='lightcoral', s=120, alpha=0.6, 
                   label=f'Hakim Çözümler (n={len(non_pareto_x)})',
                   edgecolors='darkred', linewidth=0.8, marker='o')
    
    # Pareto optimal noktaları
    plt.scatter(pareto_x, pareto_y, 
               c='darkblue', s=200, alpha=0.9, 
               label=f'Pareto Optimal (n={len(pareto_x)})',
               edgecolors='navy', linewidth=2, zorder=5, marker='*')
    
    # Pareto frontier çizgisi
    if len(pareto_x) > 1:
        plt.plot(pareto_x_sorted, pareto_y_sorted, 
                'b--', linewidth=2.5, alpha=0.8, label='Pareto Frontier', zorder=4)
    
    # Eksen ve etiketler
    plt.xlabel(f'Kolon {x_col} (Araç Varış Zamanı)', fontsize=13, fontweight='bold')
    plt.ylabel(f'Kolon {y_col} (Ürün Bekleme Zamanı)', fontsize=13, fontweight='bold')
    plt.title('Bi-Objektif Optimizasyon: Pareto Frontier Analizi', 
             fontsize=15, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
    plt.legend(loc='best', fontsize=11, framealpha=0.96, edgecolor='black')
    plt.tight_layout()
    
    # Kaydet ve göster
    plt.savefig('pareto_frontier.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # İstatistiksel özet
    print(f"\nSonuçlar:")
    print(f"  Pareto Optimal Çözüm: {len(pareto_x)}")
    print(f"  Hakim Çözüm: {len(non_pareto_x)}")
    print(f"  Verimlilik Oranı: {(len(pareto_x)/len(x_data))*100:.2f}%")
    
    print(f"\nKolon {x_col} İstatistikleri:")
    print(f"  Min: {x_data.min():.2f}, Max: {x_data.max():.2f}")
    print(f"  Ort: {x_data.mean():.2f}, Std: {x_data.std():.2f}")
    
    print(f"\nKolon {y_col} İstatistikleri:")
    print(f"  Min: {y_data.min():.2f}, Max: {y_data.max():.2f}")
    print(f"  Ort: {y_data.mean():.2f}, Std: {y_data.std():.2f}")
    
    print(f"{'='*70}\n")
    print("✓ Grafik kaydedildi: pareto_frontier.png")
    
    return pareto_x, pareto_y, non_pareto_x, non_pareto_y


# ============================================================================
# ÇALIŞMA KOMU
# ============================================================================

if __name__ == "__main__":
    
    # Seçenek 1: Örnek veri ile test et
    print("\n[1/2] Test verisi oluşturuluyor...")
    excel_file = create_sample_excel()
    
    print("\n[2/2] Pareto frontier analizi yapılıyor...")
    pareto_x, pareto_y, non_x, non_y = plot_pareto_frontier(
        excel_file=excel_file,
        sheet_name='optimization_results',
        x_col='I',
        y_col='H'
    )
    
    # Seçenek 2: Kendi dosyanızı kullanmak için aşağıdaki satırı aktif edin:
    # pareto_x, pareto_y, non_x, non_y = plot_pareto_frontier(
    #     excel_file='birlestirilmis_sonuclar.xlsx',
    #     sheet_name='optimization_results',
    #     x_col='I',
    #     y_col='H'
    # )