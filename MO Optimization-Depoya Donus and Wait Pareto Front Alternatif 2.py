import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# =====================================================================
# AYARLAR
# =====================================================================
folder = r"C:\Users\Asus\results\Pareto"
output_file = os.path.join(folder, "pareto_frontier.png")

# =====================================================================
# EXCEL DOSYALARINI OKU
# =====================================================================
print("="*80)
print("EXCEL DOSYALARI OKUNUYOR")
print("="*80)

excel_files = glob.glob(os.path.join(folder, "*.xlsx"))
print(f"Bulunan dosya sayısı: {len(excel_files)}\n")

data = []

for file in excel_files:
    try:
        filename = os.path.basename(file)
        print(f"Okunuyor: {filename}")
        
        df = pd.read_excel(file, sheet_name='optimization_results')
        
        epsilon = df['epsilon_wait_upper'].values[0]
        arrival = df['total_arrival_times'].values[0]
        waiting = df['total_wait_minutes'].values[0]
        status = df['status'].values[0]
        mip_gap = df['mip_gap'].values[0] * 100
        
        if status == 2:
            status_text = "OPTIMAL"
        elif status == 9:
            status_text = "SUBOPTIMAL"
        elif status == 3:
            status_text = "INFEASIBLE"
        else:
            status_text = f"STATUS_{status}"
        
        data.append({
            'epsilon': epsilon,
            'arrival': arrival,
            'waiting': waiting,
            'status': status_text,
            'mip_gap': mip_gap
        })
        
        print(f"  ✓ ε={epsilon:.1f}, Arrival={arrival:.2f}, Waiting={waiting:.2f}, Status={status_text}")
        
    except Exception as e:
        print(f"  ✗ HATA: {str(e)}")

print(f"\n✓ Toplam {len(data)} dosya okundu\n")

# =====================================================================
# DATAFRAME OLUŞTUR
# =====================================================================
df = pd.DataFrame(data)
df = df.sort_values('epsilon', ascending=False).reset_index(drop=True)

df_feasible = df[df['status'].isin(['OPTIMAL', 'SUBOPTIMAL'])].copy()
df_infeasible = df[df['status'] == 'INFEASIBLE'].copy()

print("="*80)
print("VERİ DURUMU")
print("="*80)
print(f"Feasible: {len(df_feasible)}")
print(f"Infeasible: {len(df_infeasible)}")
print()

if not df_feasible.empty:
    print("FEASIBLE ÇÖZÜMLER:")
    print(df_feasible[['epsilon', 'arrival', 'waiting', 'mip_gap', 'status']].to_string())
    print()

# =====================================================================
# AKADEMİK STIL PARETO FRONTIER (ScienceDirect Tarzı)
# =====================================================================
print("="*80)
print("AKADEMİK PARETO FRONTIER ÇİZİLİYOR")
print("="*80)

# Beyaz arka plan, temiz stil
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
ax.set_facecolor('white')

if not df_feasible.empty:
    
    # Kaliteli ve düşük kaliteli çözümleri ayır
    df_quality = df_feasible[df_feasible['mip_gap'] < 5.0].copy()
    df_poor = df_feasible[df_feasible['mip_gap'] >= 5.0].copy()
    
    # ============================================================
    # 1. DOMINATED SOLUTIONS (Gri kareler - sağ üstte)
    # ============================================================
    if len(df_poor) > 0:
        ax.scatter(df_poor['arrival'], df_poor['waiting'],
                  s=180, marker='s', c='#9CA3AF',
                  edgecolors='#6B7280', linewidth=1.5, alpha=0.6,
                  label='Dominated solutions', zorder=2)
    
    # ============================================================
    # 2. NON-DOMINATED SOLUTIONS (Pareto üzerindeki noktalar)
    # ============================================================
    if not df_quality.empty:
        # Kırmızı daireler - kalın kenar
        ax.scatter(df_quality['arrival'], df_quality['waiting'],
                  s=200, c='#DC2626', marker='o',
                  edgecolors='black', linewidth=2.5, alpha=0.9,
                  label='Non-dominated\nsolutions', zorder=4)
    
    # ============================================================
    # 3. PARETO FRONTIER ÇİZGİSİ
    # ============================================================
    if len(df_quality) > 1:
        df_sorted = df_quality.sort_values('arrival')
        
        # Kırmızı eğri çizgi
        ax.plot(df_sorted['arrival'], df_sorted['waiting'],
               'r-', linewidth=3, alpha=0.8, zorder=3)
        
        # Pareto fronts etiketleri (her noktanın üstünde)
        for idx, row in df_sorted.iterrows():
            ax.annotate('', xy=(row['arrival'], row['waiting']),
                       xytext=(row['arrival'], row['waiting'] + 
                              (df_sorted['waiting'].max() - df_sorted['waiting'].min()) * 0.15),
                       arrowprops=dict(arrowstyle='->', color='black', 
                                     lw=1.5, alpha=0.7))
    
    # ============================================================
    # 4. MİNİMİZASYON YÖNÜ OKLARI (Sol-alt köşe)
    # ============================================================
    x_min = df_feasible['arrival'].min()
    x_max = df_feasible['arrival'].max()
    y_min = df_feasible['waiting'].min()
    y_max = df_feasible['waiting'].max()
    
    # Sol-alt köşeye yerleştir
    arrow_start_x = x_min - (x_max - x_min) * 0.05
    arrow_start_y = y_max + (y_max - y_min) * 0.05
    arrow_length = min(x_max - x_min, y_max - y_min) * 0.12
    
    # X ekseni oku (sola doğru)
    ax.annotate('', xy=(arrow_start_x - arrow_length, arrow_start_y),
               xytext=(arrow_start_x, arrow_start_y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax.text(arrow_start_x - arrow_length/2, arrow_start_y + (y_max-y_min)*0.02,
           r'$Min(f_1)$', fontsize=11, ha='center', style='italic')
    
    # Y ekseni oku (aşağı doğru)
    ax.annotate('', xy=(arrow_start_x, arrow_start_y - arrow_length),
               xytext=(arrow_start_x, arrow_start_y),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
    ax.text(arrow_start_x - (x_max-x_min)*0.03, arrow_start_y - arrow_length/2,
           r'$Min(f_2)$', fontsize=11, ha='center', style='italic', rotation=90)
    
    # ============================================================
    # 5. "Pareto fronts" ETİKETİ (Üst kısımda)
    # ============================================================
    # En sol 3 noktayı işaretle
    if len(df_quality) >= 3:
        df_top3 = df_sorted.head(3)
        mid_y = df_sorted['waiting'].max() + (y_max - y_min) * 0.18
        
        ax.text(df_top3['arrival'].mean(), mid_y,
               'Pareto fronts', fontsize=11, ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='black', linewidth=1))

# ============================================================
# 6. GRAFIK ÖZELLİKLERİ
# ============================================================
# Eksen etiketleri
ax.set_xlabel(r'$f_1$: Total Arrival Time (minutes)', 
             fontsize=13, fontweight='bold')
ax.set_ylabel(r'$f_2$: Total Waiting Time (minutes)', 
             fontsize=13, fontweight='bold')

# Grid - ince çizgiler
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#E5E7EB')
ax.set_axisbelow(True)

# Legend - sağ üst
ax.legend(fontsize=10, loc='upper right', framealpha=1, 
         edgecolor='black', fancybox=False, frameon=True)

# Eksen renkleri ve kalınlıkları
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(2)
    ax.spines[spine].set_color('black')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Tick ayarları
ax.tick_params(axis='both', which='major', labelsize=11, 
              width=1.5, length=6, direction='out')

# Margin ayarları
if not df_feasible.empty:
    x_range = df_feasible['arrival'].max() - df_feasible['arrival'].min()
    y_range = df_feasible['waiting'].max() - df_feasible['waiting'].min()
    
    ax.set_xlim(df_feasible['arrival'].min() - x_range*0.2,
                df_feasible['arrival'].max() + x_range*0.1)
    ax.set_ylim(df_feasible['waiting'].min() - y_range*0.1,
                df_feasible['waiting'].max() + y_range*0.25)

# Aspect ratio
ax.set_aspect('auto')

plt.tight_layout()

# Kaydet
plt.savefig(output_file, dpi=400, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"\n✓ Akademik stil grafik kaydedildi: {output_file}")
print("="*80)

plt.show()