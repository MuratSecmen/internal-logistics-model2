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
# AKADEMİK STIL PARETO FRONTIER GRAFİĞİ
# =====================================================================
print("="*80)
print("AKADEMİK PARETO FRONTIER ÇİZİLİYOR")
print("="*80)

# Grafik ayarları - Akademik stil
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(10, 8))

if not df_feasible.empty:
    
    # Kaliteli çözümleri ayır
    df_quality = df_feasible[df_feasible['mip_gap'] < 5.0].copy()
    df_poor = df_feasible[df_feasible['mip_gap'] >= 5.0].copy()
    
    # ============================================================
    # 1. DOMINATED BÖLGE (Gri arka plan)
    # ============================================================
    if len(df_quality) > 0:
        # Pareto noktalarının üst-sağ bölgesi dominated
        max_arrival = df_quality['arrival'].max()
        max_waiting = df_quality['waiting'].max()
        
        # Dominated bölgeyi göster
        x_extend = (max_arrival - df_quality['arrival'].min()) * 0.3
        y_extend = (max_waiting - df_quality['waiting'].min()) * 0.3
        
        ax.fill_between([max_arrival, max_arrival + x_extend], 
                       [max_waiting, max_waiting],
                       [max_waiting + y_extend, max_waiting + y_extend],
                       alpha=0.1, color='gray', zorder=0,
                       label='Dominated Region')
    
    # ============================================================
    # 2. PARETO FRONTIER ÇİZGİSİ (Kalın, vurgulu)
    # ============================================================
    if len(df_quality) > 1:
        df_sorted = df_quality.sort_values('arrival')
        
        # Kalın kırmızı çizgi
        ax.plot(df_sorted['arrival'], df_sorted['waiting'],
               'r-', linewidth=3.5, alpha=0.8,
               label='Pareto Frontier', zorder=4)
        
        # Gölge efekti
        ax.plot(df_sorted['arrival'], df_sorted['waiting'],
               'r-', linewidth=6, alpha=0.2, zorder=3)
    
    # ============================================================
    # 3. PARETO OPTIMAL NOKTALAR
    # ============================================================
    # Kaliteli çözümler (büyük noktalar)
    if not df_quality.empty:
        ax.scatter(df_quality['arrival'], df_quality['waiting'],
                  s=200, c='#1e3a8a', marker='o', 
                  edgecolors='black', linewidth=2, alpha=0.8,
                  label='Pareto Optimal Solutions', zorder=5)
    
    # Düşük kaliteli çözümler (küçük, yarı saydam)
    if not df_poor.empty:
        ax.scatter(df_poor['arrival'], df_poor['waiting'],
                  s=150, c='#dc2626', marker='s', 
                  edgecolors='black', linewidth=1.5, alpha=0.5,
                  label='Suboptimal Solutions', zorder=5)
    
    # ============================================================
    # 4. EPSILON ETİKETLERİ (Profesyonel)
    # ============================================================
    for idx, row in df_feasible.iterrows():
        # Etiket pozisyonu - offset ayarla
        if row['mip_gap'] < 5.0:
            xytext = (15, -15)
            bbox_color = 'white'
            fontsize = 9
        else:
            xytext = (15, 15)
            bbox_color = 'lightyellow'
            fontsize = 8
        
        ax.annotate(f"ε={row['epsilon']:.0f}",
                   (row['arrival'], row['waiting']),
                   textcoords="offset points",
                   xytext=xytext,
                   ha='left', fontsize=fontsize,
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=bbox_color, 
                           edgecolor='gray',
                           alpha=0.7),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0.2',
                                 color='gray', lw=1, alpha=0.5))
    
    # ============================================================
    # 5. MİNİMİZASYON YÖNÜ OKLARI
    # ============================================================
    # Sol-alt köşeye oklar ekle
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    arrow_x = x_min + (x_max - x_min) * 0.05
    arrow_y = y_max - (y_max - y_min) * 0.05
    arrow_length = (x_max - x_min) * 0.08
    
    # X ekseni oku (sol)
    ax.arrow(arrow_x + arrow_length, arrow_y, -arrow_length, 0,
            head_width=(y_max-y_min)*0.02, head_length=(x_max-x_min)*0.015,
            fc='green', ec='green', linewidth=2, alpha=0.7, zorder=2)
    ax.text(arrow_x, arrow_y, 'Better', fontsize=10, 
           color='green', fontweight='bold', va='center')
    
    # Y ekseni oku (aşağı)
    ax.arrow(arrow_x, arrow_y - arrow_length, 0, arrow_length,
            head_width=(x_max-x_min)*0.02, head_length=(y_max-y_min)*0.015,
            fc='green', ec='green', linewidth=2, alpha=0.7, zorder=2)
    ax.text(arrow_x, arrow_y - arrow_length*2, 'Better', fontsize=10,
           color='green', fontweight='bold', ha='center', rotation=90)

# ============================================================
# 6. GRAFIK ÖZELLİKLERİ (Akademik Stil)
# ============================================================
ax.set_xlabel('$f_1$: Total Arrival Time (minutes)', 
             fontsize=13, fontweight='bold', family='serif')
ax.set_ylabel('$f_2$: Total Waiting Time (minutes)', 
             fontsize=13, fontweight='bold', family='serif')
ax.set_title('Pareto Frontier for Multi-Objective Optimization\n' +
            'Minimizing Arrival Time vs Waiting Time',
             fontsize=14, fontweight='bold', family='serif', pad=20)

# Grid - akademik stil
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
ax.set_axisbelow(True)

# Legend - sağ üst köşe, dışarıda
ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1), 
         framealpha=0.95, edgecolor='black', fancybox=False)

# Tick'ler
ax.tick_params(axis='both', which='major', labelsize=10)

# Kenarlıklar
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

# Margin
if not df_feasible.empty:
    x_range = df_feasible['arrival'].max() - df_feasible['arrival'].min()
    y_range = df_feasible['waiting'].max() - df_feasible['waiting'].min()
    
    ax.set_xlim(df_feasible['arrival'].min() - x_range*0.15,
                df_feasible['arrival'].max() + x_range*0.15)
    ax.set_ylim(df_feasible['waiting'].min() - y_range*0.15,
                df_feasible['waiting'].max() + y_range*0.15)

# Aspect ratio - square görünüm
ax.set_aspect('auto')

plt.tight_layout()

# Kaydet - yüksek çözünürlük
plt.savefig(output_file, dpi=400, bbox_inches='tight', facecolor='white')
print(f"\n✓ Akademik stil grafik kaydedildi: {output_file}")
print("="*80)

plt.show()