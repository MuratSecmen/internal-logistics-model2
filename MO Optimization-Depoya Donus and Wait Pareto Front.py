import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# =====================================================================
# AYARLAR
# =====================================================================
folder = r"C:\Users\Asus\results\Pareto"  # Excel dosyaları burda
output_file = os.path.join(folder, "pareto_frontier.png")  # Grafik buraya

# =====================================================================
# TÜM EXCEL DOSYALARINI OKU
# =====================================================================
print("="*80)
print("EXCEL DOSYALARI OKUNUYOR")
print("="*80)

# Tüm .xlsx dosyalarını bul
excel_files = glob.glob(os.path.join(folder, "*.xlsx"))
print(f"Bulunan dosya sayısı: {len(excel_files)}\n")

data = []

for file in excel_files:
    try:
        filename = os.path.basename(file)
        print(f"Okunuyor: {filename}")
        
        # optimization_results sekmesini oku
        df = pd.read_excel(file, sheet_name='optimization_results')
        
        # Verileri çek
        epsilon = df['epsilon_wait_upper'].values[0]
        arrival = df['total_arrival_times'].values[0]   # X ekseni (1. amaç - depoya dönüş)
        waiting = df['total_wait_minutes'].values[0]    # Y ekseni (2. amaç - bekleme)
        status = df['status'].values[0]
        mip_gap = df['mip_gap'].values[0] * 100
        
        # Status yorumla
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
            'arrival': arrival,      # X ekseni (total_arrival_times)
            'waiting': waiting,      # Y ekseni (total_wait_minutes)
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

# Epsilon'a göre sırala
df = df.sort_values('epsilon', ascending=False).reset_index(drop=True)

# Feasible/Infeasible ayır
df_feasible = df[df['status'].isin(['OPTIMAL', 'SUBOPTIMAL'])].copy()
df_infeasible = df[df['status'] == 'INFEASIBLE'].copy()

print("="*80)
print("VERİ DURUMU")
print("="*80)
print(f"Feasible: {len(df_feasible)}")
print(f"Infeasible: {len(df_infeasible)}")
print()

# Feasible verileri göster
if not df_feasible.empty:
    print("FEASIBLE ÇÖZÜMLER:")
    print(df_feasible[['epsilon', 'arrival', 'waiting', 'mip_gap', 'status']].to_string())
    print()

# =====================================================================
# PARETO FRONTIER ÇİZ
# =====================================================================
print("="*80)
print("PARETO FRONTIER ÇİZİLİYOR")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 10))

# Feasible noktalar
if not df_feasible.empty:
    # Kaliteye göre renk
    excellent = df_feasible[df_feasible['mip_gap'] < 1.0]
    good = df_feasible[(df_feasible['mip_gap'] >= 1.0) & (df_feasible['mip_gap'] < 5.0)]
    poor = df_feasible[df_feasible['mip_gap'] >= 5.0]
    
    if not excellent.empty:
        ax.scatter(excellent['arrival'], excellent['waiting'],
                  s=250, c='#2E86DE', marker='o', alpha=0.7,
                  edgecolors='black', linewidth=2,
                  label='Excellent (gap<1%)', zorder=3)
    
    if not good.empty:
        ax.scatter(good['arrival'], good['waiting'],
                  s=250, c='#FFA502', marker='o', alpha=0.7,
                  edgecolors='black', linewidth=2,
                  label='Good (1%≤gap<5%)', zorder=3)
    
    if not poor.empty:
        ax.scatter(poor['arrival'], poor['waiting'],
                  s=250, c='#EE5A6F', marker='o', alpha=0.7,
                  edgecolors='black', linewidth=2,
                  label='Poor (gap≥5%)', zorder=3)
    
    # Pareto frontier çizgisi (sadece kaliteli çözümler)
    df_quality = df_feasible[df_feasible['mip_gap'] < 5.0].copy()
    if len(df_quality) > 1:
        df_sorted = df_quality.sort_values('arrival')
        ax.plot(df_sorted['arrival'], df_sorted['waiting'],
               'r--', alpha=0.6, linewidth=2.5,
               label='Pareto Frontier', zorder=2)
    
    # Epsilon etiketleri
    for _, row in df_feasible.iterrows():
        ax.annotate(f"ε={row['epsilon']:.0f}",
                   (row['arrival'], row['waiting']),
                   textcoords="offset points",
                   xytext=(50, 5),
                   ha='left', fontsize=10, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='yellow', alpha=0.4))

# Infeasible gösterimi (varsa)
if not df_infeasible.empty:
    # Infeasible için arrival/waiting değerleri olmayabilir
    # Bu durumda sadece bilgi olarak gösterelim
    print(f"NOT: {len(df_infeasible)} infeasible çözüm var (grafikte gösterilemiyor)")

# Grafik özellikleri
ax.set_xlabel('1st Objective: Total Arrival Time (minutes)', 
             fontsize=14, fontweight='bold')
ax.set_ylabel('2nd Objective: Total Waiting Time (minutes)', 
             fontsize=14, fontweight='bold')
ax.set_title('Pareto Frontier: Epsilon-Constraint Method\n' +
            'Arrival Time (1st Obj) vs Waiting Time (2nd Obj)',
             fontsize=16, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best', framealpha=0.9)

# Margin ekle
if not df_feasible.empty:
    x_range = df_feasible['arrival'].max() - df_feasible['arrival'].min()
    y_range = df_feasible['waiting'].max() - df_feasible['waiting'].min()
    
    ax.set_xlim(df_feasible['arrival'].min() - x_range*0.1,
                df_feasible['arrival'].max() + x_range*0.1)
    ax.set_ylim(df_feasible['waiting'].min() - y_range*0.1,
                df_feasible['waiting'].max() + y_range*0.1)

plt.tight_layout()

# Kaydet
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Grafik kaydedildi: {output_file}")
print("="*80)

plt.show()