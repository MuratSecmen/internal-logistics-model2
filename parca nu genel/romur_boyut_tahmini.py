"""
romur_boyut_tahmini.py
======================
Fikir:
  Aynı romurla (taşıma aracı) aynı teslim adresine yakın zamanda
  giden parçalar 20 m² yüzey kapasiteli bir araçta taşınmıştır.
  Boyutu bilinen parçalar kapasiteden düşülür; kalan alan boyutsuz
  parçalara eşit dağıtılarak YUZEY ALANI ve HACIM tahmin edilir.

Veri akışı:
  masalar.xlsx  (ARACID + TESLIM_ETME_TARIHI + TEBINA + MOVE_ORDER_DETAIL_NO)
       ↓  JOIN via SOIRNO = MOVE_ORDER_DETAIL_NO
  parça_2025 / parça_2026  (GAGE / WIDTH / LENGTH → boyut kaynağı)

Algoritma (her romur grubu için):
  1. SOIRNO join → boyutu bilinenler → yüzey alanı hesapla
  2. Kapasite_kalan = 200_000 cm² − bilinen_toplam
  3. Boyutsuz parça sayısı N_bilinmeyen
  4. Tahmin_yuzey  = kapasite_kalan / N_bilinmeyen  (cm²)
  5. Tahmin_gage   = DIMENSIONCODE grubundan medyan GAGE  (cm)
  6. Tahmin_width  = sqrt(tahmin_yuzey) × oran_w  (cm) — tipik en/boy oranı
  7. Tahmin_length = sqrt(tahmin_yuzey) × oran_l  (cm)

Kullanım:
  pip install pandas openpyxl scipy
  python romur_boyut_tahmini.py
"""

import pandas as pd
import numpy as np
import openpyxl
import warnings
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# ▶ KULLANICI AYARLARI
# ════════════════════════════════════════════════════════════

F_MASALAR = r"320420masalar.xlsx"
F_2025    = r"2025_parça_detayları.xlsx"
F_2026    = r"2026_taşınan_parça_nu_s.xlsx"
OUT       = r"romur_boyut_tahmini.xlsx"

ROMUR_KAPASITE_M2 = 20          # romur / taşıma aracı kapasitesi
ROMUR_KAPASITE_CM2 = ROMUR_KAPASITE_M2 * 10_000   # 200 000 cm²

# Zaman penceresi: aynı gün + bu saat farkı içindekiler aynı romur sayılır
# (masalar verisi doğrudan ARACID içerdiği için bu sadece SOIRNO gruplama için kullanılır)
ZAMAN_PENCERE_SAAT = 4

# Birim ayarları (parca_analiz.py ile aynı)
INCH_PROJECTS = {'GMH', 'LGM'}
MM_THRESHOLD  = 500
INCH_TO_CM    = 2.54
MM_TO_CM      = 0.10

# Tipik en/boy oranı (bilinmeyen parçalar için WIDTH/LENGTH bölme oranı)
# 1.0 = kare varsayımı. Değiştirebilirsiniz.
ORAN_WIDTH  = 1.0   # sqrt(YUZEY) × bu oran = WIDTH tahmini
ORAN_LENGTH = 1.0   # sqrt(YUZEY) × bu oran = LENGTH tahmini

# ════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ════════════════════════════════════════════════════════════
print("=" * 65)
print("Romur Boyut Tahmin Aracı")
print("=" * 65)

print("\n[1/6] Dosyalar yükleniyor...")
df_m  = pd.read_excel(F_MASALAR, sheet_name='Sheet1')
df25  = pd.read_excel(F_2025, sheet_name='son')
df26  = pd.read_excel(F_2026, sheet_name='PART NU_İLK 5 AY')

# ════════════════════════════════════════════════════════════
# 2. PARÇA VERİSİ TEMİZLİK + BİRİM DÖNÜŞÜMÜ
# ════════════════════════════════════════════════════════════
print("[2/6] Parça verileri hazırlanıyor (birim dönüşümü dahil)...")

def hazirla_parca(df, kaynak):
    df = df.copy()
    df['PROJE']     = df['PROJE'].astype(str).str.strip()
    df['MALZEMENO'] = df['MALZEMENO'].astype(str).str.strip()
    df['SOIRNO']    = df['SOIRNO'].astype(str).str.strip()
    df['KAYNAK']    = kaynak
    for c in ['GAGE', 'WIDTH', 'LENGTH']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def birim_donustur(row):
    p, g, w, l = row['PROJE'], row['GAGE'], row['WIDTH'], row['LENGTH']
    if p in INCH_PROJECTS:
        return g * INCH_TO_CM, w * INCH_TO_CM, l * INCH_TO_CM, 'inç→cm'
    elif pd.notna(w) and pd.notna(l) and (w > MM_THRESHOLD or l > MM_THRESHOLD):
        return g * MM_TO_CM, w * MM_TO_CM, l * MM_TO_CM, 'mm→cm'
    return g, w, l, 'cm'

df25 = hazirla_parca(df25, '2025')
df26 = hazirla_parca(df26, '2026')

for df in [df25, df26]:
    res = df.apply(birim_donustur, axis=1, result_type='expand')
    res.columns = ['GAGE_cm', 'WIDTH_cm', 'LENGTH_cm', 'BIRIM']
    df[['GAGE_cm', 'WIDTH_cm', 'LENGTH_cm', 'BIRIM']] = res

# Tüm parçaları birleştir
parca_all = pd.concat([df25, df26], ignore_index=True)

# Boyutlu / boyutsuz ayır
has_dim = (
    parca_all['GAGE_cm'].notna() &
    parca_all['WIDTH_cm'].notna() &
    parca_all['LENGTH_cm'].notna() &
    (parca_all['GAGE_cm'] >= 0.01) &
    (parca_all['WIDTH_cm'] >= 0.1) &
    (parca_all['LENGTH_cm'] >= 0.1)
)
parca_boyutlu  = parca_all[has_dim].copy()
parca_boyutsuz = parca_all[~has_dim].copy()

parca_boyutlu['YUZEY_cm2'] = (parca_boyutlu['WIDTH_cm'] * parca_boyutlu['LENGTH_cm']).round(2)
parca_boyutlu['HACIM_cm3'] = (parca_boyutlu['GAGE_cm'] * parca_boyutlu['YUZEY_cm2']).round(2)

print(f"  Boyutlu parça satırı   : {len(parca_boyutlu):,}")
print(f"  Boyutsuz parça satırı  : {len(parca_boyutsuz):,}")

# ════════════════════════════════════════════════════════════
# 3. ROMUR GRUPLARI — MASALAR DOSYASINDAN
# ════════════════════════════════════════════════════════════
print("[3/6] Romur grupları oluşturuluyor...")

df_m['MOVE_ORDER_DETAIL_NO'] = df_m['MOVE_ORDER_DETAIL_NO'].astype(str).str.strip()
df_m['TEBINA']   = df_m['TEBINA'].astype(str).str.strip()
df_m['TESLIM_DT'] = pd.to_datetime(df_m['TESLIM_ETME_TARIHI'], errors='coerce')
df_m['TESLIM_GUN']= df_m['TESLIM_DT'].dt.date

# Her romur: ARACID + TARİH + TEBINA
romur_grp = df_m.groupby(['ARACID', 'TESLIM_GUN', 'TEBINA']).agg(
    N_PARCA      = ('MOVE_ORDER_DETAIL_NO', 'count'),
    SOIRNO_LIST  = ('MOVE_ORDER_DETAIL_NO', lambda x: list(x.dropna().unique())),
    ILK_TESLIM   = ('TESLIM_DT', 'min'),
    SON_TESLIM   = ('TESLIM_DT', 'max'),
).reset_index()

romur_grp = romur_grp[romur_grp['N_PARCA'] > 0].copy()
romur_grp.insert(0, 'ROMUR_ID', range(1, len(romur_grp) + 1))

print(f"  Aktif romur sayısı: {len(romur_grp)}")

# ════════════════════════════════════════════════════════════
# 4. HER ROMUR İÇİN BOYUT HESABI + TAHMİN
# ════════════════════════════════════════════════════════════
print("[4/6] Romur bazlı boyut analizi ve tahmin yapılıyor...")

# SOIRNO → boyut lookup (bir SOIRNO birden fazla satırda olabilir, en büyüğü al)
parca_lookup = (
    parca_boyutlu
    .sort_values('YUZEY_cm2', ascending=False)
    .groupby('SOIRNO')
    .agg(
        MALZEMENO    = ('MALZEMENO', 'first'),
        PROJE        = ('PROJE', 'first'),
        GAGE_cm      = ('GAGE_cm', 'max'),
        WIDTH_cm     = ('WIDTH_cm', 'max'),
        LENGTH_cm    = ('LENGTH_cm', 'max'),
        YUZEY_cm2    = ('YUZEY_cm2', 'max'),
        HACIM_cm3    = ('HACIM_cm3', 'max'),
        BIRIM        = ('BIRIM', 'first'),
        DIMENSIONCODE= ('DIMENSIONCODE', 'first'),
    )
    .reset_index()
)

# DIMENSIONCODE bazında medyan GAGE (bilinmeyen parçalar için gage tahmini)
gage_medyan = (
    parca_boyutlu
    .groupby('DIMENSIONCODE')['GAGE_cm']
    .median()
    .to_dict()
)
gage_medyan_global = parca_boyutlu['GAGE_cm'].median()

detay_rows = []

for _, romur in romur_grp.iterrows():
    romur_id   = romur['ROMUR_ID']
    aracid     = romur['ARACID']
    tebina     = romur['TEBINA']
    gun        = romur['TESLIM_GUN']
    soirno_lst = romur['SOIRNO_LIST']
    n_parca    = romur['N_PARCA']

    # Her SOIRNO için boyut bilgisi bul
    for soirno in soirno_lst:
        hit = parca_lookup[parca_lookup['SOIRNO'] == soirno]
        if not hit.empty:
            row = hit.iloc[0]
            detay_rows.append({
                'ROMUR_ID': romur_id, 'ARACID': aracid,
                'TEBINA': tebina, 'TESLIM_GUN': gun,
                'SOIRNO': soirno,
                'MALZEMENO': row['MALZEMENO'],
                'PROJE': row['PROJE'],
                'BIRIM': row['BIRIM'],
                'DIMENSIONCODE': row['DIMENSIONCODE'],
                'GAGE_cm': row['GAGE_cm'],
                'WIDTH_cm': row['WIDTH_cm'],
                'LENGTH_cm': row['LENGTH_cm'],
                'YUZEY_cm2': row['YUZEY_cm2'],
                'HACIM_cm3': row['HACIM_cm3'],
                'BOYUT_DURUMU': 'BİLİNİYOR',
                'TAHMİN_NOTU': '',
            })
        else:
            # Boyutsuz — masalar verisinde MALZEMENO yok; SOIRNO bilgisi tutulur
            detay_rows.append({
                'ROMUR_ID': romur_id, 'ARACID': aracid,
                'TEBINA': tebina, 'TESLIM_GUN': gun,
                'SOIRNO': soirno,
                'MALZEMENO': '— (sistemde boyut yok)',
                'PROJE': '', 'BIRIM': '',
                'DIMENSIONCODE': '',
                'GAGE_cm': np.nan, 'WIDTH_cm': np.nan, 'LENGTH_cm': np.nan,
                'YUZEY_cm2': np.nan, 'HACIM_cm3': np.nan,
                'BOYUT_DURUMU': 'TAHMİN GEREKLİ',
                'TAHMİN_NOTU': '',
            })

df_detay = pd.DataFrame(detay_rows)

# ── Romur bazlı tahmin ────────────────────────────────────────
tahmin_rows = []

for romur_id, grp in df_detay.groupby('ROMUR_ID'):
    bilinen  = grp[grp['BOYUT_DURUMU'] == 'BİLİNİYOR']
    bilinmeyen = grp[grp['BOYUT_DURUMU'] == 'TAHMİN GEREKLİ']

    toplam_bilinen_yuzey = bilinen['YUZEY_cm2'].sum()
    kalan_kapasite = max(0, ROMUR_KAPASITE_CM2 - toplam_bilinen_yuzey)
    n_bilinmeyen = len(bilinmeyen)

    if n_bilinmeyen > 0:
        tahmin_yuzey = round(kalan_kapasite / n_bilinmeyen, 2)
        tahmin_kenar = round(np.sqrt(tahmin_yuzey), 2)  # kare varsayımı
        tahmin_width  = round(tahmin_kenar * ORAN_WIDTH,  2)
        tahmin_length = round(tahmin_kenar * ORAN_LENGTH, 2)
        tahmin_gage   = round(gage_medyan.get('', gage_medyan_global), 3)
        tahmin_hacim  = round(tahmin_gage * tahmin_yuzey, 2)

        notu = (f"Kapasite={ROMUR_KAPASITE_M2}m²; "
                f"Bilinen toplam={toplam_bilinen_yuzey/10000:.2f}m²; "
                f"Kalan={kalan_kapasite/10000:.2f}m²; "
                f"÷{n_bilinmeyen} parça → {tahmin_yuzey/10000:.2f}m²/parça")

        for idx in bilinmeyen.index:
            df_detay.at[idx, 'GAGE_cm']    = tahmin_gage
            df_detay.at[idx, 'WIDTH_cm']   = tahmin_width
            df_detay.at[idx, 'LENGTH_cm']  = tahmin_length
            df_detay.at[idx, 'YUZEY_cm2']  = tahmin_yuzey
            df_detay.at[idx, 'HACIM_cm3']  = tahmin_hacim
            df_detay.at[idx, 'BOYUT_DURUMU'] = 'TAHMİN'
            df_detay.at[idx, 'TAHMİN_NOTU'] = notu

    # Romur özet satırı
    toplam_son = df_detay[df_detay['ROMUR_ID'] == romur_id]['YUZEY_cm2'].sum()
    doluluk_pct = round(toplam_son / ROMUR_KAPASITE_CM2 * 100, 1)
    tahmin_rows.append({
        'ROMUR_ID': romur_id,
        'ARACID': grp['ARACID'].iloc[0],
        'TEBINA': grp['TEBINA'].iloc[0],
        'TESLIM_GUN': grp['TESLIM_GUN'].iloc[0],
        'N_PARCA_TOPLAM': len(grp),
        'N_BOYUTLU': len(bilinen),
        'N_TAHMİN': n_bilinmeyen,
        'BİLİNEN_YUZEY_m2': round(toplam_bilinen_yuzey / 10000, 3),
        'KALAN_KAPASİTE_m2': round(kalan_kapasite / 10000, 3),
        'TAHMİN_YUZEY_m2': round(kalan_kapasite / 10000 / max(n_bilinmeyen, 1), 3) if n_bilinmeyen else 0,
        'TOPLAM_YUZEY_m2': round(toplam_son / 10000, 3),
        'DOLULUK_PCT': doluluk_pct,
    })

df_romur_ozet = pd.DataFrame(tahmin_rows)

print(f"  Analiz edilen romur : {len(df_romur_ozet)}")
print(f"  Tahmin yapılan parça: {(df_detay['BOYUT_DURUMU']=='TAHMİN').sum()}")

# ════════════════════════════════════════════════════════════
# 5. SOIRNO BAZLI GRUPLAMA (masalar olmadan)
# ════════════════════════════════════════════════════════════
print("[5/6] SOIRNO bazlı tahmin (masalar olmadan alternatif)...")

# Aynı BLDNG + aynı gün → aynı romur varsayımı
parca_all['PRODCOMPLETEDDATE'] = pd.to_datetime(
    parca_all['PRODCOMPLETEDDATE'], errors='coerce')
parca_all['TESLIM_GUN'] = parca_all['PRODCOMPLETEDDATE'].dt.date

soirno_grp = parca_all.groupby(['SOIRNO', 'BLDNG', 'TESLIM_GUN']).agg(
    N_SATIR       = ('MALZEMENO', 'count'),
    BILINEN_YUZEY = (
        'YUZEY_cm2' if 'YUZEY_cm2' in parca_all.columns else 'MALZEMENO',
        lambda x: x.sum() if 'YUZEY_cm2' in parca_all.columns else 0
    ),
    MALZEME_LIST  = ('MALZEMENO', lambda x: ', '.join(sorted(set(x.astype(str).str.strip()))[:3])),
).reset_index()

# ════════════════════════════════════════════════════════════
# 6. EXCEL ÇIKTISI
# ════════════════════════════════════════════════════════════
print(f"[6/6] Excel kaydediliyor → {OUT}")

DARK='1F3864'; MID='2E75B6'; LIGHT='D6E4F7'
GREEN='375623'; GBKG='E2EFDA'; OBKG='FCE4D6'; ORG='833C00'
WHT='FFFFFF'; ALT='EBF3FB'; GRY='F2F2F2'; YEL='FFF2CC'

t=Side(style='thin',color='AAAAAA'); m=Side(style='medium',color=MID)
bdr=Border(left=t,right=t,top=t,bottom=t); mbdr=Border(left=m,right=m,top=m,bottom=m)
def F(c,s=10,b=False): return Font(name='Arial',size=s,bold=b,color=c)
def P(c): return PatternFill('solid',fgColor=c)
def A(h='center',w=True): return Alignment(horizontal=h,vertical='center',wrap_text=w)

def title(ws,txt,n,r=1):
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=n)
    c=ws.cell(row=r,column=1,value=txt)
    c.font=F(WHT,13,True);c.fill=P(DARK);c.alignment=A('center',False)
    ws.row_dimensions[r].height=26

def subhd(ws,txt,n,r):
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=n)
    c=ws.cell(row=r,column=1,value=txt)
    c.font=F(WHT,11,True);c.fill=P(MID);c.alignment=A('left',False)
    ws.row_dimensions[r].height=20

def info(ws,txt,n,r,bg=LIGHT,tc='000000'):
    ws.merge_cells(start_row=r,start_column=1,end_row=r,end_column=n)
    c=ws.cell(row=r,column=1,value=txt)
    c.font=Font(name='Arial',size=10,color=tc,italic=True)
    c.fill=P(bg);c.alignment=A('left',True);c.border=mbdr
    ws.row_dimensions[r].height=30

def hdr(ws,headers,widths,row):
    for i,(h,w) in enumerate(zip(headers,widths),1):
        c=ws.cell(row=row,column=i,value=h)
        c.font=F(WHT,10,True);c.fill=P(MID)
        c.alignment=A('center',True);c.border=bdr
        ws.column_dimensions[get_column_letter(i)].width=w
    ws.row_dimensions[row].height=28

def datarows(ws,rows,start):
    for ri,row in enumerate(rows,start):
        for ci,val in enumerate(row,1):
            c=ws.cell(row=ri,column=ci,value=val)
            bg=ALT if ri%2==0 else WHT
            # TAHMİN satırları sarı, BİLİNİYOR yeşil
            if isinstance(val,str) and val=='TAHMİN': bg=YEL
            elif isinstance(val,str) and val=='BİLİNİYOR': bg=GBKG
            c.font=F('000000',10);c.fill=P(bg)
            c.alignment=A('left',True);c.border=bdr
            ws.row_dimensions[ri].height=14
            if isinstance(val,float) and not isinstance(val,bool):
                c.number_format='#,##0.00'
    return ri

wb=openpyxl.Workbook();wb.remove(wb.active)

# ── Sekme 1: Romur Özeti ──────────────────────────────────────
ws1=wb.create_sheet("Romur Özeti")
ws1.sheet_view.showGridLines=False;ws1.freeze_panes='A4'
NC=12
title(ws1,"Romur Bazlı Yüzey Alanı Özeti",NC,1)
info(ws1,
    f"Romur kapasitesi: {ROMUR_KAPASITE_M2} m²  |  "
    "Kaynak: masalar dosyası (ARACID + tarih + teslim binası)  |  "
    "Doluluk = Toplam Yüzey / 200 000 cm²  |  Sarı = tahmin yapıldı",
    NC,2,LIGHT)
hdr(ws1,
    ["ROMUR_ID","ARACID","TESLİM BİNASI","TESLİM GÜNÜ",
     "PARCA (Toplam)","Boyutlu","Tahmin","Bilinen (m²)",
     "Kalan Kap. (m²)","Tahmin/parça (m²)","Toplam (m²)","Doluluk %"],
    [8,16,20,12,9,8,8,12,14,16,11,10],3)
oz_cols=['ROMUR_ID','ARACID','TEBINA','TESLIM_GUN','N_PARCA_TOPLAM',
         'N_BOYUTLU','N_TAHMİN','BİLİNEN_YUZEY_m2','KALAN_KAPASİTE_m2',
         'TAHMİN_YUZEY_m2','TOPLAM_YUZEY_m2','DOLULUK_PCT']
datarows(ws1,[tuple(r) for r in df_romur_ozet[oz_cols].itertuples(index=False)],4)

# ── Sekme 2: Parça Detay ─────────────────────────────────────
ws2=wb.create_sheet("Parça Detay + Tahmin")
ws2.sheet_view.showGridLines=False;ws2.freeze_panes='A4'
NC2=14
title(ws2,"Parça Bazlı Boyut Detayı (Bilinen & Tahmin)",NC2,1)
info(ws2,
    "Yeşil satır = boyut sistemde mevcut (parça dosyasından)  |  "
    "Sarı satır = tahmin (romur kapasitesinden dağıtım)  |  "
    "TAHMİN_NOTU sütununda hesap detayı var",
    NC2,2,GBKG,GREEN)
hdr(ws2,
    ["ROMUR_ID","ARACID","TESLİM BİNASI","SOIRNO","MALZEMENO","PROJE",
     "BOYUT_DURUMU","GAGE (cm)","WIDTH (cm)","LENGTH (cm)",
     "YÜZEY (cm²)","YÜZEY (m²)","HACİM (cm³)","TAHMİN_NOTU"],
    [8,16,18,11,25,9,12,9,9,10,12,10,13,35],3)

def to_m2(v): return round(v/10000,4) if pd.notna(v) and v else np.nan

det_rows=[]
for _,r in df_detay.iterrows():
    det_rows.append((
        r['ROMUR_ID'],r['ARACID'],r['TEBINA'],r['SOIRNO'],
        r['MALZEMENO'],r['PROJE'],r['BOYUT_DURUMU'],
        r['GAGE_cm'],r['WIDTH_cm'],r['LENGTH_cm'],
        r['YUZEY_cm2'],to_m2(r['YUZEY_cm2']),r['HACIM_cm3'],r['TAHMİN_NOTU']
    ))
datarows(ws2,det_rows,4)

# ── Sekme 3: Metodoloji ────────────────────────────────────────
ws3=wb.create_sheet("Metodoloji")
ws3.sheet_view.showGridLines=False
NC3=6
title(ws3,"Romur Boyut Tahmin Metodolojisi",NC3,1)
rows_m=[
    ("ADIM","AÇIKLAMA","FORMÜL / KURAL"),
    ("1","Romur grupları oluştur",
     "ARACID + TESLIM_ETME_TARIHI + TEBINA → her grup = 1 romur"),
    ("2","Parça boyutlarını join et",
     "masalar.MOVE_ORDER_DETAIL_NO = parça_dosyası.SOIRNO"),
    ("3","Bilinen yüzey alanı topla",
     "Σ (WIDTH_cm × LENGTH_cm) — boyutu sisteme girilmiş parçalar"),
    ("4","Kalan kapasiteyi bul",
     f"Kapasite_kalan = {ROMUR_KAPASITE_M2} m² × 10000 − bilinen_toplam_cm²"),
    ("5","Bilinmeyen boyutları tahmin et",
     "Tahmin_yuzey = kapasite_kalan ÷ N_boyutsuz_parca"),
    ("6","WIDTH ve LENGTH'i tahmin et",
     "WIDTH = LENGTH = √(tahmin_yuzey)  [kare varsayımı; oran değiştirilebilir]"),
    ("7","GAGE'i tahmin et",
     "Aynı DIMENSIONCODE grubunun medyan GAGE'i kullanılır"),
    ("8","HACİM hesapla",
     "HACIM_cm³ = GAGE_cm × YUZEY_cm²"),
    ("—","Validasyon",
     "DOLULUK_PCT > 100 ise veri hatası veya kapasite hatalı"),
]
hdr(ws3,["ADIM","AÇIKLAMA","FORMÜL / KURAL"],[8,35,55],2)
for ri,row in enumerate(rows_m[1:],3):
    bg=ALT if ri%2==0 else WHT
    for ci,val in enumerate(row,1):
        c=ws3.cell(row=ri,column=ci,value=val)
        c.font=F('000000',10);c.fill=P(bg)
        c.alignment=A('left',True);c.border=bdr
        ws3.row_dimensions[ri].height=16
info(ws3,
    "Kare varsayımı (WIDTH=LENGTH=√YUZEY) gerçek parçalar için hatalı olabilir.  "
    "Eğer parçanın tipik en/boy oranı biliniyorsa ORAN_WIDTH ve ORAN_LENGTH "
    "değişkenlerini kodun üst bölümünden güncelleyebilirsiniz.",
    NC3,len(rows_m)+3,OBKG,ORG)

wb.save(OUT)
print(f"\n✓ Tamamlandı → {OUT}")
print(f"  Sekmeler: 'Romur Özeti' | 'Parça Detay + Tahmin' | 'Metodoloji'")
print()

# Konsol özet
print("══ Romur Özeti ══")
print(df_romur_ozet[['ARACID','TESLIM_GUN','N_PARCA_TOPLAM',
                      'N_BOYUTLU','N_TAHMİN','BİLİNEN_YUZEY_m2',
                      'TOPLAM_YUZEY_m2','DOLULUK_PCT']].to_string(index=False))
