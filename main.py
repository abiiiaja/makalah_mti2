import numpy as np
import pandas as pd

# ============================================================
# TAHAP 1: REPRESENTASI DATA DOKTER SEBAGAI VEKTOR (d)
# ============================================================
data_dokter = {
    'Kabupaten': [
        'BANGKALAN', 'BANYUWANGI', 'BLITAR', 'BOJONEGORO', 'BONDOWOSO',
        'GRESIK', 'JEMBER', 'JOMBANG', 'KEDIRI', 'KOTA BATU', 'KOTA BLITAR',
        'KOTA KEDIRI', 'KOTA MADIUN', 'KOTA MALANG', 'KOTA MOJOKERTO',
        'KOTA PASURUAN', 'KOTA PROBOLINGGO', 'KOTA SURABAYA', 'LAMONGAN',
        'LUMAJANG', 'MADIUN', 'MAGETAN', 'MALANG', 'MOJOKERTO', 'NGANJUK',
        'NGAWI', 'PACITAN', 'PAMEKASAN', 'PASURUAN', 'PONOROGO', 'PROBOLINGGO',
        'SAMPANG', 'SIDOARJO', 'SITUBONDO', 'SUMENEP', 'TRENGGALEK', 'TUBAN',
        'TULUNGAGUNG'
    ],
    'Sp.A':   [4,4,3,7,3,5,7,6,5,3,3,6,6,6,4,3,4,58,7,6,6,3,8,8,5,4,3,5,8,4,4,5,7,4,7,4,6,4],
    'Sp.B':   [3,7,4,8,2,4,11,5,4,4,2,5,7,2,3,2,6,38,7,5,5,2,6,6,5,6,2,9,4,4,5,4,6,4,2,3,5,7],
    'Sp.OG':  [4,4,4,8,3,5,9,5,8,4,3,6,5,8,3,3,4,56,7,4,6,3,7,8,7,6,2,6,8,6,7,5,7,5,7,4,7,7],
    'Sp.PD':  [4,5,9,11,3,10,13,9,6,7,4,10,9,14,3,3,8,70,11,8,7,4,7,10,7,7,3,8,11,3,7,3,14,8,5,6,7,7],
    'Sp.An':  [4,9,4,8,2,5,6,8,5,3,2,5,5,5,3,2,3,81,6,4,5,2,6,8,4,3,2,7,6,4,6,4,8,5,5,4,7,6],
    'Sp.Rad': [3,2,3,7,3,4,7,4,2,3,2,5,5,5,2,1,4,21,5,4,2,2,5,3,3,3,2,7,8,3,5,3,7,4,2,3,6,3],
    'Sp.PK':  [2,2,2,5,1,3,5,4,3,3,2,4,3,3,2,1,2,26,6,3,3,2,3,4,2,4,2,4,2,3,3,1,2,3,2,3,4,3]
}

df_dokter = pd.DataFrame(data_dokter)
SPESIALIS = ['Sp.A', 'Sp.B', 'Sp.OG', 'Sp.PD', 'Sp.An', 'Sp.Rad', 'Sp.PK']

# ============================================================
# TAHAP 2: DATA POPULASI RIIL BPS PER KABUPATEN/KOTA
# ============================================================
populasi_bps = {
    'BANGKALAN':      1060377, 'BANYUWANGI':     1708114, 'BLITAR':         1223745, 'BOJONEGORO':     1301635,
    'BONDOWOSO':        776151, 'GRESIK':         1311215, 'JEMBER':         2536729, 'JOMBANG':        1318062,
    'KEDIRI':         1635294, 'KOTA BATU':        213046, 'KOTA BLITAR':      149149, 'KOTA KEDIRI':      286796,
    'KOTA MADIUN':      195175, 'KOTA MALANG':      843810, 'KOTA MOJOKERTO':   132434, 'KOTA PASURUAN':    208006,
    'KOTA PROBOLINGGO': 239649, 'KOTA SURABAYA':  2874314, 'LAMONGAN':       1344165, 'LUMAJANG':       1119251,
    'MADIUN':           744350, 'MAGETAN':          670812, 'MALANG':         2654448, 'MOJOKERTO':      1119209,
    'NGANJUK':        1103902, 'NGAWI':            870057, 'PACITAN':          586110, 'PAMEKASAN':        850057,
    'PASURUAN':       1605969, 'PONOROGO':         949318, 'PROBOLINGGO':    1152537, 'SAMPANG':          969694,
    'SIDOARJO':       2082801, 'SITUBONDO':        685967, 'SUMENEP':        1124436, 'TRENGGALEK':       731125,
    'TUBAN':          1198012, 'TULUNGAGUNG':    1089775
}

wilayah_list = df_dokter['Kabupaten'].tolist()
total_populasi_provinsi = sum(populasi_bps.values())

# ============================================================
# TAHAP 3: VEKTOR DATA SIMULASI PASIEN (p)
# ============================================================
np.random.seed(42)
N_SPESIALIS = len(SPESIALIS)
data_pasien_sim = {'Kabupaten': wilayah_list}

for sp in SPESIALIS:
    pasien_per_wilayah = []
    for wil in wilayah_list:
        pop = populasi_bps[wil]
        batas_atas = max(pop // N_SPESIALIS, 1)
        nilai = np.random.randint(1, batas_atas + 1)
        pasien_per_wilayah.append(nilai)
    data_pasien_sim[sp] = pasien_per_wilayah

df_pasien = pd.DataFrame(data_pasien_sim)

# ============================================================
# TAHAP 4: NORMALISASI SKALAR (PENERAPAN RASIO IDEAL 1:3571)
# ============================================================
RASIO_IDEAL = 3571
df_pasien_ideal = df_pasien.copy()
for sp in SPESIALIS:
    df_pasien_ideal[sp] = np.ceil(df_pasien_ideal[sp] / RASIO_IDEAL)

# ============================================================
# TAHAP 5: ANALISIS KESELARASAN DENGAN COSINE SIMILARITY
# ============================================================
def cosine_similarity(p, d):
    dot_product = np.dot(p, d)
    norm_p = np.linalg.norm(p)
    norm_d = np.linalg.norm(d)
    return dot_product / (norm_p * norm_d) if norm_p > 0 and norm_d > 0 else 0.0

cos_sim_scores = []
for i in range(len(wilayah_list)):
    p = df_pasien_ideal[SPESIALIS].iloc[i].to_numpy(dtype=float)
    d = df_dokter[SPESIALIS].iloc[i].to_numpy(dtype=float)
    cos_sim_scores.append(cosine_similarity(p, d))

ranking_cos = pd.DataFrame({
    'Kabupaten/Kota': wilayah_list,
    'Cosine Similarity': [round(x, 4) for x in cos_sim_scores]
}).sort_values('Cosine Similarity', ascending=False).reset_index(drop=True)
ranking_cos['Ranking'] = range(1, len(ranking_cos) + 1)

# ============================================================
# TAHAP 6: DIAGNOSTIK INDEKS KEPARAHAN (NORMA SELISIH VEKTOR)
# ============================================================
def selisih_vektor(p, d):
    selisih = p - d
    return selisih, np.linalg.norm(selisih)

norma_selisih = []
for i in range(len(wilayah_list)):
    p = df_pasien_ideal[SPESIALIS].iloc[i].to_numpy(dtype=float)
    d = df_dokter[SPESIALIS].iloc[i].to_numpy(dtype=float)
    _, norma = selisih_vektor(p, d)
    norma_selisih.append(norma)

ranking_selisih = pd.DataFrame({
    'Kabupaten/Kota': wilayah_list,
    'Norma Selisih': [round(x, 2) for x in norma_selisih]
}).sort_values('Norma Selisih', ascending=False).reset_index(drop=True)
ranking_selisih['Ranking'] = range(1, len(ranking_selisih) + 1)

# ============================================================
# TAHAP 7: DIAGNOSTIK KESENJANGAN RIIL (JUMLAH KEPALA DOKTER)
# ============================================================
df_gap_riil = df_pasien_ideal[SPESIALIS] - df_dokter[SPESIALIS]
df_gap_riil['Total Kekurangan'] = df_gap_riil[SPESIALIS].clip(lower=0).sum(axis=1).astype(int)
df_gap_riil['Kabupaten/Kota'] = wilayah_list

ranking_kesenjangan_riil = df_gap_riil[['Kabupaten/Kota'] + SPESIALIS + ['Total Kekurangan']].sort_values('Total Kekurangan', ascending=False).reset_index(drop=True)
ranking_kesenjangan_riil['Ranking'] = range(1, len(ranking_kesenjangan_riil) + 1)


# ============================================================
# TAHAP 8: OUTPUT HASIL PEMROSESAN DATA
# ============================================================
print(f"Total Populasi Provinsi Jawa Timur (BPS): {total_populasi_provinsi:,} jiwa")

print("\n" + "=" * 80)
print("             SAMPEL BEBAN PASIEN RIIL YANG DIBANGKITKAN")
print("=" * 80)
print(df_pasien[['Kabupaten'] + SPESIALIS].to_string(index=False))

print("\n" + "=" * 80)
print("             SAMPEL KUOTA KEBUTUHAN DOKTER IDEAL (HASIL BAGI 1:3571)")
print("=" * 80)
print(df_pasien_ideal[['Kabupaten'] + SPESIALIS].to_string(index=False))

print("\n" + "=" * 80)
print("       NILAI VEKTOR KEBUTUHAN IDEAL (p_ideal) VS VEKTOR DOKTER (d)")
print("                 Format Vektor: [Sp.A, Sp.B, Sp.OG, Sp.PD, Sp.An, Sp.Rad, Sp.PK]")
print("=" * 80)
for i, wil in enumerate(wilayah_list):
    p_vec = df_pasien_ideal[SPESIALIS].iloc[i].to_numpy().astype(int)
    d_vec = df_dokter[SPESIALIS].iloc[i].to_numpy().astype(int)
    print(f"{wil:<20} -> p_ideal: {p_vec.tolist()}  |  d: {d_vec.tolist()}")

print("\n" + "=" * 60)
print("             RANKING COSINE SIMILARITY")
print("  (Mendekati 1 = Pola distribusi paling selaras)")
print("=" * 60)
print(ranking_cos[['Ranking', 'Kabupaten/Kota', 'Cosine Similarity']].to_string(index=False))

print("\n" + "=" * 60)
print("             RANKING SELISIH VEKTOR (INDEKS KEPARAHAN)")
print("  (Nilai Norma Jarak euclidean terbesar = Krisis paling parah)")
print("=" * 60)
print(ranking_selisih[['Ranking', 'Kabupaten/Kota', 'Norma Selisih']].to_string(index=False))

print("\n" + "=" * 95)
print("                 RANKING KESENJANGAN RIIL (ALOKASI FISIK DOKTER)")
print("    (Angka Positif = Jumlah Kekurangan Dokter | Angka Negatif = Kelebihan/Surplus Dokter)")
print("=" * 95)
print(ranking_kesenjangan_riil[['Ranking', 'Kabupaten/Kota'] + SPESIALIS + ['Total Kekurangan']].to_string(index=False))