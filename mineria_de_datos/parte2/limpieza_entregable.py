"""
limpieza_entregable2.py
=======================
Pipeline de preparación de datos — Entregable 2
Barómetro Social CIS Estudio 1162 — Constitución y Elecciones (I)

Ejecutar:
    python limpieza_entregable2.py

Genera:
    dataset_analitico_base.csv    → dataset completo limpio (1.192 filas, todas las variables)
    dataset_ds1_clasificacion.csv → subset DS-1 (539 filas, P15 válido)
    dataset_ds2_clustering.csv    → subset DS-2 (367 filas, escalas válidas)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
RUTA_EXCEL  = BASE_DIR.parent / "datos" / "1162.xlsx"
OUT_DIR     = BASE_DIR / "datos_limpios"
OUT_DIR.mkdir(exist_ok=True)
SEP = "─" * 70

def titulo(texto):
    print(f"\n{SEP}")
    print(f"  {texto}")
    print(SEP)

def subtitulo(texto):
    print(f"\n  ── {texto} ──")


# ══════════════════════════════════════════════════════════════════════════════
# 0. CARGA
# ══════════════════════════════════════════════════════════════════════════════
titulo("0. CARGA DEL DATASET ORIGINAL")

df_raw = pd.read_excel(RUTA_EXCEL)
print(f"\n  Filas:     {df_raw.shape[0]}")
print(f"  Columnas:  {df_raw.shape[1]}")
print(f"\n  Tipos de columna:\n{df_raw.dtypes.value_counts().to_string()}")

df = df_raw.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 1. ELIMINACIÓN DE VARIABLES ADMINISTRATIVAS
# ══════════════════════════════════════════════════════════════════════════════
titulo("1. ELIMINACIÓN DE VARIABLES ADMINISTRATIVAS")

VARS_ADMIN = ["ESTU", "T1", "T2", "ENTREV"]

subtitulo("Criterio: valores únicos / constantes por columna")
for col in VARS_ADMIN:
    n_unique = df[col].nunique()
    sample   = df[col].iloc[0]
    print(f"    {col:10s} → {n_unique} valor(es) único(s)  |  muestra: {repr(sample)}")

df = df.drop(columns=VARS_ADMIN)
print(f"\n  Variables eliminadas : {VARS_ADMIN}")
print(f"  Variables restantes  : {df.shape[1]}  (antes: {df_raw.shape[1]})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. VERIFICACIÓN DE DUPLICADOS
# ══════════════════════════════════════════════════════════════════════════════
titulo("2. VERIFICACIÓN DE DUPLICADOS")

n_dup_filas = df.duplicated().sum()
n_dup_cues  = df["CUES"].duplicated().sum()
print(f"\n  Filas completamente duplicadas : {n_dup_filas}")
print(f"  CUES duplicados (clave primaria): {n_dup_cues}")
print("\n  → No se elimina ninguna fila.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRATAMIENTO DE VALORES NS/NC Y «NO PROCEDE»
# ══════════════════════════════════════════════════════════════════════════════
titulo("3. TRATAMIENTO DE NS/NC Y «NO PROCEDE»")

NSYNC_VALS  = ["NS/NC", "N/S - N/C", "N.S.", "N.C.", "No sabe",
               "No contesta", "N/C"]
NO_PROCEDE  = "No procede"

# ── 3a. Diagnóstico previo ────────────────────────────────────────────────────
subtitulo("Diagnóstico previo: ocurrencias totales en columnas de texto")

cols_texto = df.select_dtypes(include="object").columns.tolist()
total_nsync    = sum(df[c].isin(NSYNC_VALS).sum() for c in cols_texto)
total_noproc   = sum((df[c] == NO_PROCEDE).sum() for c in cols_texto)
total_nan_prev = df.isnull().sum().sum()

print(f"\n  Valores NS/NC textuales  : {total_nsync:>6,}")
print(f"  Valores «No procede»     : {total_noproc:>6,}")
print(f"  NaN ya presentes (Excel) : {total_nan_prev:>6,}")

# ── 3b. Variables con «No procede» — justificación de no eliminar filas ───────
subtitulo("Variables afectadas por «No procede» estructural")
noproc_por_var = {c: (df[c] == NO_PROCEDE).sum()
                  for c in cols_texto if (df[c] == NO_PROCEDE).any()}
noproc_df = (pd.Series(noproc_por_var)
               .sort_values(ascending=False)
               .rename("No_procede_N"))
noproc_df["No_procede_%"] = (noproc_df / len(df) * 100).round(1)
print(f"\n{noproc_df.head(15).to_string()}")

filas_con_noproc = df[cols_texto].isin([NO_PROCEDE]).any(axis=1).sum()
print(f"\n  Filas con al menos un «No procede» : {filas_con_noproc}  "
      f"({filas_con_noproc/len(df)*100:.1f}% del total)")
print("  → Si se eliminaran estas filas, el dataset quedaría vacío.")
print("  → Decisión: reemplazar «No procede» por NaN, conservar todas las filas.")

# ── 3c. Variables con NS/NC — porcentaje por variable ─────────────────────────
subtitulo("Variables con mayor tasa de NS/NC")
nsync_por_var = {c: df[c].isin(NSYNC_VALS).sum()
                 for c in cols_texto if df[c].isin(NSYNC_VALS).any()}
nsync_df = (pd.Series(nsync_por_var)
              .sort_values(ascending=False)
              .rename("NSYNC_N"))
nsync_df["NSYNC_%"] = (nsync_df / len(df) * 100).round(1)
print(f"\n{nsync_df.head(15).to_string()}")

# ── 3d. Sustitución ───────────────────────────────────────────────────────────
reemplazos_nsync   = 0
reemplazos_noproc  = 0
for col in cols_texto:
    mask_ns = df[col].isin(NSYNC_VALS)
    mask_np = df[col] == NO_PROCEDE
    reemplazos_nsync  += mask_ns.sum()
    reemplazos_noproc += mask_np.sum()
    df.loc[mask_ns | mask_np, col] = np.nan

total_nan_post = df.isnull().sum().sum()
print(f"\n  NS/NC reemplazados por NaN    : {reemplazos_nsync:>6,}")
print(f"  «No procede» reemplazados NaN : {reemplazos_noproc:>6,}")
print(f"  Total NaN antes del paso      : {total_nan_prev:>6,}")
print(f"  Total NaN después del paso    : {total_nan_post:>6,}")
print(f"  Incremento                    : {total_nan_post - total_nan_prev:>6,}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. RECODIFICACIÓN DE P24 (ingresos → escala ordinal 1–9)
# ══════════════════════════════════════════════════════════════════════════════
titulo("4. RECODIFICACIÓN P24 → P24_num (escala ordinal 1–9)")

subtitulo("Valores únicos en P24 antes de recodificar")
print(f"\n{df['P24'].value_counts(dropna=False).to_string()}")

P24_MAP = {
    "Hasta 6.000 ptas mensuales"                   : 1,
    "De 6.000 ptas a 11.000 ptas mensuales"        : 2,
    "De 12.000 ptas a 18.000 ptas mensuales"       : 3,
    "De 19.000 ptas a 25.000 ptas mensuales"       : 4,
    "De 26.000 ptas a 35.000 ptas mensuales"       : 5,
    "De 36.000 ptas a 45.000 ptas mensuales"       : 6,
    "De 46.000 ptas a 65.000 ptas mensuales"       : 7,
    "De 66.000 ptas a 85.000 ptas mensuales"       : 8,
    "Más de 85.000 ptas mensuales"                 : 9,
}
df["P24_num"] = df["P24"].map(P24_MAP)   # NaN para los NS/NC ya sustituidos

subtitulo("Distribución de P24_num tras recodificación")
dist = df["P24_num"].value_counts(sort=False, dropna=False).sort_index()
dist_pct = (dist / len(df) * 100).round(1)
resumen_p24 = pd.DataFrame({"N": dist, "%": dist_pct})
print(f"\n{resumen_p24.to_string()}")
print(f"\n  Válidos  : {df['P24_num'].notna().sum()}")
print(f"  NaN      : {df['P24_num'].isna().sum()}  "
      f"({df['P24_num'].isna().sum()/len(df)*100:.1f}%)")
print(f"  Media    : {df['P24_num'].mean():.2f}")
print(f"  Mediana  : {df['P24_num'].median():.0f}")
print(f"  Std      : {df['P24_num'].std():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONVERSIÓN P801–P807 (texto → float)
# ══════════════════════════════════════════════════════════════════════════════
titulo("5. CONVERSIÓN P801–P807 (texto → float64)")

ESCALAS = [f"P80{i}" for i in range(1, 8)]

subtitulo("Estadísticos por escala antes/después de la conversión")
print(f"\n  {'Variable':<10} {'N válido':>9} {'NaN':>6} {'NaN%':>7} "
      f"{'Media':>7} {'Std':>6} {'Min':>4} {'Max':>4}")
print("  " + "-"*62)

for col in ESCALAS:
    col_num = col + "_num"
    df[col_num] = pd.to_numeric(df[col], errors="coerce")
    n_val  = df[col_num].notna().sum()
    n_nan  = df[col_num].isna().sum()
    pct    = n_nan / len(df) * 100
    media  = df[col_num].mean()
    std    = df[col_num].std()
    mn     = df[col_num].min()
    mx     = df[col_num].max()
    print(f"  {col:<10} {n_val:>9} {n_nan:>6} {pct:>6.1f}% "
          f"{media:>7.2f} {std:>6.2f} {mn:>4.0f} {mx:>4.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRATAMIENTO DE P23A / P23B (ocupación)
# ══════════════════════════════════════════════════════════════════════════════
titulo("6. DIAGNÓSTICO P23A / P23B (ocupación asalariado / autónomo)")

for col in ["P23A", "P23B"]:
    n_val = df[col].notna().sum()
    n_nan = df[col].isna().sum()
    print(f"\n  {col}:")
    print(f"    Válidos : {n_val}  ({n_val/len(df)*100:.1f}%)")
    print(f"    NaN     : {n_nan}  ({n_nan/len(df)*100:.1f}%)")
    top = df[col].value_counts().head(6)
    print(f"    Top categorías:\n{top.to_string()}")

mutuamente_excl = (df["P23A"].notna() & df["P23B"].notna()).sum()
print(f"\n  Filas con P23A y P23B ambas válidas (debería ser ~0): {mutuamente_excl}")
print("  → Son variables mutuamente excluyentes. Se mantienen separadas.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DISCRETIZACIÓN P20 → P20_grupo
# ══════════════════════════════════════════════════════════════════════════════
titulo("7. DISCRETIZACIÓN P20 → P20_grupo (grupos decenales)")

subtitulo("Estadísticos de P20 (edad exacta)")
print(f"\n{df['P20'].describe().round(1).to_string()}")

# Cálculo IQR para justificar ausencia de outliers
Q1  = df["P20"].quantile(0.25)
Q3  = df["P20"].quantile(0.75)
IQR = Q3 - Q1
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR
n_out_inf = (df["P20"] < lim_inf).sum()
n_out_sup = (df["P20"] > lim_sup).sum()
print(f"\n  Q1={Q1}, Q3={Q3}, IQR={IQR}")
print(f"  Límite inferior IQR : {lim_inf:.1f}  →  outliers por abajo: {n_out_inf}")
print(f"  Límite superior IQR : {lim_sup:.1f}  →  outliers por arriba: {n_out_sup}")
print("  → Sin outliers estadísticos en la edad.")

BINS   = [17, 25, 35, 45, 55, 65, 75, 100]
LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
df["P20_grupo"] = pd.cut(df["P20"], bins=BINS, labels=LABELS)

subtitulo("Distribución de P20_grupo")
dist_edad = df["P20_grupo"].value_counts().sort_index()
dist_edad_pct = (dist_edad / dist_edad.sum() * 100).round(1)
print(f"\n{pd.DataFrame({'N': dist_edad, '%': dist_edad_pct}).to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CREACIÓN DE P15_bin (variable target DS-1)
# ══════════════════════════════════════════════════════════════════════════════
titulo("8. CREACIÓN DE P15_bin (target binario DS-1)")

subtitulo("Distribución de P15 original (tras sustitución de NaN)")
print(f"\n{df_raw['P15'].value_counts(dropna=False).to_string()}")

df["P15_bin"] = df["P15"].map({"Sí": 1, "No": 0})

n_valido  = df["P15_bin"].notna().sum()
n_nan     = df["P15_bin"].isna().sum()
n_si      = (df["P15_bin"] == 1).sum()
n_no      = (df["P15_bin"] == 0).sum()
ratio     = n_si / n_no if n_no > 0 else float("inf")

print(f"\n  P15_bin = 1 (Sí) : {n_si}  ({n_si/n_valido*100:.1f}% sobre válidos)")
print(f"  P15_bin = 0 (No) : {n_no}  ({n_no/n_valido*100:.1f}% sobre válidos)")
print(f"  NaN (excluidos)  : {n_nan}")
print(f"  Total válido     : {n_valido}")
print(f"\n  Ratio desbalanceo Sí/No : {ratio:.1f}:1")
print("  → AVISO: fuerte desbalanceo de clases. En DS-1 usar class_weight='balanced'.")


# ══════════════════════════════════════════════════════════════════════════════
# 9. RESUMEN GLOBAL DE MISSINGS EN EL DATASET LIMPIO
# ══════════════════════════════════════════════════════════════════════════════
titulo("9. RESUMEN GLOBAL DE MISSINGS (dataset tras limpieza)")

miss = df.isnull().sum().sort_values(ascending=False)
miss_pct = (miss / len(df) * 100).round(1)
miss_df = pd.DataFrame({"NaN_N": miss, "NaN_%": miss_pct})

# Solo variables con al menos 1 missing
miss_df = miss_df[miss_df["NaN_N"] > 0]

subtitulo(f"Variables con missing (N={len(miss_df)} de {df.shape[1]} variables):")
print(f"\n{miss_df.to_string()}")

# Resumen por rangos
print("\n  Resumen por rango de missing%:")
rangos = [(90, 100, ">90%"), (50, 90, "50–90%"), (15, 50, "15–50%"),
          (1, 15, "1–15%"), (0, 0, "0% (completas)")]
for lo, hi, label in rangos:
    if label == "0% (completas)":
        n = (miss_pct == 0).sum()
    else:
        n = ((miss_pct > lo) & (miss_pct <= hi)).sum()
    print(f"    {label:15s}: {n} variables")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CONSTRUCCIÓN DE LOS DATASETS ANALÍTICOS
# ══════════════════════════════════════════════════════════════════════════════
titulo("10. CONSTRUCCIÓN DE DATASETS ANALÍTICOS")

# ── Dataset base ──────────────────────────────────────────────────────────────
subtitulo("Dataset base (limpio, todas las filas)")
print(f"\n  Filas    : {df.shape[0]}")
print(f"  Columnas : {df.shape[1]}")

# ── Dataset DS-1 (clasificación) ──────────────────────────────────────────────
FEATURES_DS1 = ["P18", "P20", "P21", "P22", "P23A", "P23B",
                "P24_num", "P25", "P27", "P28"]
TARGET_DS1   = "P15_bin"

ds1 = df[FEATURES_DS1 + [TARGET_DS1]].dropna(subset=[TARGET_DS1]).copy()

subtitulo("Dataset DS-1 — Clasificación (P15_bin válido)")
print(f"\n  Filas    : {ds1.shape[0]}")
print(f"  Columnas : {ds1.shape[1]}")
print(f"\n  Missing% por feature:")
miss_ds1 = (ds1[FEATURES_DS1].isnull().sum() / len(ds1) * 100).round(1)
print(miss_ds1.to_string())

# ── Dataset DS-2 (clustering) ─────────────────────────────────────────────────
ESCALAS_NUM = [f"P80{i}_num" for i in range(1, 8)]
DEMO_DS2    = ["P18", "P20", "P21", "P24_num"]

ds2 = df[ESCALAS_NUM + DEMO_DS2].copy()
ds2 = ds2.dropna(subset=["P801_num", "P802_num", "P803_num"])
ds2 = ds2.fillna(ds2.median(numeric_only=True))

subtitulo("Dataset DS-2 — Clustering (al menos P801–P803 válidas)")
print(f"\n  Filas    : {ds2.shape[0]}")
print(f"  Columnas : {ds2.shape[1]}")
print(f"\n  Reducción de muestra:")
print(f"    Total original               : {len(df)}")
print(f"    Con P801+P802+P803 no nulas  : {ds2.shape[0]}  "
      f"({ds2.shape[0]/len(df)*100:.1f}%)")

subtitulo("Estadísticos de escalas en DS-2 (tras imputar mediana)")
print(f"\n{ds2[ESCALAS_NUM].describe().round(2).to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. EXPORTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
titulo("11. EXPORTACIÓN DE DATASETS")

df.to_csv(OUT_DIR / "dataset_analitico_base.csv", index=False)
ds1.to_csv(OUT_DIR / "dataset_ds1_clasificacion.csv", index=False)
ds2.to_csv(OUT_DIR / "dataset_ds2_clustering.csv",    index=False)

print(f"\n  ✓  {OUT_DIR / 'dataset_analitico_base.csv'}       → {df.shape[0]} filas × {df.shape[1]} cols")
print(f"  ✓  {OUT_DIR / 'dataset_ds1_clasificacion.csv'}    → {ds1.shape[0]} filas × {ds1.shape[1]} cols")
print(f"  ✓  {OUT_DIR / 'dataset_ds2_clustering.csv'}       → {ds2.shape[0]} filas × {ds2.shape[1]} cols")

titulo("PIPELINE COMPLETADO")
print()