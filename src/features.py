"""
src/features.py
===============
MOETC-BD — Ingeniería de variables (Feature Engineering)
Capa 1: Data Layer — Fase 4 de CRISP-DM

Corresponde a la sección 3.1.4.3 de la tesis:
"Variables del modelo"

CAMBIOS v2.0 (datos reales GPS):
  - delta_t, ind_retraso, riesgo_cat ya existen en el CSV real;
    NO se recalculan, solo se validan con verificación de presencia.
  - FEATURES_MODELO actualizado: usa SOLO variables disponibles en el dataset real.
    Se eliminaron fill_rate, log_carga_kg y viajes_conductor (requieren
    carga_kg y chofer_id que la empresa no registra).
  - Añadidas: costo_estimado, costo_km_est, log_km (ya vienen del pipeline).
  - crear_variables_carga(): omitida del pipeline principal (sin carga_kg).
  - crear_variable_experiencia_conductor(): omitida (sin chofer_id individual).
  - crear_variables_costo(): ahora trabaja con costo_estimado en lugar de costo_total.
  - ejecutar_feature_engineering(): adaptado para no recalcular variables objetivo.
"""

import pandas as pd
import numpy as np
import os

FINAL_DIR   = "data/final"
REPORT_DIR  = "models/reports"
os.makedirs(FINAL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# VARIABLES OBJETIVO
# ══════════════════════════════════════════════════════════════════════════════

def crear_variable_delta_t(df):
    """
    MP-1: Variable objetivo continua.
    delta_t = t_real_min - t_est_min

    CAMBIO v2.0: Si delta_t ya existe en el DataFrame (viene del CSV real),
    se omite el cálculo y se imprime un aviso informativo.
    Solo crea la variable si falta, para mantener compatibilidad
    con pipelines que pasen datos sin ella.
    """
    if 'delta_t' in df.columns:
        print("[INFO] delta_t ya existe en el dataset (leída del CSV real). No se recalcula.")
        return df

    assert 't_real_min' in df.columns, "Falta t_real_min"
    assert 't_est_min'  in df.columns, "Falta t_est_min"

    df['delta_t'] = df['t_real_min'] - df['t_est_min']
    return df


def crear_variable_ind_retraso(df, umbral_min=15):
    """
    MP-3: Variable objetivo binaria.
    ind_retraso = 1 si delta_t > umbral_min, 0 si no.

    CAMBIO v2.0: Si ind_retraso ya existe en el DataFrame (viene del CSV real),
    se omite el cálculo. Solo se imprime la distribución real para validación.
    """
    if 'ind_retraso' in df.columns:
        pct_retraso = df['ind_retraso'].mean() * 100
        print(f"[INFO] ind_retraso ya existe (leída del CSV real). No se recalcula.")
        print(f"[OTIF] Tasa de retraso real: {pct_retraso:.1f}%")
        print(f"[OTIF] OTIF AS IS real:       {100 - pct_retraso:.1f}%")
        return df

    assert 'delta_t' in df.columns, "Ejecutar crear_variable_delta_t primero"
    df['ind_retraso'] = (df['delta_t'] > umbral_min).astype(int)

    pct_retraso = df['ind_retraso'].mean() * 100
    print(f"[OTIF] Tasa de retraso (umbral {umbral_min} min): {pct_retraso:.1f}%")
    print(f"[OTIF] OTIF AS IS: {100 - pct_retraso:.1f}%")

    return df


def crear_variable_riesgo(df, umbral_bajo=15, umbral_alto=30):
    """
    MP-2: Variable objetivo de clasificación multiclase.

    BAJO  : delta_t <= umbral_bajo
    MEDIO : umbral_bajo < delta_t <= umbral_alto
    ALTO  : delta_t > umbral_alto

    CAMBIO v2.0: Si riesgo_cat ya existe en el DataFrame (viene del CSV real),
    solo se muestra la distribución para validación y se crea riesgo_num
    (codificación numérica requerida por los modelos ML).
    """
    if 'riesgo_cat' in df.columns:
        print("[INFO] riesgo_cat ya existe (leída del CSV real). No se recalcula.")
        print("[RIESGO] Distribución de clases (real):")
        print(df['riesgo_cat'].value_counts(normalize=True).mul(100).round(1).to_string())
    else:
        assert 'delta_t' in df.columns, "Ejecutar crear_variable_delta_t primero"
        df['riesgo_cat'] = pd.cut(
            df['delta_t'],
            bins=[-np.inf, umbral_bajo, umbral_alto, np.inf],
            labels=['BAJO', 'MEDIO', 'ALTO']
        )
        print("[RIESGO] Distribución de clases:")
        print(df['riesgo_cat'].value_counts(normalize=True).mul(100).round(1).to_string())

    # Crear codificación numérica (siempre necesaria para los modelos)
    mapa_riesgo = {'BAJO': 0, 'MEDIO': 1, 'ALTO': 2}
    df['riesgo_num'] = df['riesgo_cat'].map(mapa_riesgo)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# VARIABLES PREDICTORAS
# ══════════════════════════════════════════════════════════════════════════════

def crear_variables_carga(df):
    """
    Variables relacionadas con la carga y utilización del vehículo.

    CAMBIO v2.0: La empresa no registra carga_kg ni capacidad_kg.
    Esta función emite un aviso claro y retorna el DataFrame sin cambios.
    fill_rate y log_carga_kg han sido eliminados de FEATURES_MODELO.
    No se inventan datos ficticios.
    """
    if 'carga_kg' not in df.columns:
        print("[AVISO] carga_kg no disponible en el dataset real.")
        print("        fill_rate y log_carga_kg omitidos del modelo.")
        print("        Para habilitarlos, registra kg de carga por viaje en el sistema GPS.")
        return df

    # Si por algún motivo el dato existe en el futuro
    if 'capacidad_kg' in df.columns:
        df['fill_rate'] = (
            df['carga_kg'] / df['capacidad_kg'].replace(0, np.nan)
        ).clip(0, 1)
        fr_medio = df['fill_rate'].mean()
        print(f"[FLOTA] Fill Rate promedio AS IS: {fr_medio*100:.1f}%")
        if fr_medio < 0.75:
            print(f"        ⚠ Subutilización detectada (meta >= 80%)")

    df['log_carga_kg'] = np.log1p(df['carga_kg'])
    return df


def crear_variables_costo(df):
    """
    Variables relacionadas con el costo operativo.

    CAMBIO v2.0: Trabaja con costo_estimado (calculado en data_pipeline.py)
    en lugar de costo_total (que la empresa no registra).

    costo_km_est = costo_estimado / km_ruta
    Si costo_km_est ya existe (calculado en data_pipeline), lo preserva.
    """
    # Preferir costo_estimado; fallback a costo_total si existe
    col_costo = None
    if 'costo_estimado' in df.columns:
        col_costo = 'costo_estimado'
    elif 'costo_total' in df.columns:
        col_costo = 'costo_total'

    if col_costo is None:
        print("[AVISO] No se encontró columna de costo (costo_estimado ni costo_total).")
        return df

    if 'km_ruta' in df.columns and 'costo_km_est' not in df.columns:
        df['costo_km_est'] = (
            df[col_costo] / df['km_ruta'].replace(0, np.nan)
        )
        # Eliminar outliers extremos (> percentil 99)
        p99 = df['costo_km_est'].quantile(0.99)
        df['costo_km_est'] = df['costo_km_est'].clip(upper=p99)

    return df


def crear_variables_temporales(df):
    """
    Variables extraídas de la fecha/hora de inicio del viaje.

    CAMBIO v2.0: Las variables temporales ya existen en el CSV real
    (hora_inicio, dia_semana_num, es_fin_semana, mes, franja_horaria).
    Si faltan por algún razón, se recalculan desde t_inicio.
    """
    vars_temporales = ['hora_inicio', 'dia_semana_num', 'es_fin_semana', 'mes']
    ya_existen = [v for v in vars_temporales if v in df.columns]

    if len(ya_existen) == len(vars_temporales):
        print(f"[INFO] Variables temporales ya existen en el dataset real: {ya_existen}")
        return df

    # Recalcular desde t_inicio si faltan
    col_fecha = 't_inicio' if 't_inicio' in df.columns else 'fecha_salida'
    if col_fecha not in df.columns:
        print("[WARN] No se encontró t_inicio ni fecha_salida. Variables temporales omitidas.")
        return df

    fecha = pd.to_datetime(df[col_fecha], errors='coerce')

    if 'hora_inicio' not in df.columns:
        df['hora_inicio']      = fecha.dt.hour
    if 'dia_semana_num' not in df.columns:
        df['dia_semana_num']   = fecha.dt.dayofweek
    if 'es_fin_semana' not in df.columns:
        df['es_fin_semana']    = df['dia_semana_num'].isin([5, 6]).astype(int)
    if 'mes' not in df.columns:
        df['mes']              = fecha.dt.month
    if 'franja_horaria' not in df.columns:
        df['franja_horaria']   = pd.cut(
            df['hora_inicio'],
            bins=[-1, 11, 17, 24],
            labels=['MAÑANA', 'TARDE', 'NOCHE']
        )

    return df


def crear_variable_experiencia_conductor(df):
    """
    Proxy de experiencia del conductor: viajes acumulados.

    CAMBIO v2.0: El dataset real no tiene chofer_id individual
    (solo vehiculo_id). Esta función emite un aviso y retorna sin cambios.
    viajes_conductor fue eliminada de FEATURES_MODELO.
    """
    if 'chofer_id' not in df.columns:
        print("[AVISO] chofer_id no disponible en el dataset real.")
        print("        viajes_conductor omitido del modelo.")
        print("        Para habilitarlo, registra el conductor en el sistema GPS.")
        return df

    if 't_inicio' in df.columns:
        df = df.sort_values('t_inicio')
    df['viajes_conductor'] = df.groupby('chofer_id').cumcount()
    return df


def crear_variable_paradas_por_km(df):
    """
    Densidad de paradas: n_paradas / km_ruta

    CAMBIO v2.0: densidad_paradas ya existe en el CSV real.
    Si existe, se preserva. Si falta, se recalcula.
    """
    if 'densidad_paradas' in df.columns:
        print("[INFO] densidad_paradas ya existe en el dataset real.")
        return df

    if 'n_paradas' in df.columns and 'km_ruta' in df.columns:
        df['densidad_paradas'] = (
            df['n_paradas'] / df['km_ruta'].replace(0, np.nan)
        ).clip(upper=df['n_paradas'].max() / 1)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ══════════════════════════════════════════════════════════════════════════════

def codificar_categoricas(df, columnas_ohe=None, columnas_label=None):
    """
    Codifica variables categóricas para los modelos de ML.

    One-Hot Encoding (OHE): para variables nominales sin orden.
    Label Encoding: para variables ordinales (riesgo_cat → riesgo_num).

    CAMBIO v2.0: franja_horaria puede venir como string o Categorical
    del CSV real. Se maneja ambos casos.
    """
    if columnas_ohe is None:
        columnas_ohe = [c for c in ['ruta', 'franja_horaria'] if c in df.columns]

    if columnas_label is None:
        if 'riesgo_cat' in df.columns and 'riesgo_num' not in df.columns:
            mapa_riesgo = {'BAJO': 0, 'MEDIO': 1, 'ALTO': 2}
            df['riesgo_num'] = df['riesgo_cat'].map(mapa_riesgo)

    # OHE — solo columnas que existen
    cols_existentes = [c for c in columnas_ohe if c in df.columns]
    if cols_existentes:
        # Convertir Categorical a str para evitar errores en get_dummies
        for c in cols_existentes:
            if hasattr(df[c], 'cat'):
                df[c] = df[c].astype(str)
        df = pd.get_dummies(df, columns=cols_existentes, drop_first=True, dtype=int)
        print(f"[OHE] Columnas codificadas: {cols_existentes}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

# CAMBIO v2.0: Lista de features actualizada a variables REALMENTE disponibles
# en el dataset GPS real. Se eliminaron fill_rate, log_carga_kg, viajes_conductor.
# Se agregaron costo_estimado, costo_km_est (calculados) y log_km (ya en el CSV).
FEATURES_MODELO = [
    "km_ruta",           # distancia recorrida (GPS)
    "log_km",            # log(km_ruta) — normalización de escala
    "vel_max",           # velocidad máxima registrada (GPS)
    "vel_prom",          # velocidad promedio (GPS)
    "t_est_min",         # tiempo estimado de entrega
    "hora_inicio",       # hora de salida del vehículo
    "dia_semana_num",    # día de la semana (0=Lun, 6=Dom)
    "es_fin_semana",     # indicador binario fin de semana
    "mes",               # mes del año (estacionalidad)
    "n_paradas",         # número de paradas en la ruta
    "densidad_paradas",  # paradas por km (complejidad de ruta)
]

TARGETS = {
    "regresion":              "delta_t",      # MP-1: retraso en minutos
    "clasificacion_riesgo":   "riesgo_num",   # MP-2: BAJO/MEDIO/ALTO
    "clasificacion_otif":     "ind_retraso",  # MP-3: binario
}


def ejecutar_feature_engineering(df):
    """
    Ejecuta el pipeline completo de ingeniería de variables.

    CAMBIO v2.0: Adaptado para el dataset real:
      - Las variables objetivo (delta_t, ind_retraso, riesgo_cat) ya EXISTEN
        en el CSV, por lo tanto se verifican y validan en lugar de calcularse.
      - crear_variables_carga() se llama pero emite aviso (sin carga_kg).
      - crear_variable_experiencia_conductor() se llama pero emite aviso (sin chofer_id).
      - FEATURES_MODELO usa solo variables disponibles en el dataset real.

    Orden de ejecución:
      1. Verificación/validación de variables objetivo
      2. Variables predictoras (costo, temporales, paradas)
      3. Codificación de categóricas
      4. Verificación del dataset final
    """
    print("\n" + "="*60)
    print("MOETC-BD — INGENIERÍA DE VARIABLES v2.0 (Datos Reales GPS)")
    print("Fase 4 de CRISP-DM")
    print("="*60)

    # [1/4] Validar / crear variables objetivo
    print("\n[1/4] Verificando variables objetivo (del CSV real)...")
    df = crear_variable_delta_t(df)       # preserva si ya existe
    df = crear_variable_ind_retraso(df)   # preserva si ya existe, muestra OTIF real
    df = crear_variable_riesgo(df)        # preserva riesgo_cat, crea riesgo_num

    # [2/4] Variables predictoras
    print("\n[2/4] Creando variables predictoras...")
    crear_variables_carga(df)             # aviso: carga_kg no disponible
    df = crear_variables_costo(df)        # usa costo_estimado del pipeline
    df = crear_variables_temporales(df)   # verifica / recalcula si falta
    crear_variable_experiencia_conductor(df)  # aviso: sin chofer_id
    df = crear_variable_paradas_por_km(df)    # preserva densidad_paradas si existe

    # [3/4] Codificación
    print("\n[3/4] Codificando variables categóricas...")
    df = codificar_categoricas(df)

    # [4/4] Verificación
    print("\n[4/4] Verificando dataset final...")
    features_disponibles = [f for f in FEATURES_MODELO if f in df.columns]
    features_faltantes   = [f for f in FEATURES_MODELO if f not in df.columns]

    print(f"  Features disponibles: {len(features_disponibles)}/{len(FEATURES_MODELO)}")
    if features_faltantes:
        print(f"  [AVISO] Features faltantes: {features_faltantes}")
        print("           Estas columnas no se incluirán en el entrenamiento.")

    # Eliminar filas con nulos en variables objetivo críticas
    targets_disponibles = [v for v in TARGETS.values() if v in df.columns]
    n_antes = len(df)
    df = df.dropna(subset=targets_disponibles)
    print(f"  Filas eliminadas por nulos en variables objetivo: {n_antes - len(df)}")
    print(f"  Dataset final: {df.shape[0]} filas × {df.shape[1]} columnas")

    # Guardar
    path = os.path.join(FINAL_DIR, "dataset_final.csv")
    df.to_csv(path, index=False)
    print(f"\n[OK] Dataset final guardado: {path}")

    return df


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_pipeline import ejecutar_pipeline

    df_integrado = ejecutar_pipeline()
    df_final     = ejecutar_feature_engineering(df_integrado)

    features_en_df = [f for f in FEATURES_MODELO if f in df_final.columns]
    targets_en_df  = [v for v in TARGETS.values() if v in df_final.columns]
    print(df_final[features_en_df + targets_en_df].head())
