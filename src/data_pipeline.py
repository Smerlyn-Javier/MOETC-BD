"""
src/data_pipeline.py
====================
MOETC-BD — Pipeline de integración y limpieza de datos
Capa 1: Data Layer del modelo MOETC-BD

Corresponde a la sección 3.1.4 de la tesis:
"Componente I: Pipeline de datos e integración Big Data"

Fases implementadas:
  1. Ingesta y consolidación
  2. Auditoría de calidad
  3. Limpieza y tratamiento
  4. Integración de fuentes
  5. Exportación del repositorio analítico

CAMBIOS v2.0 (datos reales GPS):
  - cargar_viajes(): lee dataset_gps_procesado.csv real (4,608 viajes, 22 vars)
    en lugar de generar datos sintéticos.
  - cargar_costos(): calcula costo_estimado con fórmula calibrada para la flota;
    no usa archivo externo (la empresa no tiene costo_total registrado).
  - cargar_ordenes(): deriva otif directamente de ind_retraso (ya en el CSV).
  - cargar_flota(): retorna catálogo de los 7 vehículos reales identificados.
  - limpiar_viajes(): adaptado a columnas t_inicio/t_fin del CSV real.
  - integrar_datasets(): simplificado; merge principal es viajes + costos.
  - Eliminados: _generar_viajes_sinteticos(), _generar_ordenes_sinteticas(),
    _generar_costos_sinteticos() y _generar_catalogo_flota() (ya no se usan).
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# ── RUTAS ──────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
FINAL_DIR     = "data/final"
REPORT_DIR    = "models/reports"

# Rutas de los CSV reales
CSV_VIAJES = os.path.join(RAW_DIR, "dataset_gps_procesado.csv")
CSV_GPS    = os.path.join(RAW_DIR, "puntos_entrega_gps.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── PARÁMETROS DE COSTO (calibrados para López Laureano Distribución) ─────────
# CAMBIO: la empresa no registra costo_total; se estima con estos parámetros.
CONSUMO_GALON_POR_KM = 0.167   # gal/km  (promedio flota ≈ 6 km/gal)
PRECIO_GALON_RD      = 295     # RD$ por galón (gasoil)
PAGO_CHOFER_FIJO_RD  = 1200    # RD$ por viaje (estimación conservadora)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: INGESTA
# ══════════════════════════════════════════════════════════════════════════════

def cargar_viajes(filepath=None):
    """
    Carga el dataset real de viajes GPS de la flota.

    CAMBIO v2.0: Lee data/raw/dataset_gps_procesado.csv en lugar de
    generar datos sintéticos. El archivo CSV usa coma como separador
    decimal (formato europeo/español) y ya contiene las variables
    objetivo (delta_t, ind_retraso, riesgo_cat).

    Columnas en el CSV real:
      vehiculo_id, t_inicio, t_fin, fecha, km_ruta, log_km,
      vel_max, vel_prom, t_real_min, t_est_min, hora_inicio,
      dia_semana_num, es_fin_semana, mes, franja_horaria,
      zona_destino, ruta, n_paradas, densidad_paradas,
      delta_t, ind_retraso, riesgo_cat

    Período: marzo 2025 – marzo 2026 | 7 vehículos reales
    """
    ruta = filepath if filepath else CSV_VIAJES

    if not os.path.exists(ruta):
        print(f"[AVISO] Archivo de viajes no encontrado: {ruta}")
        print("        Verifica que el CSV esté en data/raw/dataset_gps_procesado.csv")
        print("        El pipeline continuará con un DataFrame vacío.")
        return pd.DataFrame()

    # decimal=',' porque el CSV usa coma como separador decimal (formato es-DO)
    df = pd.read_csv(ruta, decimal=',', encoding='utf-8')

    # Convertir fechas a datetime
    for col in ['t_inicio', 't_fin', 'fecha']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Generar viaje_id secuencial como clave primaria
    df.insert(0, 'viaje_id', [f"VJ-{i:05d}" for i in range(1, len(df) + 1)])

    print(f"[OK] Viajes cargados: {len(df)} registros | {df['vehiculo_id'].nunique()} vehículos")
    print(f"     Período: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"     Variables: {', '.join(df.columns.tolist())}")

    return df


def cargar_ordenes(df_viajes=None):
    """
    Deriva las órdenes de entrega directamente del dataset de viajes.

    CAMBIO v2.0: La empresa no tiene un archivo de órdenes separado.
    Se construye el mini-dataset de ordenes a partir del DataFrame de viajes,
    usando ind_retraso (ya calculado en el CSV) como campo OTIF.

    NOTA: ind_retraso=0 → entregado a tiempo (OTIF=1)
          ind_retraso=1 → retrasado (OTIF=0)
    """
    if df_viajes is None or len(df_viajes) == 0:
        print("[AVISO] No hay datos de viajes para derivar órdenes.")
        return pd.DataFrame()

    cols_requeridas = ['viaje_id', 't_est_min', 't_real_min', 'ind_retraso']
    cols_disponibles = [c for c in cols_requeridas if c in df_viajes.columns]

    df_ord = df_viajes[cols_disponibles].copy()

    # OTIF = 1 si NO hay retraso (ind_retraso == 0)
    if 'ind_retraso' in df_ord.columns:
        df_ord['otif'] = (df_ord['ind_retraso'] == 0).astype(int)

    print(f"[OK] Órdenes derivadas: {len(df_ord)} registros")
    otif_pct = df_ord['otif'].mean() * 100 if 'otif' in df_ord.columns else 0
    print(f"     OTIF AS IS (real): {otif_pct:.1f}%")

    return df_ord


def cargar_costos(df_viajes=None, filepath=None):
    """
    Calcula el costo operativo estimado por viaje.

    CAMBIO v2.0: La empresa no tiene costo_total registrado.
    Se estima con la fórmula calibrada para la flota:

      costo_estimado = km_ruta × consumo_galon × precio_galon_rd + pago_chofer
      costo_km_est   = costo_estimado / km_ruta

    Parámetros (constantes al tope del módulo):
      CONSUMO_GALON_POR_KM = 0.167  (6 km/gal promedio flota)
      PRECIO_GALON_RD      = 295    (RD$ por galón gasoil)
      PAGO_CHOFER_FIJO_RD  = 1200   (RD$ por viaje, estimación conservadora)

    Si se provee filepath apuntando a un Excel real de costos, lo carga.
    De lo contrario, calcula la estimación a partir del DataFrame de viajes.
    """
    # Si existe un archivo externo de costos, usarlo directamente
    if filepath and os.path.exists(filepath):
        df = pd.read_excel(filepath)
        print(f"[OK] Costos cargados desde {filepath}: {len(df)} registros")
        return df

    # Sin archivo externo → calcular estimación
    if df_viajes is None or len(df_viajes) == 0:
        print("[AVISO] No hay datos de viajes para calcular costos estimados.")
        return pd.DataFrame()

    if 'km_ruta' not in df_viajes.columns:
        print("[AVISO] Columna km_ruta no encontrada. No se puede calcular costo.")
        return pd.DataFrame()

    df_costos = pd.DataFrame()
    df_costos['viaje_id'] = df_viajes['viaje_id']

    km = df_viajes['km_ruta'].fillna(df_viajes['km_ruta'].median())

    # Fórmula de estimación de costos
    df_costos['combustible_est_rd'] = (km * CONSUMO_GALON_POR_KM * PRECIO_GALON_RD).round(2)
    df_costos['pago_chofer_rd']     = PAGO_CHOFER_FIJO_RD
    df_costos['costo_estimado']     = (df_costos['combustible_est_rd'] + PAGO_CHOFER_FIJO_RD).round(2)
    df_costos['costo_km_est']       = (df_costos['costo_estimado'] / km.replace(0, np.nan)).round(4)

    costo_prom = df_costos['costo_estimado'].mean()
    costo_min  = df_costos['costo_estimado'].min()
    costo_max  = df_costos['costo_estimado'].max()

    print(f"[OK] Costos estimados calculados: {len(df_costos)} viajes")
    print(f"     Fórmula: km × {CONSUMO_GALON_POR_KM} gal/km × RD${PRECIO_GALON_RD} + RD${PAGO_CHOFER_FIJO_RD}")
    print(f"     Costo estimado — promedio: RD${costo_prom:,.0f} | min: RD${costo_min:,.0f} | max: RD${costo_max:,.0f}")

    return df_costos


def cargar_flota(filepath=None):
    """
    Carga el catálogo de los 7 vehículos reales de la flota.

    CAMBIO v2.0: Se reemplaza el generador sintético por el catálogo
    de los vehículos identificados en el dataset GPS real.
    Las capacidades son estimaciones operativas (empresa no las registra).

    Vehículos: L322837, L330617, L344749, L354062, L386793, L412568, L462601
    """
    if filepath and os.path.exists(filepath):
        df = pd.read_excel(filepath)
        print(f"[OK] Flota cargada desde {filepath}")
        return df

    # Catálogo de los 7 vehículos reales identificados en el GPS
    df = pd.DataFrame({
        'vehiculo_id':  ['L322837', 'L330617', 'L344749', 'L354062',
                         'L386793', 'L412568', 'L462601'],
        'tipo':         ['Camion Mediano', 'Camion Mediano', 'Camion Mediano',
                         'Camion Mediano', 'Camion Liviano', 'Camion Liviano',
                         'Furgon'],
        'capacidad_kg': [5000, 5000, 3500, 3500, 6000, 6000, 2000],
        'capacidad_m3': [25, 25, 18, 18, 30, 30, 10],
        'anio':         [2019, 2020, 2021, 2022, 2018, 2023, 2020],
        'estado':       ['Activo'] * 7
    })

    print(f"[OK] Catálogo de flota: {len(df)} vehículos reales cargados")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: AUDITORÍA DE CALIDAD
# ══════════════════════════════════════════════════════════════════════════════

def auditar_calidad(df, nombre, guardar=True):
    """
    Genera un diagnóstico completo de calidad del DataFrame.

    Reporta:
      - Dimensiones (filas × columnas)
      - Valores nulos por columna (cantidad y porcentaje)
      - Tipos de datos
      - Estadísticas descriptivas básicas
      - Detección de outliers extremos (método IQR)

    Returns:
        dict: Resumen de calidad con métricas clave
    """
    print(f"\n{'='*60}")
    print(f"AUDITORÍA DE CALIDAD: {nombre}")
    print(f"{'='*60}")
    print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

    # Nulos
    nulos     = df.isnull().sum()
    pct_nulos = (nulos / len(df) * 100).round(2)
    df_nulos  = pd.DataFrame({
        "Nulos": nulos,
        "Porcentaje (%)": pct_nulos
    }).query("Nulos > 0")

    if len(df_nulos) > 0:
        print(f"\nColumnas con valores nulos:")
        print(df_nulos.to_string())
    else:
        print("\n✓ Sin valores nulos detectados.")

    # Duplicados
    n_dup = df.duplicated().sum()
    print(f"\nRegistros duplicados: {n_dup} ({n_dup/len(df)*100:.1f}%)")

    # Tipos de datos
    print(f"\nTipos de datos:\n{df.dtypes.to_string()}")

    # Estadísticas numéricas
    print(f"\nEstadísticas descriptivas:")
    print(df.describe().round(2).to_string())

    # Resumen
    reporte = {
        "dataset":           nombre,
        "filas":             df.shape[0],
        "columnas":          df.shape[1],
        "nulos_total":       int(nulos.sum()),
        "pct_nulos":         float(pct_nulos.max()) if len(pct_nulos) > 0 else 0,
        "duplicados":        int(n_dup),
        "timestamp":         datetime.now().isoformat()
    }

    if guardar:
        path = os.path.join(REPORT_DIR, f"calidad_{nombre.lower().replace(' ','_')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(reporte, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Reporte guardado en {path}")

    return reporte


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: LIMPIEZA Y TRATAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def limpiar_viajes(df):
    """
    Limpieza del dataset de viajes GPS reales.

    CAMBIO v2.0: Adaptado para las columnas del CSV real:
      - Usa t_inicio/t_fin en lugar de fecha_salida/fecha_llegada
      - Las variables objetivo (delta_t, ind_retraso, riesgo_cat) ya existen;
        NO se recalculan, solo se validan.
      - Filtro de duraciones: t_real_min entre 1 y 720 minutos (12h)
      - No se imputar carga_kg ni capacidad_kg (no existen en el dataset)

    Operaciones:
      1. Eliminación de registros con km_ruta nulo o negativo
      2. Filtro de duraciones imposibles (t_real_min < 1 o > 720 min)
      3. Imputación de nulos numéricos con mediana
      4. Eliminación de duplicados exactos
    """
    df = df.copy()
    n_original = len(df)

    # 1. Eliminar km_ruta inválidos
    if 'km_ruta' in df.columns:
        df = df[df['km_ruta'] > 0]

    # 2. Filtrar duraciones imposibles en t_real_min
    if 't_real_min' in df.columns:
        df = df[(df['t_real_min'] > 0) & (df['t_real_min'] < 720)]

    # 3. Imputar numéricas con mediana (excepto variables objetivo)
    vars_objetivo = {'delta_t', 'ind_retraso'}
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_num = [c for c in cols_num if c not in vars_objetivo]

    for col in cols_num:
        n_nulos = df[col].isnull().sum()
        if n_nulos > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(f"[IMPUT] {col}: {n_nulos} nulos → mediana ({mediana:.2f})")

    # 4. Eliminar duplicados
    df = df.drop_duplicates()

    n_final = len(df)
    print(f"\n[LIMPIEZA] Viajes: {n_original} → {n_final} registros "
          f"({n_original - n_final} eliminados, {(n_original-n_final)/n_original*100:.1f}%)")

    return df


def limpiar_ordenes(df):
    """
    Limpieza del dataset de órdenes de entrega.

    CAMBIO v2.0: Como las órdenes se derivan del CSV de viajes,
    solo se valida la coherencia de t_real_min y t_est_min.
    El campo otif se preserva tal cual viene de cargar_ordenes().
    """
    df = df.copy()

    # Validar tiempos positivos
    for col in ['t_est_min', 't_real_min']:
        if col in df.columns:
            df = df[df[col] > 0]

    df = df.drop_duplicates()
    return df


def limpiar_costos(df):
    """
    Limpieza del dataset de costos operativos estimados.

    CAMBIO v2.0: Valida que costo_estimado sea positivo.
    No hay componentes reales de combustible/viáticos/mantenimiento.
    """
    df = df.copy()

    col_costo = 'costo_estimado' if 'costo_estimado' in df.columns else 'costo_total'

    if col_costo in df.columns:
        df = df[df[col_costo] > 0]

    df = df.drop_duplicates()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: INTEGRACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def integrar_datasets(df_viajes, df_ordenes, df_costos, df_flota):
    """
    Integra los datasets usando viaje_id como llave primaria.

    CAMBIO v2.0: Simplificado respecto a la versión anterior:
      - El merge de órdenes solo agrega otif (ind_retraso ya está en viajes)
      - El merge de costos agrega costo_estimado y costo_km_est
      - El merge de flota agrega tipo y capacidad_kg por vehiculo_id
      - No hay columnas carga_kg / capacidad_kg en viajes (se omiten)

    Estrategia: left join sobre df_viajes para preservar todos los registros.

    Returns:
        pd.DataFrame: Dataset integrado con todas las variables disponibles
    """
    print("\n[INTEGRACIÓN] Uniendo datasets...")

    n_base = len(df_viajes)

    # Join con costos estimados
    if df_costos is not None and len(df_costos) > 0:
        cols_costos = ['viaje_id'] + [
            c for c in df_costos.columns
            if c not in df_viajes.columns and c != 'viaje_id'
        ]
        cols_costos = [c for c in cols_costos if c in df_costos.columns]
        df = df_viajes.merge(df_costos[cols_costos], on='viaje_id', how='left')
        match_costos = df['costo_estimado'].notna().sum() if 'costo_estimado' in df.columns else 0
        print(f"  Viajes con costo estimado: {match_costos}/{n_base} ({match_costos/n_base*100:.1f}%)")
    else:
        df = df_viajes.copy()
        print("  [AVISO] Sin datos de costos para integrar.")

    # Join con flota (agrega tipo y capacidad estimada)
    if df_flota is not None and len(df_flota) > 0 and 'vehiculo_id' in df_flota.columns:
        cols_flota = ['vehiculo_id'] + [
            c for c in ['tipo', 'capacidad_kg', 'capacidad_m3']
            if c in df_flota.columns and c not in df.columns
        ]
        df = df.merge(df_flota[cols_flota], on='vehiculo_id', how='left')
        match_flota = df['tipo'].notna().sum() if 'tipo' in df.columns else 0
        print(f"  Viajes con datos de flota: {match_flota}/{n_base} ({match_flota/n_base*100:.1f}%)")

    print(f"\n[OK] Dataset integrado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5: EXPORTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def exportar_repositorio(df, nombre="dataset_integrado"):
    """
    Guarda el repositorio analítico integrado en processed/ y en final/.
    Genera también un diccionario de datos automático.
    """
    path_processed = os.path.join(PROCESSED_DIR, f"{nombre}.csv")
    df.to_csv(path_processed, index=False)
    print(f"\n[OK] Repositorio analítico guardado: {path_processed}")

    # Diccionario de datos
    diccionario = []
    for col in df.columns:
        diccionario.append({
            "variable": col,
            "tipo":     str(df[col].dtype),
            "nulos":    int(df[col].isnull().sum()),
            "pct_nulos": round(df[col].isnull().sum() / len(df) * 100, 2),
            "n_unicos": int(df[col].nunique()),
            "ejemplo":  str(df[col].dropna().iloc[0]) if df[col].notna().any() else "N/A"
        })

    df_dict = pd.DataFrame(diccionario)
    path_dict = os.path.join(REPORT_DIR, "diccionario_datos.csv")
    df_dict.to_csv(path_dict, index=False)
    print(f"[OK] Diccionario de datos guardado: {path_dict}")

    return path_processed


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_pipeline():
    """
    Punto de entrada principal del pipeline de datos.
    Ejecuta las 5 fases en orden y exporta el repositorio analítico.

    CAMBIO v2.0: Ahora carga los CSV reales de GPS en lugar de datos sintéticos.
    La firma de la función se mantiene igual para compatibilidad con features.py,
    models.py y el dashboard.
    """
    print("\n" + "="*60)
    print("MOETC-BD — PIPELINE DE DATOS v2.0 (Datos Reales GPS)")
    print("Capa 1: Data Layer")
    print("="*60)

    # FASE 1: Ingesta de datos reales
    print("\n[FASE 1] Ingesta de datos reales GPS...")
    df_viajes  = cargar_viajes()               # Lee dataset_gps_procesado.csv
    df_costos  = cargar_costos(df_viajes)      # Calcula costo_estimado con fórmula
    df_ordenes = cargar_ordenes(df_viajes)     # Deriva otif de ind_retraso
    df_flota   = cargar_flota()                # Catálogo de 7 vehículos reales

    if len(df_viajes) == 0:
        print("\n[ERROR CRÍTICO] No se pudieron cargar los datos de viajes.")
        print("  Verifica que data/raw/dataset_gps_procesado.csv existe y es accesible.")
        return pd.DataFrame()

    # FASE 2: Auditoría de calidad
    print("\n[FASE 2] Auditoría de calidad AS IS...")
    reportes = {}
    for df, nombre in [(df_viajes, "Viajes"), (df_costos, "Costos_Estimados")]:
        if len(df) > 0:
            reportes[nombre] = auditar_calidad(df, nombre)

    # FASE 3: Limpieza
    print("\n[FASE 3] Limpieza y tratamiento...")
    df_viajes  = limpiar_viajes(df_viajes)
    df_ordenes = limpiar_ordenes(df_ordenes)
    df_costos  = limpiar_costos(df_costos)

    # FASE 4: Integración
    print("\n[FASE 4] Integración de fuentes...")
    df_integrado = integrar_datasets(df_viajes, df_ordenes, df_costos, df_flota)

    # FASE 5: Exportación
    print("\n[FASE 5] Exportación del repositorio analítico...")
    path = exportar_repositorio(df_integrado)

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO")
    print(f"   Dataset: {df_integrado.shape[0]} registros × {df_integrado.shape[1]} variables")
    print(f"   Guardado en: {path}")
    print("="*60)

    return df_integrado


if __name__ == "__main__":
    df = ejecutar_pipeline()
