"""
src/kpis.py
===========
MOETC-BD — Cálculo de KPIs logísticos
Capa 4: BI Layer del modelo MOETC-BD

Corresponde a la sección 3.1.7 de la tesis:
"Componente IV: Dashboard de KPIs e Inteligencia de Negocios"

KPIs implementados (8 indicadores del modelo):
  1. OTIF (%)           — On Time In Full
  2. Fill Rate (%)      — Tasa de ocupación de flota
  3. Retraso promedio   — Promedio de delta_t en viajes retrasados
  4. Costo por viaje    — Promedio de costo operativo por viaje
  5. Km vacío (%)       — Porcentaje de km sin carga útil
  6. CO2 evitado (kg)   — Emisiones evitadas vs. AS IS
  7. Efficiency Score E — Índice compuesto E = α·OTIF − β·C − γ·D
  8. Utilización flota  — Vehículos activos / total flota

CAMBIOS v2.0 (datos reales GPS):
  - OTIF_AS_IS_REAL = 76.2%: constante documental del estado base real
    (calculado directamente del CSV: 3512/4608 viajes sin retraso).
  - ALERTAS["otif"]: umbrales actualizados verde=90%, amarillo=83%
    (antes: verde=95%, amarillo=90%). Alineados con benchmark sector RD.
  - calcular_otif(): prioriza ind_retraso si disponible (directo del CSV);
    fallback a delta_t o campo otif genérico.
  - calcular_costo_promedio_viaje(): busca costo_estimado además de costo_total.
  - calcular_efficiency_score(): usa costo_estimado si costo_total no existe.
  - Efficiency Score: α=0.50, β=0.30, γ=0.20 (sin cambio en pesos).
"""

import numpy as np
import pandas as pd

# Pesos del Efficiency Score (ajustables)
ALPHA = 0.50   # peso del OTIF
BETA  = 0.30   # peso del costo (penaliza costos altos)
GAMMA = 0.20   # peso del retraso (penaliza retrasos)

# NUEVO v2.0: OTIF real calculado del dataset GPS (3512/4608 viajes sin retraso)
# Valor de referencia AS IS para el modelo y la tesis. No se modifica en tiempo de ejecución.
OTIF_AS_IS_REAL = 76.2  # %

# Umbrales de alertas (semáforo: verde / amarillo / rojo)
# CAMBIO v2.0: OTIF verde ≥ 90%, amarillo ≥ 83% (antes 95%/90%)
# Justificación: benchmark sector logístico RD y gap realista desde el 76.2% actual.
ALERTAS = {
    "otif":           {"verde": 90, "amarillo": 83},   # % — ACTUALIZADO v2.0
    "fill_rate":      {"verde": 80, "amarillo": 70},   # %
    "retraso_prom":   {"verde": 15, "amarillo": 25},   # minutos (menor = mejor)
    "costo_viaje":    {"verde": None, "amarillo": None},  # dinámico
    "km_vacio_pct":   {"verde": 10, "amarillo": 15},   # %
}


# ══════════════════════════════════════════════════════════════════════════════
# KPIs INDIVIDUALES
# ══════════════════════════════════════════════════════════════════════════════

def calcular_otif(df, umbral_retraso_min=15):
    """
    OTIF = (Viajes entregados a tiempo / Total viajes) × 100

    CAMBIO v2.0: Orden de prioridad para calcular OTIF:
      1. ind_retraso (columna real del CSV: 0=a tiempo, 1=retrasado)
      2. delta_t (calcula: a tiempo si delta_t <= umbral_retraso_min)
      3. otif (campo clásico 0/1 de versiones anteriores)

    El valor real del dataset AS IS es 76.2% (constante OTIF_AS_IS_REAL).
    """
    if len(df) == 0:
        return None

    # Prioridad 1: ind_retraso del CSV real (0=a tiempo, 1=retrasado)
    if "ind_retraso" in df.columns:
        a_tiempo = (df["ind_retraso"] == 0).sum()
        return round(a_tiempo / len(df) * 100, 2)

    # Prioridad 2: delta_t calculado
    if "delta_t" in df.columns:
        a_tiempo = (df["delta_t"] <= umbral_retraso_min).sum()
        return round(a_tiempo / len(df) * 100, 2)

    # Prioridad 3: campo otif genérico
    if "otif" in df.columns:
        a_tiempo = df["otif"].sum()
        return round(a_tiempo / len(df) * 100, 2)

    return None


def calcular_fill_rate(df):
    """
    Fill Rate = Promedio de (carga_kg / capacidad_kg) × 100
    Mide la utilización promedio de la capacidad de carga.

    NOTA v2.0: La empresa no registra carga_kg. Si fill_rate no está disponible
    retorna None sin lanzar excepción.
    """
    if "fill_rate" in df.columns:
        return round(df["fill_rate"].mean() * 100, 2)
    if "carga_kg" in df.columns and "capacidad_kg" in df.columns:
        fr = (df["carga_kg"] / df["capacidad_kg"].replace(0, np.nan)).mean()
        return round(fr * 100, 2)
    return None


def calcular_retraso_promedio(df):
    """
    Retraso promedio = Promedio de delta_t en viajes con delta_t > 15 min.
    Solo considera viajes efectivamente retrasados.
    """
    if "delta_t" not in df.columns:
        return None
    retrasados = df[df["delta_t"] > 15]["delta_t"]
    if len(retrasados) == 0:
        return 0.0
    return round(retrasados.mean(), 1)


def calcular_costo_promedio_viaje(df):
    """
    Costo operativo promedio por viaje en RD$.

    CAMBIO v2.0: Busca costo_estimado (calculado en data_pipeline.py)
    además de costo_total (que la empresa no registra directamente).
    """
    # Prioridad 1: costo_estimado (fórmula del pipeline v2.0)
    if "costo_estimado" in df.columns:
        return round(df["costo_estimado"].mean(), 2)
    # Prioridad 2: costo_total (compatibilidad con versiones anteriores)
    if "costo_total" in df.columns:
        return round(df["costo_total"].mean(), 2)
    return None


def calcular_km_vacio_pct(df):
    """
    Km vacío % = (Km recorridos sin carga / Total km) × 100
    Proxy: viajes con fill_rate < 0.10 se consideran "vacíos".

    NOTA v2.0: Requiere fill_rate y km_ruta. Retorna None si fill_rate
    no está disponible (empresa no registra carga_kg).
    """
    if "fill_rate" not in df.columns or "km_ruta" not in df.columns:
        return None
    km_vacio = df[df["fill_rate"] < 0.10]["km_ruta"].sum()
    km_total = df["km_ruta"].sum()
    if km_total == 0:
        return 0.0
    return round(km_vacio / km_total * 100, 2)


def calcular_co2_evitado(km_as_is, km_to_be, factor_emision=0.35):
    """
    CO2 evitado = (km_as_is - km_to_be) × factor_emision (kgCO2/km)
    Fórmula: CO2_evitado = Σ(km_actual − km_opt) × EF
    Alineado con ODS 9 y ODS 13.
    """
    ahorro_km = km_as_is - km_to_be
    return round(max(ahorro_km, 0) * factor_emision, 2)


def calcular_efficiency_score(df, alpha=ALPHA, beta=BETA, gamma=GAMMA,
                               km_as_is=None, km_to_be=None):
    """
    E = α·OTIF_norm − β·C_norm − γ·D_norm

    donde:
      OTIF_norm = OTIF / 100
      C_norm    = costo_actual / costo_max_historico  (normalizado 0-1)
      D_norm    = retraso_actual / 60 (normalizado, máx 1 hora)

    Restricción: α + β + γ = 1
    Pesos: α=0.50, β=0.30, γ=0.20 (sin cambio v2.0)

    CAMBIO v2.0: Busca costo_estimado si costo_total no existe.
    """
    assert abs(alpha + beta + gamma - 1.0) < 1e-6, "α + β + γ debe ser igual a 1"

    otif_val    = calcular_otif(df)
    retraso_val = calcular_retraso_promedio(df)
    costo_val   = calcular_costo_promedio_viaje(df)

    if otif_val is None:
        return None

    otif_norm  = otif_val / 100
    delay_norm = min((retraso_val or 0) / 60, 1.0)

    # Normalización del costo usando la columna disponible
    col_costo = None
    if "costo_estimado" in df.columns:
        col_costo = "costo_estimado"
    elif "costo_total" in df.columns:
        col_costo = "costo_total"

    if costo_val and col_costo:
        costo_max  = df[col_costo].max()
        costo_norm = costo_val / costo_max if costo_max > 0 else 0
    else:
        costo_norm = 0

    E = alpha * otif_norm - beta * costo_norm - gamma * delay_norm
    return round(float(E), 4)


# ══════════════════════════════════════════════════════════════════════════════
# SEMÁFORO DE KPIs
# ══════════════════════════════════════════════════════════════════════════════

def evaluar_semaforo(kpi_nombre, valor):
    """
    Evalúa el estado de un KPI según los umbrales de alerta definidos.

    Returns:
        str: "VERDE" | "AMARILLO" | "ROJO"
    """
    if kpi_nombre not in ALERTAS or valor is None:
        return "N/A"

    umbrales = ALERTAS[kpi_nombre]
    verde    = umbrales["verde"]
    amarillo = umbrales["amarillo"]

    if verde is None:
        return "N/A"

    # KPIs donde más es mejor (OTIF, fill_rate)
    if kpi_nombre in ("otif", "fill_rate"):
        if valor >= verde:
            return "VERDE"
        elif valor >= amarillo:
            return "AMARILLO"
        else:
            return "ROJO"

    # KPIs donde menos es mejor (retraso, km_vacio)
    else:
        if valor <= verde:
            return "VERDE"
        elif valor <= amarillo:
            return "AMARILLO"
        else:
            return "ROJO"


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def calcular_resumen_kpis(df, km_as_is=None, km_to_be=None, label=""):
    """
    Calcula todos los KPIs del modelo MOETC-BD y devuelve
    un diccionario completo con valores y estados de semáforo.

    Args:
        df        : DataFrame del período a evaluar
        km_as_is  : km totales históricos (para CO2 evitado)
        km_to_be  : km optimizados por VRP (para CO2 evitado)
        label     : etiqueta del período ("AS IS" / "TO BE")

    Returns:
        dict: KPIs con valor + semáforo
    """
    otif      = calcular_otif(df)
    fill_rate = calcular_fill_rate(df)
    retraso   = calcular_retraso_promedio(df)
    costo     = calcular_costo_promedio_viaje(df)
    km_vacio  = calcular_km_vacio_pct(df)
    e_score   = calcular_efficiency_score(df)
    co2       = calcular_co2_evitado(km_as_is, km_to_be) if (km_as_is and km_to_be) else None

    resumen = {
        "periodo":              label,
        "n_viajes":             len(df),
        "OTIF (%)":             otif,
        "Fill Rate (%)":        fill_rate,
        "Retraso prom. (min)":  retraso,
        "Costo prom./viaje":    costo,
        "Km vacio (%)":         km_vacio,
        "CO2 evitado (kg)":     co2,
        "Efficiency Score (E)": e_score,
        "semaforo_otif":        evaluar_semaforo("otif", otif),
        "semaforo_fill_rate":   evaluar_semaforo("fill_rate", fill_rate),
        "semaforo_retraso":     evaluar_semaforo("retraso_prom", retraso),
        "semaforo_km_vacio":    evaluar_semaforo("km_vacio_pct", km_vacio),
    }

    print(f"\n{'='*50}")
    print(f"KPIs — {label or 'Período analizado'}")
    print(f"{'='*50}")
    for k, v in resumen.items():
        if k not in ("periodo", "n_viajes") and not k.startswith("semaforo"):
            estado = resumen.get(f"semaforo_{k.split('(')[0].strip().lower().replace(' ','_')}", "")
            icono  = {"VERDE": "✓", "AMARILLO": "⚠", "ROJO": "✗"}.get(estado, " ")
            print(f"  {icono} {k:<28}: {v}")

    return resumen


def comparar_as_is_to_be(df_as_is, df_to_be,
                          km_as_is=None, km_to_be=None):
    """
    Genera la tabla comparativa AS IS vs TO BE para el Capítulo IV.

    Returns:
        pd.DataFrame: tabla lista para insertar en la tesis
    """
    kpis_as_is = calcular_resumen_kpis(df_as_is, label="AS IS")
    kpis_to_be = calcular_resumen_kpis(df_to_be, km_as_is=km_as_is,
                                        km_to_be=km_to_be, label="TO BE")

    metricas = ["OTIF (%)", "Fill Rate (%)", "Retraso prom. (min)",
                "Costo prom./viaje", "Km vacio (%)", "Efficiency Score (E)"]

    filas = []
    for m in metricas:
        v_as = kpis_as_is.get(m)
        v_to = kpis_to_be.get(m)
        if v_as is not None and v_to is not None:
            delta = v_to - v_as
            pct   = (delta / abs(v_as) * 100) if v_as != 0 else 0
            filas.append({
                "KPI":           m,
                "AS IS":         round(v_as, 2),
                "TO BE (proy.)": round(v_to, 2),
                "Δ Absoluto":    round(delta, 2),
                "Δ (%)":         round(pct, 1),
            })

    df_comp = pd.DataFrame(filas)
    df_comp.to_csv("models/reports/comparativo_as_is_to_be.csv", index=False)
    print("\n[OK] Tabla comparativa guardada en models/reports/")
    return df_comp
