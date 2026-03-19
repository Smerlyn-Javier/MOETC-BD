"""
dashboard/app.py
================
MOETC-BD — Dashboard de KPIs e Inteligencia de Negocios
Capa 4: BI Layer del modelo MOETC-BD

Ejecutar con:  streamlit run dashboard/app.py
URL local:     http://localhost:8501

Vistas disponibles:
  1. Vista Operativa  — Estado diario de rutas y alertas
  2. Vista Analítica  — Tendencias semanales y modelos ML
  3. Vista Financiera — Costos y eficiencia económica
  4. Vista Ejecutiva  — Efficiency Score E y comparativo AS IS vs TO BE
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.kpis import calcular_resumen_kpis, evaluar_semaforo

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOETC-BD Dashboard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS PERSONALIZADO ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  .kpi-card { background:#1E3A5F; border-radius:10px; padding:16px; text-align:center; }
  .kpi-valor { font-size:2rem; font-weight:bold; color:#42A5F5; }
  .kpi-label { font-size:0.85rem; color:#90A4AE; }
  .verde   { color:#66BB6A; }
  .amarillo{ color:#FFA726; }
  .rojo    { color:#EF5350; }
</style>
""", unsafe_allow_html=True)


# ── DATOS ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    path = "data/final/dataset_final.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["t_inicio"])
    else:
        # Datos demo si no hay dataset real
        from src.data_pipeline import ejecutar_pipeline
        from src.features      import ejecutar_feature_engineering
        df = ejecutar_feature_engineering(ejecutar_pipeline())
    return df

df_global = cargar_datos()


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=MOETC-BD", width=200)
    st.markdown("### Filtros")

    if "t_inicio" in df_global.columns:
        fecha_min = df_global["t_inicio"].min().date()
        fecha_max = df_global["t_inicio"].max().date()
        rango_fecha = st.date_input("Período", [fecha_min, fecha_max])
    else:
        rango_fecha = None

    if "ruta" in df_global.columns:
        rutas_disp = sorted(df_global["ruta"].dropna().unique().tolist())
        rutas_sel  = st.multiselect("Rutas", rutas_disp, default=rutas_disp)
    else:
        rutas_sel = []

    vista = st.radio("Vista", [
        "🏠 Operativa",
        "📈 Analítica",
        "💰 Financiera",
        "🎯 Ejecutiva",
    ])

    st.divider()
    st.caption("MOETC-BD v1.0 · López Laureano Distribución · 2026")


# ── FILTRAR DATOS ──────────────────────────────────────────────────────────────
df = df_global.copy()
if rango_fecha and len(rango_fecha) == 2 and "t_inicio" in df.columns:
    df = df[(df["t_inicio"].dt.date >= rango_fecha[0]) &
            (df["t_inicio"].dt.date <= rango_fecha[1])]
if rutas_sel and "ruta" in df.columns:
    df = df[df["ruta"].isin(rutas_sel)]


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.title("MOETC-BD — Dashboard de Eficiencia Logística")
st.markdown(f"**López Laureano Distribución** · Santo Domingo, RD &nbsp;|&nbsp; "
            f"**{len(df):,}** viajes en el período seleccionado")
st.divider()


# ── KPIs PRINCIPALES ───────────────────────────────────────────────────────────
def render_kpi(col, label, valor, unidad="", meta=None, invert=False):
    """Renderiza una tarjeta KPI con semáforo."""
    if valor is None:
        col.metric(label, "N/D")
        return

    texto = f"{valor:,.1f}{unidad}"
    delta_txt = f"Meta: {meta}{unidad}" if meta else None

    if meta:
        cumple = valor >= meta if not invert else valor <= meta
        color  = "normal" if cumple else "inverse"
    else:
        color = "off"

    col.metric(label, texto, delta=delta_txt, delta_color=color)


kpis = calcular_resumen_kpis(df, label="Período seleccionado")

c1, c2, c3, c4, c5 = st.columns(5)
render_kpi(c1, "OTIF",              kpis.get("OTIF (%)"),            "%",   95)
render_kpi(c2, "Fill Rate",         kpis.get("Fill Rate (%)"),       "%",   80)
render_kpi(c3, "Retraso Prom.",     kpis.get("Retraso prom. (min)"), " min", 15, invert=True)
render_kpi(c4, "Costo/Viaje",       kpis.get("Costo prom./viaje"),   " RD$")
render_kpi(c5, "Efficiency Score",  kpis.get("Efficiency Score (E)"))

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# VISTA OPERATIVA
# ══════════════════════════════════════════════════════════════════════════════
if vista == "🏠 Operativa":
    st.subheader("Vista Operativa — Estado de Rutas")

    col_a, col_b = st.columns(2)

    with col_a:
        if "ruta" in df.columns and "ind_retraso" in df.columns:
            df_ruta = (df.groupby("ruta")["ind_retraso"]
                         .agg(["mean", "count"])
                         .reset_index()
                         .rename(columns={"mean": "tasa_retraso", "count": "n_viajes"}))
            df_ruta["tasa_retraso_pct"] = df_ruta["tasa_retraso"] * 100

            fig = px.bar(
                df_ruta.sort_values("tasa_retraso_pct", ascending=True),
                x="tasa_retraso_pct", y="ruta",
                orientation="h",
                color="tasa_retraso_pct",
                color_continuous_scale="RdYlGn_r",
                range_color=[0, 40],
                title="Tasa de Retraso por Ruta (%)",
                labels={"tasa_retraso_pct": "% Retrasos", "ruta": "Ruta"},
            )
            fig.add_vline(x=5, line_dash="dash", line_color="green",
                          annotation_text="Meta 5%")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "riesgo_cat" in df.columns:
            df_riesgo = df["riesgo_cat"].value_counts().reset_index()
            df_riesgo.columns = ["Riesgo", "Viajes"]
            color_map = {"BAJO": "#66BB6A", "MEDIO": "#FFA726", "ALTO": "#EF5350"}

            fig2 = px.pie(
                df_riesgo, names="Riesgo", values="Viajes",
                title="Distribución de Riesgo de Viajes",
                color="Riesgo", color_discrete_map=color_map,
                hole=0.4,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Tabla de últimos viajes con alerta
    if "delta_t" in df.columns:
        cols_mostrar = [c for c in
            ["viaje_id","ruta","t_inicio","delta_t","riesgo_cat","otif"]
            if c in df.columns]
        df_alertas = df[df.get("delta_t", pd.Series()) > 15][cols_mostrar].head(10)
        if len(df_alertas) > 0:
            st.subheader("⚠ Viajes con Retraso Detectado")
            st.dataframe(df_alertas, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VISTA ANALÍTICA
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "📈 Analítica":
    st.subheader("Vista Analítica — Tendencias y Modelos ML")

    if "t_inicio" in df.columns and "delta_t" in df.columns:
        df_trend = (df.set_index("t_inicio")
                      .resample("W")["delta_t"]
                      .agg(["mean", "median", "count"])
                      .reset_index())

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df_trend["t_inicio"], y=df_trend["mean"],
            name="Retraso prom.", line=dict(color="#42A5F5")),
            secondary_y=False)
        fig.add_trace(go.Bar(
            x=df_trend["t_inicio"], y=df_trend["count"],
            name="N° Viajes", opacity=0.3, marker_color="#90A4AE"),
            secondary_y=True)
        fig.add_hline(y=15, line_dash="dash", line_color="red",
                      annotation_text="Umbral 15 min")
        fig.update_layout(title="Evolución Semanal del Retraso Promedio")
        st.plotly_chart(fig, use_container_width=True)

    if "fill_rate" in df.columns and "franja_horaria" in df.columns:
        fig3 = px.box(
            df, x="franja_horaria", y="delta_t",
            title="Distribución de Retraso por Franja Horaria",
            color="franja_horaria",
            color_discrete_map={"MAÑANA":"#42A5F5","TARDE":"#FFA726","NOCHE":"#7E57C2"},
        )
        st.plotly_chart(fig3, use_container_width=True)

    img_path = "models/reports/importancia_variables_mp-1.png"
    if os.path.exists(img_path):
        st.subheader("Importancia de Variables — Modelo MP-1")
        st.image(img_path)

    # Gráfica real vs predicho
    img_rvp = "models/reports/real_vs_predicho_mp-1_regresión.png"
    if os.path.exists(img_rvp):
        st.subheader("Evaluación de Predicciones — MP-1")
        st.image(img_rvp)

    # Evaluación y Métricas (Confusion Matrix y ROC Curve)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        img_conf = "models/reports/confusion_mp-3_otif.png"
        if os.path.exists(img_conf):
            st.subheader("Matriz de Confusión — MP-3")
            st.image(img_conf)
    
    with col_m2:
        img_roc = "models/reports/roc_mp-3_otif.png"
        if os.path.exists(img_roc):
            st.subheader("Curva ROC — MP-3")
            st.image(img_roc)


# ══════════════════════════════════════════════════════════════════════════════
# VISTA FINANCIERA
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "💰 Financiera":
    st.subheader("Vista Financiera — Análisis de Costos")

    if "t_inicio" in df.columns and "costo_estimado" in df.columns:
        df_costo = (df.set_index("t_inicio")
                      .resample("M")["costo_estimado"]
                      .agg(["mean", "sum"])
                      .reset_index())

        col_x, col_y = st.columns(2)
        with col_x:
            fig_c1 = px.line(df_costo, x="t_inicio", y="mean",
                             title="Costo Promedio por Viaje (RD$)",
                             markers=True)
            st.plotly_chart(fig_c1, use_container_width=True)
        with col_y:
            fig_c2 = px.bar(df_costo, x="t_inicio", y="sum",
                            title="Costo Total Mensual (RD$)")
            st.plotly_chart(fig_c2, use_container_width=True)

    if "costo_km" in df.columns and "ruta" in df.columns:
        fig_km = px.box(df, x="ruta", y="costo_km",
                        title="Costo por Km según Ruta (RD$/km)",
                        color="ruta")
        st.plotly_chart(fig_km, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VISTA EJECUTIVA
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "🎯 Ejecutiva":
    st.subheader("Vista Ejecutiva — Efficiency Score y Proyección TO BE")

    e_score = kpis.get("Efficiency Score (E)")
    if e_score is not None:
        col_e1, col_e2, col_e3 = st.columns([1, 2, 1])
        with col_e2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=e_score,
                title={"text": "Efficiency Score (E)<br><span style='font-size:0.7em'>E = α·OTIF − β·C − γ·D</span>"},
                delta={"reference": 0.5, "increasing": {"color": "green"}},
                gauge={
                    "axis": {"range": [-0.5, 1]},
                    "bar":  {"color": "#1565C0"},
                    "steps": [
                        {"range": [-0.5, 0.2], "color": "#EF5350"},
                        {"range": [0.2, 0.5],  "color": "#FFA726"},
                        {"range": [0.5, 1.0],  "color": "#66BB6A"},
                    ],
                    "threshold": {"value": 0.5, "line": {"color": "white", "width": 3}},
                },
            ))
            fig_gauge.update_layout(height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Tabla comparativa AS IS vs TO BE
    comp_path = "models/reports/comparativo_as_is_to_be.csv"
    if os.path.exists(comp_path):
        st.subheader("Comparativo AS IS vs TO BE")
        df_comp = pd.read_csv(comp_path)
        st.dataframe(
            df_comp.style.applymap(
                lambda v: "color: green; font-weight: bold"
                if isinstance(v, (int, float)) and v > 0 else "",
                subset=["Δ (%)"]
            ),
            use_container_width=True
        )
    else:
        st.info("Ejecuta el notebook 05_kpis.ipynb para generar el comparativo AS IS vs TO BE.")
