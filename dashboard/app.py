"""
dashboard/app.py
================
MOETC-BD — Dashboard de KPIs e Inteligencia de Negocios
One-Page Report — Modern UI Style
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib 
import datetime 
from src.kpis import calcular_resumen_kpis, evaluar_semaforo

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOETC-BD | BlueSky Simulator",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS PERSONALIZADO (BlueSky Style) ──────────────────────────────────────────
st.markdown("""
<style>
    /* Ocultar barra lateral, header default y padding extra */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
    header {visibility: hidden;}
    .block-container { 
        padding-top: 2rem !important; 
        padding-bottom: 2rem !important; 
        max-width: 1500px !important; 
        background-color: #f7f9fc;
    }
    .stApp { background-color: #f7f9fc; }
    
    /* Tipografía y Textos */
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    h1, h2, h3, h4 { color: #111827; font-weight: 600; margin-bottom: 0.5rem; }
    p { color: #6b7280; }

    /* Tarjetas principales (Cards) */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
        background-color: white;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.03);
        border: 1px solid #f3f4f6;
    }

    /* Modificadores de Métricas de Streamlit nativas */
    div[data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700 !important; color: #111827 !important; }
    div[data-testid="stMetricDelta"] { font-size: 0.95rem !important; }
    
    /* Píldoras Personalizadas / Radio Buttons transformados */
    div.row-widget.stRadio > div {
        display: flex;
        flex-direction: row;
        background-color: white;
        padding: 8px;
        border-radius: 100px;
        box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.03);
        width: fit-content;
        gap: 8px;
    }
    div.row-widget.stRadio > div > label {
        background-color: transparent !important;
        border-radius: 100px !important;
        padding: 8px 16px !important;
        cursor: pointer;
    }
    div.row-widget.stRadio > div > label[data-baseweb="radio"] > div:first-child { display: none; } /* Ocultar circulito */
    
    /* Botones primarios */
    button[kind="primary"] {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 100px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3) !important;
    }
    button[kind="primary"]:hover { background-color: #2563eb !important; }
    
    /* Tarjeta Dark (BlueSky Premium) */
    .premium-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border-radius: 20px;
        padding: 30px;
        color: white;
        margin-top: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    .premium-card h3 { color: white; margin-bottom: 5px; font-size: 1.5rem; }
    .premium-card p { color: #9ca3af; font-size: 0.95rem; }
    .premium-deco {
        position: absolute; right: -20%; bottom: -40%;
        width: 250px; height: 250px;
        background: #3b82f6;
        opacity: 0.8;
        transform: rotate(30deg);
        border-radius: 40px;
    }

    /* Badges */
    .badge {
        display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;
    }
    .bg-green { background: #dcfce7; color: #16a34a; }
    .bg-yellow{ background: #fef08a; color: #ca8a04; }
    .bg-red   { background: #fee2e2; color: #dc2626; }
    .bg-blue  { background: #e0f2fe; color: #0284c7; }

    /* Inputs y Tablas */
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; padding: 12px; border-bottom: 2px solid #f3f4f6; color: #6b7280; font-weight: 500; font-size: 0.85rem;}
    td { padding: 16px 12px; border-bottom: 1px solid #f3f4f6; color: #111827; font-weight: 500; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)


# ── CARGA DE DATOS Y MODELOS ───────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    path = "data/final/dataset_final.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["t_inicio"])
    else:
        # Fallback local
        from src.data_pipeline import ejecutar_pipeline
        from src.features      import ejecutar_feature_engineering
        return ejecutar_feature_engineering(ejecutar_pipeline())

df_global = cargar_datos()

modelos = {}
try:
    modelos["mp1"] = joblib.load("models/saved/mp1_regresion.pkl")
    modelos["mp2"] = joblib.load("models/saved/mp2_riesgo.pkl")
    modelos["mp3"] = joblib.load("models/saved/mp3_otif.pkl")
except FileNotFoundError:
    st.error("⚠ Modelos ML no encontrados. Por favor entrena los modelos con `python -m src.models`")
    st.stop()


# NUEVO: VISTA DIAGNÓSTICO AS IS
@st.cache_data
def cargar_datos_diagnostico():
    path_gps = "data/processed/dataset_integrado.csv"
    path_puntos = "data/processed/puntos_entrega_gps.csv"
    
    try:
        df_gps = pd.read_csv(path_gps, parse_dates=["t_inicio"]) if os.path.exists(path_gps) else None
    except Exception:
        df_gps = pd.read_csv(path_gps) if os.path.exists(path_gps) else None
        
    df_puntos = pd.read_csv(path_puntos) if os.path.exists(path_puntos) else None
    if df_puntos is not None and "lat" in df_puntos.columns and "lon" in df_puntos.columns:
        df_puntos = df_puntos[(df_puntos["lat"].between(17.5, 20.0)) & (df_puntos["lon"].between(-71.5, -68.0))]
        if "dur_park_min" in df_puntos.columns:
            df_puntos = df_puntos[df_puntos["dur_park_min"] > 30]
            
    return df_gps, df_puntos

def vista_diagnostico_as_is():
    df, df_puntos = cargar_datos_diagnostico()
    if df is None:
        st.error("⚠ 'dataset_integrado.csv' no encontrado. Ejecuta primero: python -m src.data_pipeline && python -m src.features")
        return
        
    AZUL, VERDE, ROJO, NARANJA, GRIS = "#1565C0", "#1B5E20", "#B71C1C", "#E65100", "#546E7A"
    
    st.markdown("## 📊 Diagnóstico Operativo AS IS")
    st.markdown("Estado actual de la flota basado en datos GPS reales.")
    
    t1, t2, t3, t4 = st.tabs(["KPIs Ejecutivos", "Flota", "Tiempo y Tráfico", "Geografía"])
    
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        otif_asis = (1 - df["ind_retraso"].mean()) * 100
        retraso_prom = df[df["delta_t"] > 15]["delta_t"].mean()
        km_prom = df["km_ruta"].mean()
        
        c1.metric("OTIF AS IS", f"{otif_asis:.1f}%", delta="-16.8pp vs meta 93%", delta_color="inverse")
        c2.metric("Retraso Promedio", f"{retraso_prom:.0f} min", delta="meta ≤15 min", delta_color="inverse")
        c3.metric("Km prom/viaje", f"{km_prom:.1f} km")
        c4.metric("Total viajes analizados", f"{len(df):,.0f}")
        
        st.divider()
        st.markdown("**Figura 1. OTIF por vehículo — AS IS**<br>*Revela cuáles unidades rompen el Service Level.*", unsafe_allow_html=True)
        
        df_veh = df.groupby("vehiculo_id").agg(
            n_viajes=("vehiculo_id", "count"),
            otif=("ind_retraso", lambda x: (1 - x.mean())*100),
            km_total=("km_ruta", "sum"),
            km_prom=("km_ruta", "mean"),
            retraso_prom_min=("delta_t", lambda x: x[x>15].mean() if len(x[x>15])>0 else 0)
        ).reset_index().sort_values("otif", ascending=True)
        
        df_veh["color"] = df_veh["otif"].apply(lambda x: ROJO if x < 76 else NARANJA if x <= 85 else VERDE)
        fig1 = px.bar(df_veh, x="otif", y="vehiculo_id", orientation="h", template="plotly_white")
        fig1.update_traces(marker_color=df_veh["color"])
        fig1.add_vline(x=76.2, line_dash="dash", line_color=GRIS, annotation_text="Promedio 76.2%")
        fig1.add_vline(x=93.0, line_dash="dash", line_color=VERDE, annotation_text="Meta 93%")
        fig1.update_layout(height=400, font_size=12, title_font_size=14, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig1, use_container_width=True)
        
        st.dataframe(df_veh.style.format({"otif": "{:.1f}%", "km_total": "{:.0f}", "km_prom": "{:.1f}", "retraso_prom_min": "{:.0f}"}), use_container_width=True)

    with t2:
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown("**Figura 2. Distribución de Riesgo de Viajes**<br>*Clasificación del nivel de riesgo histórico.*", unsafe_allow_html=True)
            riesgo = df["riesgo_cat"].value_counts().reset_index()
            riesgo.columns = ["Riesgo", "Cantidad"]
            fig2 = px.pie(riesgo, names="Riesgo", values="Cantidad", hole=0.5, color="Riesgo", color_discrete_map={"BAJO": VERDE, "MEDIO": NARANJA, "ALTO": ROJO}, template="plotly_white")
            fig2.update_traces(textposition='inside', textinfo='percent+value')
            fig2.update_layout(height=380, font_size=12, title_font_size=14)
            st.plotly_chart(fig2, use_container_width=True)
            
        with c_p2:
            st.markdown("**Figura 3. Kilómetros por vehículo y mes — Utilización de flota**<br>*Desbalance en asignación de carga.*", unsafe_allow_html=True)
            if "t_inicio" in df.columns:
                df["mes_yyyy_mm"] = pd.to_datetime(df["t_inicio"]).dt.to_period("M").astype(str)
                df_mes = df.groupby(["mes_yyyy_mm", "vehiculo_id"])["km_ruta"].sum().reset_index()
                fig3 = px.bar(df_mes, x="mes_yyyy_mm", y="km_ruta", color="vehiculo_id", barmode="stack", template="plotly_white")
                fig3.update_layout(height=380, font_size=12, title_font_size=14)
                st.plotly_chart(fig3, use_container_width=True)
                
        st.divider()
        st.markdown("**Figura 4. Variabilidad del tiempo de entrega por ruta (delta_t en min)**<br>*Caja estrecha = ruta predecible · Caja ancha = ruta ineficiente.*", unsafe_allow_html=True)
        top_rutas = df["ruta"].value_counts().head(8).index
        df_box = df[df["ruta"].isin(top_rutas)].copy()
        df_box["delta_t_clip"] = df_box["delta_t"].clip(-60, 180)
        
        fig4 = px.box(df_box, x="ruta", y="delta_t_clip", color="ruta", template="plotly_white")
        fig4.add_hline(y=0, line_dash="solid", line_color=VERDE)
        fig4.add_hline(y=15, line_dash="dash", line_color=NARANJA)
        fig4.add_hline(y=30, line_dash="solid", line_color=ROJO)
        fig4.update_layout(height=450, font_size=12, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with t3:
        st.markdown("**Figura 5. Velocidad promedio por hora del día**<br>*La hora de salida explica el 38% de la variabilidad.*", unsafe_allow_html=True)
        df_h = df.groupby("hora_inicio")["vel_prom"].mean().reset_index()
        df_h["color"] = df_h["hora_inicio"].apply(lambda h: ROJO if 6<=h<=9 or 16<=h<=20 else (NARANJA if 10<=h<=15 else VERDE))
        fig5 = px.bar(df_h, x="hora_inicio", y="vel_prom", template="plotly_white")
        fig5.update_traces(marker_color=df_h["color"])
        fig5.add_hline(y=35, line_dash="dash", line_color=AZUL)
        fig5.update_layout(height=400, font_size=12, showlegend=False, xaxis=dict(dtick=1))
        st.plotly_chart(fig5, use_container_width=True)
        
        st.divider()
        st.markdown("**Figura 6. Tasa de retraso (%) por día y franja horaria**<br>*Zonas oscuras = epicentros de ineficiencia.*", unsafe_allow_html=True)
        dias_map = {0:'Lun', 1:'Mar', 2:'Mié', 3:'Jue', 4:'Vie', 5:'Sáb', 6:'Dom'}
        if "franja_horaria" in df.columns:
            pivot = df.pivot_table(index="dia_semana_num", columns="franja_horaria", values="ind_retraso", aggfunc="mean") * 100
            pivot.index = pivot.index.map(dias_map)
            cols = [c for c in ["MAÑANA", "TARDE", "NOCHE"] if c in pivot.columns]
            pivot = pivot[cols] if cols else pivot
            fig6 = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn_r", aspect="auto", template="plotly_white")
            fig6.update_layout(height=450, font_size=12)
            st.plotly_chart(fig6, use_container_width=True)
            st.info("🔴 Domingo mañana: mayor riesgo (42% retraso) · 🟢 Noche entre semana: menor riesgo (7-11%)")

    with t4:
        st.markdown("**Figura 7. Historial de Puntos de Entrega Reales**<br>*Concentración de paradas logísticas.*", unsafe_allow_html=True)
        if df_puntos is not None and not df_puntos.empty:
            vehs = df_puntos["vehiculo_id"].unique().tolist() if "vehiculo_id" in df_puntos.columns else []
            v_sel = st.multiselect("Filtrar por vehículo:", vehs, default=vehs[:2] if len(vehs)>=2 else vehs)
            df_m = df_puntos[df_puntos["vehiculo_id"].isin(v_sel)] if v_sel else df_puntos
            st.map(df_m)
            st.caption(f"📍 Mostrando {len(df_m):,} puntos operativos.")
            
        st.divider()
        st.markdown("**Figura 8. Top 10 destinos por frecuencia de viaje**<br>*Áreas urbanas con mayor densidad de operaciones.*", unsafe_allow_html=True)
        top10 = df["ruta"].value_counts().head(10).reset_index()
        top10.columns = ["Ruta", "Viajes"]
        fig7 = px.bar(top10.sort_values("Viajes", ascending=True), y="Ruta", x="Viajes", orientation="h", template="plotly_white", color_discrete_sequence=[AZUL])
        fig7.update_layout(height=400, font_size=12)
        st.plotly_chart(fig7, use_container_width=True)


# ── HEADER (Top Navigation) ────────────────────────────────────────────────────
top_col1, top_col2, top_col3 = st.columns([2, 8, 1])
with top_col1:
    st.markdown("### MOETC-BD")
with top_col2:
    vista = st.radio("", ["Operational", "Analytics", "Financial", "Executive", "AS-IS Diagnosis", "Simulation"], horizontal=True, label_visibility="collapsed", index=5)
with top_col3:
    st.markdown("<div style='text-align:right;'><img src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png' width='40' style='border-radius:50%; border:2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.1)'></div>", unsafe_allow_html=True)

st.write("") # Spacer


# ── FUNCIONES DEL SIMULADOR ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_vel_lookup():
    # CORREGIDO 3 — Tabla lookup de velocidad por hora REAL
    path_gps = "data/processed/dataset_integrado.csv"
    if os.path.exists(path_gps):
        df = pd.read_csv(path_gps)
        if 'hora_inicio' in df.columns:
            return df.groupby('hora_inicio')[['vel_prom','vel_max']].mean().round(1)
    return pd.DataFrame()

def build_feature_vector(inputs):
    # CORREGIDO 1 — Separar el modelo ML del bloque financiero
    # SEPARACIÓN DE RESPONSABILIDADES: el modelo ML predice TIEMPO,
    # el bloque financiero calcula COSTO. Nunca mezclar ambos vectores.
    
    km_ruta = inputs['km_ruta']
    log_km = np.log1p(km_ruta)
    hora_inicio = inputs['hora_inicio']
    vel_lookup = get_vel_lookup()
    
    if hora_inicio in vel_lookup.index:
        vel_prom = float(vel_lookup.loc[hora_inicio, 'vel_prom'])
        vel_max  = float(vel_lookup.loc[hora_inicio, 'vel_max'])
    else:
        vel_prom, vel_max = 35.0, 80.0
        
    t_est_min = (km_ruta / vel_prom) * 60.0 if vel_prom > 0 else 60.0
    
    ohe_cols = ['ruta_Cibao-Norte', 'ruta_Cibao-Sur', 'ruta_Este-Turístico', 'ruta_La-Romana', 'ruta_Nordeste', 'ruta_Norte', 'ruta_Otras', 'ruta_SD-Centro', 'ruta_SD-Este', 'ruta_SD-Norte', 'ruta_SD-Oeste', 'ruta_Santiago', 'ruta_Sur', 'franja_horaria_NOCHE', 'franja_horaria_TARDE']
    ohe_valores = {c: 0.0 for c in ohe_cols}
    if f"ruta_{inputs['destino']}" in ohe_valores: ohe_valores[f"ruta_{inputs['destino']}"] = 1.0
    if 18 <= hora_inicio <= 23 or 0 <= hora_inicio <= 5: ohe_valores['franja_horaria_NOCHE'] = 1.0
    elif 13 <= hora_inicio <= 17: ohe_valores['franja_horaria_TARDE'] = 1.0
    
    vector = [
        km_ruta, log_km, vel_max, vel_prom, t_est_min, hora_inicio,
        inputs['dia_semana_num'], inputs['es_fin_semana'], inputs['mes'],
        inputs['n_paradas'], float(inputs['n_paradas'] / km_ruta) if km_ruta > 0 else 0.0
    ] + [ohe_valores[c] for c in ohe_cols]
    
    columnas = [
        "km_ruta", "log_km", "vel_max", "vel_prom", "t_est_min", "hora_inicio",
        "dia_semana_num", "es_fin_semana", "mes", "n_paradas", "densidad_paradas"
    ] + ohe_cols
    
    return pd.DataFrame([vector], columns=columnas), t_est_min

def calcular_financiero(km, precio_galon, pago_chofer, ingreso):
    # CORREGIDO 1 — Bloque B Análisis financiero
    combustible = km * 0.167 * precio_galon
    mantenimiento = km * 3.25
    costo_total = combustible + pago_chofer + mantenimiento
    margen_bruto = ingreso - costo_total
    margen_pct = (margen_bruto / ingreso * 100) if ingreso > 0 else 0.0
    costo_km = costo_total / km if km > 0 else 0.0
    return costo_total, margen_bruto, margen_pct, costo_km

def render_simulador():
    left_col, right_col = st.columns([7, 3], gap="large")
    
    with right_col:
        with st.container():
            st.markdown("### 🎛 Configurar Viaje")
            st.markdown("<p style='font-size: 0.85rem;'>Configura las variables para la simulación ML.</p>", unsafe_allow_html=True)

            zonas_reales = df_global["zona_destino"].dropna().unique().tolist() if "zona_destino" in df_global.columns else ["Santo Domingo", "Santiago", "Este"]
            rutas_reales = df_global["ruta"].dropna().unique().tolist() if "ruta" in df_global.columns else ["SD-Centro", "SD-Oeste", "Santiago", "Nordeste", "Sur", "Otras"]
            vehiculos_reales = df_global["vehiculo_id"].dropna().unique().tolist() if "vehiculo_id" in df_global.columns else ["L322837", "L330617", "L344749", "L354062"]

            cp1, cp2 = st.columns(2)
            punto_partida = cp1.selectbox("Origen (Zona)", zonas_reales)
            destino = cp2.selectbox("Destino (Ruta)", rutas_reales)
            
            # CORREGIDO 2 — Distancia exacta para el simulador
            km_hist_ruta = float(df_global[df_global['ruta']==destino]['km_ruta'].mean()) if "ruta" in df_global.columns and not df_global[df_global['ruta']==destino].empty else 50.0
            n_viajes_ruta = df_global[df_global['ruta']==destino].shape[0] if "ruta" in df_global.columns else 0
            
            if n_viajes_ruta < 5:
                st.warning(f"Pocos datos históricos para {destino} ({n_viajes_ruta} viajes). Mapeando distancia sugerida obligatoria.")
                usar_km_exacto = True
            else:
                usar_km_exacto = st.checkbox("📍 Ingresar distancia exacta al cliente", value=False, help="Actívalo si conoces la distancia real.")
                
            if usar_km_exacto:
                km_ruta_final = st.number_input("Distancia exacta (km)", min_value=1.0, max_value=600.0, value=float(km_hist_ruta), step=0.5)
                st.caption(f"Promedio histórico para {destino}: {km_hist_ruta:.0f} km")
            else:
                km_ruta_final = km_hist_ruta
                st.caption(f"Usando histórico: {km_hist_ruta:.0f} km ({n_viajes_ruta} viajes)")
            
            c_hora, c_paradas = st.columns(2)
            hora_salida = c_hora.time_input("Hora Salida", datetime.time(8, 0))
            n_paradas = c_paradas.number_input("Paradas", min_value=1, max_value=15, value=1)
            
            ct1, ct2 = st.columns(2)
            tipo_camion = ct1.selectbox("Tipo de camión", ["Grande (6000kg)", "Med." , "Pequeño"]) 
            chofer_id = ct2.selectbox("Vehículo (Chofer)", vehiculos_reales)
            
            st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px dashed #e5e7eb;'>", unsafe_allow_html=True)
            st.markdown("#### 💰 Financiero y Ambiental")
            
            cf1, cf2 = st.columns(2)
            ingreso_viaje_rd = cf1.number_input("Ingreso (RD$)", value=15000.0, step=1000.0)
            pago_chofer_rd = cf2.number_input("Pago (RD$)", value=1200.0, step=100.0)
            
            ca1, ca2 = st.columns(2)
            precio_galon = ca1.number_input("Combustible (RD$)", value=295.0)
            pct_opt_ruta = ca2.number_input("Optimización (%)", value=85.0, help="100% = recta ideal")

            btn_calcular = st.button("Simular Despacho", use_container_width=True, type="primary")

        st.markdown("<div class='premium-card'><div class='premium-deco'></div><h3 style='position: relative; z-index: 2;'>Logistics AI</h3><p style='position: relative; z-index: 2;'>Powered by Random Forest.<br>Tesis MOETC-BD UAPA.</p></div>", unsafe_allow_html=True)

    # ── LOGICA Y LLAMADAS ──
    fecha_viaje = datetime.date.today()
    inputs = {
        'km_ruta': km_ruta_final,
        'hora_inicio': int(hora_salida.hour),
        'dia_semana_num': int(fecha_viaje.weekday()),
        'es_fin_semana': 1 if int(fecha_viaje.weekday()) >= 5 else 0,
        'mes': int(fecha_viaje.month),
        'n_paradas': int(n_paradas),
        'destino': destino
    }
    
    # 1. Feature ML
    try:
        X_pred, t_est_min = build_feature_vector(inputs)
        delta_t_predicho = float(modelos["mp1"].predict(X_pred)[0])
        prob_retraso_raw = float(modelos["mp3"].predict_proba(X_pred)[0][1])
        riesgo_pred = float(modelos["mp2"].predict(X_pred)[0])
    except ValueError as e:
        # Fallback en caso de mismatch temporal con los modelos viejos.
        st.error(f"Incompatibilidad de variables detectada con los modelos ML entrenados previamente. Debes reentrenarlos. Error: {e}")
        return

    # 2. Financiero
    costo_estimado, margen, margen_pct, costo_km_est = calcular_financiero(km_ruta_final, precio_galon, pago_chofer_rd, ingreso_viaje_rd)
    
    # Post-Procesamiento Resultados
    otif_pred = 1.0 - prob_retraso_raw
    tt_min = int(t_est_min + delta_t_predicho)
    prob_retraso = prob_retraso_raw * 100.0

    r_str = str(riesgo_pred)
    if "0" in r_str: riesgo_badge = "<span class='badge bg-green'>BAJO RIESGO</span>"
    elif "1" in r_str: riesgo_badge = "<span class='badge bg-yellow'>MEDIO RIESGO</span>"
    else: riesgo_badge = "<span class='badge bg-red'>ALTO RIESGO</span>"

    C_norm = min(costo_estimado / 8500.0, 1.0)
    D_norm = min(max(delta_t_predicho, 0) / 60.0, 1.0)
    e_score_viaje = 0.50 * otif_pred + 0.30 * (1.0 - C_norm) + 0.20 * (1.0 - D_norm)
    if margen < 0: e_score_viaje = min(e_score_viaje, 0.29)

    km_opt = km_ruta_final * (pct_opt_ruta / 100.0)
    co2_evitado = (km_ruta_final - km_opt) * 1.25

    # ── RENDER PANTALLA PRINCIPAL (Izquierda) ──
    with left_col:
        with st.container():
            st.markdown("### Simulation Overview")
            otif_badge = f"<span class='badge bg-green'>+{otif_pred*100:.1f}% Éxito</span>" if otif_pred > 0.6 else f"<span class='badge bg-red'>-{prob_retraso:.1f}% Retraso</span>"
            str_retraso = f"+{int(delta_t_predicho)}m Demora" if delta_t_predicho > 0 else "✓ A tiempo"
            color_ret = "bg-red" if delta_t_predicho > 0 else "bg-green"
            
            html_metrics = f'''<div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:10px;"><div style="flex:1;"><p style="margin:0; font-size:0.9rem;">Estimated Time</p><div style="font-size:2.8rem; font-weight:700; color:#111827;">{tt_min} <span style="font-size:1.2rem; color:#6b7280; font-weight:500;">min</span></div><span class='badge {color_ret}'>{str_retraso}</span></div><div style="flex:1; border-left:1px solid #f3f4f6; padding-left:20px;"><p style="margin:0; font-size:0.9rem;">Service Level (OTIF)</p><div style="font-size:2.8rem; font-weight:700; color:#111827;">{otif_pred*100:.1f} <span style="font-size:1.2rem; color:#6b7280; font-weight:500;">%</span></div>{otif_badge}</div><div style="flex:1; border-left:1px solid #f3f4f6; padding-left:20px;"><p style="margin:0; font-size:0.9rem;">Risk Assessment</p><div style="margin-top:20px;">{riesgo_badge}</div></div></div>'''
            st.markdown(html_metrics, unsafe_allow_html=True)
            st.write("")

        col_cash, col_env = st.columns(2)
        with col_cash:
            with st.container():
                st.markdown("### Cashflow Analysis ↗")
                st.markdown(f"<div style='font-size:2.4rem; font-weight:700;'>RD$ {margen:,.2f}</div>", unsafe_allow_html=True)
                mbadge = f"<span class='badge bg-green'>+{margen_pct:.1f}% Margin</span>" if margen_pct >= 15 else f"<span class='badge bg-red'>{margen_pct:.1f}% Alert</span>"
                st.markdown(mbadge, unsafe_allow_html=True)
                st.markdown(f'''<div style="display:flex; justify-content:space-between; margin-top:30px;"><div><div style="font-size:1.2rem; font-weight:600;">RD$ {ingreso_viaje_rd:,.0f}</div><div style="font-size:0.8rem; color:#6b7280;">Revenue</div></div><div style="text-align:right;"><div style="font-size:1.2rem; font-weight:600; color:#ef4444;">RD$ {costo_estimado:,.0f}</div><div style="font-size:0.8rem; color:#6b7280;">Total OPEX</div></div></div>''', unsafe_allow_html=True)

        with col_env:
            with st.container():
                st.markdown("### E-Score & Environment ↗")
                eco_color = "#10b981" if e_score_viaje >= 0.5 else "#f59e0b" if e_score_viaje >= 0.3 else "#ef4444"
                fig_e = go.Figure(go.Indicator(mode="number+gauge", value=e_score_viaje * 100, number={'suffix': "%", 'font': {'size': 35, 'color': eco_color}}, gauge={'axis': {'range': [None, 100], 'visible': False}, 'bar': {'color': eco_color, 'thickness': 0.7}, 'bordercolor': "rgba(0,0,0,0)"}))
                fig_e.update_layout(height=110, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_e, use_container_width=True)
                st.markdown(f'''<div style="display:flex; justify-content:space-between; margin-top:5px; padding-top:10px; border-top:1px dashed #e5e7eb;"><div><div style="font-size:0.8rem; color:#6b7280;">CO₂ Avoided</div><div style="font-size:1.1rem; font-weight:600; color:#10b981;">{co2_evitado:.1f} kg <span class='badge bg-green'>↑</span></div></div><div style="text-align:right;"><div style="font-size:0.8rem; color:#6b7280;">Optimal Dist.</div><div style="font-size:1.1rem; font-weight:600;">{km_opt:.1f} km</div></div></div>''', unsafe_allow_html=True)

        with st.container():
            st.markdown("### Benchmarking History")
            otif_asis = 76.2
            t_est_asis = df_global["t_real_min"].mean() if "t_real_min" in df_global.columns else 120.0
            c_km_asis = df_global["costo_km_est"].mean() if "costo_km_est" in df_global.columns else 85.0
            e_asis = 0.45 
            
            html_table = f"""<table><thead><tr><th>Metric ID</th><th>Date</th><th>AS-IS (Historical)</th><th>Prediction (TO-BE)</th><th>Status</th></tr></thead><tbody><tr><td><b>OTIF Success</b></td><td>Today</td><td style='color:#6b7280;'>{otif_asis}%</td><td><b>{otif_pred*100:.1f}%</b></td><td><span class='badge bg-{"green" if otif_pred*100 > otif_asis else "red"}'>{"Success" if otif_pred*100 > otif_asis else "Alert"}</span></td></tr><tr><td><b>Avg. Duration</b></td><td>Today</td><td style='color:#6b7280;'>{t_est_asis:.0f} min</td><td><b>{tt_min} min</b></td><td><span class='badge bg-{"green" if tt_min < t_est_asis else "red"}'>{"Success" if tt_min < t_est_asis else "Alert"}</span></td></tr><tr><td><b>Efficiency Score (E)</b></td><td>Today</td><td style='color:#6b7280;'>{e_asis*100:.0f}%</td><td><b>{e_score_viaje*100:.0f}%</b></td><td><span class='badge bg-{"green" if e_score_viaje > e_asis else "yellow"}'>{"Excellent" if e_score_viaje > e_asis else "Monitor"}</span></td></tr></tbody></table>"""
            st.markdown(html_table, unsafe_allow_html=True)


# ── VISTAS DINÁMICAS ──────────────────────────────────────────────────────────
kpis = calcular_resumen_kpis(df_global, label="Global")

if vista == "Simulation":
    render_simulador()

# ══════════════════════════════════════════════════════════════════════════════
# VISTA OPERACIONAL (Legacy)
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "Operational":
    with st.container():
        st.subheader("Vista Operativa — Estado de Rutas")
        df = df_global.copy()
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
                    title="Tasa de Retraso por Ruta (%)"
                )
                fig.add_vline(x=5, line_dash="dash", line_color="green", annotation_text="Meta 5%")
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "riesgo_cat" in df.columns:
                df_riesgo = df["riesgo_cat"].value_counts().reset_index()
                df_riesgo.columns = ["Riesgo", "Viajes"]
                color_map = {"BAJO": "#66BB6A", "MEDIO": "#FFA726", "ALTO": "#EF5350"}

                fig2 = px.pie(
                    df_riesgo, names="Riesgo", values="Viajes",
                    title="Distribución de Riesgo de Viajes",
                    color="Riesgo", color_discrete_map=color_map, hole=0.4
                )
                st.plotly_chart(fig2, use_container_width=True)

    with st.container():
        if "delta_t" in df.columns:
            cols_mostrar = [c for c in ["viaje_id","ruta","t_inicio","delta_t","riesgo_cat","otif"] if c in df.columns]
            df_alertas = df[df.get("delta_t", pd.Series(dtype=float)) > 15][cols_mostrar].head(10)
            if len(df_alertas) > 0:
                st.subheader("⚠ Viajes con Retraso Detectado")
                st.dataframe(df_alertas, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VISTA ANALÍTICA (Legacy)
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "Analytics":
    with st.container():
        st.subheader("Vista Analítica — Tendencias y Modelos ML")
        df = df_global.copy()
        
        if "t_inicio" in df.columns and "delta_t" in df.columns:
            df_trend = (df.set_index("t_inicio")
                          .resample("W")["delta_t"]
                          .agg(["mean", "median", "count"])
                          .reset_index())

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df_trend["t_inicio"], y=df_trend["mean"], name="Retraso prom.", line=dict(color="#42A5F5")), secondary_y=False)
            fig.add_trace(go.Bar(x=df_trend["t_inicio"], y=df_trend["count"], name="N° Viajes", opacity=0.3, marker_color="#90A4AE"), secondary_y=True)
            fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Umbral 15 min")
            fig.update_layout(title="Evolución Semanal del Retraso Promedio")
            st.plotly_chart(fig, use_container_width=True)

    with st.container():
        img_rvp = "models/reports/real_vs_predicho_mp-1_regresión.png"
        if os.path.exists(img_rvp):
            st.subheader("Evaluación de Predicciones — MP-1")
            st.image(img_rvp)

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
# VISTA FINANCIERA (Legacy)
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "Financial":
    with st.container():
        st.subheader("Vista Financiera — Análisis de Costos")
        df = df_global.copy()
        if "t_inicio" in df.columns and "costo_estimado" in df.columns:
            df_costo = (df.set_index("t_inicio").resample("M")["costo_estimado"].agg(["mean", "sum"]).reset_index())

            col_x, col_y = st.columns(2)
            with col_x:
                fig_c1 = px.line(df_costo, x="t_inicio", y="mean", title="Costo Promedio por Viaje (RD$)", markers=True)
                st.plotly_chart(fig_c1, use_container_width=True)
            with col_y:
                fig_c2 = px.bar(df_costo, x="t_inicio", y="sum", title="Costo Total Mensual (RD$)")
                st.plotly_chart(fig_c2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# VISTA EJECUTIVA (Legacy)
# ══════════════════════════════════════════════════════════════════════════════
elif vista == "Executive":
    with st.container():
        st.subheader("Vista Ejecutiva — Efficiency Score y Proyección TO BE")
        
        e_score = kpis.get("Efficiency Score (E)")
        if e_score is not None:
            col_e1, col_e2, col_e3 = st.columns([1, 2, 1])
            with col_e2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="number+gauge",
                    value=e_score * 100,
                    number={'suffix': "%"},
                    title={"text": "Global Efficiency Score (E)"},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#3b82f6"}}
                ))
                fig_gauge.update_layout(height=320)
                st.plotly_chart(fig_gauge, use_container_width=True)

    with st.container():
        comp_path = "models/reports/comparativo_as_is_to_be.csv"
        if os.path.exists(comp_path):
            st.subheader("Comparativo AS IS vs TO BE (Dataset completo)")
            df_comp = pd.read_csv(comp_path)
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.info("Ejecuta el notebook 05_kpis.ipynb para generar el comparativo AS IS vs TO BE.")


# NUEVO: VISTA DIAGNÓSTICO AS IS
elif vista == "AS-IS Diagnosis":
    vista_diagnostico_as_is()
