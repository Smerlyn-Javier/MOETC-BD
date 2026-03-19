# MOETC-BD — Guía del Desarrollador

> **Modelo de Optimización de la Eficiencia del Transporte de Carga basado en Big Data**  
> López Laureano Distribución · Santo Domingo, RD · 2026  
> Tesis de Maestría en Big Data e Inteligencia de Negocios — UAPA

---

## Índice

1. [Configuración del entorno](#1-configuración-del-entorno)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Pipeline de datos](#3-pipeline-de-datos)
4. [EDA y diagnóstico AS IS](#4-eda-y-diagnóstico-as-is)
5. [Ingeniería de variables](#5-ingeniería-de-variables)
6. [Modelos de Machine Learning](#6-modelos-de-machine-learning)
7. [Motor VRP](#7-motor-vrp)
8. [Dashboard KPIs](#8-dashboard-kpis)
9. [Validaciones y tests](#9-validaciones-y-tests)
10. [Orden de ejecución](#10-orden-de-ejecución)

---

## 1. Configuración del entorno

```bash
# Clonar / crear la carpeta del proyecto
cd MOETC-BD

# Instalar dependencias
pip install -r requirements.txt

# Registrar kernel para Jupyter
python -m ipykernel install --user --name moetc-bd --display-name "MOETC-BD"

# Abrir Jupyter
jupyter notebook
```

---

## 2. Estructura del proyecto

```
MOETC-BD/
├── data/
│   ├── raw/          ← Datos originales (NO modificar)
│   ├── processed/    ← Datos limpios
│   ├── final/        ← Dataset listo para modelos
│   └── external/     ← Rutas, distancias, coordenadas
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_features.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_vrp.ipynb
│   └── 05_kpis.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── features.py
│   ├── models.py
│   ├── vrp_solver.py
│   ├── kpis.py
│   └── utils.py
├── dashboard/
│   ├── app.py
│   └── pages/
├── models/
│   ├── saved/        ← .pkl de modelos entrenados
│   └── reports/      ← Métricas, gráficas, reportes
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

---

## 10. Orden de ejecución (Paso a paso)

Para ejecutar el proyecto completo de manera secuencial desde la terminal, sigue estos pasos:

```bash
# 1. Activar el entorno virtual e instalar dependencias (si no lo has hecho)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Pipeline de datos
# Ejecuta la limpieza, transformación y carga inicial de los datos
python -m src.data_pipeline

# 3. Ingeniería de variables (Features)
# Construye las nuevas variables necesarias para el modelo
python -m src.features

# 4. Modelos de Machine Learning
# Entrena el modelo, evalúa métricas y genera las gráficas de rendimiento
python -m src.models

# 5. Motor de Optimización de Rutas (VRP)
# Ejecuta el algoritmo para crear rutas óptimas de despacho
python -m src.vrp_solver

# 6. Cálculo y guardado de KPIs
# Procesa y consolida las métricas finales de negocio
python -m src.kpis

# 7. Desplegar el Dashboard
# Inicia la interfaz gráfica que consolida resultados (accesible en localhost)
streamlit run dashboard/app.py
```

> **Nota:** También puedes ejecutar el análisis interactivo de cada fase abriendo los archivos de Jupyter en la carpeta `notebooks/` (`jupyter notebook`).
