"""
src/models.py
=============
MOETC-BD — Módulo de Analítica Predictiva y Machine Learning
Capa 2: Analytics Layer del modelo MOETC-BD

Corresponde a la sección 3.1.5 de la tesis:
"Componente II: Módulo de analítica predictiva y Machine Learning"

Implementa los tres sub-modelos:
  MP-1: Regresión        → predice delta_t (minutos de retraso)
  MP-2: Clasificación    → predice riesgo (ALTO / MEDIO / BAJO)
  MP-3: Clasificación    → predice OTIF binario (1/0)

Protocolo de entrenamiento:
  - División 80/20 (train/test), stratify en clasificadores
  - Validación cruzada k-fold (k=5)
  - Grid Search para optimización de hiperparámetros
  - Evaluación final en test set
  - Exportación del mejor modelo como .pkl
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.tree          import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble      import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics       import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.preprocessing import LabelEncoder
from xgboost               import XGBRegressor, XGBClassifier

from src.features import FEATURES_MODELO, TARGETS

MODELS_DIR  = "models/saved"
REPORT_DIR  = "models/reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREPARACIÓN DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════

def preparar_datasets(df):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba
    para los tres sub-modelos.
    
    Estrategia:
      - test_size = 0.20 (20% para evaluación final)
      - random_state = 42 (reproducibilidad)
      - stratify para clasificadores (balance de clases)
    
    Returns:
        dict con las divisiones para cada sub-modelo
    """
    # Seleccionar solo features que existan en el DataFrame
    features = [f for f in FEATURES_MODELO if f in df.columns]
    
    # Agregar features OHE generadas dinámicamente
    ohe_cols = [c for c in df.columns if c.startswith("ruta_") or c.startswith("franja_")]
    features = features + ohe_cols
    features = list(dict.fromkeys(features))  # deduplicar preservando orden
    
    print(f"[FEATURES] {len(features)} variables predictoras: {features}")
    
    X = df[features].copy()
    
    splits = {}
    
    # MP-1: Regresión (delta_t)
    if TARGETS["regresion"] in df.columns:
        y_reg = df[TARGETS["regresion"]]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_reg, test_size=0.20, random_state=42
        )
        splits["mp1"] = {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te}
        print(f"[MP-1] Train: {len(X_tr)} | Test: {len(X_te)}")
    
    # MP-3: Clasificación OTIF (binaria)
    if TARGETS["clasificacion_otif"] in df.columns:
        y_otif = df[TARGETS["clasificacion_otif"]]
        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
            X, y_otif, test_size=0.20, random_state=42, stratify=y_otif
        )
        splits["mp3"] = {"X_train": Xc_tr, "X_test": Xc_te, "y_train": yc_tr, "y_test": yc_te}
        print(f"[MP-3] Train: {len(Xc_tr)} | Test: {len(Xc_te)} | Balance: {y_otif.mean():.2%} retrasos")
    
    # MP-2: Clasificación de riesgo (multiclase)
    if TARGETS["clasificacion_riesgo"] in df.columns:
        y_riesgo = df[TARGETS["clasificacion_riesgo"]]
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X, y_riesgo, test_size=0.20, random_state=42, stratify=y_riesgo
        )
        splits["mp2"] = {"X_train": Xr_tr, "X_test": Xr_te, "y_train": yr_tr, "y_test": yr_te}
        print(f"[MP-2] Train: {len(Xr_tr)} | Test: {len(Xr_te)}")
    
    # Guardar datasets
    for nombre, split in splits.items():
        for key, data in split.items():
            data.to_csv(f"data/final/{nombre}_{key}.csv", index=False)
    
    return splits, features


# ══════════════════════════════════════════════════════════════════════════════
# SUB-MODELO MP-1: REGRESIÓN
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_mp1_regresion(splits):
    """
    Entrena y evalúa los cuatro algoritmos de regresión del sub-modelo MP-1.
    
    Algoritmos evaluados:
      1. Regresión Lineal Múltiple (línea base interpretable)
      2. Árbol de Decisión CART
      3. Random Forest (ensamblado bagging)
      4. XGBoost (ensamblado boosting)
    
    Protocolo:
      - KFold k=5 sobre training set (validación cruzada)
      - Métricas en test set: RMSE, MAE, R²
      - Selección por menor RMSE test + estabilidad CV
    
    Returns:
        dict: resultados comparativos + modelo seleccionado
    """
    print("\n" + "="*60)
    print("MP-1: REGRESIÓN — Predicción de delta_t (minutos)")
    print("="*60)
    
    X_train = splits["mp1"]["X_train"]
    X_test  = splits["mp1"]["X_test"]
    y_train = splits["mp1"]["y_train"]
    y_test  = splits["mp1"]["y_test"]
    
    # Línea base trivial: predice siempre la media
    media_train = y_train.mean()
    y_base      = np.full(len(y_test), media_train)
    rmse_base   = np.sqrt(mean_squared_error(y_test, y_base))
    print(f"\n[LÍNEA BASE] Media constante: RMSE = {rmse_base:.2f} min")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    candidatos = {
        "Regresion Lineal":  LinearRegression(),
        "Arbol Decision":    DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":           XGBRegressor(n_estimators=200, learning_rate=0.1,
                                          max_depth=6, random_state=42, verbosity=0),
    }
    
    resultados = {}
    modelos_entrenados = {}
    
    for nombre, modelo in candidatos.items():
        # Validación cruzada
        cv_scores = -cross_val_score(
            modelo, X_train, y_train,
            cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        
        # Entrenamiento completo
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        # Métricas
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        
        resultados[nombre] = {
            "CV_RMSE_mean":   round(float(cv_scores.mean()), 3),
            "CV_RMSE_std":    round(float(cv_scores.std()), 3),
            "RMSE_test":      round(float(rmse), 3),
            "MAE_test":       round(float(mae), 3),
            "R2_test":        round(float(r2), 4),
            "supera_base":    bool(rmse < rmse_base),
        }
        modelos_entrenados[nombre] = modelo
        
        estado = "✓" if rmse < rmse_base else "✗"
        print(f"\n{estado} {nombre}:")
        print(f"  CV RMSE: {cv_scores.mean():.2f} ± {cv_scores.std():.2f} min")
        print(f"  Test   → RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    
    # Seleccionar mejor modelo
    df_res = pd.DataFrame(resultados).T.sort_values("RMSE_test")
    mejor_nombre = df_res.index[0]
    mejor_modelo = modelos_entrenados[mejor_nombre]
    
    print(f"\n[SELECCIONADO] {mejor_nombre} — RMSE test: {df_res['RMSE_test'].iloc[0]:.2f} min")
    
    # Guardar
    joblib.dump(mejor_modelo, f"{MODELS_DIR}/mp1_regresion.pkl")
    df_res.to_csv(f"{REPORT_DIR}/mp1_resultados.csv")
    
    # Gráfica de importancia (para modelos basados en árboles)
    if hasattr(mejor_modelo, "feature_importances_"):
        _graficar_importancia(mejor_modelo, X_train.columns.tolist(), "MP-1")
    
    # Gráfica real vs predicho
    y_pred_final = mejor_modelo.predict(X_test)
    _graficar_real_vs_predicho(y_test, y_pred_final, "MP-1 Regresión")
    
    print("\n[OK] Resultados MP-1:")
    print(df_res.to_string())
    
    return {
        "resultados": resultados,
        "mejor_nombre": mejor_nombre,
        "mejor_modelo": mejor_modelo,
        "rmse_base": rmse_base,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUB-MODELO MP-3: CLASIFICACIÓN OTIF (binaria)
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_mp3_otif(splits):
    """
    Entrena y evalúa los clasificadores binarios del sub-modelo MP-3.
    Predice si un viaje cumplirá o no el OTIF (1 = a tiempo, 0 = retrasado).
    
    Algoritmos:
      1. Regresión Logística
      2. Random Forest Classifier
      3. XGBoost Classifier
    
    Métricas:
      - F1-Score (equilibrio precisión/recall)
      - AUC-ROC (capacidad discriminatoria)
      - Matriz de confusión
    """
    print("\n" + "="*60)
    print("MP-3: CLASIFICACIÓN OTIF — Predicción binaria")
    print("="*60)
    
    X_train = splits["mp3"]["X_train"]
    X_test  = splits["mp3"]["X_test"]
    y_train = splits["mp3"]["y_train"]
    y_test  = splits["mp3"]["y_test"]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    candidatos = {
        "Reg. Logistica":  LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest":   RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":         XGBClassifier(n_estimators=200, learning_rate=0.1,
                                          max_depth=5, random_state=42,
                                          eval_metric="logloss", verbosity=0),
    }
    
    resultados = {}
    modelos_entrenados = {}
    
    for nombre, modelo in candidatos.items():
        cv_f1 = cross_val_score(modelo, X_train, y_train, cv=skf, scoring="f1")
        
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        resultados[nombre] = {
            "CV_F1_mean": round(float(cv_f1.mean()), 4),
            "CV_F1_std":  round(float(cv_f1.std()), 4),
            "F1_test":    round(float(f1), 4),
            "AUC_ROC":    round(float(auc), 4),
        }
        modelos_entrenados[nombre] = modelo
        
        print(f"\n{nombre}: CV F1={cv_f1.mean():.4f}±{cv_f1.std():.4f} | Test F1={f1:.4f} | AUC={auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["A tiempo", "Retrasado"]))
    
    # Seleccionar mejor
    df_res = pd.DataFrame(resultados).T.sort_values("F1_test", ascending=False)
    mejor_nombre = df_res.index[0]
    mejor_modelo = modelos_entrenados[mejor_nombre]
    
    print(f"\n[SELECCIONADO] {mejor_nombre} — F1: {df_res['F1_test'].iloc[0]:.4f}")
    
    # Guardar
    joblib.dump(mejor_modelo, f"{MODELS_DIR}/mp3_otif.pkl")
    df_res.to_csv(f"{REPORT_DIR}/mp3_resultados.csv")
    
    # Gráficas
    y_pred_final = mejor_modelo.predict(X_test)
    y_prob_final = mejor_modelo.predict_proba(X_test)[:, 1]
    _graficar_confusion(y_test, y_pred_final, "MP-3 OTIF")
    _graficar_roc(y_test, y_prob_final, "MP-3 OTIF")
    
    return {
        "resultados": resultados,
        "mejor_nombre": mejor_nombre,
        "mejor_modelo": mejor_modelo,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUB-MODELO MP-2: CLASIFICACIÓN DE RIESGO (multiclase)
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_mp2_riesgo(splits):
    """
    Entrena clasificadores multiclase para predecir el nivel de riesgo.
    Clases: BAJO (0), MEDIO (1), ALTO (2)
    """
    print("\n" + "="*60)
    print("MP-2: CLASIFICACIÓN RIESGO — BAJO / MEDIO / ALTO")
    print("="*60)
    
    if "mp2" not in splits:
        print("[WARN] No se encontró el split mp2. Omitiendo MP-2.")
        return {}
    
    X_train = splits["mp2"]["X_train"]
    X_test  = splits["mp2"]["X_test"]
    y_train = splits["mp2"]["y_train"]
    y_test  = splits["mp2"]["y_test"]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    candidatos = {
        "Arbol Decision":  DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":   RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":         XGBClassifier(n_estimators=200, random_state=42,
                                          eval_metric="mlogloss", verbosity=0),
    }
    
    resultados = {}
    modelos_entrenados = {}
    
    for nombre, modelo in candidatos.items():
        cv_f1 = cross_val_score(modelo, X_train, y_train, cv=skf, scoring="f1_macro")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        
        resultados[nombre] = {
            "CV_F1_macro_mean": round(float(cv_f1.mean()), 4),
            "CV_F1_macro_std":  round(float(cv_f1.std()), 4),
            "F1_macro_test":    round(float(f1_mac), 4),
        }
        modelos_entrenados[nombre] = modelo
        
        print(f"\n{nombre}: CV F1-macro={cv_f1.mean():.4f} | Test={f1_mac:.4f}")
        print(classification_report(y_test, y_pred, target_names=["BAJO","MEDIO","ALTO"]))
    
    df_res = pd.DataFrame(resultados).T.sort_values("F1_macro_test", ascending=False)
    mejor_nombre = df_res.index[0]
    mejor_modelo = modelos_entrenados[mejor_nombre]
    
    joblib.dump(mejor_modelo, f"{MODELS_DIR}/mp2_riesgo.pkl")
    df_res.to_csv(f"{REPORT_DIR}/mp2_resultados.csv")
    
    _graficar_confusion(y_test, mejor_modelo.predict(X_test), "MP-2 Riesgo",
                        labels=["BAJO","MEDIO","ALTO"])
    
    return {"resultados": resultados, "mejor_nombre": mejor_nombre, "mejor_modelo": mejor_modelo}


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

def optimizar_hiperparametros_mp1(splits, modelo_base="Random Forest"):
    """
    Grid Search sobre el modelo seleccionado en MP-1.
    Solo ejecutar si el tiempo de cómputo lo permite.
    """
    print(f"\n[GRID SEARCH] Optimizando hiperparámetros de {modelo_base} MP-1...")
    
    X_train = splits["mp1"]["X_train"]
    y_train = splits["mp1"]["y_train"]
    
    param_grid = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [None, 10, 20],
        "min_samples_leaf":[1, 2, 5],
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    gs = GridSearchCV(
        rf, param_grid,
        cv=5, scoring="neg_root_mean_squared_error",
        n_jobs=-1, verbose=1
    )
    gs.fit(X_train, y_train)
    
    print(f"[OK] Mejores hiperparámetros: {gs.best_params_}")
    print(f"     CV RMSE: {-gs.best_score_:.2f} min")
    
    joblib.dump(gs.best_estimator_, f"{MODELS_DIR}/mp1_regresion_optimizado.pkl")
    return gs.best_estimator_, gs.best_params_


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

def _graficar_importancia(modelo, feature_names, tag):
    importancias = pd.Series(
        modelo.feature_importances_, index=feature_names
    ).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
    importancias.plot(kind="barh", ax=ax, color="#1565C0")
    ax.set_title(f"Importancia de Variables — {tag}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importancia relativa")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/importancia_variables_{tag.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gráfica importancia guardada: {REPORT_DIR}/importancia_variables_{tag}.png")


def _graficar_real_vs_predicho(y_true, y_pred, tag):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#1565C0", s=20)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="Predicción perfecta")
    ax.set_xlabel("Valor real (min)")
    ax.set_ylabel("Valor predicho (min)")
    ax.set_title(f"Real vs. Predicho — {tag}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/real_vs_predicho_{tag.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def _graficar_confusion(y_true, y_pred, tag, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_labels = labels if labels else [str(l) for l in sorted(set(y_true))]
    disp = ConfusionMatrixDisplay(cm, display_labels=disp_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de Confusión — {tag}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/confusion_{tag.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def _graficar_roc(y_true, y_prob, tag):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title(f"Curva ROC — {tag}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/roc_{tag.lower().replace(' ','_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_todos_los_modelos(df):
    """
    Ejecuta el entrenamiento completo de los tres sub-modelos.
    Genera todos los reportes y gráficas para el Capítulo IV.
    """
    print("\n" + "="*60)
    print("MOETC-BD — MÓDULO DE MACHINE LEARNING")
    print("Capa 2: Analytics Layer")
    print("="*60)
    
    splits, features = preparar_datasets(df)
    
    resultados_globales = {}
    
    if "mp1" in splits:
        resultados_globales["mp1"] = entrenar_mp1_regresion(splits)
    
    if "mp3" in splits:
        resultados_globales["mp3"] = entrenar_mp3_otif(splits)
    
    if "mp2" in splits:
        resultados_globales["mp2"] = entrenar_mp2_riesgo(splits)
    
    # Guardar resumen
    resumen = {
        k: {
            "mejor_algoritmo": v.get("mejor_nombre", "N/A"),
            "metricas": list(v.get("resultados", {}).get(
                v.get("mejor_nombre", ""), {}
            ).items())
        }
        for k, v in resultados_globales.items()
    }
    with open(f"{REPORT_DIR}/resumen_modelos.json", "w") as f:
        json.dump(resumen, f, indent=2)
    
    print("\n✅ Entrenamiento completo. Modelos guardados en models/saved/")
    print("   Reportes guardados en models/reports/")
    
    return resultados_globales


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_pipeline import ejecutar_pipeline
    from src.features      import ejecutar_feature_engineering
    
    df = ejecutar_pipeline()
    df = ejecutar_feature_engineering(df)
    resultados = entrenar_todos_los_modelos(df)
