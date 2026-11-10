"""
train.py - Pipeline de Entrenamiento y Modelado No Supervisado (Clustering)
=============================================================================
Reto de Machine Learning 2 - Ruta B: Aprendizaje No Supervisado
Integrante 1: ML Engineer & Data Pipeline

Este script implementa un pipeline reproducible, modular y 100% adaptable
(agnostico al dataset). Soporta tanto el dataset de Minifiguras de LEGO
como cualquier otro dataset tabular que se provea en el futuro.

Uso:
    python train.py
    python train.py --data data/datos.csv --k 3
    python train.py --data lego_minifigures_1975_2025.csv --evaluate-k
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(data_path: str) -> pd.DataFrame:
    """Carga un dataset tabular desde un archivo CSV."""
    path = Path(data_path)
    if not path.exists():
        # Fallback inteligente si se llamo por defecto y el archivo esta en otra ubicacion comun
        fallbacks = [
            Path("data/datos.csv"),
            Path("lego_minifigures_1975_2025.csv"),
            Path("../data/datos.csv"),
        ]
        for fb in fallbacks:
            if fb.exists():
                print(f"[INFO] '{data_path}' no encontrado. Usando fallback: '{fb}'")
                path = fb
                break
        else:
            raise FileNotFoundError(f"No se encontro el archivo de datos en: {data_path}")

    print(f"[INFO] Cargando dataset desde: {path.resolve()}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset cargado exitosamente. Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas.")
    return df


def detect_features(df: pd.DataFrame, explicit_features: Optional[List[str]] = None) -> List[str]:
    """
    Detecta automaticamente las caracteristicas numericas relevantes para clustering,
    excluyendo identificadores unicos, URLs o campos no informativos.
    """
    if explicit_features:
        missing = [f for f in explicit_features if f not in df.columns]
        if missing:
            raise ValueError(f"Las siguientes columnas explicitas no existen en el dataset: {missing}")
        return explicit_features

    numeric_cols = []
    total_rows = len(df)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_lower = col.lower()
        # Filtrar identificadores comunes o URLs si fuesen numericos
        is_id_name = (
            col_lower == "id"
            or col_lower.endswith("_id")
            or col_lower.startswith("id_")
        )
        # Si todos los valores son unicos y correlativos, probablemente es un indice
        is_unique_index = (df[col].nunique() == total_rows and total_rows > 100)

        if not is_id_name and not is_unique_index:
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No se encontraron columnas numericas validas para clustering en el dataset.")

    print(f"[INFO] Variables numericas detectadas automaticamente ({len(numeric_cols)}): {numeric_cols}")
    return numeric_cols


def compute_feature_stats(df: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, float]]:
    """Calcula estadisticas resumen de cada variable para el frontend de Streamlit."""
    stats = {}
    for f in features:
        series = pd.to_numeric(df[f], errors="coerce").dropna()
        stats[f] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(round(series.mean(), 2)),
            "median": float(round(series.median(), 2)),
            "std": float(round(series.std(), 2)) if len(series) > 1 else 0.0,
            "q25": float(round(series.quantile(0.25), 2)),
            "q75": float(round(series.quantile(0.75), 2)),
        }
    return stats


def build_pipeline(n_clusters: int = 3, random_state: int = 42) -> Pipeline:
    """
    Construye el Pipeline estandar de Scikit-Learn:
    Imputacion por mediana + Estandarizacion + K-Means.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
    ])
    return pipeline


def evaluate_k_range(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 7) -> None:
    """Evalua un rango de k e imprime la tabla comparativa de Inercia y Silueta."""
    print("\n" + "=" * 70)
    print("  EVALUACION DE HIPERPARAMETRO k (METODO DEL CODO Y COEFICIENTE DE SILUETA)")
    print("=" * 70)
    print(f"{'k':>4} | {'Inercia (WCSS)':>16} | {'Silueta (Silhouette)':>22} | {'Davies-Bouldin':>16}")
    print("-" * 70)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # Calculo de silueta muestreado para agilidad si el dataset es masivo
        sample_size = min(5000, len(X_scaled)) if len(X_scaled) > 5000 else None
        sil = silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=42)
        db = davies_bouldin_score(X_scaled, labels)

        print(f"{k:>4} | {km.inertia_:>16.2f} | {sil:>22.4f} | {db:>16.4f}")
    print("=" * 70 + "\n")


def profile_clusters(
    df: pd.DataFrame,
    features: List[str],
    labels: np.ndarray,
    n_clusters: int,
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Genera perfiles estadisticos y descripciones semanticas claras e interpretables
    para cada cluster, con soporte para el caso base LEGO y generico para otros datasets.
    """
    df_clustered = df.copy()
    df_clustered["_cluster"] = labels

    profiles_table = (
        df_clustered.groupby("_cluster")[features]
        .mean()
        .round(2)
    )

    counts = df_clustered["_cluster"].value_counts().sort_index()
    total_samples = len(df_clustered)

    descriptions = {}
    is_lego = set(features) == {"num_parts", "total_sets_appeared"} or (
        "num_parts" in features and "total_sets_appeared" in features
    )

    for cluster_id in range(n_clusters):
        count_c = int(counts.get(cluster_id, 0))
        pct_c = (count_c / total_samples) * 100
        means = profiles_table.loc[cluster_id]

        if is_lego and n_clusters == 3:
            # Perfilamiento semantico contextualizado para el caso de LEGO Minifigures
            mean_parts = means["num_parts"]
            mean_sets = means["total_sets_appeared"]

            if mean_parts > 20:
                name = "Minifiguras Complejas / MOCs de Gran Escala"
                desc = (
                    f"Perfil de Alta Complejidad: figuras con promedio de {mean_parts:.1f} piezas "
                    f"(muy por encima de la media), presentes en pocos sets ({mean_sets:.1f} sets). "
                    f"Representa el {pct_c:.1f}% del catalogo ({count_c:,} figuras)."
                )
            elif mean_sets > 3.0:
                name = "Minifiguras Iconicas / Ubicuas"
                desc = (
                    f"Perfil Iconico y Recurrente: piezas estandar ({mean_parts:.1f} partes) pero con "
                    f"altisima presencia transversal en multiples sets ({mean_sets:.1f} sets en promedio, hasta 40+). "
                    f"Representa el {pct_c:.1f}% del catalogo ({count_c:,} figuras)."
                )
            else:
                name = "Minifiguras Estandar / Sets Especiales"
                desc = (
                    f"Perfil Estandar y Especifico: figuras convencionales ({mean_parts:.1f} partes) "
                    f"distribuidas en 1 o 2 sets unicos ({mean_sets:.1f} sets). "
                    f"Representa la gran mayoria del catalogo: {pct_c:.1f}% ({count_c:,} figuras)."
                )
            descriptions[cluster_id] = f"{name}: {desc}"

        else:
            # Perfilamiento semantico dinamico agnostico a cualquier dataset
            traits = []
            for feat in features:
                val = means[feat]
                pop_mean = df[feat].mean()
                pop_std = df[feat].std() if df[feat].std() > 0 else 1.0
                z = (val - pop_mean) / pop_std

                if z > 0.6:
                    traits.append(f"Alto {feat} ({val:.2f})")
                elif z < -0.6:
                    traits.append(f"Bajo {feat} ({val:.2f})")
                else:
                    traits.append(f"{feat} promedio ({val:.2f})")

            summary_traits = ", ".join(traits)
            descriptions[cluster_id] = (
                f"Cluster {cluster_id} ({count_c:,} obs, {pct_c:.1f}%): Caracterizado por {summary_traits}."
            )

    profiles_dict = profiles_table.to_dict(orient="index")
    return descriptions, profiles_dict


def train(
    data_path: str = "data/datos.csv",
    n_clusters: int = 3,
    output_path: str = "model.joblib",
    features: Optional[List[str]] = None,
    run_eval_k: bool = False,
) -> Dict[str, Any]:
    """Ejecuta el ciclo de entrenamiento completo y genera el artefacto model.joblib."""
    print("=" * 70)
    print("   INICIANDO PIPELINE DE ENTRENAMIENTO DE MACHINE LEARNING (CLUSTERING)")
    print("=" * 70)

    # 1. Cargar datos
    df = load_dataset(data_path)

    # 2. Seleccionar caracteristicas
    selected_features = detect_features(df, features)
    X = df[selected_features].copy()

    # 3. Construir e inicializar pipeline
    pipeline = build_pipeline(n_clusters=n_clusters, random_state=42)

    # Preprocesar datos para evaluacion
    X_preprocessed = pipeline.named_steps["scaler"].fit_transform(
        pipeline.named_steps["imputer"].fit_transform(X)
    )

    if run_eval_k:
        evaluate_k_range(X_preprocessed, k_min=2, k_max=7)

    # 4. Ajustar modelo final
    print(f"[INFO] Ajustando K-Means con k={n_clusters} clusters...")
    pipeline.fit(X)
    labels = pipeline.named_steps["cluster"].labels_

    # 5. Evaluacion cuantitativa
    inertia = float(pipeline.named_steps["cluster"].inertia_)
    sample_size = min(5000, len(X)) if len(X) > 5000 else None
    sil_score = float(silhouette_score(X_preprocessed, labels, sample_size=sample_size, random_state=42))
    db_score = float(davies_bouldin_score(X_preprocessed, labels))
    ch_score = float(calinski_harabasz_score(X_preprocessed, labels))

    print("\n" + "-" * 50)
    print(" METRICAS DE EVALUACION (Ruta B: Clustering)")
    print("-" * 50)
    print(f" • Numero de Clusters (k):        {n_clusters}")
    print(f" • Coeficiente de Silueta:        {sil_score:.4f} (Rango [-1, 1], mayor es mejor)")
    print(f" • Inercia (WCSS):                {inertia:.2f}")
    print(f" • Indice Davies-Bouldin:         {db_score:.4f} (menor es mejor)")
    print(f" • Indice Calinski-Harabasz:      {ch_score:.2f} (mayor es mejor)")
    print("-" * 50 + "\n")

    # 6. Estadisticas de variables para el frontend
    stats = compute_feature_stats(df, selected_features)

    # 7. Perfilamiento e interpretacion
    descriptions, profiles = profile_clusters(df, selected_features, labels, n_clusters)

    print("PERFILES INTERPRETATIVOS DE CADA CLUSTER:")
    for cid, desc in descriptions.items():
        print(f"  [{cid}] {desc}")
    print("\n" + "=" * 70)

    # 8. Empaquetar Artefacto estandar (Contrato entre Integrante 1 e Integrante 2)
    artifact = {
        "model": pipeline,
        "features": selected_features,
        "numeric_features": selected_features,
        "categorical_features": [],
        "feature_stats": stats,
        "task": "clustering",
        "k": n_clusters,
        "silhouette_score": sil_score,
        "inertia": inertia,
        "davies_bouldin_score": db_score,
        "calinski_harabasz_score": ch_score,
        "cluster_descriptions": descriptions,
        "cluster_profiles": profiles,
        "dataset_name": Path(data_path).name,
        "total_samples": len(df),
    }

    out_file = Path(output_path)
    joblib.dump(artifact, out_file)
    print(f"[EXITO] Artefacto exportado correctamente a: {out_file.resolve()}")
    print("=" * 70 + "\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento agnostico de clustering para el Reto de Machine Learning 2."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/datos.csv",
        help="Ruta al archivo CSV de entrada (default: data/datos.csv)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Numero de clusters para K-Means (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.joblib",
        help="Ruta donde se guardara el artefacto model.joblib (default: model.joblib)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Lista de features separadas por comas (opcional)",
    )
    parser.add_argument(
        "--evaluate-k",
        action="store_true",
        help="Ejecuta una evaluacion de k=2..7 mostrando inercia y silueta antes del entrenamiento.",
    )

    args = parser.parse_args()

    feats = [f.strip() for f in args.features.split(",")] if args.features else None

    train(
        data_path=args.data,
        n_clusters=args.k,
        output_path=args.output,
        features=feats,
        run_eval_k=args.evaluate_k,
    )


if __name__ == "__main__":
    main()
