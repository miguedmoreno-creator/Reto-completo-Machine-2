# app.py - Streamlit frontend for the ML clustering pipeline
"""
Interactive Streamlit application that loads the artefact produced by `train.py`
(`model.joblib`) and provides a dynamic UI based on the numeric features
detected during training. The UI allows the user to input values, predicts the
cluster, and shows a human‑readable description.

Optional visualization: a 2‑D PCA projection of the original data coloured by
cluster and the new observation.
"""

import pathlib
import joblib
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load artefact ----------------------------------------------------------------
# ---------------------------------------------------------------------------
ARTIFACT_PATH = pathlib.Path(__file__).parent / "model.joblib"
if not ARTIFACT_PATH.exists():
    st.error(f"❌ Artefacto no encontrado en {ARTIFACT_PATH}")
    st.stop()

artifact = joblib.load(ARTIFACT_PATH)

# Extract useful pieces --------------------------------------------------------
model = artifact["model"]
features = artifact["features"]  # list of numeric feature names
stats = artifact["feature_stats"]  # dict with min, max, mean, median, etc.
cluster_desc = artifact["cluster_descriptions"]

st.set_page_config(page_title="ML Clustering – Demo", layout="centered")
st.title("🚀 Demo de Clustering – Interfaz Streamlit")
st.caption("Integrante 2 (Frontend & DevOps) – Interfaz dinámica basada en el artefacto creado por el Integrante 1.")

# ---------------------------------------------------------------------------
# Sidebar – input widgets ------------------------------------------------------
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Parámetros de entrada")
user_input = {}
for feat in features:
    f_stats = stats[feat]
    # Use median as default value; step = (max-min)/100 for reasonable granularity
    default = f_stats.get("median", f_stats.get("mean", (f_stats["min"] + f_stats["max"]) / 2))
    step = max((f_stats["max"] - f_stats["min"]) / 100, 0.01)
    val = st.sidebar.number_input(
        label=feat,
        min_value=float(f_stats["min"]),
        max_value=float(f_stats["max"]),
        value=float(default),
        step=step,
        format="%.4f",
    )
    user_input[feat] = val

# ---------------------------------------------------------------------------
# Predict ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
if st.sidebar.button("🔍 Predecir"):
    # Build a one‑row dataframe with the same column order as training
    df_input = pd.DataFrame([user_input])
    pred = model.predict(df_input)[0]
    desc = cluster_desc.get(int(pred), "Descripción no disponible")
    st.success(f"**Cluster asignado:** {pred}")
    st.info(desc)

    # -----------------------------------------------------------------------
    # Visualización opcional -------------------------------------------------
    # -----------------------------------------------------------------------
    if st.checkbox("Mostrar visualización (PCA 2‑D)"):
        # Retrieve original training data (the model pipeline contains the imputer & scaler)
        # We'll reconstruct the processed X from the pipeline steps.
        # The pipeline is: imputer -> scaler -> cluster
        # We can get the pre‑processed matrix via pipeline.named_steps["scaler"].transform(...)
        # Load the raw data used during training – it is stored in the artefact as well.
        # Unfortunately the raw dataframe is not stored; we can approximate by reading the CSV again.
        raw_path = pathlib.Path(__file__).parent / "data" / "datos.csv"
        if raw_path.exists():
            raw_df = pd.read_csv(raw_path)
            X_raw = raw_df[features]
            # Apply the same preprocessing steps as the pipeline
            X_pre = model.named_steps["imputer"].fit_transform(X_raw)
            X_pre = model.named_steps["scaler"].transform(X_pre)
            # Reduce to 2 dimensions for plotting
            pca = PCA(n_components=2)
            X2d = pca.fit_transform(X_pre)
            # Transform the new sample using the same steps
            new_pre = model.named_steps["imputer"].transform(df_input)
            new_pre = model.named_steps["scaler"].transform(new_pre)
            new2d = pca.transform(new_pre)
            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(
                X2d[:, 0], X2d[:, 1], c=model.named_steps["cluster"].labels_,
                cmap="viridis", s=20, alpha=0.6, label="Datos de entrenamiento"
            )
            ax.scatter(new2d[0, 0], new2d[0, 1], color="red", s=80, marker="X", label="Nueva observación")
            ax.set_xlabel("PCA‑1")
            ax.set_ylabel("PCA‑2")
            ax.set_title("Proyección PCA 2‑D de los clusters")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No se encontró el archivo de datos para la visualización.")
else:
    st.info("Introduce los valores en la barra lateral y pulsa **Predecir**.")

# ---------------------------------------------------------------------------
# Footer ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
st.caption("Este proyecto está pensado para ser agnóstico al dataset; basta con reemplazar `data/datos.csv` y volver a ejecutar `train.py` para generar un nuevo artefacto.")

