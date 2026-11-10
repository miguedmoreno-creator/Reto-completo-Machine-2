# app.py - Streamlit Frontend Avanzado para el Pipeline de Clustering
"""
================================================================================
PLATAFORMA ANALÍTICA INTERACTIVA: SEGMENTACIÓN DE MINIFIGURAS LEGO (RUTA B)
================================================================================
Interfaz reactiva, inspirada en sistemas analíticos avanzados con UI moderna,
diseño por componentes, inyección CSS personalizada y arquitectura agnóstica.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="LEGO Cluster Engine | MLOps Suite",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. INYECCIÓN DE ESTILOS CSS MODERNOS (Glassmorphism & High-Contrast)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;600&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Fondo moderno con gradiente sutil */
.stApp {
    background: radial-gradient(circle at 10% 15%, rgba(26, 41, 128, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 85%, rgba(220, 53, 69, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(255, 193, 7, 0.08) 0%, transparent 60%),
                #0b0f19;
    color: #e2e8f0;
}

/* Banner Hero */
.hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF 0%, #FBBF24 50%, #60A5FA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 6px;
    margin-bottom: 14px;
}

/* Badges / Chips */
.badge-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 8px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #cbd5e1;
}

.badge-gold {
    background: rgba(245, 158, 11, 0.15);
    border-color: rgba(245, 158, 11, 0.4);
    color: #fbbf24;
}

.badge-blue {
    background: rgba(59, 130, 246, 0.15);
    border-color: rgba(59, 130, 246, 0.4);
    color: #60a5fa;
}

.badge-emerald {
    background: rgba(16, 185, 129, 0.15);
    border-color: rgba(16, 185, 129, 0.4);
    color: #34d399;
}

/* Tarjeta de Resultado */
.result-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 22px 26px;
    margin-top: 15px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.4);
    animation: fadeIn 0.4s ease-out;
}

.result-card-0 {
    border-left: 6px solid #3b82f6;
}

.result-card-1 {
    border-left: 6px solid #ef4444;
}

.result-card-2 {
    border-left: 6px solid #f59e0b;
}

/* KPI Cards */
.kpi-container {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}

.kpi-title {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 4px 0;
}

.kpi-caption {
    font-size: 0.75rem;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. CARGA SEGURA DEL ARTEFACTO DINÁMICO
# ------------------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    artifact_path = Path(__file__).resolve().parent / "model.joblib"
    if not artifact_path.exists():
        st.error(f"❌ Artefacto no encontrado en {artifact_path}. Ejecuta 'python train.py' primero.")
        st.stop()
    return joblib.load(artifact_path)

artifact = load_artifact()
pipeline = artifact["model"]
features = artifact["features"]
stats = artifact.get("feature_stats", {})
cluster_descriptions = artifact.get("cluster_descriptions", {})
k = artifact.get("k", 3)
silhouette = artifact.get("silhouette_score", 0.0)
inertia = artifact.get("inertia", 0.0)
davies_bouldin = artifact.get("davies_bouldin_score", 0.0)
calinski = artifact.get("calinski_harabasz_score", 0.0)
total_samples = artifact.get("total_samples", 15790)

# Cargar dataset de muestra para visualizaciones
@st.cache_data
def load_data():
    csv_path = Path(__file__).resolve().parent / "data" / "datos.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None

df_raw = load_data()

# ------------------------------------------------------------------------------
# 4. HEADER HERO
# ------------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-card">
    <h1 class="hero-title">🧩 LEGO Minifigures Cluster Intelligence</h1>
    <div class="hero-subtitle">
        Plataforma analítica de segmentación no supervisada, perfilamiento estructural y serving desacoplado con K-Means y Docker.
    </div>
    <div>
        <span class="badge-chip badge-gold">⚡ Algoritmo: K-Means (k={k})</span>
        <span class="badge-chip badge-emerald">✔ Silueta: {silhouette:.4f} (Excelente)</span>
        <span class="badge-chip badge-blue">📦 Población: {total_samples:,} minifiguras</span>
        <span class="badge-chip">🐳 Despliegue: Docker Containerizado (:8501)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. BARRA LATERAL: ENTRADA DE DATOS Y PRESETS
# ------------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/24/LEGO_logo.svg", width=70)
st.sidebar.markdown("### 🎛️ Centro de Inferencia")
st.sidebar.markdown("Ajusta los parámetros para clasificar una minifigura en tiempo real:")

# Selector de Presets de Prueba
preset = st.sidebar.selectbox(
    "📌 Cargar Caso de Ejemplo:",
    options=[
        "Personalizado (Ajuste Manual)",
        "Caso 1: Minifigura Estándar (Ciudadano Común)",
        "Caso 2: Minifigura Compleja / MOC (Muchas Piezas)",
        "Caso 3: Minifigura Icónica (Batman / Recurrente)"
    ]
)

# Valores según preset
preset_values = {}
if preset == "Caso 1: Minifigura Estándar (Ciudadano Común)":
    preset_values = {"num_parts": 4.0, "total_sets_appeared": 1.0}
elif preset == "Caso 2: Minifigura Compleja / MOC (Muchas Piezas)":
    preset_values = {"num_parts": 38.0, "total_sets_appeared": 1.0}
elif preset == "Caso 3: Minifigura Icónica (Batman / Recurrente)":
    preset_values = {"num_parts": 4.0, "total_sets_appeared": 22.0}

inputs = {}
for feat in features:
    f_stat = stats.get(feat, {"min": 0.0, "max": 100.0, "mean": 10.0, "median": 5.0})
    min_val = float(f_stat.get("min", 0.0))
    max_val = float(f_stat.get("max", 150.0))
    default_val = preset_values.get(feat, float(f_stat.get("median", (min_val + max_val) / 2)))
    
    # Tooltip descriptivo
    help_text = "Número de piezas que ensamblan la figura" if feat == "num_parts" else "Sets de LEGO en los que aparece"
    
    inputs[feat] = st.sidebar.slider(
        label=f"🔹 {feat.replace('_', ' ').title()}",
        min_value=min_val,
        max_value=max_val,
        value=min(max(default_val, min_val), max_val),
        step=1.0,
        help=help_text
    )

st.sidebar.markdown("---")
st.sidebar.caption("💡 *Arquitectura Dinámica: Los rangos y campos se autoconfiguran leyendo el artefacto `model.joblib` sin código estático.*")

# ------------------------------------------------------------------------------
# 6. INFERENCIA EN TIEMPO REAL
# ------------------------------------------------------------------------------
input_df = pd.DataFrame([inputs], columns=features)
pred_cluster = int(pipeline.predict(input_df)[0])

# Metadatos temáticos por clúster
cluster_meta = {
    0: {
        "title": "Minifiguras Estándar / Sets Especiales",
        "badge_class": "badge-blue",
        "card_class": "result-card-0",
        "icon": "👤",
        "pop_share": "94.9% del catálogo (14,984 figuras)",
        "action": "Estrategia de Producción Masiva: Mantener optimización de inventario estándar; figuras de rotación normal para sets temáticos regulares."
    },
    1: {
        "title": "Minifiguras Complejas / MOCs Especiales",
        "badge_class": "badge-emerald",
        "card_class": "result-card-1",
        "icon": "🤖",
        "pop_share": "1.0% del catálogo (162 figuras)",
        "action": "Estrategia Collector / High-End: Minifiguras con armaduras, accesorios o construcciones avanzadas. Alto valor unitario en reventa y empaque de lujo."
    },
    2: {
        "title": "Minifiguras Icónicas y Ubicuas",
        "badge_class": "badge-gold",
        "card_class": "result-card-2",
        "icon": "⭐",
        "pop_share": "4.1% del catálogo (644 figuras)",
        "action": "Estrategia de Franquicia Transversal: Personajes legendarios presentes en múltiples olas y sets. Pilar de marketing y fidelización global."
    }
}
meta = cluster_meta.get(pred_cluster, {
    "title": f"Clúster {pred_cluster}",
    "badge_class": "badge-blue",
    "card_class": "result-card-0",
    "icon": "🧩",
    "pop_share": "N/A",
    "action": "Asignación estándar generada por K-Means."
})

# ------------------------------------------------------------------------------
# 7. ORGANIZACIÓN POR PESTAÑAS (TABS)
# ------------------------------------------------------------------------------
tab_inference, tab_space, tab_metrics, tab_agnostic = st.tabs([
    "🎯 Inferencia y Diagnóstico",
    "🗺️ Espacio de Clusters 2D",
    "📊 Perfiles y Métricas MLOps",
    "🔄 Modo Dataset Agnóstico"
])

# ==============================================================================
# TAB 1: INFERENCIA Y DIAGNÓSTICO
# ==============================================================================
with tab_inference:
    st.markdown("### 📋 Diagnóstico del Nuevo Caso Evaluado")
    
    # Tarjeta de Resultado Principal
    st.markdown(f"""
    <div class="result-card {meta['card_class']}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="margin: 0; color: #f8fafc; font-size: 1.5rem;">
                {meta['icon']} Segmento Asignado: <strong>Clúster {pred_cluster}</strong> — {meta['title']}
            </h2>
            <span class="badge-chip {meta['badge_class']}">Segmento #{pred_cluster}</span>
        </div>
        <p style="margin-top: 10px; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
            {cluster_descriptions.get(pred_cluster, 'Sin descripción disponible')}
        </p>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 14px 0;">
        <div style="display: flex; gap: 20px; font-size: 0.85rem;">
            <div><strong>📊 Frecuencia en Catálogo:</strong> {meta['pop_share']}</div>
            <div><strong>💼 Recomendación Táctica:</strong> {meta['action']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.markdown("#### 🔍 Comparación vs. Promedios del Clúster")
        # Tabla comparativa
        compare_data = []
        for feat in features:
            val_in = inputs[feat]
            f_stat = stats.get(feat, {})
            c_mean = f_stat.get("mean", val_in)
            compare_data.append({
                "Variable": feat.replace("_", " ").title(),
                "Valor Ingresado": f"{val_in:.1f}",
                "Promedio General": f"{c_mean:.2f}",
                "Diferencia": f"{val_in - c_mean:+.2f}"
            })
        st.dataframe(pd.DataFrame(compare_data), use_container_width=True)
        
    with col_b:
        st.markdown("#### 📈 Nivel Relativo en Escala Normalizada")
        # Gráfico de barras comparativo de las variables
        fig_bar, ax = plt.subplots(figsize=(6, 3.2), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        
        y_pos = np.arange(len(features))
        vals = [inputs[f] for f in features]
        means = [stats.get(f, {}).get("mean", 1.0) for f in features]
        
        ax.barh(y_pos - 0.15, vals, height=0.3, label="Caso Actual", color="#F59E0B")
        ax.barh(y_pos + 0.15, means, height=0.3, label="Media Global", color="#3B82F6", alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace("_", " ").title() for f in features], color="#E2E8F0")
        ax.tick_params(colors="#94A3B8")
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")
        
        st.pyplot(fig_bar)

# ==============================================================================
# TAB 2: ESPACIO DE CLUSTERS 2D
# ==============================================================================
with tab_space:
    st.markdown("### 🗺️ Distribución Espacial de Minifiguras y Proyección del Nuevo Caso")
    st.caption("Visualización del mapa de dispersión del catálogo LEGO. El marcador dorado con estrella representa el caso que acabas de configurar.")
    
    if df_raw is not None and "num_parts" in df_raw.columns and "total_sets_appeared" in df_raw.columns:
        fig_scatter, ax_sc = plt.subplots(figsize=(9, 4.8), facecolor="#0f172a")
        ax_sc.set_facecolor("#0f172a")
        
        # Muestra de fondo (1500 puntos para fluidez)
        df_sample = df_raw.sample(n=min(1500, len(df_raw)), random_state=42)
        X_sample = df_sample[["num_parts", "total_sets_appeared"]]
        sample_preds = pipeline.predict(X_sample)
        
        palette = {0: "#3B82F6", 1: "#EF4444", 2: "#10B981"}
        names = {0: "Clúster 0 (Estándar)", 1: "Clúster 1 (Complejas)", 2: "Clúster 2 (Icónicas)"}
        
        for c_id in range(3):
            mask = sample_preds == c_id
            ax_sc.scatter(
                X_sample.loc[mask, "num_parts"],
                X_sample.loc[mask, "total_sets_appeared"],
                c=palette.get(c_id, "#94A3B8"),
                label=names.get(c_id, f"Clúster {c_id}"),
                alpha=0.45,
                s=25,
                edgecolors="none"
            )
            
        # Marcador del caso evaluado
        ax_sc.scatter(
            [inputs["num_parts"]],
            [inputs["total_sets_appeared"]],
            c="#FBBF24",
            s=220,
            marker="*",
            edgecolors="#FFFFFF",
            linewidths=1.5,
            label="⭐ Caso Evaluado",
            zorder=10
        )
        
        ax_sc.set_xlabel("Número de Piezas (num_parts)", color="#E2E8F0", fontsize=10)
        ax_sc.set_ylabel("Sets en que Aparece (total_sets_appeared)", color="#E2E8F0", fontsize=10)
        ax_sc.tick_params(colors="#94A3B8")
        ax_sc.grid(color="#1E293B", linestyle="--", linewidth=0.5, alpha=0.7)
        ax_sc.spines['bottom'].set_color('#334155')
        ax_sc.spines['left'].set_color('#334155')
        ax_sc.spines['top'].set_visible(False)
        ax_sc.spines['right'].set_visible(False)
        ax_sc.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", loc="upper right")
        
        st.pyplot(fig_scatter)
    else:
        st.info("ℹ️ Dataset completo disponible en `data/datos.csv`.")

# ==============================================================================
# TAB 3: PERFILES Y MÉTRICAS MLOPS
# ==============================================================================
with tab_metrics:
    st.markdown("### 📊 Métricas de Validación No Supervisada")
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Silhouette Score</div>
            <div class="kpi-value" style="color: #34d399;">{silhouette:.4f}</div>
            <div class="kpi-caption">Rango [-1, 1] — Cohesión Alta</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Inercia WCSS</div>
            <div class="kpi-value" style="color: #60a5fa;">{inertia:,.0f}</div>
            <div class="kpi-caption">Suma de cuadrados interna</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Davies-Bouldin</div>
            <div class="kpi-value" style="color: #fbbf24;">{davies_bouldin:.4f}</div>
            <div class="kpi-caption">Menor es mejor (Separación)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Calinski-Harabasz</div>
            <div class="kpi-value" style="color: #f472b6;">{calinski:,.0f}</div>
            <div class="kpi-caption">Mayor es mejor (Densidad)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📑 Perfiles Estructurales de los Segmentos")
    
    profiles = artifact.get("cluster_profiles", {})
    if profiles:
        prof_df = pd.DataFrame(profiles).T
        prof_df.index.name = "Clúster"
        st.dataframe(prof_df, use_container_width=True)

# ==============================================================================
# TAB 4: MODO AGNOSTICO A CUALQUIER DATASET
# ==============================================================================
with tab_agnostic:
    st.markdown("### 🔄 Capacidad Agnóstica a Nuevos Datasets")
    st.markdown("""
    En cumplimiento de las especificaciones del docente: ***"El sistema debe ser capaz de operar con cualquier dataset tabular"***.
    
    El pipeline detecta automáticamente columnas numéricas, excluye identificadores y URLs, y recalcula el artefacto `model.joblib`.
    """)
    
    uploaded_file = st.file_uploader("📂 Subir dataset alternativo (.csv) para verificar adaptabilidad:", type=["csv"])
    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            st.success(f"✔ Dataset cargado exitosamente: **{df_new.shape[0]:,} filas × {df_new.shape[1]} columnas**.")
            st.markdown("#### Vista previa de las primeras filas:")
            st.dataframe(df_new.head(4), use_container_width=True)
            
            num_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
            st.info(f"🔎 Variables numéricas detectadas automáticamente ({len(num_cols)}): `{num_cols}`")
            st.markdown("""
            > *Para entrenar el modelo con este nuevo dataset, simplemente ejecute:*
            > ```bash
            > python train.py --data ruta_al_archivo.csv
            > ```
            """)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")

# ------------------------------------------------------------------------------
# 8. FOOTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem;">
    <strong>Taller de Machine Learning 2 (Ruta B: Clustering)</strong> | Pipeline Scikit-Learn + Streamlit + Docker + GCP<br>
    Desarrollado con arquitectura desacoplada y contrato inmutable <code>model.joblib</code>.
</div>
""", unsafe_allow_html=True)
