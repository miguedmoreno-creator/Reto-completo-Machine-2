# Proyecto de Machine Learning: Ruta B - Aprendizaje No Supervisado (Clustering)

Aplicación interactiva y reproducible para segmentación y descubrimiento de patrones en datos no etiquetados mediante algoritmos de agrupamiento (*Clustering*), desarrollada con **scikit-learn**, desplegada con **Streamlit**, empaquetada con **Docker** y alojada en **Google Cloud Platform (GCP - Compute Engine)**.

> 📌 **Plan de Trabajo y Colaboración:** Consulta el documento detallado de división de roles y arquitectura agnóstica en [PLAN_DE_IMPLEMENTACION.md](PLAN_DE_IMPLEMENTACION.md).

---

## 👥 Autores
- **Integrante 1 (ML & Pipeline):** [Nombre y Apellidos] - [Correo / Usuario GitHub]
- **Integrante 2 (Frontend & DevOps):** [Nombre y Apellidos] - [Correo / Usuario GitHub]
- **Institución:** Curso de Machine Learning 2

---

## 🎯 1. Título y Objetivo del Proyecto

- **Título:** Pipeline Generalizado de Segmentación no Supervisada y Asignación de Perfiles en Tiempo Real.
- **Objetivo:** Descubrir grupos homogéneos, segmentos o perfiles latentes a partir de un conjunto de datos multidimensional no supervisado, caracterizar estadísticamente cada clúster y disponibilizar una interfaz interactiva en Streamlit contenerizada en Docker que permita clasificar y perfilar nuevas observaciones en tiempo real.

---

## 📊 2. Descripción del Dataset y Unidad de Observación

> Este proyecto está diseñado para ser **agnóstico al dataset**. Puede ser utilizado con cualquier conjunto de datos tabular (por ejemplo: clientes, transacciones, registros clínicos, sensores, etc.).

- **Dataset utilizado:** `LEGO Minifigures 1975–2025` — Catálogo exhaustivo de minifiguras de LEGO extraído de [Rebrickable](https://rebrickable.com/).
- **Ruta de almacenamiento local:** `data/datos.csv` (copia de `lego_minifigures_1975_2025.csv`)
- **Unidad de observación:** Cada fila representa **una minifigura única** identificada por su código de inventario (`fig_num`).
- **Tamaño:** `15,790 filas × 5 columnas`.
- **Formato esperado:** Archivo CSV separado por comas. Las columnas no numéricas o identificadores (`fig_num`, `name`, `img_url`) son excluidas automáticamente durante el pipeline.

---

## 🔀 3. Ruta Elegida

- **Ruta B: Aprendizaje No Supervisado (Clustering)**
- **Justificación:** Los datos no cuentan con una variable objetivo (*target* o etiqueta predefinida). El objetivo primordial es el aprendizaje estructural para agrupar instancias con características similares e identificar comportamientos o arquetipos representativos dentro del catálogo de minifiguras.

---

## 🧩 4. Variables Utilizadas

El pipeline detecta automáticamente las variables numéricas relevantes del dataset de LEGO:

| Nombre de la Variable | Tipo de Dato | Rol / Justificación | Rango / Valores |
| :--- | :--- | :--- | :--- |
| `num_parts` | Numérica Discreta | Número de piezas que componen la minifigura. Mide la complejidad física del modelo. | `2 – 143` (media: 4.90, mediana: 4.0) |
| `total_sets_appeared` | Numérica Discreta | Cantidad total de sets de LEGO en los que aparece la minifigura. Mide la ubicuidad/popularidad. | `1 – 41` (media: 1.40, mediana: 1.0) |

**Variables excluidas automáticamente:**
| Variable | Motivo de Exclusión |
| :--- | :--- |
| `fig_num` | Identificador único (no aporta información numérica para clustering) |
| `name` | Texto descriptivo de la minifigura |
| `img_url` | URL de imagen (campo no numérico) |

*(Nota: En clustering se priorizan variables numéricas continuas/discretas normalizadas para evitar distorsiones en el cálculo de distancias euclidianas).*

---

## ⚙️ 5. Preparación de Datos y Pipeline

Para garantizar la reproducibilidad y prevenir la fuga de información (*data leakage*), el preprocesamiento se encapsula completamente dentro de un `Pipeline` de scikit-learn:

1. **Tratamiento de Valores Faltantes (Imputación):**
   - Variables numéricas: Imputación por mediana (`SimpleImputer(strategy='median')`) para mitigar el impacto de valores atípicos (*outliers*).
   - El dataset actual no presenta valores faltantes (0 NaN en las 5 columnas), pero el pipeline está diseñado para manejarlos si se cambia el dataset.
2. **Escalamiento y Normalización:**
   - Estandarización de variables numéricas con `StandardScaler` ($\mu=0, \sigma=1$). Esto es indispensable ya que `num_parts` (rango 2–143) y `total_sets_appeared` (rango 1–41) tienen escalas diferentes, y K-Means basa su optimización en distancias euclidianas.
3. **Detección Automática de Features:**
   - El script `train.py` identifica automáticamente las columnas numéricas, excluyendo campos de tipo texto, URLs e identificadores únicos.

**Estructura del Pipeline en código:**
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("cluster", KMeans(n_clusters=3, random_state=42, n_init=10)),
])
```

---

## 🧠 6. Modelo de Agrupamiento y Justificación

- **Algoritmo seleccionado:** **K-Means**
- **Justificación del modelo:**
  - K-Means ofrece alta eficiencia computacional $O(k \cdot n \cdot d)$, facilidad de convergencia y rápida inferencia en producción para asignar nuevas observaciones al centroide más cercano.
  - La interpretabilidad de los centroides permite generar perfiles de clúster directamente traducibles a descripciones de negocio / catálogo.
- **Selección de hiperparámetros:**
  - Número de clusters ($k = 3$): Determinado mediante el análisis del **Método del Codo (*Elbow Method*)** analizando la inercia y validado con el coeficiente de silueta.

### Evaluación de $k$ óptimo (Método del Codo y Silueta)

| $k$ | Inercia (WCSS) | Coeficiente de Silueta | Índice Davies-Bouldin |
| :---: | ---: | ---: | ---: |
| 2 | 21,334.98 | **0.9061** | 0.4394 |
| 3 | 13,274.56 | **0.8287** | 0.5725 |
| 4 | 9,155.33 | 0.7699 | 0.5132 |
| 5 | 6,999.15 | 0.7854 | 0.5276 |
| 6 | 5,659.73 | 0.7123 | 0.5669 |
| 7 | 4,371.70 | 0.7148 | 0.5024 |

> **Decisión:** Se eligió $k = 3$ porque ofrece el mejor balance entre separabilidad (Silueta = 0.83, excelente) y granularidad interpretativa. Con $k = 2$ la silueta es mayor (0.91) pero solo genera dos grupos demasiado genéricos. Con $k = 3$ se obtienen tres perfiles bien diferenciados y relevantes para el análisis del catálogo LEGO.

---

## 📈 7. Criterios de Evaluación

En el aprendizaje no supervisado no existen matrices de confusión directas; la calidad de los clusters se evalúa combinando criterios intrínsecos de separación y cohesión:

### Métricas del modelo final ($k = 3$):

| Métrica | Valor | Interpretación |
| :--- | ---: | :--- |
| **Coeficiente de Silueta** | **0.8287** | Excelente separación (rango $[-1, 1]$; > 0.50 se considera bueno) |
| **Inercia (WCSS)** | 13,274.56 | Suma de distancias cuadráticas intra-clúster |
| **Índice Davies-Bouldin** | 0.5725 | Buena compacidad y separación (menor es mejor) |
| **Índice Calinski-Harabasz** | 10,885.03 | Alta varianza inter-clúster vs intra-clúster (mayor es mejor) |

---

## 🔍 8. Interpretación de la Salida y Perfilamiento

El entrenamiento genera perfiles estadísticos promedio de cada grupo. Al predecir sobre una nueva entrada, la aplicación no solo retorna un número de clúster, sino su **descripción cualitativa**:

| Clúster | Nombre del Perfil | Tamaño | Interpretación / Características Clave |
| :---: | :--- | :--- | :--- |
| **0** | **Minifiguras Estándar / Sets Especiales** | 14,984 (94.9%) | Figuras convencionales con ~4.6 partes, distribuidas en 1–2 sets únicos. Representan la gran mayoría del catálogo. |
| **1** | **Minifiguras Complejas / MOCs de Gran Escala** | 162 (1.0%) | Figuras con promedio de 36.2 piezas (muy por encima de la media), presentes en pocos sets (~1.1). Modelos de alta complejidad. |
| **2** | **Minifiguras Icónicas / Ubicuas** | 644 (4.1%) | Piezas estándar (~4.5 partes) pero con altísima presencia transversal: aparecen en 6.1 sets en promedio (hasta 41). Son las figuras más populares y recurrentes. |

### Perfiles numéricos detallados por clúster:

| Clúster | `num_parts` (media) | `total_sets_appeared` (media) |
| :---: | ---: | ---: |
| 0 | 4.58 | 1.20 |
| 1 | 36.20 | 1.06 |
| 2 | 4.46 | 6.11 |

---

## 📁 9. Estructura del Repositorio

```text
.
├── app.py                     # Interfaz web interactiva en Streamlit (Integrante 2)
├── train.py                   # Script de entrenamiento, evaluación y exportación del modelo
├── model.joblib               # Artefacto serializado (modelo, features y perfiles)
├── requirements.txt           # Dependencias fijadas del proyecto (Integrante 2)
├── Dockerfile                 # Definición de la imagen del contenedor (Integrante 2)
├── .dockerignore              # Archivos excluidos del build de Docker (Integrante 2)
├── README.md                  # Documentación completa y guía de reproducción
├── PLAN_DE_IMPLEMENTACION.md  # Plan de roles, arquitectura y estrategia de Git
├── lego_minifigures_1975_2025.csv  # Dataset original de minifiguras
└── data/                      # Directorio de datos
    └── datos.csv              # Dataset de entrada (copia del CSV original)
```

---

## 🚀 10. Guía de Ejecución Paso a Paso

### Prerrequisitos
- Python 3.10, 3.11 o 3.12.
- Git instalado.
- Docker instalado (para despliegue en contenedor).

---

### A. Ejecución Local (Entrenamiento e Interfaz)

1. **Clonar el repositorio y entrar al proyecto:**
   ```bash
   git clone git@github.com:miguedmoreno-creator/Reto-completo-Machine-2.git
   cd Reto-completo-Machine-2
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   # En Linux / macOS
   python3 -m venv venv
   source venv/bin/activate

   # En Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Entrenar el modelo (`train.py`):**
   Asegúrese de que el archivo `data/datos.csv` esté en su lugar y ejecute:
   ```bash
   # Entrenamiento estándar con k=3
   python train.py

   # Con evaluación de k=2..7 (tabla comparativa de métricas)
   python train.py --evaluate-k

   # Con otro dataset y otro número de clusters
   python train.py --data ruta/a/otro.csv --k 4

   # Especificando features manualmente
   python train.py --features "edad,ingreso,frecuencia" --k 5
   ```
   *Salida esperada:*
   - Tabla de evaluación de k (si se usa `--evaluate-k`).
   - Métricas de evaluación: Silhouette Score, Inercia, Davies-Bouldin, Calinski-Harabasz.
   - Tabla de perfilamiento interpretativo de cada clúster.
   - Creación del archivo `model.joblib`.

5. **Iniciar la aplicación interactiva de Streamlit (`app.py`):**
   ```bash
   streamlit run app.py
   ```
   La aplicación se abrirá automáticamente en `http://localhost:8501`.

---

### B. Ejecución con Docker

1. **Construir la imagen de Docker:**
   ```bash
   docker build -t ml-clustering-app .
   ```

2. **Verificar que la imagen se haya creado:**
   ```bash
   docker images
   ```

3. **Ejecutar el contenedor:**
   ```bash
   docker run -d --name ml-clustering-container -p 8501:8501 ml-clustering-app
   ```

4. **Verificar estado y logs:**
   ```bash
   docker ps
   docker logs ml-clustering-container
   ```
   Abra en su navegador web: `http://localhost:8501`.

5. **Detener y remover el contenedor:**
   ```bash
   docker stop ml-clustering-container
   docker rm ml-clustering-container
   ```

---

### C. Despliegue en Google Cloud Platform (GCP) - Compute Engine

1. **Crear instancia de Máquina Virtual (VM):**
   - En la consola de GCP: `Compute Engine` > `Instancias de VM` > `Crear Instancia`.
   - Elegir serie pequeña y económica (ej. `e2-micro` o `e2-small` con Debian/Ubuntu Linux).
   - Asegurarse de asignar una **IP externa** efímera o estática.
   - Habilitar tráfico HTTP/HTTPS en la sección de cortafuegos.

2. **Configurar regla de Firewall en GCP para el puerto 8501:**
   - Ir a `Red de VPC` > `Reglas de firewall` > `Crear regla de firewall`.
   - **Nombre:** `allow-streamlit-8501`
   - **Dirección del tráfico:** Entrada (*Ingress*).
   - **Destino:** Todas las instancias de la red (o por etiqueta de red).
   - **Intervalos de IPv4 de origen:** `0.0.0.0/0`
   - **Protocolos y puertos:** TCP especificado en `8501`.

3. **Conectar a la VM vía SSH y configurar el entorno:**
   ```bash
   sudo apt update
   sudo apt install -y docker.io git
   sudo systemctl enable --now docker

   # Opcional: permitir ejecutar docker sin sudo
   sudo usermod -aG docker $USER
   ```
   *(Cerrar y reabrir sesión SSH si agregó el usuario al grupo docker).*

4. **Clonar el proyecto y lanzar el contenedor:**
   ```bash
   git clone https://github.com/miguedmoreno-creator/Reto-completo-Machine-2.git
   cd Reto-completo-Machine-2
   docker build -t ml-clustering-app .
   docker run -d --name ml-clustering -p 8501:8501 --restart unless-stopped ml-clustering-app
   ```

5. **Acceso al aplicativo:**
   Navegar desde cualquier navegador a:
   ```text
   http://<IP_EXTERNA_DE_LA_VM>:8501
   ```

---

## 💻 11. Ejemplo de Entrada y Salida

### Entrada de prueba en la interfaz:
| Parámetro | Valor de Entrada |
| :--- | :--- |
| `num_parts` (Número de piezas) | `4` |
| `total_sets_appeared` (Sets en que aparece) | `1` |

### Salida generada por el modelo:
- **Resultado asignado:** `Clúster 0`
- **Etiqueta descriptiva:** `Minifiguras Estándar / Sets Especiales`
- **Detalle de la asignación:**
  > *"Perfil Estándar y Específico: figuras convencionales (4.6 partes) distribuidas en 1 o 2 sets únicos (1.2 sets). Representa la gran mayoría del catálogo: 94.9% (14,984 figuras)."*

### Otros ejemplos de predicción:
| `num_parts` | `total_sets_appeared` | Clúster Asignado | Perfil |
| ---: | ---: | :---: | :--- |
| 4 | 1 | 0 | Minifiguras Estándar |
| 40 | 1 | 1 | Minifiguras Complejas |
| 4 | 20 | 2 | Minifiguras Icónicas |

---

## 🔄 12. Procedimiento para Adaptar a un Nuevo Dataset

Para utilizar este pipeline con cualquier otro dataset:
1. Coloque su archivo CSV en `data/datos.csv`.
2. Ejecute `python train.py --evaluate-k` para ver las métricas y elegir el $k$ adecuado.
3. Ejecute `python train.py --k <valor_elegido>` para regenerar `model.joblib`.
4. Ejecute `streamlit run app.py` (o reconstruya la imagen Docker): la interfaz tomará automáticamente los nuevos campos y rangos sin modificar el frontend.

> **Nota:** No es necesario modificar el código fuente de `train.py` ni de `app.py`. El pipeline detecta automáticamente las columnas numéricas y genera los controles de forma dinámica.

---

## 📦 13. Contrato del Artefacto (`model.joblib`)

El archivo `model.joblib` contiene un diccionario Python con la siguiente estructura estándar, que funciona como **contrato** entre el Integrante 1 (ML) y el Integrante 2 (Frontend):

```python
{
    "model": Pipeline,                # Pipeline completo de sklearn (imputer+scaler+kmeans)
    "features": List[str],            # Lista de variables usadas en el entrenamiento
    "numeric_features": List[str],    # Alias de features (compatibilidad)
    "categorical_features": List,     # Variables categóricas (vacía en este caso)
    "feature_stats": {                # Estadísticas por variable para generar sliders
        "num_parts": {"min": 2.0, "max": 143.0, "mean": 4.9, "median": 4.0, ...},
        "total_sets_appeared": {"min": 1.0, "max": 41.0, "mean": 1.4, "median": 1.0, ...}
    },
    "task": "clustering",
    "k": 3,
    "silhouette_score": 0.8287,
    "inertia": 13274.56,
    "davies_bouldin_score": 0.5725,
    "calinski_harabasz_score": 10885.03,
    "cluster_descriptions": {0: "...", 1: "...", 2: "..."},
    "cluster_profiles": {0: {...}, 1: {...}, 2: {...}},
    "dataset_name": "datos.csv",
    "total_samples": 15790
}
```
