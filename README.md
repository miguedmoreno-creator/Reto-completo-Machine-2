# Proyecto de Machine Learning: Ruta B - Aprendizaje No Supervisado (Clustering)

Aplicación interactiva y reproducible para segmentación y descubrimiento de patrones en datos no etiquetados mediante algoritmos de agrupamiento (*Clustering*), desarrollada con **scikit-learn**, desplegada con **Streamlit**, empaquetada con **Docker** y alojada en **Google Cloud Platform (GCP - Compute Engine)**.

> 📌 **Plan de Trabajo y Colaboración:** Consulta el documento detallado de división de roles y arquitectura agnóstica en [PLAN_DE_IMPLEMENTACION.md](file:///C:/Users/camor/.gemini/antigravity-ide/scratch/Reto-completo-Machine-2/PLAN_DE_IMPLEMENTACION.md).

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

- **Dataset utilizado:** `[Nombre del Dataset o Fuente, ej. Segmentación de Clientes / Métricas de Uso]`
- **Ruta de almacenamiento local:** `data/datos.csv`
- **Unidad de observación:** `[Definir qué representa exactamente una fila, ej. Un cliente único registrado / Una sesión de usuario / Una transacción]`
- **Tamaño aproximado:** `[N° de filas] filas × [N° de columnas] columnas`.
- **Formato esperado:** Archivo CSV separado por comas o punto y coma (`data/datos.csv`). Si el dataset contiene identificadores (IDs) o columnas no informativas, se excluyen durante el pipeline.

---

## 🔀 3. Ruta Elegida

- **Ruta B: Aprendizaje No Supervisado (Clustering)**
- **Justificación:** Los datos no cuentan con una variable objetivo (*target* o etiqueta predefinida). El objetivo primordial es el aprendizaje estructural para agrupar instancias con características similares e identificar comportamientos o arquetipos representativos.

---

## 🧩 4. Variables Utilizadas

El pipeline procesa dinámicamente las variables relevantes definidas para el análisis de agrupamiento:

| Nombre de la Variable | Tipo de Dato | Rol / Justificación | Rango / Valores Comunes |
| :--- | :--- | :--- | :--- |
| `[variable_1]` (ej. `edad` o `monto`) | Numérica Continua | Factor de escala de magnitud del individuo | `[min - max]` |
| `[variable_2]` (ej. `frecuencia`) | Numérica Discreta | Mide recurrencia o intensidad de actividad | `[min - max]` |
| `[variable_3]` (ej. `antiguedad`) | Numérica Continua | Mide fidelidad / tiempo de exposición | `[min - max]` |
| `[variable_k]` (ej. `categoria_preferida`) | Categórica (opcional) | Segmentación por preferencia o atributo cualitativo | `[Cat A, Cat B, ...]` |

*(Nota: En clustering se priorizan variables numéricas continuas/discretas normalizadas para evitar distorsiones en el cálculo de distancias euclidianas).*

---

## ⚙️ 5. Preparación de Datos y Pipeline

Para garantizar la reproducibilidad y prevenir la fuga de información (*data leakage*), el preprocesamiento se encapsula completamente dentro de un `Pipeline` de scikit-learn:

1. **Tratamiento de Valores Faltantes (Imputación):**
   - Variables numéricas: Imputación por mediana (`SimpleImputer(strategy='median')`) para mitigar el impacto de valores atípicos (*outliers*).
   - Variables categóricas (si aplica): Imputación por valor más frecuente o categoría constante (`'desconocido'`).
2. **Escalamiento y Normalización:**
   - Estandarización de variables numéricas con `StandardScaler` ($\mu=0, \sigma=1$) o `RobustScaler`. Esto es indispensable ya que algoritmos como K-Means basan su optimización en distancias geométricas (euclidianas).
3. **Codificación Categórica (Opcional):**
   - Variables categóricas procesadas mediante `OneHotEncoder(handle_unknown='ignore')` a través de `ColumnTransformer`.

---

## 🧠 6. Modelo de Agrupamiento y Justificación

- **Algoritmo seleccionado:** **K-Means** (o *Gaussian Mixture Models* / *DBSCAN* según el caso).
- **Justificación del modelo:**
  - K-Means ofrece alta eficiencia computacional $O(k \cdot n \cdot d)$, facilidad de convergencia y rápida inferencia en producción para asignar nuevas observaciones al centroide más cercano.
  - La interpretabilidad de los centroides permite generar perfiles de clúster directamente traducibles a reglas o descripciones de negocio.
- **Selección de hiperparámetros:**
  - Número de clusters ($k$): Determinado mediante el análisis del **Método del Codo (*Elbow Method*)** analizando la inercia y validado con el coeficiente de silueta.

---

## 📈 7. Criterios de Evaluación

En el aprendizaje no supervisado no existen matrices de confusión directas; la calidad de los clusters se evalúa combinando criterios intrínsecos de separación y cohesión:

1. **Coeficiente de Silueta (*Silhouette Score*):**
   - Mide qué tan similar es un objeto a su propio clúster en comparación con otros clústeres (rango $[-1, 1]$). Valores superiores a 0.35–0.50 indican separación estructural aceptable.
2. **Inercia (*Within-Cluster Sum of Squares* - WCSS):**
   - Suma de distancias cuadráticas de las muestras a su centroide más cercano.
3. **Índice de Davies-Bouldin / Calinski-Harabasz (opcional):**
   - Métricas complementarias de dispersión intra-clúster y separación inter-clúster.
4. **Coherencia e Interpretabilidad de Negocio:**
   - Validación empírica de que los perfiles resultantes presentan diferencias estadísticas significativas en sus medias/medianas.

---

## 🔍 8. Interpretación de la Salida y Perfilamiento

El entrenamiento genera perfiles estadísticos promedio de cada grupo. Al predecir sobre una nueva entrada, la aplicación no solo retorna un número de clúster, sino su **descripción cualitativa**:

| Clúster | Nombre del Perfil | Interpretación / Características Clave |
| :---: | :--- | :--- |
| **0** | **Perfil Base / Estándar** | Valores promedio en todas las variables analizadas. |
| **1** | **Perfil Alto Valor / Intensivo** | Valores altos en frecuencia e intensidad; grupo prioritario. |
| **2** | **Perfil Esporádico / Ocasional** | Baja recurrencia, montos moderados o registros recientes. |

---

## 📁 9. Estructura del Repositorio

```text
.
├── app.py                     # Interfaz web interactiva en Streamlit
├── train.py                   # Script de entrenamiento, evaluación y exportación del modelo
├── model.joblib               # Artefacto serializado (modelo, features y perfiles)
├── requirements.txt           # Dependencias fijadas del proyecto
├── Dockerfile                 # Definición de la imagen del contenedor
├── .dockerignore              # Archivos excluidos del build de Docker
├── README.md                  # Documentación completa y guía de reproducción
└── data/                      # Directorio de datos
    └── datos.csv              # Dataset de entrada
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
   python train.py
   ```
   *Salida esperada:*
   - Mensaje con la métrica de evaluación (ej. `Silhouette Score: 0.54`).
   - Tabla de perfilamiento de clusters impresa en consola.
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
| Variable 1 (`edad` / `magnitud_1`) | `34.0` |
| Variable 2 (`ingreso` / `magnitud_2`) | `4500.0` |
| Variable 3 (`frecuencia` / `magnitud_3`) | `12` |
| Variable 4 (`ticket_promedio` / `magnitud_4`) | `150.5` |

### Salida generada por el modelo:
- **Resultado asignado:** `Clúster 1`
- **Etiqueta descriptiva:** `Perfil Alto Valor / Intensivo`
- **Detalle de la asignación:**
  > *"Esta observación presenta valores superiores al percentil 75 en frecuencia y consumo, correspondiendo al segmento de clientes altamente fidelizados."*
- **Visualización:** Gráfico interactivo que sitúa la nueva observación respecto a los centroides del modelo.

---

## 🔄 12. Procedimiento para Adaptar a un Nuevo Dataset

Para utilizar este pipeline con cualquier otro dataset:
1. Coloque su archivo en `data/datos.csv`.
2. Abra `train.py` y ajuste la lista de columnas en `numeric_features` (y `categorical_features` si aplica).
3. Ajuste el número de clústeres `n_clusters` según la necesidad o gráfica del codo.
4. Ejecute `python train.py` para regenerar `model.joblib`.
5. Ejecute `streamlit run app.py` (o reconstruya la imagen Docker): la interfaz tomará automáticamente los nuevos campos y rangos sin modificar el frontend.
