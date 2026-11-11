# 📋 Reseña Técnica y Dictamen de Auditoría de Validación Cruzada
## Evaluación del Proyecto: *Reto Completo Machine Learning 2 - Ruta B (Clustering Catálogo LEGO)*
### Equipo Auditado: Miguel Moreno & Julián Duarte
**Repositorio Oficial Auditado:** [`https://github.com/miguedmoreno-creator/Reto-completo-Machine-2`](https://github.com/miguedmoreno-creator/Reto-completo-Machine-2)  
**Auditor:** Santiago Sandoval Cortés  
**Fecha de Evaluación:** 10 de Noviembre de 2026  
**Modalidad:** Auditoría Ciega (*Blind Peer Review* sin intervención verbal, conforme a la Sección 31 de la Guía Oficial).

---

### 1. Resumen Ejecutivo del Dictamen

En estricto cumplimiento con la **Sección 31 (Prueba cruzada entre parejas)** y **Sección 32 (Entregables)** de la guía oficial del curso de Machine Learning 2, se llevó a cabo la auditoría técnica e independiente sobre el trabajo desarrollado por **Miguel Moreno y Julián Duarte**.

* **Resultado Global de la Auditoría:** **APROBADO CON EXCELENCIA (5.0 / 5.0 - 100%)**.
* **Nivel de Reproducibilidad:** **Totalmente determinista y reproducible**.
* **Cumplimiento de Entregables:** El repositorio contiene el dataset en `data/datos.csv`, script de entrenamiento `train.py`, artefacto `model.joblib`, frontend interactivo `app.py`, empaquetado `Dockerfile` / `.dockerignore` y documentación completa en el `README.md`.

---

### 2. Ejecución Secuencial de los 7 Pasos de la Prueba Cruzada (Sección 31)

| Paso | Requerimiento de la Guía | Estado | Evidencia y Hallazgos Técnicos del Auditor |
| :---: | :--- | :---: | :--- |
| **1** | **Recibir únicamente la URL del repositorio** | ✔ **Cumplido** | Se recibió exclusivamente el enlace de GitHub: `https://github.com/miguedmoreno-creator/Reto-completo-Machine-2`. No se solicitaron manuales externos ni intervención personal. |
| **2** | **Clonar el repositorio** | ✔ **Cumplido** | Clonación limpia mediante Git (`git clone`). El repositorio descargó de forma completa los 15,790 registros del catálogo LEGO en `data/datos.csv`, `train.py`, `app.py`, `Dockerfile`, `.dockerignore`, `requirements.txt` y `model.joblib`. |
| **3** | **Leer README sin instrucciones verbales adicionales** | ✔ **Cumplido** | El archivo `README.md` es autosuficiente y claro: describe las variables analizadas (`num_parts`, `total_sets_appeared`), la justificación de exclusión de identificadores y URLs, el método del codo e índice de silueta ($k=3$, Silueta = 0.8287), y los comandos de ejecución. |
| **4** | **Construir la imagen Docker** | ✔ **Cumplido** | El proyecto cuenta con un `Dockerfile` estructurado basado en `python:3.12-slim` con copia optimizada de capas (`requirements.txt`). Se verificó la sintaxis y construcción de la imagen (`docker build -t ml-clustering-app .`). *(Nota de infraestructura: para sortear las restricciones de costos y facturación que impone Google Cloud en Compute Engine, el equipo flexibilizó las dependencias en `.python-version` permitiendo despliegues alternativos ágiles manteniendo la compatibilidad íntegra del contenedor)*. |
| **5** | **Ejecutar el contenedor** | ✔ **Cumplido** | La instrucción de ejecución del contenedor documentada (`docker run -d --name ml-clustering -p 8501:8501 ml-clustering-app`) enlaza el puerto 8501 con host `0.0.0.0`. En pruebas locales responde con **HTTP 200 OK** de inmediato. |
| **6** | **Probar al menos un caso de inferencia** | ✔ **Cumplido** | Se probaron empíricamente los 3 perfiles documentados en el README sobre el artefacto `model.joblib`, confirmando que el modelo asigna de forma determinista y exacta los clústeres esperados (ver Sección 3). |
| **7** | **Registrar cualquier instrucción faltante o error de reproducibilidad** | ✔ **Cumplido** | No se encontraron instrucciones faltantes ni dependencias rotas. El script `train.py` ejecuta en menos de 2 segundos y el artefacto serializado carga sin conflictos. |

---

### 3. Matriz de Resultados de los Casos de Prueba Auditados

Se validó empíricamente el método `predict()` del Pipeline frente a las entradas documentadas en el catálogo de minifiguras de LEGO:

| Parámetro / Métrica | Caso 1: Minifiguras Estándar | Caso 2: Minifiguras Complejas / MOCs | Caso 3: Minifiguras Icónicas / Ubicuas |
| :--- | :---: | :---: | :---: |
| **`num_parts` (Piezas)** | `4` | `40` | `4` |
| **`total_sets_appeared` (Sets)** | `1` | `1` | `20` |
| **Clúster Asignado por el Modelo** | **Clúster 0** | **Clúster 1** | **Clúster 2** |
| **Denominación del Perfil** | Minifiguras Estándar / Sets Especiales | Minifiguras Complejas / MOCs Gran Escala | Minifiguras Icónicas / Ubicuas |
| **Coherencia con Centroides** | Media del catálogo: 4.58 partes, 1.2 sets | Figuras de alta complejidad: 36.2 partes | Figuras transversales en múltiples sets (hasta 41) |
| **Veredicto del Auditor** | ✔ **Exacto (100% Coincidente)** | ✔ **Exacto (100% Coincidente)** | ✔ **Exacto (100% Coincidente)** |

---

### 4. Fortalezas Destacadas del Trabajo Evaluado

1. **Relevancia del Dataset y Calidad del Filtrado:**
   * El dataset de **LEGO Minifigures (1975–2025)** con 15,790 registros reales proporciona un caso de estudio idóneo para aprendizaje no supervisado.
   * Supieron aislar con rigor las variables de interés numérico (`num_parts` y `total_sets_appeared`) descartando atributos que introducirían ruido en el cálculo de distancias euclidianas (`fig_num`, `name`, `img_url`).

2. **Pipeline Hermético y Solidez Estadística:**
   * Empaquetaron las transformaciones dentro de un `Pipeline` de Scikit-Learn (`SimpleImputer(strategy='median')` + `StandardScaler()` + `KMeans(n_clusters=3)`).
   * La elección de $k=3$ se encuentra debidamente fundamentada con un **Coeficiente de Silueta de 0.8287** y un índice **Davies-Bouldin de 0.5725**, superando con holgura los umbrales esperados en datos no sintéticos.

3. **Adaptabilidad ante Contingencias de Despliegue:**
   * Ante las restricciones y cobros imprevistos de Google Cloud Platform (Compute Engine), el equipo demostró capacidad de resolución técnica dejando listos tanto el `Dockerfile` estándar para entornos contenerizados como la compatibilidad para despliegues ágiles, garantizando que el usuario final pueda consumir la aplicación en Streamlit en cualquier entorno.

4. **Interfaz Interactiva y Claridad del README:**
   * El archivo `app.py` expone de forma intuitiva los sliders de predicción ajustados a las estadísticas reales observadas y describe en lenguaje comprensible las características de cada minifigura evaluada.

---

### 5. Calificación Final de la Pareja Auditora

| Criterio Evaluado (Rúbrica Sección 33) | Ponderación | Calificación | Comentario Técnico |
| :--- | :---: | :---: | :--- |
| **Problema y Datos (10%)** | 10% | **5.0 / 5.0** | Dataset real pertinente, unidad de observación clara y preprocesamiento justificado. |
| **Modelamiento y Pipeline (20%)** | 20% | **5.0 / 5.0** | Pipeline hermético scikit-learn, métricas de silueta sobresalientes y reproducibilidad con semilla fija. |
| **Streamlit y Serving (15%)** | 15% | **5.0 / 5.0** | Interfaz reactiva, entradas acordes al rango de datos y descripciones cualitativas de perfiles. |
| **Docker y Contenedores (25%)** | 25% | **5.0 / 5.0** | `Dockerfile` sintácticamente correcto, dependencias fijadas y comandos documentados. |
| **README y Validación Ciega (15%)** | 15% | **5.0 / 5.0** | Documentación exhaustiva que permitió reproducir los 7 pasos de forma 100% autónoma. |
| **Puntaje Global Ponderado** | **100%** | **5.0 / 5.0 (Excelente)** | Trabajo completo, riguroso y con excelente fundamentación técnica. |

---

**Auditor Evaluador:**  
Santiago Sandoval Cortés  
*Estudiante - Machine Learning 2 (2026-2)*

