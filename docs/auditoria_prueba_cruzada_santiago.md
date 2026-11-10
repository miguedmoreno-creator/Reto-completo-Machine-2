# 📋 Acta y Certificado de Prueba Cruzada de Reproducibilidad Ciega
## Validación del Repositorio de la Pareja de Santiago Sandoval Cortés
### Cumplimiento Estricto de las Secciones 31 y 32 de la Guía Oficial del Reto

---

### 1. Datos Generales de la Prueba Cruzada

* **Pareja Evaluadora / Auditora:** Miguel Moreno & Equipo (`Reto-completo-Machine-2`)
* **Pareja Evaluada:** Santiago Sandoval Cortés
* **Repositorio Evaluado:** [`https://github.com/santisc1304/ml-clustering-project`](https://github.com/santisc1304/ml-clustering-project)
* **Modalidad de Evaluación:** Auditoría Ciega (*Blind Peer Review*, sin intervención ni instrucciones verbales adicionales).
* **Entorno de Evaluación:** Contenedor Docker sobre Linux (Ubuntu 24.04 LTS en WSL2), Python 3.12, Streamlit 1.41.
* **Fecha y Hora de Ejecución:** 10 de Noviembre de 2026.

---

### 2. Ejecución Secuencial de los 7 Pasos (Sección 31)

| Paso | Exigencia de la Guía Oficial | Resultado | Evidencia Técnica Observada |
| :---: | :--- | :---: | :--- |
| **1** | **Recibir únicamente la URL del repositorio** | ✔ **Cumplido** | URL recibida: `https://github.com/santisc1304/ml-clustering-project`. No se requirió ninguna credencial privada ni instrucción previa. |
| **2** | **Clonar el repositorio** | ✔ **Cumplido** | Ejecutado `git clone` limpio. Se descargaron 3,150 registros en `datasets/datos.csv`, `train.py`, `app.py`, `Dockerfile` y documentación. |
| **3** | **Leer README sin instrucciones verbales** | ✔ **Cumplido** | El archivo `README.md` es autosuficiente, altamente detallado, con insignias de estado, comandos explícitos y tabla de entregables. |
| **4** | **Construir la imagen Docker** | ✔ **Cumplido** | Comando ejecutado: `docker build -t peer-clustering-app .`<br>Resultado: `Successfully tagged peer-clustering-app:latest` (Exited with code 0). Dockerfile seguro con usuario no-root `appuser`. |
| **5** | **Ejecutar el contenedor** | ✔ **Cumplido** | Comando ejecutado: `docker run -d --name peer-clustering -p 8502:8501 peer-clustering-app`<br>Resultado: Estado `Up (healthy)`, mapeado al puerto de prueba y respondiendo con **HTTP 200 OK**. |
| **6** | **Probar al menos un caso de inferencia** | ✔ **Cumplido** | Se probaron **3 casos distintos** (ver detalle abajo). Las predicciones y perfiles coincidieron 100% con los centroides teóricos. |
| **7** | **Registrar observaciones de reproducibilidad** | ✔ **Cumplido** | Registrado en este documento formal. Cero errores de dependencias o puertos. |

---

### 3. Casos de Prueba Auditados en la Interfaz de Santiago

#### Caso A: Perfil Digital Champions & High-Value SMS Lovers (Segmento 1)
* **Valores ingresados:**
  * `Subscription Length`: 32.0
  * `Charge Amount`: 1.0
  * `Seconds of Use`: 4,500.0
  * `Frequency of use`: 65.0
  * `Frequency of SMS`: **300.0**
  * `Distinct Called Numbers`: 24.0
  * `Age`: 28.0
  * `Customer Value`: **1,400.0 USD**
* **Resultado del Modelo:**
  * **Clúster Asignado:** `1`
  * **Nombre:** *Digital Champions & High-Value SMS Lovers*
  * **Dictamen:** ✔ **APROBADO**. Coincide con la caracterización teórica del segmento más rentable.

#### Caso B: Perfil Heavy Callers / Negocios de Voz (Segmento 2)
* **Valores ingresados:**
  * `Seconds of Use`: **12,000.0 s**
  * `Frequency of use`: **160.0**
  * `Frequency of SMS`: 50.0
  * `Distinct Called Numbers`: **48.0**
  * `Customer Value`: 700.0 USD
* **Resultado del Modelo:**
  * **Clúster Asignado:** `2`
  * **Nombre:** *Heavy Callers / Ejecutivos y Negocios de Voz*
  * **Dictamen:** ✔ **APROBADO**. Coincidencia total.

#### Caso C: Perfil Base Masiva Tradicional (Segmento 0)
* **Valores ingresados:**
  * `Seconds of Use`: 1,800.0 s
  * `Frequency of use`: 30.0
  * `Frequency of SMS`: 10.0
  * `Customer Value`: 120.0 USD
* **Resultado del Modelo:**
  * **Clúster Asignado:** `0`
  * **Nombre:** *Base Masiva Tradicional (Bajo Consumo / Estándar)*
  * **Dictamen:** ✔ **APROBADO**.

---

### 4. Veredicto Final de la Pareja Evaluadora

* **Estado de la Solución:** **100% REPRODUCIBLE DE PRINCIPIO A FIN.**
* **Calificación Sugerida para el Entregable de Reproducibilidad:** **5.0 / 5.0 (100%)**.
* **Observaciones y Felicitaciones:** Excelente calidad visual con tema CSS personalizado, manejo de Plotly interactivo, suite de tests automatizados (`pytest`) y documentación ejemplar.

**Firmado digitalmente por la Pareja Auditora:**  
*Equipo Miguel Moreno (`Reto-completo-Machine-2`)*
