# Proyectos de Universidad – Grado en Ciberseguridad e Inteligencia Artificial

Este repositorio recopila prácticas, proyectos y ejemplos desarrollados durante la carrera de **Ciberseguridad e Inteligencia Artificial**. Todo el código está implementado en **Python** y abarca áreas como optimización, aprendizaje automático, programación entera, redes neuronales, inteligencia artificial clásica y moderna, y más.

---

## Estructura del repositorio

- `p1/` – **Optimización sin restricciones (ADALINA)**
- `p2/` – **Optimización con restricciones y regularización L1 (ADALINA Lasso)**
- `p3/` – **Programación entera: Problema del Viajante de Comercio (TSP)**
- `p4/` – **Programación entera: Rejilla mágica**
- `p5/` – **Redes de Hopfield: Problema de las N torres**
- `p6/` – **Optimización evolutiva: Carteras de inversión con CMA-ES**
- `ia/` – **Ejemplos de Inteligencia Artificial supervisada y no supervisada (PyTorch y scikit-learn)**
- `modelos_lineales.ipynb` – Ejemplo de modelos lineales y logísticos en Python

---

## Resumen de proyectos y ejemplos

### `p1/` – Optimización sin restricciones (ADALINA)
Implementación de dos variantes del algoritmo ADALINA para regresión: descenso de gradiente y método de Newton. Se comparan ambos métodos en términos de eficiencia y convergencia para minimizar el error cuadrático medio.

### `p2/` – Optimización con restricciones y regularización L1
Implementación de ADALINA usando gradiente proyectado y regularización L1 (Lasso). Se utiliza backtracking line search y se analiza el impacto de la regularización en la predicción de precios de viviendas.

### `p3/` – Programación entera: Problema del Viajante de Comercio (TSP)
Resolución del TSP usando programación lineal entera binaria con la librería `pulp` y el método de Miller-Tucker-Zemlin para evitar subtours. Incluye visualización de la solución.

### `p4/` – Programación entera: Rejilla mágica
Modelado y resolución de una rejilla mágica con restricciones sobre subrejillas usando programación entera. Se busca una asignación de números enteros que cumpla condiciones de suma en submatrices.

### `p5/` – Redes de Hopfield: Problema de las N torres
Aplicación de redes neuronales de Hopfield para resolver el problema de colocar N torres en un tablero de ajedrez sin que se ataquen. Se modela el problema como un sistema dinámico que minimiza una función de energía.

### `p6/` – Optimización evolutiva: Carteras de inversión con CMA-ES
Optimización de carteras bajo restricciones de presupuesto y cardinalidad usando el algoritmo evolutivo CMA-ES. Se exploran estrategias de proyección y penalización para cumplir restricciones y se analizan los resultados obtenidos.

### `modelos_lineales.ipynb`
Ejemplo de implementación de modelos de regresión lineal y logística desde cero en Python, útil para entender los fundamentos del aprendizaje supervisado.

---

## Ejemplos de Inteligencia Artificial (`ia/`)

La carpeta `ia/` contiene ejemplos clásicos y didácticos de aprendizaje supervisado y no supervisado, implementados con **PyTorch** y **scikit-learn**. Estos scripts son ideales para demostrar conocimientos prácticos de IA y pueden servir como base para proyectos más avanzados.

### Contenido de `ia/`

- **Clasificación supervisada con PyTorch** (`supervised_classification_pytorch.py`):
  - Implementa una red neuronal simple (MLP) para clasificación binaria sobre datos sintéticos generados con `sklearn.datasets.make_classification`.
  - Incluye preprocesamiento, entrenamiento, evaluación y cálculo de precisión sobre el conjunto de test.
  - Demuestra el flujo típico de un problema supervisado: generación de datos, partición, normalización, definición de modelo, entrenamiento y evaluación.

- **Autoencoder no supervisado con PyTorch** (`unsupervised_autoencoder_pytorch.py`):
  - Implementa un autoencoder simple para reducción de dimensionalidad sobre datos sintéticos de clusters (`sklearn.datasets.make_blobs`).
  - El modelo aprende una representación latente de 2 dimensiones a partir de datos de 10 dimensiones.
  - Se muestra la reconstrucción y las primeras representaciones latentes, útil para visualización y compresión de datos.

- **Clustering no supervisado con KMeans** (`unsupervised_clustering_kmeans.py`):
  - Ejemplo clásico de clustering usando KMeans sobre datos sintéticos 2D.
  - Incluye visualización de los clusters y los centros encontrados por el algoritmo.
  - Es un ejemplo fundamental de aprendizaje no supervisado y análisis exploratorio de datos.

#### Ejemplo de uso de cada script

```bash
# Clasificación supervisada con PyTorch
python ia/supervised_classification_pytorch.py

# Autoencoder no supervisado con PyTorch
python ia/unsupervised_autoencoder_pytorch.py

# Clustering KMeans no supervisado
python ia/unsupervised_clustering_kmeans.py
```

Todos los scripts son autocontenibles y generan resultados por consola o gráficos.

#### Requisitos para la carpeta `ia/`

- Python 3.8+
- `torch`, `numpy`, `scikit-learn`, `matplotlib`

Instala las dependencias con:
```bash
pip install torch numpy scikit-learn matplotlib
```

---

## Requisitos generales del repositorio

- Python 3.8+
- Bibliotecas recomendadas:
  - `numpy`, `pandas`, `matplotlib`
  - `scikit-learn` (algunos proyectos)
  - `pulp` (programación lineal/entera)
  - `cma` (optimización evolutiva)
  - `torch` (para ejemplos de IA)

Instala todas las dependencias recomendadas con:
```bash
pip install numpy pandas matplotlib scikit-learn pulp cma torch
```

---

## ¿Cómo ejecutar los proyectos?
Cada carpeta contiene scripts y un informe explicativo (`informe.md` o `.pdf`). Para ejecutar un proyecto, navega a la carpeta correspondiente y ejecuta el script principal, por ejemplo:

```bash
cd p6
python main.py
```

Algunos proyectos requieren archivos de datos incluidos en la carpeta correspondiente.

---

## Sugerencias de mejora y ampliación
- Añadir notebooks interactivos para visualización y experimentación.
- Incluir tests automáticos para cada módulo.
- Mejorar la documentación de cada script.
- Añadir ejemplos de ciberseguridad aplicada (análisis de datos, detección de anomalías, etc).
- Ampliar los ejemplos de IA con redes convolucionales, regresión, clustering jerárquico, reducción de dimensionalidad avanzada, etc.
- Integrar pipelines de machine learning y ejemplos de despliegue de modelos.

---

## Autor
**Esteban Sánchez Gámez**

---

¡Este repositorio es una muestra de habilidades en matemáticas aplicadas, optimización, inteligencia artificial, machine learning y programación en Python! Si tienes sugerencias o quieres colaborar, no dudes en abrir un issue o pull request. 