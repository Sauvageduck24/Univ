# Práctica 2: Optimización con restricciones (Implementación del ADALINA)
### Esteban Sánchez Gámez

---

## 1. Explicación del método de Gradiente Proyectado y su aplicación a la regularización ℓ1

En esta práctica, se implementó el algoritmo ADALINA utilizando el método de gradiente proyectado para la optimización, además de la restricción ℓ1 (valor absoluto). Se usó una formulación basada en la minimización de la función de costo mediante el gradiente y el uso de un backtracking line search para la selección del tamaño del paso. A continuación, se describen los componentes principales:

- **Función de costo**: Se usa el error cuadrático medio (MSE) con una penalización \( λ \) para regularizar la solución.
- **Cálculo del gradiente**: Se obtiene derivando la función de costo con respecto a los pesos \( w^+ \) y \( w^- \), si no hacemos esto no sería posible ya que no se puede derivar el valor absoluto, por eso hacemos el cambio de variable.
- **Backtracking line search**: Se implementa un método de búsqueda de paso para optimizar la actualización de los pesos.
- **Condición de parada**: El proceso de optimización termina cuando el cambio de gradiente es suficientemente pequeño o cuando se alcanza el número máximo de iteraciones.

## 2. Número de iteraciones requeridas con \( λ = 100 \)

Para \( λ = 100 \),  el modelo converge en **50** iteraciones. Esto se determinó evaluando la norma del gradiente y verificando la condición de convergencia.

## 3. Comparación de valores reales y predichos para las primeras 5 viviendas

Los valores reales de las primeras 5 viviendas y sus predicciones obtenidas por el modelo ADALINA son:

| Vivienda | Precio Real | Precio Predicho |
|----------|------------|----------------|
| **1**        | 24          | 28.19              |
| **2**        | 21.6          | 24.82              |
| **3**        | 34.7          | 29.59              |
| **4**        | 33.4          | 29.13              |
| **5**        | 36.2          | 28.52              |

## 4. Análisis del efecto de diferentes valores de \( λ \) en el modelo

El parámetro \( λ \) controla la regularización del modelo y tiene un impacto significativo en el ajuste de los pesos. Se observaron los siguientes efectos al variar \( λ \):

- **Valores altos de \( λ \)**: Mayor regularización, lo que reduce el sobreajuste pero puede llevar a una subestimación de los valores predichos.
- **Valores bajos de \( λ \)**: Menor regularización, permitiendo que el modelo se ajuste mejor a los datos de entrenamiento, pero con mayor riesgo de sobreajuste.

A partir de los resultados obtenidos, se recomienda ajustar \( λ \) en un rango adecuado para equilibrar la precisión del modelo y su capacidad de generalización.

| Valores de λ | MSE (Predicciones) | Pasos para convergencia |
|----------|------------|----------------|
| **200**        | 0.4811          | 20              |
| **150**        | 0.3706          | 44              |
| **100**        | 0.3107          | 50              |
| **50**        | 0.2782          | 23              |
| **10**        | 0.3624          | 9              |

---

**Conclusión:** Se implementó con éxito el modelo ADALINA utilizando optimización con restricciones. El algoritmo logró ajustar los pesos de manera eficiente, y los resultados muestran una buena aproximación entre los valores predichos y los reales, dependiendo del valor de \( λ \) utilizado.