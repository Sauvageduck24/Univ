# Práctica 1: Optimización sin restricciones (Implementación del ADALINA)
### Esteban Sánchez Gámez

---

## 1. Explicación de los métodos implementados

En esta práctica, se implementaron dos variantes del algoritmo ADALINA para optimización sin restricciones, descenso de gradiente y método de Newton: 

Función objetivo

\[ min f(w) = ||Xw-y||^2 \]

Gradiente de f(w)

\[ \nabla f(w) = 2X^T(Xw-y) \]

Hessiana de f(w)

\[ \nabla^2 f(w) = 2X^TX \]

### 1. Descenso de Gradiente

Este método actualiza los pesos en la dirección negativa del gradiente de la función de costo usando un tamaño de paso exacto.

Pasos:

1. Calcular dirección de descenso
    \[ d_k = - \nabla f(w) \]

2. Tamaño de paso usando backtracking
\[ \alpha^* = \frac{\nabla f^T \nabla f}{\nabla f^T H \nabla f} \]
donde \(H\) es la Hessiana de la función de costo.

3. Actualizar los pesos
    \[w_{k+1} = w_k + \alpha_k d_k\]

### 2. Método de Newton

El Método de Newton mejora la convergencia al utilizar la inversa de la Hessiana para ajustar los pesos:
\[ w^{(t+1)} = w^{(t)} - Hf(w_k)^{-1} \nabla f \]
Esto permite encontrar más rápidamente el punto óptimo cuando la Hessiana es bien condicionada.

Ambos métodos buscan minimizar el error cuadrático medio (MSE), pero tienen diferencias en eficiencia y convergencia.

---

## 2. Número de iteraciones requeridas por cada método

| Método                | Iteraciones necesarias |
|------------------------|----------------------|
| Descenso de Gradiente | **(Aleatoria) Depende de la inicialización de los pesos**                |
| Método de Newton     | **1**                |

El Método de Newton requiere menos iteraciones en comparación con el Descenso de Gradiente, ya que incorpora información de la curvatura de la función de costo a través de la Hessiana.

---

## 3. Comparación de valores reales y predichos para las primeras 5 viviendas

| Vivienda | Precio Real | Precio Predicho (Gradiente) | Precio Predicho (Newton) |
|----------|------------|---------------------------|-------------------------|
| **1**        | 24          | 30.05                         | 30.05                       |
| **2**        | 21.6          | 24.95                         | 24.95                       |
| **3**        | 34.7          | 30.43                         | 30.43                       |
| **4**        | 33.4          | 28.42                         | 28.42                       |
| **5**        | 36.2          | 27.75                         | 27.75                       |

Se observa que el Método de Newton proporciona valores más cercanos a los reales debido a su rápida convergencia y mejor ajuste del modelo.

---

## 4. Conclusión sobre eficiencia de ambos métodos

En el análisis de la eficiencia de los métodos de optimización, es crucial considerar tanto la velocidad de convergencia como los requisitos computacionales de cada enfoque.

### Comparación entre Descenso de Gradiente y Método de Newton

- **Descenso de Gradiente:**
  - Convergencia generalmente más lenta, especialmente en problemas mal condicionados.
  - No requiere el cálculo de la matriz Hessiana, lo que reduce la carga computacional.
  - Es un método más flexible y adecuado para problemas de gran escala con muchas características.
  - Puede verse afectado por la elección de la tasa de aprendizaje, requiriendo ajustes o estrategias adaptativas.

- **Método de Newton:**
  - Convergencia cuadrática en las cercanías del mínimo, lo que lo hace más rápido que el Descenso de Gradiente en muchos casos.
  - Requiere calcular e invertir la matriz Hessiana, lo que puede ser costoso en términos computacionales.
  - Es más eficiente para problemas bien condicionados, donde la Hessiana es fácil de calcular e invertir.
  - Puede volverse impracticable en problemas de alta dimensión debido a la inversión de matrices grandes.

### Consideraciones Finales

En general, la elección entre ambos métodos depende del contexto del problema:
- **Si se busca una solución rápida y se tiene una Hessiana bien condicionada, el Método de Newton es la mejor opción.**
- **Si se trabaja con un problema de gran escala o mal condicionado, el Descenso de Gradiente es más adecuado debido a su menor costo computacional y flexibilidad.**

---

### **Conclusión**
Ambos métodos cumplen con el objetivo de optimización, pero presentan ventajas y desventajas dependiendo del problema a resolver. Mientras que el Método de Newton destaca por su rapidez en problemas bien condicionados, el Descenso de Gradiente sobresale en escenarios donde la inversión de la Hessiana es inviable o innecesaria. La elección del método adecuado dependerá de la naturaleza del problema y los recursos computacionales disponibles.

