# **Informe – Práctica 6: CMA-ES y Gestión de Restricciones**

### *Esteban Sánchez Gámez*

---

## **1. Introducción Teórica**

La construcción de carteras óptimas bajo el modelo media-varianza de Markowitz plantea desafíos cuando se imponen restricciones adicionales como:

* **Presupuesto**: la suma de los pesos debe ser 1.
* **Cardinalidad**: solo se permite invertir en un subconjunto de $K$ activos.

Estas restricciones convierten el problema en uno **no convexo** y **NP-difícil**. Para abordarlo, se utilizó **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**, un algoritmo evolutivo especialmente efectivo para optimización en espacios continuos con restricciones no lineales.

---

## **2. Conjunto de Datos y Preprocesado**

Se emplearon tres fuentes de datos:

* `media_rentabilidades_train.csv`: rentabilidades medias históricas.
* `covarianza_rentabilidades_train.csv`: matriz de covarianzas entre activos.
* `precios_test.csv`: precios diarios de validación (enero-febrero 2025).

A partir de los precios, se calcularon **rentabilidades diarias** mediante diferencias porcentuales. El universo de inversión contiene $N$ activos, con $K = 10$ activos permitidos en cada cartera.

---

## **3. Parte A – Proyección Externa**

Se implementó una estrategia basada en **proyección externa** para garantizar las restricciones. El procedimiento consiste en:

1. Eliminar pesos negativos.
2. Seleccionar los $K$ pesos más altos.
3. Normalizar para que sumen 1.

Esta proyección se aplica a cada solución generada por CMA-ES. Además, se incorporó un **término de diversificación** para evitar concentraciones excesivas:

$$
\text{Función objetivo} = -\mu^T w + \frac{1}{2} w^T \Sigma w - \lambda \cdot \text{diversidad}(w)
$$

donde la diversidad se mide como la fracción de activos con peso positivo.

---

## **4. Parte B – Penalización**

Como alternativa, se formuló una función objetivo penalizada para integrar las restricciones directamente:

$$
\text{objetivo}(w) = -\mu^T w + \frac{1}{2} w^T \Sigma w + \beta \cdot \max(0, \|w\|_0 - K) + \gamma \cdot |\sum w - 1|
$$

donde:

* $\|w\|_0$ es el número de activos con peso distinto de cero (cardinalidad).
* $\sum w = 1$ es la restricción de presupuesto.
* $\beta$ penaliza violaciones a la cardinalidad.
* $\gamma$ penaliza desbalances presupuestarios.

Se evaluaron múltiples combinaciones de $\beta \in \{0.1, 1.0, 10.0\}$ y $\gamma \in \{0.0, 10.0\}$. Al finalizar, se proyectó la solución para asegurar cardinalidad exacta.

---

## **5. Resultados Experimentales**

A continuación, se resumen los resultados obtenidos en la validación:

| Método       | β    | γ    | Rent. Media (%) | Volatilidad | Sharpe |
| ------------ | ---- | ---- | --------------- | ----------- | ------ |
| Proyección   | –    | –    | -0.0551         | 0.0128      | -0.043 |
| Penalización | 0.1  | 0.0  | 0.0229          | 0.0028      | 0.0827 |
| Penalización | 0.1  | 10.0 | 0.0255          | 0.0028      | 0.0899 |
| Penalización | 1.0  | 0.0  | 0.0229          | 0.0028      | 0.0827 |
| Penalización | 1.0  | 10.0 | 0.0255          | 0.0028      | 0.0899 |
| Penalización | 10.0 | 0.0  | 0.0229          | 0.0028      | 0.0827 |
| Penalización | 10.0 | 10.0 | 0.0255          | 0.0028      | 0.0899 |

* La proyección externa mostró un rendimiento estable pero negativo.
* Las carteras penalizadas lograron rendimientos positivos con baja volatilidad, especialmente para $\gamma = 10$, destacando la importancia de respetar el presupuesto.

---

## **6. Conclusiones**

* La **proyección externa** es una estrategia simple y robusta, aunque puede subaprovechar el potencial de la optimización.
* La **formulación penalizada** permite una mayor adaptabilidad, aunque exige calibrar cuidadosamente los parámetros $\beta$ y $\gamma$.
* En este experimento, las carteras penalizadas superaron sistemáticamente a la proyectada, tanto en rendimiento como en ratio de Sharpe.
* El uso de **CMA-ES** demostró ser eficaz para este tipo de problemas complejos con restricciones no convexas.