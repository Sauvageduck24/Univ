# Resolución del problema de las N torres usando redes de Hopfield
### Esteban Sánchez Gámez

---

## 1. Modelado del espacio de estados

Para resolver el problema de colocar N torres en un tablero de ajedrez de tamaño \( N \times N \) sin que se ataquen entre sí, se modela cada celda del tablero como una neurona binaria de una red de Hopfield. 

Cada celda del tablero \( N \times N \) se representa con una variable binaria \( x_{ij} \in \{0,1\} \):

- \( x_{ij} = 1 \): indica que hay una torre en la posición (i, j)
- \( x_{ij} = 0 \): indica que la celda está vacía

El estado del sistema se representa mediante una matriz binaria de tamaño \( N \times N \), donde se busca una configuración con exactamente una torre por fila y por columna.

---

## 2. Función de energía

La red de Hopfield minimiza una función de energía, la cual se diseña para penalizar configuraciones inválidas en las que dos torres estén en la misma fila o columna.

\[
E = \frac{A}{2} \sum_{i=1}^{N} \left( \sum_{j=1}^{N} x_{ij} - 1 \right)^2 + \frac{B}{2} \sum_{j=1}^{N} \left( \sum_{i=1}^{N} x_{ij} - 1 \right)^2
\]

Donde:
- El primer término penaliza filas con más o menos de una torre.
- El segundo término penaliza columnas con más o menos de una torre.
- A y B son constantes que controlan el peso de cada restricción.

En la práctica, esta función se implementa a través de los pesos y umbrales de las neuronas.

---

## 3. Configuración de la red de Hopfield

Los pesos sinápticos de la red se configuran de forma que reflejen las restricciones del problema. Para dos neuronas \( x_{ij} \) y \( x_{kl} \):

- Si están en la misma fila (i = k, j ≠ l): \( w_{ij,kl} = -2\mu \)
- Si están en la misma columna (j = l, i ≠ k): \( w_{ij,kl} = -2\mu \)
- Si son la misma neurona: \( w_{ij,ij} = 0 \)
- En cualquier otro caso: \( w_{ij,kl} = 0 \)

Cada neurona tiene un umbral \( \theta_{ij} = -2\mu \). Esta configuración garantiza que las neuronas "compiten" por activarse dentro de una misma fila y columna, cumpliendo así las restricciones del problema.

---

## 4. Dinámica de la red

La evolución de la red se basa en una dinámica asincrónica donde se actualiza una neurona a la vez en función del estado actual de sus vecinas.

\[
x_{ij}^{(t+1)} =
\begin{cases}
1 & \text{si } \sum w_{ij,kl} x_{kl}^{(t)} > \theta_{ij} \\
0 & \text{si no}
\end{cases}
\]

Esta actualización se repite iterativamente hasta que la red alcanza un estado estable (mínimo local de energía), que representa una solución al problema.

---

## 5. Convergencia y solución

Cuando el sistema alcanza un estado donde la energía ya no disminuye, se considera que ha llegado a un equilibrio. Si el modelo está bien parametrizado, este estado corresponde a una solución válida donde no hay conflictos entre las torres, es decir, hay exactamente una torre por fila y por columna.
