# Informe: Rejilla Mágica usando Programación Entera con PuLP

### Esteban Sánchez Gámez

## Objetivo

Dado un tamaño de rejilla \( N \times N \), encontrar una asignación de números enteros no negativos en cada celda de la rejilla, tal que:

- La suma de cada subrejilla de tamaño \( M \times L \) sea exactamente \( K \)
- La suma de cada subrejilla de tamaño \( L \times M \) sea también \( K \)

---

## Entrada de datos

El programa recibe como entrada:

- \( N \): tamaño de la rejilla \( N \times N \)
- \( M, L \): dimensiones de las subrejillas
- \( K \): suma objetivo para cada subrejilla

---

## Modelado Matemático

Definimos una variable entera no negativa para cada celda de la matriz:

\[
x_{i,j} \in \mathbb{Z}_{\geq 0}, \quad \text{para } 0 \leq i,j < N
\]

Estas variables representan el valor de la celda \( (i,j) \) de la rejilla.

### Restricciones

#### 1. Subrejillas de tamaño \( M \times L \)

Para cada posición válida de la esquina superior izquierda \( (i, j) \) de una subrejilla \( M \times L \):

\[
\sum_{a=0}^{M-1} \sum_{b=0}^{L-1} x_{i+a, j+b} = K
\]

Válido para:
\[
0 \leq i \leq N - M,\quad 0 \leq j \leq N - L
\]

#### 2. Subrejillas de tamaño \( L \times M \)

Análogamente, para subrejillas transpuestas:

\[
\sum_{a=0}^{L-1} \sum_{b=0}^{M-1} x_{i+a, j+b} = K
\]

Válido para:
\[
0 \leq i \leq N - L,\quad 0 \leq j \leq N - M
\]

### Función Objetivo

Este es un problema **de factibilidad**, por lo tanto la función objetivo es simplemente:

\[
\min 0
\]

No buscamos optimizar ningún valor, solo encontrar una solución que satisfaga todas las restricciones.

---

## Implementación en Python

Se utilizó la librería `PuLP` para modelar y resolver el problema como un programa entero.

Pasos:

1. Crear variables enteras no negativas \( x_{i,j} \)
2. Agregar todas las restricciones sobre las subrejillas
3. Resolver con un solver LP (por defecto, CBC)
4. Mostrar la matriz solución (si existe)

---

## Ejemplo

Con los parámetros:

- \( N = 6 \)
- \( M = 3 \)
- \( L = 2 \)
- \( K = 7 \)

Se obtiene una matriz que cumple las condiciones:

```plaintext
[[0 3 0 1 2 1]
 [1 0 3 0 1 2]
 [2 1 0 3 0 1]
 [1 2 1 0 3 0]
 [0 1 2 1 0 3]
 [3 0 1 2 1 0]]
```

Todas las subrejillas de tamaño \( 3 \times 2 \) y \( 2 \times 3 \) suman exactamente 7.