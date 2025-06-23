# Práctica 3: Programación de enteros (El problema de viajante de comercio)

### Esteban Sánchez Gámez

Aquí el objetivo está planteado en el código como **minimizar la distancia total recorrida**, el enfoque de programación entera es lo importante.

---

## **Resumen general**
Este código resuelve el **Problema del Viajante de Comercio (TSP)** usando **programación lineal entera binaria**, implementado con la librería `pulp`.  
También se usa el método de **Miller-Tucker-Zemlin (MTZ)** para evitar subtours (circuitos parciales).

---

## **Secciones del código y su propósito**

### **1. Cargar los datos geográficos**
```python
data = loadmat('usborder.mat')
usbborder = data['usbborder']
x = usbborder['x'][0][0].flatten()
y = usbborder['y'][0][0].flatten()
```
Se cargan las coordenadas del borde de Estados Unidos desde un archivo `.mat` de MATLAB.

---

### **2. Generar puntos aleatorios dentro del país**
```python
# Crear un polígono con Path
polygon = np.column_stack((x, y))
path = Path(polygon)

# Generar nStops puntos aleatorios dentro del polígono
```
Usa un objeto `Path` de matplotlib para comprobar si un punto está dentro del borde del país. Luego, se generan 5 puntos aleatorios dentro del polígono que representa el país.

---

### **3. Graficar puntos y borde**
```python
plt.plot(...) # borde
plt.scatter(...) # puntos de parada
```
Esto es solo para visualización de los puntos generados aleatoriamente.

---

### **4. Formulación del problema TSP**
Aquí empieza la parte de programación entera.

#### **Variables de decisión**
```python
x_vars[i, j] ∈ {0,1} 
```
- Si `x_vars[i,j] = 1` → se va directamente del nodo `i` al nodo `j`.
- Son **variables binarias**, típicas de la programación entera.

```python
u_vars[i] ∈ [0, n-1]
```
- Variables auxiliares para MTZ: permiten controlar el orden de visita y **evitar subtours** (viajes en circuitos más pequeños).

---

### **Función objetivo**
```python
prob += lpSum(dist_matrix[i][j] * x_vars[i, j] for i in nodes for j in nodes if i != j)
```
> Minimizar la **suma total de distancias** recorridas entre los nodos.

Este reemplaza al objetivo `min 0`, y tiene sentido ya que se quiere minimizar el recorrido total del viajante.

---

### **Restricciones principales:**

#### 1. **Salir de cada nodo exactamente una vez**
```python
for i in nodes:
    prob += lpSum(x_vars[i, j] for j in nodes if i != j) == 1
```

#### 2. **Entrar a cada nodo exactamente una vez**
```python
for j in nodes:
    prob += lpSum(x_vars[i, j] for i in nodes if i != j) == 1
```

Estas dos restricciones garantizan que cada ciudad se visita una y solo una vez.

#### 3. **Restricciones MTZ para evitar subtours**
```python
for i in nodes:
    for j in nodes:
        if i != j and i != 0 and j != 0:
            prob += u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2
```
Estas restricciones adicionales (llamadas MTZ) se usan para impedir que se formen ciclos pequeños dentro de la solución, y forzar un solo tour completo.

---

### **5. Resolución y visualización**
```python
prob.solve()
```
Se resuelve el modelo con un solver de programación lineal entera.

Después, se extrae la solución y se grafica el recorrido.

---

## **Interpretación con el enfoque "min 0 y Ax = b"**
En este caso:

- La **programación entera** define variables binarias \( x_{ij} \in \{0,1\} \) para tomar decisiones.
- El objetivo no es realmente "min 0", sino minimizar la función lineal de costos: \( \min \sum c_{ij} x_{ij} \)
- Las **restricciones** (del tipo \( Ax = b \)) son:
  - Que cada nodo tenga entrada y salida una vez.
  - MTZ se podría expresar como desigualdades lineales adicionales.
  
Es decir, se ha **formulado el problema como una programación entera** con:
- **Función objetivo lineal**
- **Restricciones lineales**
- **Variables binarias (enteras)**

---

¿Quieres que te explique cómo sería la matriz \( A \), vector \( x \) y \( b \) explícitamente para este problema?