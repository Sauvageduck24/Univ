from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pulp
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpBinary, LpStatus
from scipy.spatial.distance import cdist

# Cargar el archivo .mat
data = loadmat('usborder.mat')

# Acceder a la estructura 'usbborder'
usbborder = data['usbborder']

# Extraer las variables x, y, xx, yy desde la estructura
x = usbborder['x'][0][0].flatten()
y = usbborder['y'][0][0].flatten()
xx = usbborder['xx'][0][0].flatten()
yy = usbborder['yy'][0][0].flatten()

# Crear un polígono a partir de los puntos x e y
polygon = np.column_stack((x, y))  # Combinar x e y en una matriz de puntos
path = Path(polygon)  # Crear un objeto Path para el polígono

# Configuración inicial
np.random.seed(3)
nStops = 5
stopsLon = np.zeros(nStops)
stopsLat = np.zeros(nStops)
n = 0

# Generar puntos aleatorios dentro del borde de Estados Unidos
while n < nStops:
    xp = np.random.rand() * 1.5
    yp = np.random.rand()
    if path.contains_point((xp, yp)):  # Verificar si el punto está dentro del polígono
        stopsLon[n] = xp
        stopsLat[n] = yp
        n += 1

# Dibujar el borde de Estados Unidos y los puntos de parada
plt.plot(x, y, color='red', label='Borde de Estados Unidos')
plt.scatter(stopsLon, stopsLat, color='blue', marker='*', label='Puntos de parada')
plt.title('Puntos de parada del viajante de comercio')
plt.legend()
plt.show()


####### INCLUIR AQUí EL CÓDIGO DE OPTIMIZACIÓN DEL TSP

# Número de nodos/paradas
n = nStops
# Índices de los nodos
nodes = list(range(n))

# Calcular matriz de distancias euclidianas entre paradas
coords = np.column_stack((stopsLon, stopsLat))
dist_matrix = cdist(coords, coords)

# Variables binarias x[i,j] = 1 si se va de i a j
x_vars = pulp.LpVariable.dicts('x', ((i, j) for i in nodes for j in nodes if i != j), cat=LpBinary)

# Variables continuas u[i] para MTZ (para evitar subtours)
u_vars = pulp.LpVariable.dicts('u', nodes, lowBound=0, upBound=n-1, cat='Continuous')

# Crear el problema
prob = LpProblem("TSP", LpMinimize)

# Función objetivo: minimizar la distancia total
prob += lpSum(dist_matrix[i][j] * x_vars[i, j] for i in nodes for j in nodes if i != j)

# Restricciones: salir exactamente una vez de cada nodo
for i in nodes:
    prob += lpSum(x_vars[i, j] for j in nodes if i != j) == 1

# Restricciones: llegar exactamente una vez a cada nodo
for j in nodes:
    prob += lpSum(x_vars[i, j] for i in nodes if i != j) == 1

# Restricciones MTZ para evitar subtours
for i in nodes:
    for j in nodes:
        if i != j and i != 0 and j != 0:
            prob += u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2

# Resolver
prob.solve()
print("Estado:", LpStatus[prob.status])

# Extraer solución
x_tsp_sol = np.zeros((n, n), dtype=int)
P = []

for i in nodes:
    for j in nodes:
        if i != j and x_vars[i, j].varValue == 1:
            x_tsp_sol[i, j] = 1
            P.append((i, j))
P = np.array(P)

# Encontrar los segmentos de la ruta óptima
segments = np.where(x_tsp_sol == 1)[0]

# Dibujar la ruta óptima
plt.plot(x, y, color='red', label='Borde de Estados Unidos')
plt.scatter(stopsLon, stopsLat, color='blue', marker='*', label='Puntos de parada')
for seg in segments:
    plt.plot([stopsLon[P[seg, 0]], stopsLon[P[seg, 1]]], [stopsLat[P[seg, 0]], stopsLat[P[seg, 1]]], color='green')
plt.title('Solución con subtours')
plt.legend()
plt.show()
