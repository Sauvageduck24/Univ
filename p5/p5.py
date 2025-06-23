import numpy as np

# Paso 1: Inicialización del estado
def inicializar(N):
    X = np.zeros((N, N), dtype=int)
    for i in range(N):
        j = np.random.randint(0, N)
        X[i, j] = 1
    return X

# Paso 2 y 3: Definir pesos y umbrales según las restricciones
def calcular_pesos_umbral(N, mu=2):
    W = np.zeros((N, N, N, N))
    theta = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if i == k and j != l:
                        W[i, j, k, l] = -2 * mu
                    elif j == l and i != k:
                        W[i, j, k, l] = -2 * mu
            theta[i, j] = -2 * mu
    return W, theta

# Paso 4: Dinámica asincrónica de la red
def actualizar_asincrono(X, W, theta):
    N = X.shape[0]
    indices = np.random.permutation(N*N)
    for idx in indices:
        i, j = divmod(idx, N)
        suma = 0
        for k in range(N):
            for l in range(N):
                suma += W[i, j, k, l] * X[k, l]
        X[i, j] = 1 if suma > theta[i, j] else 0
    return X

# Ejecución completa del proceso
def red_hopfield_n_torres(N, iteraciones=100):
    X = inicializar(N)
    W, theta = calcular_pesos_umbral(N, mu=2)
    for _ in range(iteraciones):
        X_nuevo = actualizar_asincrono(X.copy(), W, theta)
        if np.array_equal(X_nuevo, X):
            break
        X = X_nuevo
    return X

# Prueba con N = 4
solucion = red_hopfield_n_torres(4)
print("Solución encontrada:\n", solucion)
