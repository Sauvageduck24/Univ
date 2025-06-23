import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
test_size = 5

# Cargar datos
dataframe = pd.read_csv("hou_all.csv", header=None)

# Procesamiento de datos
dataframe = dataframe.drop(dataframe.columns[-1], axis=1)  # Eliminar columna bias
dataframe_scaled = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)

# Convertir y_train y y_test a vectores columna
X_train, y_train = np.array(dataframe_scaled.iloc[test_size:, :-1]), np.array(dataframe_scaled.iloc[test_size:, -1]).reshape(-1, 1)
X_test, y_test = np.array(dataframe_scaled.iloc[:test_size, :-1]), np.array(dataframe_scaled.iloc[:test_size, -1]).reshape(-1, 1)

# Mostrar los datos escalados
print(dataframe_scaled)

# Implementación de Adaline (gradiente)

# Inicialización de datos
n_data = X_train.shape[0]
num_caracteristicas = X_train.shape[1]
pesos_solucion = np.random.rand(num_caracteristicas, 1)
contador = 0
tolerancia = 1e-6
max_iters = int(10e2)  # Convertir a entero
hessiana = 2 * X_train.T @ X_train  # H = 2X^TX

# Calcular gradiente inicial
gradiente_funcion_siguiente = 2 * X_train.T @ ((X_train @ pesos_solucion) - y_train)

print(f"\nTamaño dataframe train: {n_data}\nNúmero características train: {num_caracteristicas}\nTamaño pesos: {pesos_solucion.shape}\n")

# Entrenamiento con descenso de gradiente
while np.linalg.norm(gradiente_funcion_siguiente) > tolerancia and contador <= max_iters:
    error_mse = np.mean((X_train @ pesos_solucion - y_train) ** 2)
    direccion_descenso = -gradiente_funcion_siguiente

    # Cálculo del tamaño de paso óptimo
    numerador = gradiente_funcion_siguiente.T @ gradiente_funcion_siguiente
    denominador = gradiente_funcion_siguiente.T @ (hessiana @ gradiente_funcion_siguiente)

    if denominador == 0:
        print("Denominador es 0, deteniendo la optimización para evitar división por 0.")
        break

    tamanyo_paso = numerador / denominador  # α* = (∇f^T ∇f) / (∇f^T H ∇f)

    # Actualizar pesos
    pesos_solucion = pesos_solucion + tamanyo_paso * direccion_descenso

    # Recalcular el gradiente
    gradiente_funcion_siguiente = 2 * X_train.T @ ((X_train @ pesos_solucion) - y_train)

    print(f"Iteración: {contador}, Norma Gradiente: {np.linalg.norm(gradiente_funcion_siguiente)} Error modelo (MSE): {error_mse}")
    contador += 1

# Test
y_pred = X_test @ pesos_solucion  # Adaline es un modelo lineal

# Calcular métricas de evaluación
error_mse = np.mean((y_pred - y_test) ** 2)
print(f"\nError modelo (MSE): {error_mse}\n")

print(f"Precios reales 5 primeras viviendas: {scaler.inverse_transform(np.hstack((X_test,y_test)))[:,-1]}")
print(f"Prediccion 5 primeras viviendas: {scaler.inverse_transform(np.hstack((X_test,y_pred)))[:,-1]}\n")