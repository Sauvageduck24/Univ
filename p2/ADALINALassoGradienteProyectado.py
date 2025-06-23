import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar y procesar datos
dataframe = pd.read_csv("hou_all.csv", header=None)
dataframe = dataframe.drop(dataframe.columns[-1], axis=1)  # Eliminar columna bias
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

# Normalizar X e Y
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Separar datos de entrenamiento y prueba
test_size = 5
X_train, y_train = X[test_size:], y[test_size:]
X_test, y_test = X[:test_size], y[:test_size]

# Inicializar pesos
np.random.seed(42)
w = np.random.rand(X_train.shape[1])
w_p = np.maximum(w, 0)
w_n = np.maximum(-w, 0)

# Función para calcular el error cuadrático (MSE)
def compute_error(X, y, w, landa):
    return 0.5 * np.linalg.norm(X @ w - y) ** 2 + landa * np.linalg.norm(w, 1)

# Función de backtracking
def backtracking(w_p, d_p, w_n, d_n, X_train, y_train, landa):
    alpha = 1
    beta = 0.5
    while True:
        new_w_p = np.maximum(w_p + alpha * d_p, 0)
        new_w_n = np.maximum(w_n + alpha * d_n, 0)
        new_w = new_w_p - new_w_n
        if compute_error(X_train, y_train, new_w, landa) < compute_error(X_train, y_train, w_p - w_n, landa):
            break
        alpha *= beta
    return new_w_p, new_w_n

# Algoritmo del gradiente proyectado
tolerance = 1e-3
landa = 200
max_iter = landa
k = 0
error_anterior = float('inf')

while k < max_iter:
    w = w_p - w_n
    gradient_p = X_train.T @ X_train @ w - X_train.T @ y_train + landa * np.ones(X_train.shape[1])
    gradient_n = -X_train.T @ X_train @ w + X_train.T @ y_train + landa * np.ones(X_train.shape[1])
    d_p = -gradient_p
    d_n = -gradient_n
    
    w_p, w_n = backtracking(w_p, d_p, w_n, d_n, X_train, y_train, landa)

    error = np.mean((X_train @ (w_p - w_n) - y_train) ** 2)
    print(f"Iteración: {k}, Error modelo (MSE): {error}")
    if abs(error - error_anterior) < tolerance:
        print("Convergencia alcanzada, el error disminuye menos que la tolerancia")
        break

    error_anterior = error
    k += 1

w = w_p - w_n

if k == max_iter:
    print("Se ha llegado al máximo de iteraciones antes de alcanzar la convergencia")
else:
    print(f"\nConvergencia alcanzada en {k} iteraciones.")

# Predecir y evaluar
y_pred = X_test @ w
error_mse = np.mean((y_pred - y_test) ** 2)
print(f"\nError modelo (MSE): {error_mse}")

# Desnormalizar y mostrar resultados
y_real_first_5 = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_first_5 = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

print("\nValores reales y predichos para los 5 primeros patrones:")
for i in range(5):
    print(f"Patrón {i+1}: Real = {y_real_first_5[i]:.4f}, Predicho = {y_pred_first_5[i]:.4f}")