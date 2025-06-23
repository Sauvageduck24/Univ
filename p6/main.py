import numpy as np
import pandas as pd
import cma
import matplotlib.pyplot as plt

print()
plt.rcParams['figure.figsize'] = (10, 6)

# === UTILIDADES COMUNES ===
def proyectar_pesos(w, K):
    w = np.maximum(w, 0)  # Eliminar pesos negativos
    idx = np.argsort(w)[-K:]  # Índices de los K mayores pesos (no negativos)
    
    # Si hay menos de K pesos positivos, completar con los siguientes más grandes
    if np.sum(w > 0) < K:
        idx = np.argsort(w)[-K:]
    
    w_proj = np.zeros_like(w)
    w_proj[idx] = w[idx]
    total = np.sum(w_proj)
    
    # Evitar división por cero
    if total > 1e-8:
        w_proj /= total
    else:
        # Distribución equitativa si todos son cero
        w_proj[idx] = 1.0 / K
    return w_proj

def calcular_metricas(pesos, r_val):
    ret_cartera = r_val @ pesos
    rent_media = ret_cartera.mean()
    volatilidad = ret_cartera.std()
    sharpe = rent_media / volatilidad
    return rent_media, volatilidad, sharpe, ret_cartera

# === DATOS ===
mu_train = pd.read_csv("data/media_rentabilidades_train.csv", index_col=0).squeeze()
sigma_train = pd.read_csv("data/covarianza_rentabilidades_train.csv", index_col=0)
precios_test = pd.read_csv("data/precios_test.csv", index_col=0, parse_dates=True)
r_val = precios_test.pct_change().dropna()
N = len(mu_train)
K = 10

# ==============================================================================
# PARTE A – PROYECCIÓN EXTERNA (VERSIÓN MEJORADA)
# ==============================================================================
def objective_proyeccion(w, mu, sigma):
    # Añadir término de diversificación (evita concentración)
    diversificacion = 0.1 * np.sum(w > 0) / len(w)
    return -w @ mu + 0.5 * w @ sigma @ w - diversificacion

def optimizar_proyeccion(mu, sigma, K):
    # Configuración robusta de CMA-ES
    opts = {
        'verbose': -9,
        'popsize': 50,       # Población más grande
        'maxiter': 200,      # Máximo de iteraciones
        'tolfun': 1e-7,      # Tolerancia de convergencia
        'seed': 42           # Semilla para reproducibilidad
    }
    es = cma.CMAEvolutionStrategy(N * [1/N], 0.05, opts)
    
    # Ejecutar optimización
    while not es.stop():
        soluciones = es.ask()
        soluciones_proj = [proyectar_pesos(w, K) for w in soluciones]
        valores_obj = [objective_proyeccion(w, mu, sigma) for w in soluciones_proj]
        es.tell(soluciones, valores_obj)
    
    # Mejor solución proyectada
    mejor_w = proyectar_pesos(es.result.xbest, K)
    return mejor_w

# === EJECUCIÓN PROYECCIÓN ===
pesos_proj = optimizar_proyeccion(mu_train.values, sigma_train.values, K)
print("Pesos optimizados (K={}):".format(K))
print(pesos_proj.tolist())

rent_p, vol_p, sharpe_p, serie_proj = calcular_metricas(pesos_proj, r_val)

df_proj = pd.DataFrame({
    "metodo": ["proyeccion"],
    "beta": [None],
    "gamma": [None],
    "rentabilidad %": [rent_p*100],
    "volatilidad": [vol_p],
    "sharpe": [sharpe_p]
})

# Gráfico rendimiento acumulado (proyección)
plt.plot((1 + serie_proj).cumprod(), label="Proyección externa", linewidth=2)
plt.title("Rendimiento acumulado – Proyección externa")
plt.xlabel("Fecha")
plt.ylabel("Crecimiento del capital")
plt.grid(True)
plt.legend()
plt.show()
print()

# ==============================================================================
# PARTE B – PENALIZACIÓN
# ==============================================================================

def objective_penalizado(w, mu, sigma, K, beta, gamma):
    penal = beta * max(0, np.count_nonzero(w) - K) + gamma * abs(np.sum(w) - 1)
    return -w @ mu + 0.5 * w @ sigma @ w + penal

def optimizar_penalizado(mu, sigma, K, beta, gamma):
    es = cma.CMAEvolutionStrategy(N * [1/N], 0.1, {'verbose': -9})
    while not es.stop():
        soluciones = es.ask()
        valores = [objective_penalizado(w, mu, sigma, K, beta, gamma) for w in soluciones]
        es.tell(soluciones, valores)
    return es.result.xbest

# === EJECUCIÓN PENALIZACIÓN ===

resultados_pen = []
series_penalizadas = {}

for beta in [0.1, 1.0, 10.0]:
    for gamma in [0.0, 10.0]:
        pesos_pen = optimizar_penalizado(mu_train.values, sigma_train.values, K, beta, gamma)
        pesos_pen = proyectar_pesos(pesos_pen, K)
        print(pesos_pen.tolist())  
        rent, vol, sharpe, serie = calcular_metricas(pesos_pen, r_val)
        label = f"β={beta}, γ={gamma}"
        resultados_pen.append(["penalizacion", beta, gamma, rent*100, vol, sharpe])
        series_penalizadas[label] = serie

df_pen = pd.DataFrame(resultados_pen, columns=["metodo", "beta", "gamma", "rentabilidad %", "volatilidad", "sharpe"])

# Gráfico rendimiento acumulado (penalización)
for label, serie in series_penalizadas.items():
    plt.plot((1 + serie).cumprod(), label=label, alpha=0.7)
    
plt.title("Rendimiento acumulado – Penalización")
plt.xlabel("Fecha")
plt.ylabel("Crecimiento del capital")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==============================================================================
# ✅ COMPARACIÓN FINAL Y EXPORTACIÓN
# ==============================================================================

df_proj = df_proj.dropna(axis=1, how="all")
df_pen = df_pen.dropna(axis=1, how="all")
df_resultados = pd.concat([df_proj, df_pen], ignore_index=True)

print()
print(df_resultados)

df_resultados.to_csv("resultados/resultados_validacion.csv", index=False)
print()