import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Activos y pesos (ajusta si quieres otros)
activos = [
    "ITX.MC",
    "BKT.MC",
    "FER.MC",
    "LOG.MC",
    "TEF.MC",
    "ACS.MC",
    "BBVA.MC",
    "AMS.MC",
    "MAP.MC"
]

pesos = np.array([
    0.1290,
    0.1208,
    0.1208,
    0.1180,
    0.1122,
    0.1073,
    0.1050,
    0.0991,
    0.0879
])

pesos = pesos / pesos.sum()  # Por si acaso, normaliza

capital_inicial = 9000

# Fechas
fecha_inicio = "2000-01-01"
fecha_fin = datetime.today().strftime("%Y-%m-%d")

# Descargar precios ajustados
data = yf.download(activos, start=fecha_inicio, end=fecha_fin)["Close"]

# Elimina filas con NaN (días sin cotización en algún activo)
data = data.dropna()

# Calcula retornos diarios
retornos = data.pct_change().dropna()

# Simula la evolución del capital
ret_cartera = retornos @ pesos
capital = (1 + ret_cartera).cumprod() * capital_inicial

# Estadísticas finales
rentabilidad_total = (capital.iloc[-1] / capital.iloc[0]) - 1
volatilidad_anual = ret_cartera.std() * np.sqrt(252)
sharpe = ret_cartera.mean() / (ret_cartera.std() + 1e-8) * np.sqrt(252)

print(f"Rentabilidad total: {rentabilidad_total*100:.2f}%")
print(f"Volatilidad anualizada: {volatilidad_anual*100:.2f}%")
print(f"Sharpe anualizado: {sharpe:.2f}")
print(f"Capital final: {capital.iloc[-1]:.2f} €")

# Gráfico
plt.figure(figsize=(12,6))
plt.plot(capital, label='Evolución del capital')
plt.title('Backtest Portfolio desde 2000')
plt.xlabel('Fecha')
plt.ylabel('Capital (€)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()