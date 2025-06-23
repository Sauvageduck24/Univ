import yfinance as yf
import pandas as pd

# Paso 1: seleccionar los tickers de las acciones
symbols = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "JNJ", "V", "NVDA", "JPM", "WMT", "PG",
    "XOM", "HD", "BAC", "MA", "INTC", "T", "MRK", "PFE", "KO", "PEP"
]

# Paso 2: descargar los datos de precios históricos
data = yf.download(
    tickers=symbols,
    start="2024-01-01",
    end="2024-12-31",
    interval="1h",
)

data_close= data['Close']

# Paso 3: calcular rentabilidades diarias
rentabilidades_diarias = data_close.pct_change().dropna()

# Paso 4: vector μ (media) y matriz Σ (covarianza)
mu = rentabilidades_diarias.mean()
sigma = rentabilidades_diarias.cov()

# Paso 5: guardar en CSV
mu.to_csv("data/media_rentabilidades_train.csv", header=True)
sigma.to_csv("data/covarianza_rentabilidades_train.csv")
data_close.to_csv("data/precios_train.csv")