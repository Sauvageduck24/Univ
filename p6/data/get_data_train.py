import yfinance as yf
import pandas as pd
import os

def main(symbols=None, start=None, end=None, interval="1d"):
    if symbols is None:
        symbols = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "JNJ", "V", "NVDA", "JPM", "WMT", "PG",
            "XOM", "HD", "BAC", "MA", "INTC", "T", "MRK", "PFE", "KO", "PEP"
        ]
    if start is None:
        start = "2024-01-01"
    if end is None:
        end = "2024-12-31"
    os.makedirs("data", exist_ok=True)
    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval=interval,
    )
    if data.empty:
        print(f"Error: No se pudieron descargar datos para los símbolos: {symbols}")
        return
    data_close = data['Close']
    rentabilidades_diarias = data_close.pct_change().dropna()
    mu = rentabilidades_diarias.mean()
    sigma = rentabilidades_diarias.cov()
    mu.to_csv("data/media_rentabilidades_train.csv", header=True)
    sigma.to_csv("data/covarianza_rentabilidades_train.csv")
    data_close.to_csv("data/precios_train.csv")

if __name__ == "__main__":
    main()