import yfinance as yf
import pandas as pd

symbols = [
    "AAPL", "MSFT", "AMZN", "GOOGL"
]

print("Probando descarga de datos con yfinance...")
failed = []
for sym in symbols:
    print(f"Descargando {sym}...", end=" ")
    try:
        data = yf.download(sym, start="2024-01-01", end="2024-12-31", interval="1d")
        if data.empty:
            print("FALLÓ (vacío)")
            failed.append(sym)
        else:
            print("OK")
    except Exception as e:
        print(f"FALLÓ ({e})")
        failed.append(sym)

print("\nResumen de descarga:")
if failed:
    print(f"Tickers que fallaron: {failed}")
else:
    print("Todos los tickers descargaron correctamente.") 