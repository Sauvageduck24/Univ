import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import optuna
import argparse
import importlib
from dateutil.relativedelta import relativedelta
from typing import Tuple, Dict, List, Optional

# Configuración inicial
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# === CONSTANTES Y CONFIGURACIÓN ===
class Config:
    DEFAULT_SYMBOLS = [
        "SAN.MC", "BBVA.MC", "TEF.MC", "IBE.MC", "ITX.MC", "AMS.MC", "FER.MC", "CLNX.MC", "AENA.MC",
        "CABK.MC", "BKT.MC", "SAB.MC", "MAP.MC",       # Banca/Seguros
        "ENG.MC", "NTGY.MC",                            # Energía/Gas
        "ACS.MC",                                      # Infraestructuras
        "MEL.MC", "COL.MC", "LOG.MC",                  # Consumo/Turismo/Inmobiliario
        "GRF.MC",                                      # Salud/Farma
        "IAG.MC"                                       # Aerolíneas/Transporte
    ]
    DEFAULT_TRAIN_START = "2000-01-01"
    DEFAULT_TRAIN_END = "2023-12-31"
    DEFAULT_TEST_START = "2024-01-01"
    DEFAULT_TEST_END = "2024-12-31"
    DEFAULT_INTERVAL = "1d"
    DEFAULT_CAPITAL = 9000.0
    DEFAULT_N_TRIALS = 1000
    MIN_INVERSION = 600
    MODE='REAL'
    DEFAULT_WINDOW_TRAIN_YEARS = 5
    DEFAULT_WINDOW_TEST_MONTHS = 3
    DEFAULT_WINDOW_STEP_MONTHS = 3

# Crear carpeta de resultados si no existe
os.makedirs("resultados", exist_ok=True)

# === MANEJO DE ARGUMENTOS ===
def parse_args() -> argparse.Namespace:
    """Parse los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Optimización de carteras con Optuna")
    
    parser.add_argument('--symbols', type=str, default=','.join(Config.DEFAULT_SYMBOLS),
                       help='Lista de activos separados por coma (ej: AAPL,GOOGL)')
    parser.add_argument('--train_start', type=str, default=Config.DEFAULT_TRAIN_START, 
                       help='Fecha inicio train (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default=Config.DEFAULT_TRAIN_END, 
                       help='Fecha fin train (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default=Config.DEFAULT_TEST_START, 
                       help='Fecha inicio test (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, default=Config.DEFAULT_TEST_END, 
                       help='Fecha fin test (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default=Config.DEFAULT_INTERVAL, 
                       help='Temporalidad (ej: 1d, 1wk, 1mo)')
    parser.add_argument('--capital_inicial', type=float, default=Config.DEFAULT_CAPITAL, 
                       help='Capital inicial para la simulación')
    parser.add_argument('--n_trials', type=int, default=Config.DEFAULT_N_TRIALS, 
                       help='Número de trials para Optuna')
    parser.add_argument('--min_inversion', type=float, default=Config.MIN_INVERSION, 
                       help='Inversión mínima por activo')
    parser.add_argument('--window_train_years', type=int, default=Config.DEFAULT_WINDOW_TRAIN_YEARS, 
                       help='Años de train en cada rolling window')
    parser.add_argument('--window_test_months', type=int, default=Config.DEFAULT_WINDOW_TEST_MONTHS, 
                       help='Meses de test en cada rolling window')
    parser.add_argument('--window_step_months', type=int, default=Config.DEFAULT_WINDOW_STEP_MONTHS, 
                       help='Paso de la ventana rolling en meses')
    parser.add_argument('--no_rolling', action='store_true', default=True,
                       help='Desactiva el rolling window y usa todo el periodo de test')
    parser.add_argument('--modo', type=str, default=Config.MODE, choices=['TRAIN', 'REAL'],
                        help='Modo de operación: TRAIN (por defecto, con test/rolling) o REAL (entrena hasta hoy y muestra pesos para invertir)')
    
    return parser.parse_args()

# === FUNCIONES DE UTILIDAD ===
def load_data_modules() -> Tuple:
    """Carga los módulos de obtención de datos."""
    sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
    get_data_train = importlib.import_module('get_data_train')
    get_data_test = importlib.import_module('get_data_test')
    return get_data_train, get_data_test

def download_data(symbols: List[str], train_start: str, train_end: str, 
                 test_start: str, test_end: str, interval: str) -> None:
    """Descarga y procesa los datos de entrenamiento y test."""
    get_data_train, get_data_test = load_data_modules()
    
    print("Descargando y procesando datos de entrenamiento...")
    get_data_train.main(symbols=symbols, start=train_start, end=train_end, interval=interval)
    
    print("Descargando y procesando datos de test...")
    get_data_test.main(symbols=symbols, start=test_start, end=test_end, interval=interval)

def proyectar_pesos(w: np.ndarray, K: int) -> np.ndarray:
    """Proyecta los pesos para seleccionar los K mejores activos."""
    w = np.maximum(w, 0)
    idx = np.argsort(w)[-K:]
    w_proj = np.zeros_like(w)
    w_proj[idx] = w[idx]
    total = np.sum(w_proj)
    
    if total > 1e-8:
        w_proj /= total
    else:
        w_proj[idx] = 1.0 / K
        
    return w_proj

def calcular_metricas(pesos: np.ndarray, r_val: pd.DataFrame) -> Tuple[float, float, float, pd.Series]:
    """Calcula métricas de rentabilidad, volatilidad y ratio Sharpe."""
    ret_cartera = r_val @ pesos
    rent_media = ret_cartera.mean()
    volatilidad = ret_cartera.std()
    sharpe = rent_media / (volatilidad + 1e-8)  # Evitar división por cero
    return rent_media, volatilidad, sharpe, ret_cartera

def cargar_datos() -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, List[str]]:
    """Carga los datos desde los archivos CSV."""
    mu_train = pd.read_csv("data/media_rentabilidades_train.csv", index_col=0).squeeze()
    sigma_train = pd.read_csv("data/covarianza_rentabilidades_train.csv", index_col=0)
    precios_test = pd.read_csv("data/precios_test.csv", index_col=0, parse_dates=True)
    r_val = precios_test.pct_change().dropna()
    tickers = mu_train.index.tolist()
    
    # Validación de datos
    if mu_train.isnull().any() or sigma_train.isnull().any().any() or r_val.isnull().any().any():
        raise ValueError("Los datos contienen valores NaN. Revise la descarga de datos.")
    if len(mu_train) == 0 or sigma_train.shape[0] == 0 or r_val.shape[0] == 0:
        raise ValueError("Los datos están vacíos. Revise la descarga de datos.")
        
    return mu_train, sigma_train, r_val, tickers

# === FUNCIONES DE OPTIMIZACIÓN ===
def objective_proyeccion(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Función objetivo para el método de proyección."""
    diversificacion = 0.1 * np.sum(w > 0) / len(w)
    val = -w @ mu + 0.5 * w @ sigma @ w - diversificacion
    return val if not (np.isnan(val) or np.isinf(val)) else 1e10

def optuna_proyeccion(mu: np.ndarray, sigma: np.ndarray, K: int, 
                     n_trials: int = 100) -> np.ndarray:
    """Optimización por proyección usando Optuna."""
    N = len(mu)
    
    def objective(trial):
        w = np.array([trial.suggest_float(f'w_{i}', 0, 1) for i in range(N)])
        w = proyectar_pesos(w, K)
        return objective_proyeccion(w, mu, sigma)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_w = np.array([study.best_params[f'w_{i}'] for i in range(N)])
    return proyectar_pesos(best_w, K)

def objective_penalizado(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, 
                        K: int, beta: float, gamma: float) -> float:
    """Función objetivo para el método penalizado."""
    penal = beta * max(0, np.count_nonzero(w) - K) + gamma * abs(np.sum(w) - 1)
    val = -w @ mu + 0.5 * w @ sigma @ w + penal
    return val if not (np.isnan(val) or np.isinf(val)) else 1e10

def optuna_penalizado(mu: np.ndarray, sigma: np.ndarray, K: int, 
                      beta: float, gamma: float, n_trials: int = 100) -> np.ndarray:
    """Optimización penalizada usando Optuna."""
    N = len(mu)
    
    def objective(trial):
        w = np.array([trial.suggest_float(f'w_{i}', 0, 1) for i in range(N)])
        return objective_penalizado(w, mu, sigma, K, beta, gamma)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_w = np.array([study.best_params[f'w_{i}'] for i in range(N)])
    return proyectar_pesos(best_w, K)

# === FUNCIONES DE AJUSTE Y VISUALIZACIÓN ===
def ajustar_pesos_min_inversion(pesos: np.ndarray, tickers: List[str], 
                               capital_inicial: float, min_inversion: float) -> Tuple[np.ndarray, Dict]:
    """Ajusta los pesos para cumplir con la inversión mínima por activo."""
    pesos_dict = dict(zip(tickers, pesos))
    
    while True:
        inversiones = {activo: peso * capital_inicial for activo, peso in pesos_dict.items()}
        activos_bajos = [a for a, inv in inversiones.items() if inv < min_inversion]
        
        if not activos_bajos:
            break
            
        for a in activos_bajos:
            pesos_dict.pop(a)
            
        suma_valida = sum(pesos_dict.values())
        if suma_valida == 0:
            break
            
        pesos_dict = {a: p / suma_valida for a, p in pesos_dict.items()}
    
    # Validar suma de pesos
    suma_final = sum(pesos_dict.values())
    if abs(suma_final - 1.0) > 1e-6 and suma_final > 0:
        pesos_dict = {a: p / suma_final for a, p in pesos_dict.items()}
    
    return np.array([pesos_dict.get(t, 0.0) for t in tickers]), pesos_dict

def mostrar_pesos(pesos: np.ndarray, tickers: List[str], 
                 capital_inicial: float = None) -> None:
    """Muestra los pesos óptimos en formato de tabla."""
    df = pd.DataFrame({"Ticker": tickers, "Peso": pesos})
    df = df[df["Peso"] > 0.001].sort_values("Peso", ascending=False)
    
    if capital_inicial is not None:
        df["Inversión"] = df["Peso"] * capital_inicial
        print("\nPesos óptimos recomendados para cada activo:")
        print(df[["Ticker", "Peso", "Inversión"]].to_string(index=False))
    else:
        print("\nPesos óptimos recomendados para cada activo:")
        print(df[["Ticker", "Peso"]].to_string(index=False))
    
    df.to_csv("resultados/pesos_optimos.csv", index=False)

def plot_evolution(capital_series: pd.Series, title: str, filename: str) -> None:
    """Grafica la evolución del capital."""
    plt.figure(figsize=(12, 6))
    plt.plot(capital_series, label=title, linewidth=2)
    plt.title(f"Evolución del capital - {title}")
    plt.xlabel("Fecha")
    plt.ylabel("Capital acumulado")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados/{filename}.png")
    plt.show()

# === BACKTEST ROLLING WINDOW ===
def rolling_window_backtest(precios: pd.DataFrame, mu_full: pd.Series, sigma_full: pd.DataFrame, 
                           r_full: pd.DataFrame, tickers: List[str], window_train_years: int, 
                           window_test_months: int, window_step_months: int, capital_inicial: float, 
                           n_trials: int, min_inversion: float, test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Realiza backtesting con ventanas móviles."""
    fechas = precios.index
    start_test = pd.to_datetime(test_start)
    end_test = pd.to_datetime(test_end)
    current_test_start = start_test
    capital = capital_inicial
    capital_evol = []
    fechas_evol = []
    pesos_ventanas = []
    metricas_ventanas = []
    while current_test_start < end_test:
        current_test_end = current_test_start + relativedelta(months=window_test_months)
        
        # Definir ventanas de train y test
        train_start = current_test_start - relativedelta(years=window_train_years)
        train_end = current_test_start - pd.Timedelta(days=1)
        
        precios_train = precios[(precios.index >= train_start) & (precios.index <= train_end)]
        precios_test = precios[(precios.index >= current_test_start) & (precios.index < current_test_end)]
        
        if len(precios_train) < 10 or len(precios_test) < 2:
            current_test_start += relativedelta(months=window_step_months)
            continue
            
        r_train = precios_train.pct_change().dropna()
        r_test = precios_test.pct_change().dropna()
        mu_train = r_train.mean()
        sigma_train = r_train.cov()
        
        # Optimización para la ventana actual
        N = len(mu_train)
        K = min(10, N)
        
        # Buscar mejor estrategia en train
        mejor_sharpe = -np.inf
        mejor_pesos = None
        mejor_label = None
        
        # Probar diferentes combinaciones de parámetros
        for beta in [0.1, 1.0, 10.0]:
            for gamma in [0.0, 10.0]:
                pesos_pen = optuna_penalizado(mu_train.values, sigma_train.values, K, beta, gamma, n_trials)
                ret, vol, sharpe, _ = calcular_metricas(pesos_pen, r_train)
                
                if sharpe > mejor_sharpe:
                    mejor_sharpe = sharpe
                    mejor_pesos = pesos_pen
                    mejor_label = f"Penalización β={beta}, γ={gamma}"
        
        # Ajuste de pesos mínimos
        mejor_pesos_ajustados, _ = ajustar_pesos_min_inversion(mejor_pesos, tickers, capital, min_inversion)
        
        # Aplicar en test
        ret_test, vol_test, sharpe_test, serie_test = calcular_metricas(mejor_pesos_ajustados, r_test)
        
        # Evolución del capital
        capital_serie = (1 + serie_test).cumprod() * capital
        capital = capital_serie.iloc[-1]
        capital_evol.extend(capital_serie.tolist())
        fechas_evol.extend(capital_serie.index.tolist())
        
        # Guardar resultados de la ventana
        pesos_ventanas.append(mejor_pesos_ajustados)
        metricas_ventanas.append({
            "train_start": train_start.strftime('%Y-%m-%d'),
            "train_end": train_end.strftime('%Y-%m-%d'),
            "test_start": current_test_start.strftime('%Y-%m-%d'),
            "test_end": (current_test_end-pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
            "sharpe_test": sharpe_test,
            "rentabilidad_test": ret_test*100,
            "volatilidad_test": vol_test,
            "capital_final": capital,
            "mejor_label": mejor_label
        })
        
        current_test_start += relativedelta(months=window_step_months)
    
    # Guardar resultados
    df_evol = pd.DataFrame({"fecha": fechas_evol, "capital": capital_evol})
    df_evol.to_csv("resultados/evolucion_capital_rolling.csv", index=False)
    
    df_pesos = pd.DataFrame(pesos_ventanas, columns=tickers)
    df_pesos.to_csv("resultados/pesos_ventanas_rolling.csv", index=False)
    
    df_metricas = pd.DataFrame(metricas_ventanas)
    df_metricas.to_csv("resultados/metricas_ventanas_rolling.csv", index=False)
    
    print("\nRolling window finalizado. Resultados guardados en 'resultados/'.")
    return df_evol, df_pesos, df_metricas

# === FUNCIÓN PRINCIPAL ===
def main():
    args = parse_args()
    
    # Procesar argumentos
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    train_start = args.train_start
    train_end = args.train_end
    test_start = args.test_start
    test_end = args.test_end
    interval = args.interval
    capital_inicial = args.capital_inicial
    n_trials = args.n_trials
    min_inversion = args.min_inversion
    window_train_years = args.window_train_years
    window_test_months = args.window_test_months
    window_step_months = args.window_step_months
    
    print("\n" + "="*50)
    print("CONFIGURACIÓN DEL ANÁLISIS")
    print("="*50)
    print(f"Activos: {symbols}")
    print(f"Train: {train_start} a {train_end}")
    print(f"Test: {test_start} a {test_end}")
    print(f"Temporalidad: {interval}")
    print(f"Capital inicial: {capital_inicial:.2f} €")
    print(f"Inversión mínima por activo: {min_inversion} €")
    print(f"Número de trials Optuna: {n_trials}")
    
    if not args.no_rolling:
        print(f"\nRolling Window Config:")
        print(f"  - Train: {window_train_years} años")
        print(f"  - Test: {window_test_months} meses")
        print(f"  - Step: {window_step_months} meses")
    
    if args.modo.upper() == 'REAL':
        print("\n=== MODO REAL: Entrenando con todos los datos hasta hoy ===")
        import datetime
        fecha_hoy = pd.Timestamp.today().strftime('%Y-%m-%d')
        # Descargar datos hasta hoy
        download_data(symbols, train_start, fecha_hoy, fecha_hoy, fecha_hoy, interval)
        mu_train, sigma_train, r_val, tickers = cargar_datos()
        N = len(mu_train)
        K = min(10, N)
        # Optimización y ajuste igual que antes
        mejor_sharpe = -np.inf
        mejor_pesos = None
        mejor_label = None
        for beta in [0.1, 1.0, 10.0]:
            for gamma in [0.0, 10.0]:
                pesos_pen = optuna_penalizado(mu_train.values, sigma_train.values, K, beta, gamma, n_trials)
                ret, vol, sharpe, _ = calcular_metricas(pesos_pen, r_val)
                if sharpe > mejor_sharpe:
                    mejor_sharpe = sharpe
                    mejor_pesos = pesos_pen
                    mejor_label = f"Penalización β={beta}, γ={gamma}"
        # Ajuste de pesos mínimos
        mejor_pesos_ajustados, pesos_ajustados_dict = ajustar_pesos_min_inversion(mejor_pesos, tickers, capital_inicial, args.min_inversion)
        # Mostrar y guardar pesos
        print("\nPesos óptimos para invertir hoy:")
        for ticker, peso in sorted(pesos_ajustados_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{ticker}: {peso:.4f} → {peso * capital_inicial:.2f} €")
        pd.DataFrame({"Ticker": tickers, "Peso": mejor_pesos_ajustados}).to_csv("resultados/pesos_optimos_real.csv", index=False)
        print(f"\nCapital inicial: {capital_inicial:.2f} €")
        print(f"Archivo de pesos guardado en 'resultados/pesos_optimos_real.csv'")
        return
    
    # Descargar datos
    download_data(symbols, train_start, train_end, test_start, test_end, interval)
    
    # Cargar datos procesados
    mu_train, sigma_train, r_val, tickers = cargar_datos()
    N = len(mu_train)
    K = min(10, N)
    
    # === OPTIMIZACIÓN Y EVALUACIÓN ===
    print("\n" + "="*50)
    print("OPTIMIZANDO CARTERA")
    print("="*50)
    
    resultados = []
    series_evolucion = {}
    mejor_sharpe = -np.inf
    mejor_pesos = None
    mejor_label = None
    mejor_serie = None
    
    # Método de Proyección
    print("\nOptimizando con método de proyección...")
    pesos_proj = optuna_proyeccion(mu_train.values, sigma_train.values, K, n_trials)
    rent_p, vol_p, sharpe_p, serie_proj = calcular_metricas(pesos_proj, r_val)
    resultados.append(["proyeccion", None, None, rent_p*100, vol_p, sharpe_p])
    series_evolucion["Proyección"] = serie_proj
    
    if sharpe_p > mejor_sharpe:
        mejor_sharpe = sharpe_p
        mejor_pesos = pesos_proj
        mejor_label = "Proyección externa"
        mejor_serie = serie_proj
    
    # Método Penalizado
    print("\nOptimizando con método penalizado...")
    for beta in [0.1, 1.0, 10.0]:
        for gamma in [0.0, 10.0]:
            print(f"  - Probando beta={beta}, gamma={gamma}")
            pesos_pen = optuna_penalizado(mu_train.values, sigma_train.values, K, beta, gamma, n_trials)
            rent, vol, sharpe, serie = calcular_metricas(pesos_pen, r_val)
            label = f"Penalización β={beta}, γ={gamma}"
            resultados.append(["penalizacion", beta, gamma, rent*100, vol, sharpe])
            series_evolucion[label] = serie
            
            if sharpe > mejor_sharpe:
                mejor_sharpe = sharpe
                mejor_pesos = pesos_pen
                mejor_label = label
                mejor_serie = serie
    
    # Resultados
    df_resultados = pd.DataFrame(resultados, 
                                columns=["metodo", "beta", "gamma", "rentabilidad %", "volatilidad", "sharpe"])
    print("\nResumen de resultados de validación:")
    print(df_resultados)
    df_resultados.to_csv("resultados/resultados_validacion.csv", index=False)
    
    # Mostrar y guardar mejores pesos
    mostrar_pesos(mejor_pesos, tickers, capital_inicial)
    
    # Ajustar pesos según inversión mínima
    mejor_pesos_ajustados, pesos_ajustados = ajustar_pesos_min_inversion(
        mejor_pesos, tickers, capital_inicial, min_inversion
    )
    
    print("\nDistribución ajustada por inversión mínima:")
    for ticker, peso in sorted(pesos_ajustados.items(), key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {peso:.4f} → {peso * capital_inicial:.2f} €")
    
    # Guardar pesos ajustados
    pd.DataFrame({"Ticker": tickers, "Peso": mejor_pesos_ajustados}).to_csv(
        "resultados/pesos_optimos_ajustados.csv", index=False
    )
    
    # Calcular métricas finales con pesos ajustados
    rent_ajust, vol_ajust, sharpe_ajust, serie_ajust = calcular_metricas(mejor_pesos_ajustados, r_val)
    capital_final = (1 + serie_ajust).cumprod().iloc[-1] * capital_inicial
    
    # Guardar métricas finales
    df_metricas_final = pd.DataFrame({
        "metodo": [mejor_label],
        "sharpe": [sharpe_ajust],
        "rentabilidad %": [rent_ajust*100],
        "volatilidad": [vol_ajust],
        "capital_inicial": [capital_inicial],
        "capital_final": [capital_final],
        "beneficio": [capital_final - capital_inicial]
    })
    df_metricas_final.to_csv("resultados/metricas_finales_ajustadas.csv", index=False)
    
    # Gráfico de evolución del capital
    capital_series = (1 + mejor_serie).cumprod() * capital_inicial
    plot_evolution(capital_series, mejor_label, "evolucion_capital_mejor")
    
    # Resumen final
    print("\n" + "="*50)
    print("RESULTADOS FINALES")
    print("="*50)
    print(f"Mejor estrategia: {mejor_label}")
    print(f"Capital inicial: {capital_inicial:.2f} €")
    print(f"Capital final: {capital_final:.2f} €")
    print(f"Beneficio: {capital_final - capital_inicial:.2f} €")
    print(f"Rentabilidad: {(capital_final - capital_inicial) / capital_inicial * 100:.2f}%")
    print(f"Ratio Sharpe: {sharpe_ajust:.2f}")
    print(f"Volatilidad: {vol_ajust:.4f}")
    
    # Rolling Window si está activado
    if not args.no_rolling:
        print("\n" + "="*50)
        print("INICIANDO ROLLING WINDOW BACKTEST")
        print("="*50)
        
        precios = pd.read_csv("data/precios_test.csv", index_col=0, parse_dates=True)
        df_evol, df_pesos, df_metricas = rolling_window_backtest(
            precios, mu_train, sigma_train, r_val, tickers, 
            window_train_years, window_test_months, window_step_months, 
            capital_inicial, n_trials, min_inversion,
            test_start, test_end
        )
        
        # Gráfico de evolución del capital en rolling window
        plot_evolution(df_evol.set_index("fecha")["capital"], 
                      "Rolling Window Backtest", 
                      "evolucion_capital_rolling")

if __name__ == "__main__":
    main()