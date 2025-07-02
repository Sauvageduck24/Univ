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
from sklearn.model_selection import train_test_split
import time

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
    DEFAULT_N_TRIALS = 100
    MIN_INVERSION = 600
    MODE='TRAIN'
    DEFAULT_WINDOW_TRAIN_YEARS = 5
    DEFAULT_WINDOW_STEP_MONTHS = 3
    DEFAULT_METHOD='penalizacion_10.0_10.0'#'averiguate'
    EARLY_STOPPING_PATIENCE = 10  # Número de trials sin mejora para parar
    VALIDATION_SIZE = 0.2  # Tamaño del conjunto de validación

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
    parser.add_argument('--window_step_months', type=int, default=Config.DEFAULT_WINDOW_STEP_MONTHS, 
                       help='Paso de la ventana rolling en meses')
    parser.add_argument('--modo', type=str, default=Config.MODE, choices=['TRAIN', 'REAL'],
                        help='Modo de operación: TRAIN (por defecto, con test/rolling) o REAL (entrena hasta hoy y muestra pesos para invertir)')
    parser.add_argument('--metodo', type=str, default=Config.DEFAULT_METHOD,
                       help="Método de optimización: 'proyeccion', 'penalizacion_BETA_GAMMA' (ej: penalizacion_10.0_0.0) o 'averiguate' para probar todos y elegir el mejor")
    
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

def calcular_metricas(pesos: np.ndarray, r_val: pd.DataFrame, tickers: list = None) -> Tuple[float, float, float, pd.Series]:
    """Calcula métricas de rentabilidad, volatilidad y ratio Sharpe de forma robusta."""
    if tickers is None:
        tickers = r_val.columns.tolist()
    # Filtrar activos con peso > 0
    activos_validos = [t for t, p in zip(tickers, pesos) if p > 0]
    pesos_validos = np.array([p for t, p in zip(tickers, pesos) if p > 0])
    if len(activos_validos) == 0 or len(pesos_validos) == 0:
        return float('nan'), float('nan'), float('nan'), pd.Series(dtype=float)
    # Normalizar pesos
    pesos_validos = pesos_validos / pesos_validos.sum()
    # Filtrar r_val
    r_val_filtrado = r_val[activos_validos]
    if r_val_filtrado.shape[1] != len(pesos_validos):
        return float('nan'), float('nan'), float('nan'), pd.Series(dtype=float)
    ret_cartera = r_val_filtrado @ pesos_validos
    rent_media = ret_cartera.mean()
    volatilidad = ret_cartera.std()
    sharpe = rent_media / (volatilidad + 1e-8)
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

# === FUNCIONES DE OPTIMIZACIÓN MEJORADAS ===
def objective_proyeccion(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, 
                        r_val: pd.DataFrame) -> float:
    """Función objetivo para el método de proyección con validación."""
    diversificacion = 0.1 * np.sum(w > 0) / len(w)
    val = -w @ mu + 0.5 * w @ sigma @ w - diversificacion
    
    # Calcular Sharpe en conjunto de validación
    _, _, sharpe_val, _ = calcular_metricas(w, r_val)
    
    return sharpe_val if not (np.isnan(sharpe_val) or np.isinf(sharpe_val)) else -1e10

def optuna_proyeccion(mu: np.ndarray, sigma: np.ndarray, r_train: pd.DataFrame, 
                     r_val: pd.DataFrame, K: int, n_trials: int = 100) -> np.ndarray:
    """Optimización por proyección usando Optuna con early stopping."""
    N = len(mu)
    best_sharpe = -np.inf
    no_improvement = 0
    best_w = None
    
    def objective(trial):
        nonlocal best_sharpe, no_improvement, best_w
        
        w = np.array([trial.suggest_float(f'w_{i}', 0, 1) for i in range(N)])
        w = proyectar_pesos(w, K)
        sharpe = objective_proyeccion(w, mu, sigma, r_val)
        
        # Early stopping
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w.copy()
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= Config.EARLY_STOPPING_PATIENCE:
            trial.study.stop()
            
        return sharpe
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return best_w if best_w is not None else proyectar_pesos(np.ones(N)/N, K)

def objective_penalizado(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, 
                        r_val: pd.DataFrame, K: int, beta: float, gamma: float) -> float:
    """Función objetivo para el método penalizado con validación."""
    penal = beta * max(0, np.count_nonzero(w) - K) + gamma * abs(np.sum(w) - 1)
    val = -w @ mu + 0.5 * w @ sigma @ w + penal
    
    # Calcular Sharpe en conjunto de validación
    _, _, sharpe_val, _ = calcular_metricas(w, r_val)
    
    return sharpe_val if not (np.isnan(sharpe_val) or np.isinf(sharpe_val)) else -1e10

def optuna_penalizado(mu: np.ndarray, sigma: np.ndarray, r_train: pd.DataFrame,
                     r_val: pd.DataFrame, K: int, beta: float, gamma: float, 
                     n_trials: int = 100) -> np.ndarray:
    """Optimización penalizada usando Optuna con early stopping."""
    N = len(mu)
    best_sharpe = -np.inf
    no_improvement = 0
    best_w = None
    
    def objective(trial):
        nonlocal best_sharpe, no_improvement, best_w
        
        w = np.array([trial.suggest_float(f'w_{i}', 0, 1) for i in range(N)])
        sharpe = objective_penalizado(w, mu, sigma, r_val, K, beta, gamma)
        
        # Early stopping
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w.copy()
            no_improvement = 0
        else:
            no_improvement += 1
            
        if no_improvement >= Config.EARLY_STOPPING_PATIENCE:
            trial.study.stop()
            
        return sharpe
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return proyectar_pesos(best_w, K) if best_w is not None else proyectar_pesos(np.ones(N)/N, K)

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
    df = df[df["Peso"] > 0.0001].sort_values("Peso", ascending=False)
    if df.empty:
        print("\nNo hay activos seleccionados con peso significativo (>0.0001). Prueba con menos restricción de inversión mínima o más capital.")
        return
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

# === BACKTEST ROLLING WINDOW SOLO EN TRAIN ===
def rolling_window_train(precios_train: pd.DataFrame, tickers: List[str], 
                        window_train_years: int, window_step_months: int, 
                        capital_inicial: float, n_trials: int, min_inversion: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Realiza rolling window solo en datos de entrenamiento para validación."""
    fechas = precios_train.index
    end_date = precios_train.index[-1]
    current_start = precios_train.index[0]
    capital = capital_inicial
    capital_evol = []
    fechas_evol = []
    pesos_ventanas = []
    metricas_ventanas = []
    # Calcular el número total de ventanas aproximado
    total_ventanas = 0
    temp_start = current_start
    while temp_start < end_date:
        temp_end = temp_start + relativedelta(years=window_train_years)
        if temp_end > end_date:
            break
        total_ventanas += 1
        temp_start += relativedelta(months=window_step_months)
    ventana_actual = 0
    while current_start < end_date:
        current_end = current_start + relativedelta(years=window_train_years)
        if current_end > end_date:
            break
        ventana_actual += 1
        print(f"Ventana {ventana_actual}/{total_ventanas}")
        precios_window = precios_train[(precios_train.index >= current_start) & 
                                      (precios_train.index <= current_end)]
        if len(precios_window) < 10:
            current_start += relativedelta(months=window_step_months)
            continue
        split_idx = int(len(precios_window) * 0.8)
        precios_train_window = precios_window.iloc[:split_idx]
        precios_val_window = precios_window.iloc[split_idx:]
        r_train = precios_train_window.pct_change(fill_method=None).dropna()
        r_val = precios_val_window.pct_change(fill_method=None).dropna()
        # Chequeo de tamaño suficiente para evitar warnings
        if r_train.shape[0] < 2 or r_train.shape[1] == 0 or r_val.shape[0] < 2 or r_val.shape[1] == 0:
            current_start += relativedelta(months=window_step_months)
            continue
        mu_train = r_train.mean()
        sigma_train = r_train.cov()
        N = len(mu_train)
        K = min(10, N)
        mejor_sharpe = -np.inf
        mejor_pesos = None
        mejor_label = None
        for beta in [0.1, 1.0, 10.0]:
            for gamma in [0.0, 1.0, 10.0]:
                pesos_pen = optuna_penalizado(mu_train.values, sigma_train.values, 
                                            r_train, r_val, K, beta, gamma, n_trials)
                # Chequeo de pesos válidos
                if pesos_pen is None or np.allclose(pesos_pen, 0):
                    continue
                _, _, sharpe_val, _ = calcular_metricas(pesos_pen, r_val, mu_train.index.tolist())
                if np.isnan(sharpe_val):
                    continue
                if sharpe_val > mejor_sharpe:
                    mejor_sharpe = sharpe_val
                    mejor_pesos = pesos_pen
                    mejor_label = f"Penalización β={beta}, γ={gamma}"
        # Si no hay pesos válidos, saltar ventana
        if mejor_pesos is None or np.allclose(mejor_pesos, 0):
            current_start += relativedelta(months=window_step_months)
            continue
        # Aplicar en validation
        ret_val, vol_val, sharpe_val, serie_val = calcular_metricas(mejor_pesos, r_val, mu_train.index.tolist())
        if np.isnan(sharpe_val):
            current_start += relativedelta(months=window_step_months)
            continue
        capital_serie = (1 + serie_val).cumprod() * capital
        capital = capital_serie.iloc[-1]
        capital_evol.extend(capital_serie.tolist())
        fechas_evol.extend(capital_serie.index.tolist())
        pesos_ventanas.append(mejor_pesos)
        metricas_ventanas.append({
            "train_start": current_start.strftime('%Y-%m-%d'),
            "train_end": precios_train_window.index[-1].strftime('%Y-%m-%d'),
            "val_start": precios_val_window.index[0].strftime('%Y-%m-%d'),
            "val_end": precios_val_window.index[-1].strftime('%Y-%m-%d'),
            "sharpe_val": sharpe_val,
            "rentabilidad_val": ret_val*100,
            "volatilidad_val": vol_val,
            "capital_final": capital,
            "mejor_label": mejor_label
        })
        current_start += relativedelta(months=window_step_months)
    df_evol = pd.DataFrame({"fecha": fechas_evol, "capital": capital_evol})
    df_evol.to_csv("resultados/evolucion_capital_rolling_train.csv", index=False)
    df_pesos = pd.DataFrame(pesos_ventanas)
    df_pesos.to_csv("resultados/pesos_ventanas_rolling_train.csv", index=False)
    df_metricas = pd.DataFrame(metricas_ventanas)
    df_metricas.to_csv("resultados/metricas_ventanas_rolling_train.csv", index=False)
    print("\nRolling window en train finalizado. Resultados guardados en 'resultados/'.")
    return df_evol, df_metricas

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
    window_step_months = args.window_step_months
    metodo = args.metodo.lower()
    
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
    
    print(f"\nRolling Window Config (solo en train):")
    print(f"  - Train window: {window_train_years} años")
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
        # Dividir en train y validation
        precios = pd.read_csv("data/precios_test.csv", index_col=0, parse_dates=True)
        r_full = precios.pct_change(fill_method=None).dropna()
        r_train, r_val_split = train_test_split(r_full, test_size=Config.VALIDATION_SIZE, shuffle=False)
        # Optimización y ajuste igual que antes
        mejor_sharpe = -np.inf
        mejor_pesos = None
        mejor_label = None
        for beta in [0.1, 1.0, 10.0]:
            for gamma in [0.0, 1.0, 10.0]:
                pesos_pen = optuna_penalizado(mu_train.values, sigma_train.values, 
                                            r_train, r_val_split, K, beta, gamma, n_trials)
                ret, vol, sharpe, _ = calcular_metricas(pesos_pen, r_val_split)
                if sharpe > mejor_sharpe:
                    mejor_sharpe = sharpe
                    mejor_pesos = pesos_pen
                    mejor_label = f"Penalización β={beta}, γ={gamma}"
        # Ajuste de pesos mínimos
        mejor_pesos_ajustados, pesos_ajustados_dict = ajustar_pesos_min_inversion(
            mejor_pesos, tickers, capital_inicial, args.min_inversion
        )
        # Mostrar y guardar pesos
        print(f"\nCapital inicial: {capital_inicial:.2f} €")
        print(f"Archivo de pesos guardado en 'resultados/pesos_optimos_real.csv'")
        # Rolling window en modo REAL
        print("\nValidación rolling window en modo REAL...")
        precios_train = pd.read_csv("data/precios_train.csv", index_col=0, parse_dates=True)
        df_evol_train, df_metricas_train = rolling_window_train(
            precios_train, tickers, window_train_years, window_step_months, 
            capital_inicial, n_trials, min_inversion
        )
        avg_sharpe = df_metricas_train['sharpe_val'].mean()
        print(f"\nSharpe promedio en validación (rolling, modo REAL): {avg_sharpe:.2f}")
        print("\nPesos óptimos para invertir hoy:")
        mostrar_pesos(mejor_pesos_ajustados, tickers, capital_inicial)
        if np.allclose(mejor_pesos_ajustados, 0):
            print("\nNo hay activos seleccionados tras el ajuste de inversión mínima. Prueba con menos restricción o más capital.")
        pd.DataFrame({"Ticker": tickers, "Peso": mejor_pesos_ajustados}).to_csv(
            "resultados/pesos_optimos_real.csv", index=False
        )
        return
    
    fecha_inicio=time.time()

    # Descargar datos
    download_data(symbols, train_start, train_end, test_start, test_end, interval)
    
    # Cargar datos procesados
    mu_train, sigma_train, r_test, tickers = cargar_datos()
    N = len(mu_train)
    K = min(10, N)
    
    # Dividir train en train y validation
    precios_train = pd.read_csv("data/precios_train.csv", index_col=0, parse_dates=True)
    r_full_train = precios_train.pct_change(fill_method=None).dropna()
    r_train, r_val = train_test_split(r_full_train, test_size=Config.VALIDATION_SIZE, shuffle=False)
    
    # === OPTIMIZACIÓN Y EVALUACIÓN ===
    print("\n" + "="*50)
    print("OPTIMIZANDO CARTERA")
    print("="*50)
    
    # Primero validación con rolling window en train
    print("\nRealizando validación con rolling window en datos de entrenamiento...")
    df_evol_train, df_metricas_train = rolling_window_train(
        precios_train, tickers, window_train_years, window_step_months, 
        capital_inicial, n_trials, min_inversion
    )
    
    # Analizar métricas de las ventanas para seleccionar mejores parámetros
    avg_sharpe = df_metricas_train['sharpe_val'].mean()
    print(f"\nSharpe promedio en validación (rolling): {avg_sharpe:.2f}")
    
    # Optimización final con todos los datos de entrenamiento
    print("\nOptimizando con todos los datos de entrenamiento...")
    mu_final = r_full_train.mean()
    sigma_final = r_full_train.cov()
    
    resultados = []
    mejor_sharpe = -np.inf
    mejor_pesos = None
    mejor_label = None
    
    if metodo == 'proyeccion':
        print("\nOptimizando con método de proyección...")
        pesos_proj = optuna_proyeccion(mu_final.values, sigma_final.values, 
                                      r_train, r_val, K, n_trials)
        rent_p, vol_p, sharpe_p, serie_proj = calcular_metricas(pesos_proj, r_test)
        resultados.append(["proyeccion", None, None, rent_p*100, vol_p, sharpe_p])
        mejor_sharpe = sharpe_p
        mejor_pesos = pesos_proj
        mejor_label = "Proyección externa"
    elif metodo.startswith('penalizacion_'):
        try:
            _, beta_str, gamma_str = metodo.split('_')
            beta = float(beta_str)
            gamma = float(gamma_str)
        except Exception as e:
            print("Error en el formato de --metodo para penalización. Usa: penalizacion_BETA_GAMMA, ej: penalizacion_10.0_0.0")
            return
        print(f"\nOptimizando con método penalizado (beta={beta}, gamma={gamma})...")
        pesos_pen = optuna_penalizado(mu_final.values, sigma_final.values, 
                                     r_train, r_val, K, beta, gamma, n_trials)
        rent, vol, sharpe, serie = calcular_metricas(pesos_pen, r_test)
        label = f"Penalización β={beta}, γ={gamma}"
        resultados.append(["penalizacion", beta, gamma, rent*100, vol, sharpe])
        mejor_sharpe = sharpe
        mejor_pesos = pesos_pen
        mejor_label = label
    elif metodo == 'averiguate':
        print("\nProbando todos los métodos para averiguar el mejor...")
        # Proyección
        pesos_proj = optuna_proyeccion(mu_final.values, sigma_final.values, 
                                      r_train, r_val, K, n_trials)
        rent_p, vol_p, sharpe_p, serie_proj = calcular_metricas(pesos_proj, r_test)
        resultados.append(["proyeccion", None, None, rent_p*100, vol_p, sharpe_p])
        if sharpe_p > mejor_sharpe:
            mejor_sharpe = sharpe_p
            mejor_pesos = pesos_proj
            mejor_label = "Proyección externa"
        # Penalizaciones
        for beta in [0.1, 1.0, 10.0]:
            for gamma in [0.0, 1.0, 10.0]:
                print(f"  - Probando penalización beta={beta}, gamma={gamma}")
                pesos_pen = optuna_penalizado(mu_final.values, sigma_final.values, 
                                             r_train, r_val, K, beta, gamma, n_trials)
                rent, vol, sharpe, serie = calcular_metricas(pesos_pen, r_test)
                label = f"Penalización β={beta}, γ={gamma}"
                resultados.append(["penalizacion", beta, gamma, rent*100, vol, sharpe])
                if sharpe > mejor_sharpe:
                    mejor_sharpe = sharpe
                    mejor_pesos = pesos_pen
                    mejor_label = label
        print(f"\nEl mejor método es: {mejor_label} (Sharpe={mejor_sharpe:.3f})")
    else:
        print("\nMétodo no reconocido. Usando penalización por defecto (beta=10, gamma=0)")
        pesos_pen = optuna_penalizado(mu_final.values, sigma_final.values, 
                                    r_train, r_val, K, 10.0, 0.0, n_trials)
        rent, vol, sharpe, serie = calcular_metricas(pesos_pen, r_test)
        label = "Penalización β=10.0, γ=0.0"
        resultados.append(["penalizacion", 10.0, 0.0, rent*100, vol, sharpe])
        mejor_sharpe = sharpe
        mejor_pesos = pesos_pen
        mejor_label = label
    
    # Resultados
    df_resultados = pd.DataFrame(resultados, 
                               columns=["metodo", "beta", "gamma", "rentabilidad %", "volatilidad", "sharpe"])
    print("\nResumen de resultados en test:")
    print(df_resultados)
    df_resultados.to_csv("resultados/resultados_test.csv", index=False)
    
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
    rent_ajust, vol_ajust, sharpe_ajust, serie_ajust = calcular_metricas(mejor_pesos_ajustados, r_test)
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
    capital_series = (1 + serie_ajust).cumprod() * capital_inicial
    plot_evolution(capital_series, mejor_label, "evolucion_capital_test")
    
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

    print(f"\nTiempo de ejecución: {time.time() - fecha_inicio:.2f} segundos")
if __name__ == "__main__":
    main()