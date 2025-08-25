import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
from pathlib import Path
import sys

# df = load_dataset("dataset.txt")
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path,
                     sep=';',
                     na_values='?',
                     low_memory=False)

   
    df['datetime'] = pd.to_datetime(
        df['Date'].astype(str) + ' ' + df['Time'].astype(str),
        format="%d/%m/%Y %H:%M:%S",   
        errors='coerce'
    )

   
    df.drop(columns=['Date', 'Time'] , inplace = True)
    df.set_index('datetime', inplace=True)

    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

 
    df = df.sort_index().ffill()

    return df



def GAPH(df: pd.DataFrame) -> pd.Series:
    
    if 'Global_active_power' not in df.columns:
        raise KeyError('Global active column not found ')
    
    hourly= df['Global_active_power'].resample("h").mean()
    return hourly

def data_par(series: pd.Series, n_lags: int= 24):
    s= series.copy().astype(float)
    dfx= pd.DataFrame({f"lag_{i}": s.shift(i)for i in range (1, n_lags+1)}, index = s.index)
    dfx= dfx.dropna()
    y= s.loc[dfx.index]
    return dfx, y 


def P(y_true: pd.Series, y_predict: pd.Series, name:str):
    
    mae= mean_absolute_error(y_true,y_predict)
    rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    eps=1e-10
    mape= (np.abs((y_true -y_predict)/ np.clip(np.abs(y_true), eps, None))).mean() *100
    print(f"{name:>14} -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
    return mae, rmse, mape

def G(y_test: pd.Series, lin_pred: pd.Series, rf_pred: pd.Series, outpath: Path):
    
    N= 24 * 15
    plt.figure(figsize=(10,5))
    y_test.tail(N).plot(label="ACTUAL")
    lin_pred.tail(N).plot(label= "Linear Regression")
    rf_pred.tail(N).plot(label="RandomForest")
    plt.legend()
    plt.title("ACTUAL VS PREDICTION")
    plt.xlabel("Datetime")
    plt.ylabel("Global Active Power (kW)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    
    
def main():
    parser = argparse.ArgumentParser(description="Energy Forecast (UCI) — end-to-end")
    parser.add_argument("--data", type=str, required=True,
                        help="Ruta al archivo : data/household_power_consumption.txt")
    parser.add_argument("--results", type=str, default="results",
                        help="Directorio para guardar salidas (predicciones, gráficos, métricas).")
    parser.add_argument("--n-lags", type=int, default=24, help="Número de lags (horas previas).")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporción del set de prueba (0-1).")
    args = parser.parse_args()
    
    data_load= Path(args.data)
    results_dir=Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Cargando y limpiando Datos")
    df= load_dataset(data_load)
    print(f"Shape (minutos): {df.shape}")
    
    
    print("Obteniendo kW por hora")
    hourly = GAPH(df)
    hourly_dir = results_dir / "Hourly_kW"
    hourly_dir.mkdir(parents=True, exist_ok=True)

    hourly_file = hourly_dir / "Hourly_kW.csv"
    hourly.to_csv(hourly_file, index=True)

    
    X,y = data_par(hourly, n_lags= args.n_lags)
    
    test_size = args.test_size
    if not (0.0 < test_size< 1):
        print (" --test size  shoul be between 0,1 . Using 0.2 by default ")
        test_size= 0.2
    split_idx = int (len (X) * (1.0-test_size))
    XTR, XTE = X.iloc[:split_idx], X.iloc[split_idx:]
    yTR, yTE = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train: {len(XTR)} muestras | Test: {len(XTE)} muestras")
    

    print("Entrenando modelos...")
    lin = LinearRegression().fit(XTR, yTR)
    rf = RandomForestRegressor(n_estimators=500, random_state=7, n_jobs=-1).fit(XTR, yTR)

    lin_p = pd.Series(lin.predict(XTE), index=yTE.index, name="lin_pred")
    rf_p  = pd.Series(rf.predict(XTE),  index=yTE.index, name="rf_pred")   
    
    print("\nEvaluación (conjunto de prueba):")
    lin_m = P(yTE, lin_p, "LinearRegression")
    rf_m  = P(yTE, rf_p,  "RandomForest")
    preds_o = results_dir / "predictions_test.csv"
    pd.concat([yTE.rename("actual"), lin_p, rf_p], axis=1).to_csv(preds_o, index=True)
    print(f"Predicciones guardadas en: {preds_o}")
    
    fig_o = results_dir / "actual_vs_pred_last15d.png"
    G(yTE, lin_p, rf_p, fig_o)
    print(f"Gráfica guardada en: {fig_o}")
  
    met_o = results_dir / "metrics.txt"
    with open(met_o, "w", encoding="utf-8") as f:
        f.write("Metrics (test set)\n")
        f.write("------------------\n")
        f.write(f"LinearRegression -> MAE: {lin_m[0]:.6f} | RMSE: {lin_m[1]:.6f} | MAPE: {lin_m[2]:.4f}%\n")
        f.write(f"RandomForest     -> MAE: {rf_m[0]:.6f} | RMSE: {rf_m[1]:.6f} | MAPE: {rf_m[2]:.4f}%\n")
    print(f"Métricas guardadas en: {met_o}\n")
    
    
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(" Error:", e, file=sys.stderr)
        sys.exit(1)