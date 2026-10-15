# Hourly Household Power Forecast

**Energy data · Time-series regression · Python / scikit-learn**

[Engineering portfolio](https://github.com/anga-as) · [Source code](P2cargas.py) · [Saved metrics](results%20example/metrics.txt)

An end-to-end study of **one-step-ahead hourly mean active power** using the UCI Individual Household Electric Power Consumption dataset. Previous hourly measurements are used to predict the next hour, comparing linear regression with a random forest.

![Actual and predicted power over the last 15 days of the saved test period](results%20example/actual_vs_pred_last15d.png)

## Pipeline

1. Read semicolon-separated minute data and parse `Date` / `Time`.
2. Convert missing markers to NaN, sort chronologically and forward-fill values.
3. Average `Global_active_power` into hourly **kW** values.
4. Construct lagged inputs, using 24 previous hours by default.
5. Split chronologically, retaining the last 20% as the default test set.
6. Fit linear regression and a 500-tree random forest; save predictions, metrics and a comparison plot.

The target is mean active **power in kW**. It is not an accumulated energy series in kWh.

## Run

From the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows Command Prompt, or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install -r requirements.txt
python P2cargas.py --data path/to/household_power_consumption.txt --results results
```

The raw minute dataset is not bundled. Supply a local file in the original UCI format, with `Date`, `Time` and `Global_active_power` columns. The command uses the exact, case-sensitive filename **`P2cargas.py`**.

```bash
python P2cargas.py --data path/to/household_power_consumption.txt --n-lags 48 --test-size 0.2 --results results_lag48
```

The saved hourly CSV is an output of preprocessing; it cannot be passed directly to the raw-data loader.

## Saved example results

The values below are transcribed from the repository's existing `metrics.txt`; they have not been recomputed from the full UCI dataset for this documentation update.

| Model | MAE (kW) | RMSE (kW) | MAPE |
|---|---:|---:|---:|
| Linear regression | 0.347206 | 0.492878 | 47.8623% |
| Random forest | 0.327448 | 0.481410 | 42.9354% |

Random forest has lower error in this saved comparison. This does not establish superiority across other households, time periods or validation schemes.

## Outputs

| File under `--results` | Contents |
|---|---|
| `Hourly_kW/Hourly_kW.csv` | Hourly mean active-power series |
| `predictions_test.csv` | Actual, linear-regression and random-forest values |
| `metrics.txt` | MAE, RMSE and MAPE |
| `actual_vs_pred_last15d.png` | Final 15 days of the test period |

[Browse the existing examples](results%20example)

## Evaluation limits

- Test inputs include prior **observed** values, including observations within the test period. This represents sequential one-step forecasting, not a recursive 15-day forecast.
- MAPE can become large at low power values; interpret it alongside MAE and RMSE.
- Forward-filling can smooth missing periods. Leading gaps or remaining NaNs require input inspection.
- The current study uses one chronological holdout. Persistence baselines and walk-forward validation are useful next steps, not implemented results.
- Dependency versions are lower bounds, not a fully pinned environment. The full training run requires the external raw dataset.

### En español

Pronóstico de potencia activa media de la siguiente hora a partir de mediciones anteriores. El proyecto incluye procesamiento de datos, dos modelos y evaluación cronológica. Las métricas corresponden al ejemplo guardado en este repositorio.
