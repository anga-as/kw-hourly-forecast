# Energy Consumption Forecast (UCI Dataset) — `p2cargas.py`

This project predicts **hourly** household electricity consumption using the UCI *“Individual household electric power consumption”* dataset.  
It demonstrates an end-to-end data science + machine learning pipeline: from raw data to model training, evaluation, and visualization.

---

## 📂 Project Overview

- **Preprocessing:** load the raw file, clean missing values (`?`), and resample from **1-minute to 1-hour** averages.  
- **Feature engineering:** create **lag features** (last *n* hours, default 24) to predict the **next hour**.  
- **Models:**
  - Linear Regression (baseline)
  - RandomForestRegressor (non-linear)
- **Evaluation:** MAE, RMSE, and MAPE.  
- **Outputs:** predictions CSV, metrics text file, and comparison plots.

---

## 🗃 Dataset

- **Source:** UCI Machine Learning Repository  
- **Original file:** `household_power_consumption.txt` (~20 MB)  
- **Period:** 2006–2010  

- **Target variable:** `Global_active_power` (kW)

> You can keep any filename locally (e.g., `dataset.txt`) and pass it via `--data`.

---

## ⚙️ Installation

Create a virtual environment and install dependencies from `requirements.txt`.

**Using `pip`:**
```bash
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
pip install -r requirements.txt end
```


## ▶️ Usage

Run the script pointing to your data file:
```bash
python p2cargas.py --data dataset.txt
```
Optional arguments:

--results → output directory (default: results/)

--n-lags → number of previous hours used as features (default: 24)

--test-size → test split proportion (default: 0.2)
## Example
```
# 24 lags (1 day), results in ./results_lag24
python p2cargas.py --data dataset.txt --n-lags 24 --results results_lag24

# 48 lags (2 days), results in ./results_lag48
python p2cargas.py --data dataset.txt --n-lags 48 --results results_lag48
```

## Outputs

The following files are created under --results:

Hourly_KW/Hourly_KW.csv → hourly series of Global_active_power.

predictions_test.csv → actual vs predicted values (datetime index).

actual_vs_pred_last15d.png → comparison plot for the last 15 days of the test set.

metrics.txt → MAE, RMSE, MAPE for both models.

## Example Plot

Predicted vs Actual — Global Active Power (last 15 days)
Shows three lines: ACTUAL, Linear Regression, and RandomForest over the final two weeks of the test period.

## Notes

Linear Regression serves as a simple baseline.

Random Forest captures non-linear patterns and typically performs better.

Future work: try Gradient Boosting, ARIMA/Prophet, or LSTM/Transformers for time-series forecasting


## License

MIT License.
