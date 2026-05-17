import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

DATA_PATH = "output/tables/dataset_cu_transformari.csv"

TEST_SIZE = 24  # ultimele 24 luni pentru test


# 1. INCARCARE DATASET

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Folosim seria logaritmata a pretului Bitcoin
df["log_btc"] = np.log(df["btc_close"])

# Setam data ca index
df = df.set_index("date")

# Frecventa lunara. MS = month start
df = df.asfreq("MS")

series = df["log_btc"].dropna()

print("=" * 80)
print("MODELE UNIVARIATE PENTRU BITCOIN")
print("=" * 80)
print("Perioada:", series.index.min(), "->", series.index.max())
print("Numar observatii:", len(series))


# 2. TRAIN / TEST SPLIT

train = series.iloc[:-TEST_SIZE]
test = series.iloc[-TEST_SIZE:]

print("\nTrain:", train.index.min(), "->", train.index.max(), "| Observatii:", len(train))
print("Test:", test.index.min(), "->", test.index.max(), "| Observatii:", len(test))


# Grafic train-test
plt.figure(figsize=(11, 5))
plt.plot(train.index, train, label="Training", linewidth=1.8)
plt.plot(test.index, test, label="Test", linewidth=1.8)
plt.title("Impartirea seriei log(Bitcoin) in set de training si test")
plt.xlabel("Data")
plt.ylabel("log(Bitcoin)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/univariate_train_test.png", dpi=150)
plt.close()


# 3.FUNCTII METRICI

def mae(actual, forecast):
    return np.mean(np.abs(actual - forecast))


def rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))


def mape(actual, forecast):
    actual_price = np.exp(actual)
    forecast_price = np.exp(forecast)
    return np.mean(np.abs((actual_price - forecast_price) / actual_price)) * 100


# 4. MODEL 1: HOLT EXPONENTIAL SMOOTHING

# Holt: model cu trend, fara sezonalitate.
# Pe date lunare Bitcoin, sezonalitatea nu este neaparat stabila,
# de aceea folosim trend aditiv.
holt_model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal=None,
    initialization_method="estimated"
)

holt_fit = holt_model.fit(optimized=True)
holt_forecast = holt_fit.forecast(steps=len(test))

holt_mae = mae(test, holt_forecast)
holt_rmse = rmse(test, holt_forecast)
holt_mape = mape(test, holt_forecast)

print("\n" + "=" * 80)
print("MODEL HOLT / EXPONENTIAL SMOOTHING")
print("=" * 80)
print("MAE:", round(holt_mae, 4))
print("RMSE:", round(holt_rmse, 4))
print("MAPE:", round(holt_mape, 2), "%")


# Grafic Holt
plt.figure(figsize=(11, 5))
plt.plot(train.index, train, label="Training", linewidth=1.5)
plt.plot(test.index, test, label="Valori reale test", linewidth=1.8)
plt.plot(test.index, holt_forecast, label="Forecast Holt", linewidth=1.8)

plt.title("Prognoza log(Bitcoin) folosind Holt Exponential Smoothing")
plt.xlabel("Data")
plt.ylabel("log(Bitcoin)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/univariate_holt_forecast.png", dpi=150)
plt.close()


# 5. MODEL 2: ARIMA

# ARIMA(1,1,1) pe log(Bitcoin)
# d=1 este justificat de testele de stationaritate.
arima_order = (1, 1, 1)

arima_model = ARIMA(
    train,
    order=arima_order
)

arima_fit = arima_model.fit()

arima_forecast_result = arima_fit.get_forecast(steps=len(test))
arima_forecast = arima_forecast_result.predicted_mean
arima_conf_int = arima_forecast_result.conf_int(alpha=0.05)

arima_mae = mae(test, arima_forecast)
arima_rmse = rmse(test, arima_forecast)
arima_mape = mape(test, arima_forecast)

print("\n" + "=" * 80)
print("MODEL ARIMA")
print("=" * 80)
print("Ordin ARIMA:", arima_order)
print("AIC:", round(arima_fit.aic, 4))
print("BIC:", round(arima_fit.bic, 4))
print("MAE:", round(arima_mae, 4))
print("RMSE:", round(arima_rmse, 4))
print("MAPE:", round(arima_mape, 2), "%")


# Grafic ARIMA cu interval de incredere
plt.figure(figsize=(11, 5))
plt.plot(train.index, train, label="Training", linewidth=1.5)
plt.plot(test.index, test, label="Valori reale test", linewidth=1.8)
plt.plot(test.index, arima_forecast, label="Forecast ARIMA(1,1,1)", linewidth=1.8)

plt.fill_between(
    test.index,
    arima_conf_int.iloc[:, 0],
    arima_conf_int.iloc[:, 1],
    alpha=0.2,
    label="Interval incredere 95%"
)

plt.title("Prognoza log(Bitcoin) folosind ARIMA(1,1,1)")
plt.xlabel("Data")
plt.ylabel("log(Bitcoin)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/univariate_arima_forecast.png", dpi=150)
plt.close()


# 6. COMPARATIE FORECAST HOLT VS ARIMA

plt.figure(figsize=(11, 5))
plt.plot(test.index, test, label="Valori reale test", linewidth=2.0)
plt.plot(test.index, holt_forecast, label="Forecast Holt", linewidth=1.8)
plt.plot(test.index, arima_forecast, label="Forecast ARIMA(1,1,1)", linewidth=1.8)

plt.title("Compararea prognozelor univariate pentru log(Bitcoin)")
plt.xlabel("Data")
plt.ylabel("log(Bitcoin)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/univariate_forecast_comparison.png", dpi=150)
plt.close()


# 7. EXPORT TABELE

accuracy = pd.DataFrame({
    "Model": [
        "Holt Exponential Smoothing",
        "ARIMA(1,1,1)"
    ],
    "MAE": [
        holt_mae,
        arima_mae
    ],
    "RMSE": [
        holt_rmse,
        arima_rmse
    ],
    "MAPE_percent": [
        holt_mape,
        arima_mape
    ]
})

accuracy.to_csv("output/tables/univariate_accuracy.csv", index=False)

forecasts = pd.DataFrame({
    "date": test.index,
    "actual_log_btc": test.values,
    "holt_forecast_log_btc": holt_forecast.values,
    "arima_forecast_log_btc": arima_forecast.values,
    "actual_btc_price": np.exp(test.values),
    "holt_forecast_price": np.exp(holt_forecast.values),
    "arima_forecast_price": np.exp(arima_forecast.values)
})

forecasts.to_csv("output/tables/univariate_forecasts.csv", index=False)


print("\n" + "=" * 80)
print("COMPARATIE ACURATETE")
print("=" * 80)
print(accuracy)

best_model_rmse = accuracy.loc[accuracy["RMSE"].idxmin(), "Model"]
best_model_mape = accuracy.loc[accuracy["MAPE_percent"].idxmin(), "Model"]

print("\nCel mai bun model dupa RMSE:", best_model_rmse)
print("Cel mai bun model dupa MAPE:", best_model_mape)

print("\nFisiere generate:")
print("output/figures/univariate_train_test.png")
print("output/figures/univariate_holt_forecast.png")
print("output/figures/univariate_arima_forecast.png")
print("output/figures/univariate_forecast_comparison.png")
print("output/tables/univariate_accuracy.csv")
print("output/tables/univariate_forecasts.csv")