import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")


# 0. SETARI GENERALE

os.makedirs("output", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"


# 1. INCARCARE DATE

# Bitcoin
btc = pd.read_csv("data/processed/Bitcoin.csv", sep=";")
btc["date"] = pd.to_datetime(btc["timeOpen"].str[:10])

btc = btc[["date", "close", "marketCap"]].rename(columns={
    "close": "btc_close",
    "marketCap": "btc_marketcap"
})

btc = btc[
    (btc["date"] >= START_DATE) &
    (btc["date"] <= END_DATE)
].copy()

btc = btc.sort_values("date").reset_index(drop=True)


# Gold
gold = pd.read_csv("data/processed/gold.csv")
gold["date"] = pd.to_datetime(gold["date"])

gold = gold[
    (gold["date"] >= START_DATE) &
    (gold["date"] <= END_DATE)
].copy()

gold = gold.sort_values("date").reset_index(drop=True)


# S&P 500
sp = pd.read_csv("data/processed/S&P 500.csv")
sp["date"] = pd.to_datetime(sp["Date"], format="%m/%d/%Y")

sp["sp500_price"] = (
    sp["Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

sp = sp[["date", "sp500_price"]]

sp = sp[
    (sp["date"] >= START_DATE) &
    (sp["date"] <= END_DATE)
].copy()

sp = sp.sort_values("date").reset_index(drop=True)


# Inflatie
inf = pd.read_csv("data/processed/inflation_2015_2025.csv")
inf["date"] = pd.to_datetime(inf["date"])

inf = inf[
    (inf["date"] >= START_DATE) &
    (inf["date"] <= END_DATE)
].copy()

inf = inf.sort_values("date").reset_index(drop=True)


# 2. UNIRE DATASETURI
df = btc.merge(gold, on="date", how="inner")
df = df.merge(sp, on="date", how="inner")
df = df.merge(inf, on="date", how="inner")

df = df.sort_values("date").reset_index(drop=True)

print("=" * 80)
print("EDA - DATASET FINAL")
print("=" * 80)
print("Perioada:", df["date"].min(), "->", df["date"].max())
print("Numar observatii:", len(df))
print("\nValori lipsa:")
print(df.isna().sum())


# 3. TRANSFORMARI

# Logaritmi
df["log_btc"] = np.log(df["btc_close"])
df["log_gold"] = np.log(df["gold_price"])
df["log_sp500"] = np.log(df["sp500_price"])
df["log_cpi"] = np.log(df["CPIAUCSL"])

# Randamente logaritmice lunare
df["btc_return"] = df["log_btc"].diff() * 100
df["gold_return"] = df["log_gold"].diff() * 100
df["sp500_return"] = df["log_sp500"].diff() * 100
df["cpi_return"] = df["log_cpi"].diff() * 100

# Indici baza 100, utili pentru compararea vizuala a activelor
df["btc_index"] = df["btc_close"] / df["btc_close"].iloc[0] * 100
df["gold_index"] = df["gold_price"] / df["gold_price"].iloc[0] * 100
df["sp500_index"] = df["sp500_price"] / df["sp500_price"].iloc[0] * 100

df.to_csv("output/tables/eda_dataset.csv", index=False)


# 4. GRAFIC BITCOIN IN NIVEL

plt.figure(figsize=(11, 5))
plt.plot(df["date"], df["btc_close"], linewidth=1.8)
plt.title("Evolutia lunara a pretului Bitcoin")
plt.xlabel("Data")
plt.ylabel("Pret Bitcoin, USD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/eda_btc_price.png", dpi=150)
plt.close()


# 5. COMPARATIE BTC, GOLD, S&P 500, BAZA 100

plt.figure(figsize=(11, 5))

plt.plot(df["date"], df["btc_index"], label="Bitcoin", linewidth=1.8)
plt.plot(df["date"], df["gold_index"], label="Aur", linewidth=1.8)
plt.plot(df["date"], df["sp500_index"], label="S&P 500", linewidth=1.8)

plt.yscale("log")

plt.title("Comparatie intre Bitcoin, aur si S&P 500, indice baza 100, scara logaritmica")
plt.xlabel("Data")
plt.ylabel("Indice, prima luna = 100, scara logaritmica")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("output/figures/eda_all_prices_indexed_logscale.png", dpi=150)
plt.close()

# 6. GRAFICE RANDAMENTE LOGARITMICE

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

axes[0].plot(df["date"], df["btc_return"], linewidth=1.4)
axes[0].set_title("Randamente lunare Bitcoin")
axes[0].set_ylabel("Randament (%)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["date"], df["gold_return"], linewidth=1.4)
axes[1].set_title("Randamente lunare aur")
axes[1].set_ylabel("Randament (%)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(df["date"], df["sp500_return"], linewidth=1.4)
axes[2].set_title("Randamente lunare S&P 500")
axes[2].set_xlabel("Data")
axes[2].set_ylabel("Randament (%)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/figures/eda_returns.png", dpi=150)
plt.close()


# 7. GRAFIC INFLATIE

plt.figure(figsize=(11, 5))
plt.plot(df["date"], df["inflation_yoy"], linewidth=1.8)
plt.title("Evolutia inflatiei anuale in SUA")
plt.xlabel("Data")
plt.ylabel("Inflatie YoY (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/eda_inflation.png", dpi=150)
plt.close()


# 8. STATISTICI DESCRIPTIVE

eda_vars = [
    "btc_return",
    "gold_return",
    "sp500_return",
    "inflation_yoy"
]

desc = df[eda_vars].describe().T
desc["skewness"] = df[eda_vars].skew()
desc["kurtosis"] = df[eda_vars].kurtosis()

desc = desc.rename(columns={
    "count": "Numar observatii",
    "mean": "Medie",
    "std": "Abatere standard",
    "min": "Minim",
    "25%": "Q1",
    "50%": "Mediana",
    "75%": "Q3",
    "max": "Maxim",
    "skewness": "Asimetrie",
    "kurtosis": "Kurtosis"
})

desc.to_csv("output/tables/eda_descriptive_statistics.csv")

print("\nStatistici descriptive:")
print(desc)


# 9. MATRICE DE CORELATIE

corr_vars = [
    "btc_return",
    "gold_return",
    "sp500_return",
    "inflation_yoy"
]

corr = df[corr_vars].corr()
corr.to_csv("output/tables/eda_correlation_matrix.csv")

plt.figure(figsize=(7, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Coeficient de corelatie")

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)

for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.title("Matricea de corelatie")
plt.tight_layout()
plt.savefig("output/figures/eda_correlation_matrix.png", dpi=150)
plt.close()

print("\nMatrice de corelatie:")
print(corr)


# 10. ACF SI PACF PENTRU BITCOIN

btc_return_clean = df["btc_return"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

plot_acf(
    btc_return_clean,
    lags=24,
    ax=axes[0],
    zero=False
)
axes[0].set_title("ACF pentru randamentele Bitcoin")

plot_pacf(
    btc_return_clean,
    lags=24,
    ax=axes[1],
    method="ywm",
    zero=False
)
axes[1].set_title("PACF pentru randamentele Bitcoin")

plt.tight_layout()
plt.savefig("output/figures/eda_btc_acf_pacf.png", dpi=150)
plt.close()


# 11. REZUMAT AUTOMAT PENTRU ORIENTARE

btc_std = desc.loc["btc_return", "Abatere standard"]
gold_std = desc.loc["gold_return", "Abatere standard"]
sp_std = desc.loc["sp500_return", "Abatere standard"]

corr_btc_gold = corr.loc["btc_return", "gold_return"]
corr_btc_sp = corr.loc["btc_return", "sp500_return"]
corr_btc_inf = corr.loc["btc_return", "inflation_yoy"]

print("\n" + "=" * 80)
print("REZUMAT EDA")
print("=" * 80)

print(f"Volatilitate BTC: {btc_std:.4f}")
print(f"Volatilitate aur: {gold_std:.4f}")
print(f"Volatilitate S&P 500: {sp_std:.4f}")

print(f"Corelatie BTC - aur: {corr_btc_gold:.4f}")
print(f"Corelatie BTC - S&P 500: {corr_btc_sp:.4f}")
print(f"Corelatie BTC - inflatie: {corr_btc_inf:.4f}")

if btc_std > gold_std and btc_std > sp_std:
    print("Observatie: Bitcoin are cea mai ridicata volatilitate dintre activele analizate.")

if abs(corr_btc_sp) > abs(corr_btc_gold):
    print("Observatie: Bitcoin este mai corelat cu S&P 500 decat cu aurul.")
else:
    print("Observatie: Bitcoin este mai corelat cu aurul decat cu S&P 500.")

print("\nFisiere generate:")
print("output/figures/eda_btc_price.png")
print("output/figures/eda_all_prices_indexed.png")
print("output/figures/eda_returns.png")
print("output/figures/eda_inflation.png")
print("output/figures/eda_correlation_matrix.png")
print("output/figures/eda_btc_acf_pacf.png")
print("output/tables/eda_descriptive_statistics.csv")
print("output/tables/eda_correlation_matrix.csv")
print("output/tables/eda_dataset.csv")

print("\nGata.")