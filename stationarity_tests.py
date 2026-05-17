import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss

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

# 6,845.50  ->  6845.50
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


# 2. MERGE DATASETURI

df = btc.merge(gold, on="date", how="inner")
df = df.merge(sp, on="date", how="inner")
df = df.merge(inf, on="date", how="inner")

df = df.sort_values("date").reset_index(drop=True)

print("=" * 80)
print("DATASET FINAL")
print("=" * 80)
print("Perioada:", df["date"].min(), "->", df["date"].max())
print("Numar observatii:", len(df))
print("\nPrimele randuri:")
print(df.head())
print("\nUltimele randuri:")
print(df.tail())
print("\nValori lipsa:")
print(df.isna().sum())


# 3. DEFINIRE SERII

# Logaritmi pentru seriile in nivel
df["log_btc"] = np.log(df["btc_close"])
df["log_gold"] = np.log(df["gold_price"])
df["log_sp500"] = np.log(df["sp500_price"])
df["log_cpi"] = np.log(df["CPIAUCSL"])

# Prima diferenta a logaritmilor = aproximativ randament lunar
df["dlog_btc"] = df["log_btc"].diff()
df["dlog_gold"] = df["log_gold"].diff()
df["dlog_sp500"] = df["log_sp500"].diff()
df["dlog_cpi"] = df["log_cpi"].diff()

# Diferenta inflatiei anuale
df["dinf_yoy"] = df["inflation_yoy"].diff()

# Dictionare pentru raport
serii_nivel = {
    "log_btc": "log(Bitcoin)",
    "log_gold": "log(Aur)",
    "log_sp500": "log(S&P 500)",
    "log_cpi": "log(CPI)",
    "inflation_yoy": "Inflatie YoY"
}

serii_diferenta = {
    "dlog_btc": "Dlog(Bitcoin)",
    "dlog_gold": "Dlog(Aur)",
    "dlog_sp500": "Dlog(S&P 500)",
    "dlog_cpi": "Dlog(CPI)",
    "dinf_yoy": "DInflatie YoY"
}


# 4. FUNCTII TESTE ADF SI KPSS

def run_adf(series, regression="ct"):
    """
    Test ADF.
    H0: seria are radacina unitara, deci este non-stationara.
    H1: seria este stationara.

    regression:
    - "ct" = constanta + trend
    - "c"  = constanta
    - "n"  = fara constanta
    """

    s = series.dropna()

    result = adfuller(s, autolag="AIC", regression=regression)

    stat = result[0]
    pval = result[1]
    lags_used = result[2]
    critical_values = result[4]

    reject_h0 = pval < 0.05

    return {
        "ADF stat": round(stat, 4),
        "ADF p-value": round(pval, 4),
        "ADF lags": lags_used,
        "ADF VC 1%": round(critical_values["1%"], 3),
        "ADF VC 5%": round(critical_values["5%"], 3),
        "ADF VC 10%": round(critical_values["10%"], 3),
        "ADF decision": "STATIONARA" if reject_h0 else "NON-STATIONARA"
    }


def run_kpss(series, regression="ct"):
    """
    Test KPSS.
    H0: seria este stationara.
    H1: seria este non-stationara.

    regression:
    - "ct" = trend stationar
    - "c"  = nivel stationar
    """

    s = series.dropna()

    stat, pval, lags_used, critical_values = kpss(
        s,
        regression=regression,
        nlags="auto"
    )

    # La KPSS respingem H0 daca statistica este mai mare decat valoarea critica.
    reject_h0 = stat > critical_values["5%"]

    return {
        "KPSS stat": round(stat, 4),
        "KPSS p-value": round(pval, 4),
        "KPSS lags": lags_used,
        "KPSS VC 1%": round(critical_values["1%"], 3),
        "KPSS VC 5%": round(critical_values["5%"], 3),
        "KPSS VC 10%": round(critical_values["10%"], 3),
        "KPSS decision": "NON-STATIONARA" if reject_h0 else "STATIONARA"
    }


def interpret_combined(adf_decision, kpss_decision):
    """
    Interpretare combinata ADF + KPSS.

    ADF:
    - STATIONARA inseamna ca respingem H0 de radacina unitara.

    KPSS:
    - STATIONARA inseamna ca nu respingem H0 de stationaritate.
    """

    adf_stationary = adf_decision == "STATIONARA"
    kpss_stationary = kpss_decision == "STATIONARA"

    if adf_stationary and kpss_stationary:
        return "I(0) - stationara"
    elif not adf_stationary and not kpss_stationary:
        return "I(1) - non-stationara, necesita diferentiere"
    elif adf_stationary and not kpss_stationary:
        return "Ambiguu - posibil trend stationar sau specificatie diferita"
    else:
        return "Ambiguu - verificati specificatia testelor"


# 5. RULARE TESTE PENTRU SERII IN NIVEL

print("\n" + "=" * 80)
print("TESTE DE STATIONARITATE - SERII IN NIVEL")
print("Specificatie: constanta + trend, regression='ct'")
print("=" * 80)

rezultate_nivel = []

for col, label in serii_nivel.items():
    adf_res = run_adf(df[col], regression="ct")
    kpss_res = run_kpss(df[col], regression="ct")

    concluzie = interpret_combined(
        adf_res["ADF decision"],
        kpss_res["KPSS decision"]
    )

    print("\n" + "-" * 70)
    print(label)
    print("-" * 70)
    print(
        "ADF  | Statistica:",
        adf_res["ADF stat"],
        "| p-value:",
        adf_res["ADF p-value"],
        "| Decizie:",
        adf_res["ADF decision"]
    )
    print(
        "ADF  | Valori critice:",
        "1% =", adf_res["ADF VC 1%"],
        "| 5% =", adf_res["ADF VC 5%"],
        "| 10% =", adf_res["ADF VC 10%"]
    )
    print(
        "KPSS | Statistica:",
        kpss_res["KPSS stat"],
        "| p-value:",
        kpss_res["KPSS p-value"],
        "| Decizie:",
        kpss_res["KPSS decision"]
    )
    print(
        "KPSS | Valori critice:",
        "1% =", kpss_res["KPSS VC 1%"],
        "| 5% =", kpss_res["KPSS VC 5%"],
        "| 10% =", kpss_res["KPSS VC 10%"]
    )
    print("CONCLUZIE:", concluzie)

    rezultate_nivel.append({
        "Serie": label,
        "Tip": "Nivel",
        "ADF stat": adf_res["ADF stat"],
        "ADF p-value": adf_res["ADF p-value"],
        "ADF lags": adf_res["ADF lags"],
        "ADF VC 1%": adf_res["ADF VC 1%"],
        "ADF VC 5%": adf_res["ADF VC 5%"],
        "ADF VC 10%": adf_res["ADF VC 10%"],
        "ADF decision": adf_res["ADF decision"],
        "KPSS stat": kpss_res["KPSS stat"],
        "KPSS p-value": kpss_res["KPSS p-value"],
        "KPSS lags": kpss_res["KPSS lags"],
        "KPSS VC 1%": kpss_res["KPSS VC 1%"],
        "KPSS VC 5%": kpss_res["KPSS VC 5%"],
        "KPSS VC 10%": kpss_res["KPSS VC 10%"],
        "KPSS decision": kpss_res["KPSS decision"],
        "Concluzie": concluzie
    })


# 6. RULARE TESTE PENTRU PRIMA DIFERENTA

print("\n" + "=" * 80)
print("TESTE DE STATIONARITATE - PRIMA DIFERENTA")
print("Specificatie: constanta, regression='c'")
print("=" * 80)

rezultate_diff = []

for col, label in serii_diferenta.items():
    adf_res = run_adf(df[col], regression="c")
    kpss_res = run_kpss(df[col], regression="c")

    concluzie = interpret_combined(
        adf_res["ADF decision"],
        kpss_res["KPSS decision"]
    )

    print("\n" + "-" * 70)
    print(label)
    print("-" * 70)
    print(
        "ADF  | Statistica:",
        adf_res["ADF stat"],
        "| p-value:",
        adf_res["ADF p-value"],
        "| Decizie:",
        adf_res["ADF decision"]
    )
    print(
        "ADF  | Valori critice:",
        "1% =", adf_res["ADF VC 1%"],
        "| 5% =", adf_res["ADF VC 5%"],
        "| 10% =", adf_res["ADF VC 10%"]
    )
    print(
        "KPSS | Statistica:",
        kpss_res["KPSS stat"],
        "| p-value:",
        kpss_res["KPSS p-value"],
        "| Decizie:",
        kpss_res["KPSS decision"]
    )
    print(
        "KPSS | Valori critice:",
        "1% =", kpss_res["KPSS VC 1%"],
        "| 5% =", kpss_res["KPSS VC 5%"],
        "| 10% =", kpss_res["KPSS VC 10%"]
    )
    print("CONCLUZIE:", concluzie)

    rezultate_diff.append({
        "Serie": label,
        "Tip": "Prima diferenta",
        "ADF stat": adf_res["ADF stat"],
        "ADF p-value": adf_res["ADF p-value"],
        "ADF lags": adf_res["ADF lags"],
        "ADF VC 1%": adf_res["ADF VC 1%"],
        "ADF VC 5%": adf_res["ADF VC 5%"],
        "ADF VC 10%": adf_res["ADF VC 10%"],
        "ADF decision": adf_res["ADF decision"],
        "KPSS stat": kpss_res["KPSS stat"],
        "KPSS p-value": kpss_res["KPSS p-value"],
        "KPSS lags": kpss_res["KPSS lags"],
        "KPSS VC 1%": kpss_res["KPSS VC 1%"],
        "KPSS VC 5%": kpss_res["KPSS VC 5%"],
        "KPSS VC 10%": kpss_res["KPSS VC 10%"],
        "KPSS decision": kpss_res["KPSS decision"],
        "Concluzie": concluzie
    })


# 7. EXPORT REZULTATE IN CSV

df_nivel = pd.DataFrame(rezultate_nivel)
df_diff = pd.DataFrame(rezultate_diff)

rezultate_finale = pd.concat([df_nivel, df_diff], ignore_index=True)

rezultate_finale.to_csv(
    "output/tables/rezultate_stationaritate.csv",
    index=False
)

print("\nTabel rezultate salvat in:")
print("output/tables/rezultate_stationaritate.csv")


# =============================================================================
# 8. GRAFIC REZULTATE TESTE, CU LINII VC 5% MEDII
# =============================================================================

def color_by_decision(decision):
    if decision == "STATIONARA":
        return "#16A34A"
    return "#DC2626"


labels_nivel = df_nivel["Serie"].tolist()
labels_diff = df_diff["Serie"].tolist()

adf_stats_nivel = df_nivel["ADF stat"].tolist()
adf_colors_nivel = [color_by_decision(x) for x in df_nivel["ADF decision"]]

kpss_stats_nivel = df_nivel["KPSS stat"].tolist()
kpss_colors_nivel = [color_by_decision(x) for x in df_nivel["KPSS decision"]]

adf_stats_diff = df_diff["ADF stat"].tolist()
adf_colors_diff = [color_by_decision(x) for x in df_diff["ADF decision"]]

kpss_stats_diff = df_diff["KPSS stat"].tolist()
kpss_colors_diff = [color_by_decision(x) for x in df_diff["KPSS decision"]]

# Valori critice medii la pragul de 5%
adf_vc5_nivel = df_nivel["ADF VC 5%"].mean()
kpss_vc5_nivel = df_nivel["KPSS VC 5%"].mean()

adf_vc5_diff = df_diff["ADF VC 5%"].mean()
kpss_vc5_diff = df_diff["KPSS VC 5%"].mean()


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "Teste de stationaritate ADF si KPSS\nSerii in nivel vs. prima diferenta",
    fontsize=14,
    fontweight="bold"
)

# -----------------------------
# ADF nivel
# -----------------------------
axes[0, 0].barh(
    labels_nivel,
    adf_stats_nivel,
    color=adf_colors_nivel,
    alpha=0.85
)
axes[0, 0].axvline(
    adf_vc5_nivel,
    color="black",
    linewidth=1.5,
    linestyle="--",
    label=f"VC 5% medie = {adf_vc5_nivel:.2f}"
)
axes[0, 0].set_title("ADF - Serii in nivel")
axes[0, 0].set_xlabel("Statistica ADF")
axes[0, 0].grid(axis="x", alpha=0.3)
axes[0, 0].legend(fontsize=8)

# -----------------------------
# KPSS nivel
# -----------------------------
axes[0, 1].barh(
    labels_nivel,
    kpss_stats_nivel,
    color=kpss_colors_nivel,
    alpha=0.85
)
axes[0, 1].axvline(
    kpss_vc5_nivel,
    color="black",
    linewidth=1.5,
    linestyle="--",
    label=f"VC 5% medie = {kpss_vc5_nivel:.2f}"
)
axes[0, 1].set_title("KPSS - Serii in nivel")
axes[0, 1].set_xlabel("Statistica KPSS")
axes[0, 1].grid(axis="x", alpha=0.3)
axes[0, 1].legend(fontsize=8)

# -----------------------------
# ADF prima diferenta
# -----------------------------
axes[1, 0].barh(
    labels_diff,
    adf_stats_diff,
    color=adf_colors_diff,
    alpha=0.85
)
axes[1, 0].axvline(
    adf_vc5_diff,
    color="black",
    linewidth=1.5,
    linestyle="--",
    label=f"VC 5% medie = {adf_vc5_diff:.2f}"
)
axes[1, 0].set_title("ADF - Prima diferenta")
axes[1, 0].set_xlabel("Statistica ADF")
axes[1, 0].grid(axis="x", alpha=0.3)
axes[1, 0].legend(fontsize=8)

# -----------------------------
# KPSS prima diferenta
# -----------------------------
axes[1, 1].barh(
    labels_diff,
    kpss_stats_diff,
    color=kpss_colors_diff,
    alpha=0.85
)
axes[1, 1].axvline(
    kpss_vc5_diff,
    color="black",
    linewidth=1.5,
    linestyle="--",
    label=f"VC 5% medie = {kpss_vc5_diff:.2f}"
)
axes[1, 1].set_title("KPSS - Prima diferenta")
axes[1, 1].set_xlabel("Statistica KPSS")
axes[1, 1].grid(axis="x", alpha=0.3)
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(
    "output/figures/fig_stationaritate.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("Grafic salvat in:")
print("output/figures/fig_stationaritate.png")


# 9. EXPORT DATASET CU TRANSFORMARI

df.to_csv("output/tables/dataset_cu_transformari.csv", index=False)

print("Dataset cu transformari salvat in:")
print("output/tables/dataset_cu_transformari.csv")

print("\nObservatie pentru raport:")
print(
    "In grafic, liniile punctate reprezinta valorile critice medii la pragul de 5%. "
    "Deciziile finale trebuie interpretate pe baza tabelului, unde sunt raportate "
    "valorile critice specifice fiecarei serii."
)

print("\nGata.")