import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")


# 0. SETARI GENERALE

os.makedirs("output", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
os.makedirs("output/tables", exist_ok=True)

DATA_PATH = "output/tables/dataset_cu_transformari.csv"


# 1. INCARCARE DATE

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 80)
print("ANALIZA MULTIVARIATA - BITCOIN, AUR, S&P 500, INFLATIE")
print("=" * 80)
print("Perioada:", df["date"].min(), "->", df["date"].max())
print("Numar observatii:", len(df))


# =============================================================================
# 4.4.1 ALEGEREA VARIABILELOR PENTRU ANALIZA MULTIVARIATA
# =============================================================================
# Pe baza testelor de stationaritate din stationarity_tests.py:
# - dlog_btc:   I(0) - stationara (ADF si KPSS ambele confirma)
# - dlog_gold:  Ambiguu, dar ADF confirma stationaritate -> folosim dlog_gold
# - dlog_sp500: I(0) - stationara (ADF si KPSS ambele confirma)
# - dinf_yoy:   I(0) - stationara (ADF si KPSS ambele confirma)
# Concluzie: folosim randamentele logaritmice si diferenta inflatiei -> VAR pe serii stationare

print("\n" + "=" * 80)
print("4.4.1 ALEGEREA VARIABILELOR")
print("=" * 80)

# Construim datasetul multivariat cu seriile stationare
var_vars = ["dlog_btc", "dlog_gold", "dlog_sp500", "dinf_yoy"]

df_var = df[["date"] + var_vars].dropna().copy()
df_var = df_var.set_index("date")
df_var = df_var.asfreq("MS")

print("Variabile selectate:", var_vars)
print("Observatii dupa eliminarea NaN:", len(df_var))
print("\nPrimele randuri:")
print(df_var.head())
print("\nStatistici descriptive:")
print(df_var.describe().round(4))


# =============================================================================
# 4.4.2 ANALIZA DE COINTEGRARE JOHANSEN
# =============================================================================
# Aplicam testul Johansen pe seriile in nivel (log) pentru a verifica
# daca exista o relatie de echilibru pe termen lung intre cele 4 active.

print("\n" + "=" * 80)
print("4.4.2 ANALIZA DE COINTEGRARE JOHANSEN")
print("=" * 80)

# Serii in nivel pentru testul Johansen
nivel_vars = ["log_btc", "log_gold", "log_sp500", "log_cpi"]
df_nivel = df[["date"] + nivel_vars].dropna().copy()
df_nivel = df_nivel.set_index("date")

johansen_data = df_nivel.values

# det_order=1: constanta in relatia de cointegrare (standard pentru serii financiare)
# k_ar_diff=2: 2 laguri in diferente (echivalent VAR(3))
johansen_result = coint_johansen(johansen_data, det_order=1, k_ar_diff=2)

print("\nTestul Johansen - statistici trace:")
print(f"{'Ipoteza nula':<25} {'Statistica trace':<20} {'VC 5%':<15} {'Decizie'}")
print("-" * 75)

ipoteze = [
    "r = 0 (fara cointegrare)",
    "r <= 1",
    "r <= 2",
    "r <= 3"
]

trace_stats = johansen_result.lr1
trace_cv = johansen_result.cvt[:, 1]  # valori critice 5%

nr_relatii_cointegrare = 0
for i in range(len(ipoteze)):
    stat = trace_stats[i]
    cv = trace_cv[i]
    decizie = "RESPINGE H0" if stat > cv else "NU respinge H0"
    if stat > cv:
        nr_relatii_cointegrare = i + 1
    print(f"{ipoteze[i]:<25} {stat:<20.4f} {cv:<15.4f} {decizie}")

print(f"\nNumar relatii de cointegrare identificate: {nr_relatii_cointegrare}")

if nr_relatii_cointegrare == 0:
    print("Concluzie: Nu exista cointegrare -> se utilizeaza modelul VAR pe serii stationare.")
    use_vecm = False
else:
    print(f"Concluzie: Exista {nr_relatii_cointegrare} relatie(i) de cointegrare -> se poate utiliza VECM.")
    use_vecm = True

# Salvam rezultatele Johansen
johansen_rows = []
for i in range(len(ipoteze)):
    johansen_rows.append({
        "Ipoteza nula": ipoteze[i],
        "Statistica trace": round(trace_stats[i], 4),
        "Valoare critica 5%": round(trace_cv[i], 4),
        "Decizie": "RESPINGE H0" if trace_stats[i] > trace_cv[i] else "NU respinge H0"
    })

pd.DataFrame(johansen_rows).to_csv(
    "output/tables/johansen_cointegrare.csv", index=False
)
print("Tabel Johansen salvat in: output/tables/johansen_cointegrare.csv")


# =============================================================================
# 4.4.3 LAG OPTIM SI ESTIMAREA MODELULUI VAR
# =============================================================================

print("\n" + "=" * 80)
print("4.4.3 LAG OPTIM SI ESTIMAREA MODELULUI VAR")
print("=" * 80)

# Selectia lagului optim pe baza criteriilor informationale
model_select = VAR(df_var)
lag_results = model_select.select_order(maxlags=12)

print("\nCriterii informationale pentru selectia lagului:")
print(lag_results.summary())

# Extragem lagurile optime pentru fiecare criteriu
lag_aic  = lag_results.aic
lag_bic  = lag_results.bic
lag_hqic = lag_results.hqic
lag_fpe  = lag_results.fpe

print(f"\nLag optim AIC:  {lag_aic}")
print(f"Lag optim BIC:  {lag_bic}")
print(f"Lag optim HQIC: {lag_hqic}")
print(f"Lag optim FPE:  {lag_fpe}")

# Folosim lagul BIC (cel mai parcimonios, preferat pentru esantioane mici)
lag_ales = lag_bic
if lag_ales == 0:
    lag_ales = 1  # minimum 1 lag

print(f"\nLag ales pentru estimare (BIC): {lag_ales}")

# Estimarea modelului VAR
var_model = VAR(df_var)
var_fit = var_model.fit(lag_ales)

print("\nRezumat model VAR:")
print(var_fit.summary())

# Salvam coeficientii VAR
try:
    coef_df = pd.DataFrame(var_fit.params)
    coef_df.to_csv("output/tables/var_coeficienti.csv")
    print("Coeficienti VAR salvati in: output/tables/var_coeficienti.csv")
except Exception:
    pass

# Grafic: evolutia reziduurilor VAR
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
labels_plot = ["dlog_btc", "dlog_gold", "dlog_sp500", "dinf_yoy"]
titles_plot = [
    "Reziduuri VAR - Bitcoin",
    "Reziduuri VAR - Aur",
    "Reziduuri VAR - S&P 500",
    "Reziduuri VAR - Inflatie YoY"
]

resid = var_fit.resid
for i, ax in enumerate(axes):
    ax.plot(df_var.index[lag_ales:], resid.iloc[:, i], linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(titles_plot[i])
    ax.set_ylabel("Reziduu")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Data")
plt.suptitle("Reziduurile modelului VAR", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("output/figures/var_reziduuri.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafic reziduuri salvat in: output/figures/var_reziduuri.png")


# =============================================================================
# 4.4.4 TESTE DE CAUZALITATE GRANGER
# =============================================================================

print("\n" + "=" * 80)
print("4.4.4 TESTE DE CAUZALITATE GRANGER")
print("=" * 80)
print(f"Laguri utilizate: {lag_ales}")
print("H0: variabila X NU Granger-cauzeaza variabila Y")
print("Daca p-value < 0.05 -> respingem H0 -> X Granger-cauzeaza Y")

granger_results = []

# Testam: fiecare variabila -> dlog_btc
# (vrem sa stim ce variabile ajuta la prognoza Bitcoin)
variabile_cauza = ["dlog_gold", "dlog_sp500", "dinf_yoy"]
variabila_efect = "dlog_btc"

print(f"\nTestare cauzalitate Granger catre {variabila_efect}:")
print("-" * 60)

for cauza in variabile_cauza:
    test_data = df_var[[variabila_efect, cauza]].dropna()
    try:
        rezultat = grangercausalitytests(test_data, maxlag=lag_ales, verbose=False)
        # Luam p-value pentru lagul ales
        pval = rezultat[lag_ales][0]["ssr_ftest"][1]
        fstat = rezultat[lag_ales][0]["ssr_ftest"][0]
        decizie = "GRANGER-CAUZEAZA" if pval < 0.05 else "NU Granger-cauzeaza"
        print(f"{cauza} -> {variabila_efect}: F={fstat:.4f}, p={pval:.4f} -> {decizie}")
        granger_results.append({
            "Cauza": cauza,
            "Efect": variabila_efect,
            "F-statistica": round(fstat, 4),
            "p-value": round(pval, 4),
            "Laguri": lag_ales,
            "Decizie": decizie
        })
    except Exception as e:
        print(f"{cauza} -> {variabila_efect}: Eroare - {e}")

# Testam si in sens invers: dlog_btc -> fiecare variabila
print(f"\nTestare cauzalitate Granger din {variabila_efect}:")
print("-" * 60)

for efect in variabile_cauza:
    test_data = df_var[[efect, variabila_efect]].dropna()
    try:
        rezultat = grangercausalitytests(test_data, maxlag=lag_ales, verbose=False)
        pval = rezultat[lag_ales][0]["ssr_ftest"][1]
        fstat = rezultat[lag_ales][0]["ssr_ftest"][0]
        decizie = "GRANGER-CAUZEAZA" if pval < 0.05 else "NU Granger-cauzeaza"
        print(f"{variabila_efect} -> {efect}: F={fstat:.4f}, p={pval:.4f} -> {decizie}")
        granger_results.append({
            "Cauza": variabila_efect,
            "Efect": efect,
            "F-statistica": round(fstat, 4),
            "p-value": round(pval, 4),
            "Laguri": lag_ales,
            "Decizie": decizie
        })
    except Exception as e:
        print(f"{variabila_efect} -> {efect}: Eroare - {e}")

# Salvam rezultatele Granger
pd.DataFrame(granger_results).to_csv(
    "output/tables/granger_cauzalitate.csv", index=False
)
print("\nTabel Granger salvat in: output/tables/granger_cauzalitate.csv")

# Grafic heatmap cauzalitate Granger
granger_df = pd.DataFrame(granger_results)
pivot_pval = granger_df.pivot(index="Cauza", columns="Efect", values="p-value")

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(pivot_pval.values, aspect="auto", vmin=0, vmax=0.1)
plt.colorbar(im, ax=ax, label="p-value")

ax.set_xticks(range(len(pivot_pval.columns)))
ax.set_yticks(range(len(pivot_pval.index)))
ax.set_xticklabels(pivot_pval.columns, rotation=30, ha="right")
ax.set_yticklabels(pivot_pval.index)

for i in range(len(pivot_pval.index)):
    for j in range(len(pivot_pval.columns)):
        val = pivot_pval.values[i, j]
        if not np.isnan(val):
            semn = "*" if val < 0.05 else ""
            ax.text(j, i, f"{val:.3f}{semn}", ha="center", va="center", fontsize=9)

ax.set_title(
    "Heatmap cauzalitate Granger (p-values)\n* = semnificativ la 5%",
    fontweight="bold"
)
plt.tight_layout()
plt.savefig("output/figures/granger_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafic Granger salvat in: output/figures/granger_heatmap.png")


# =============================================================================
# 4.4.5 FUNCTII DE RASPUNS LA IMPULS (IRF)
# =============================================================================

print("\n" + "=" * 80)
print("4.4.5 FUNCTII DE RASPUNS LA IMPULS (IRF)")
print("=" * 80)

ORIZONT_IRF = 20  # 20 luni

irf = var_fit.irf(ORIZONT_IRF)

# Grafic IRF: raspunsul Bitcoin la socuri din celelalte variabile
# Ordinea variabilelor in model: dlog_btc=0, dlog_gold=1, dlog_sp500=2, dinf_yoy=3
# Identificare Cholesky: ordinea conteaza -> BTC primul (cel mai exogen)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

socuri = [
    (1, 0, "Soc in Aur -> raspuns Bitcoin"),
    (2, 0, "Soc in S&P 500 -> raspuns Bitcoin"),
    (3, 0, "Soc in Inflatie -> raspuns Bitcoin"),
]

for idx, (impulse_idx, response_idx, titlu) in enumerate(socuri):
    irf_vals = irf.irfs[:, response_idx, impulse_idx]
    lower = irf.stderr_cov if hasattr(irf, "stderr_cov") else None

    # Intervale de incredere bootstrap
    try:
        irf_ci = irf.cum_effect_stderr if hasattr(irf, "cum_effect_stderr") else None
        lower_ci = irf.irfs_lower[:, response_idx, impulse_idx]
        upper_ci = irf.irfs_upper[:, response_idx, impulse_idx]
        has_ci = True
    except Exception:
        has_ci = False

    ax = axes[idx]
    ax.plot(range(ORIZONT_IRF + 1), irf_vals, linewidth=2, label="IRF")
    if has_ci:
        ax.fill_between(
            range(ORIZONT_IRF + 1),
            lower_ci,
            upper_ci,
            alpha=0.2,
            label="IC 95%"
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(titlu, fontsize=10)
    ax.set_xlabel("Luni")
    ax.set_ylabel("Raspuns")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle(
    "Functii de raspuns la impuls - Raspunsul Bitcoin la socuri externe",
    fontsize=12,
    fontweight="bold"
)
plt.tight_layout()
plt.savefig("output/figures/irf_bitcoin.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Grafic IRF salvat in: output/figures/irf_bitcoin.png")

# Grafic IRF complet: toti respondentii la toate socurile
print("Generare grafic IRF complet...")
try:
    irf_plot = irf.plot(orth=False, figsize=(14, 12))
    irf_plot.suptitle(
        "Functii de raspuns la impuls - toate combinatiile",
        fontsize=12,
        fontweight="bold"
    )
    irf_plot.tight_layout()
    irf_plot.savefig("output/figures/irf_complet.png", dpi=120, bbox_inches="tight")
    plt.close("all")
    print("Grafic IRF complet salvat in: output/figures/irf_complet.png")
except Exception as e:
    print(f"Grafic IRF complet: {e}")


# =============================================================================
# 4.4.6 DESCOMPUNEREA VARIANTEI (FEVD)
# =============================================================================

print("\n" + "=" * 80)
print("4.4.6 DESCOMPUNEREA VARIANTEI ERORII DE PROGNOZA (FEVD)")
print("=" * 80)

ORIZONT_FEVD = 20

fevd = var_fit.fevd(ORIZONT_FEVD)

# Extragem FEVD pentru Bitcoin (variabila 0)
fevd_btc = fevd.decomp[0]  # shape: (orizont, n_vars)

print("\nDescompunerea variantei Bitcoin (% din varianta explicata):")
print(f"{'Luna':<8} {'BTC':<12} {'Aur':<12} {'S&P500':<12} {'Inflatie':<12}")
print("-" * 55)

for t in [1, 3, 6, 12, 20]:
    if t <= ORIZONT_FEVD:
        row = fevd_btc[t - 1] * 100
        print(f"{t:<8} {row[0]:<12.2f} {row[1]:<12.2f} {row[2]:<12.2f} {row[3]:<12.2f}")

# Salvam FEVD complet
fevd_rows = []
for t in range(ORIZONT_FEVD):
    row = fevd_btc[t] * 100
    fevd_rows.append({
        "Luna": t + 1,
        "BTC (%)": round(row[0], 4),
        "Aur (%)": round(row[1], 4),
        "S&P500 (%)": round(row[2], 4),
        "Inflatie (%)": round(row[3], 4)
    })

pd.DataFrame(fevd_rows).to_csv(
    "output/tables/fevd_bitcoin.csv", index=False
)
print("\nTabel FEVD salvat in: output/tables/fevd_bitcoin.csv")

# Grafic FEVD - stacked area chart pentru Bitcoin
fig, ax = plt.subplots(figsize=(11, 6))

luni = range(1, ORIZONT_FEVD + 1)
btc_share   = [fevd_btc[t][0] * 100 for t in range(ORIZONT_FEVD)]
gold_share  = [fevd_btc[t][1] * 100 for t in range(ORIZONT_FEVD)]
sp500_share = [fevd_btc[t][2] * 100 for t in range(ORIZONT_FEVD)]
inf_share   = [fevd_btc[t][3] * 100 for t in range(ORIZONT_FEVD)]

ax.stackplot(
    luni,
    btc_share,
    gold_share,
    sp500_share,
    inf_share,
    labels=["Bitcoin", "Aur", "S&P 500", "Inflatie YoY"],
    alpha=0.85
)

ax.set_xlabel("Orizont de prognoza (luni)")
ax.set_ylabel("Proportie din varianta explicata (%)")
ax.set_title(
    "Descompunerea variantei erorii de prognoza - Bitcoin",
    fontweight="bold"
)
ax.legend(loc="upper right")
ax.set_xlim(1, ORIZONT_FEVD)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/figures/fevd_bitcoin.png", dpi=150, bbox_inches="tight")
plt.close()
print("Grafic FEVD salvat in: output/figures/fevd_bitcoin.png")

# Grafic FEVD complet pentru toate variabilele
try:
    fevd_plot = fevd.plot(figsize=(13, 10))
    fevd_plot.suptitle(
        "Descompunerea variantei erorii de prognoza - toate variabilele",
        fontsize=12,
        fontweight="bold"
    )
    fevd_plot.tight_layout()
    fevd_plot.savefig("output/figures/fevd_complet.png", dpi=120, bbox_inches="tight")
    plt.close("all")
    print("Grafic FEVD complet salvat in: output/figures/fevd_complet.png")
except Exception as e:
    print(f"Grafic FEVD complet: {e}")


# =============================================================================
# REZUMAT FINAL
# =============================================================================

print("\n" + "=" * 80)
print("REZUMAT ANALIZA MULTIVARIATA")
print("=" * 80)

print(f"\nVariabile folosite: {var_vars}")
print(f"Lag ales (BIC): {lag_ales}")
print(f"Cointegrare Johansen: {nr_relatii_cointegrare} relatie(i)")
print(f"Model estimat: {'VECM' if use_vecm else 'VAR'}")

print("\nCauzalitate Granger catre Bitcoin:")
for r in granger_results:
    if r["Efect"] == "dlog_btc":
        print(f"  {r['Cauza']} -> Bitcoin: p={r['p-value']} -> {r['Decizie']}")

print("\nDescompunerea variantei Bitcoin la orizont 12 luni:")
row12 = fevd_btc[11] * 100
print(f"  BTC propriu: {row12[0]:.2f}%")
print(f"  Aur:         {row12[1]:.2f}%")
print(f"  S&P 500:     {row12[2]:.2f}%")
print(f"  Inflatie:    {row12[3]:.2f}%")

print("\nFisiere generate:")
print("  output/tables/johansen_cointegrare.csv")
print("  output/tables/var_coeficienti.csv")
print("  output/tables/granger_cauzalitate.csv")
print("  output/tables/fevd_bitcoin.csv")
print("  output/figures/var_reziduuri.png")
print("  output/figures/granger_heatmap.png")
print("  output/figures/irf_bitcoin.png")
print("  output/figures/irf_complet.png")
print("  output/figures/fevd_bitcoin.png")
print("  output/figures/fevd_complet.png")