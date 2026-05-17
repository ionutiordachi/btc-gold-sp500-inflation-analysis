import pandas as pd

btc = pd.read_csv("data/processed/Bitcoin.csv", sep=';')
btc['date'] = pd.to_datetime(btc['timeOpen'].str[:10])  # pastram doar YYYY-MM-DD
btc = btc[['date', 'close', 'volume', 'marketCap']].rename(columns={
    'close': 'btc_close',
    'volume': 'btc_volume',
    'marketCap': 'btc_marketcap'
})
btc = btc[btc['date'] >= '2015-01-01'].reset_index(drop=True)

# Gold 
gold = pd.read_csv("data/processed/gold.csv")
gold['date'] = pd.to_datetime(gold['date'])

#  S&P 500 
sp = pd.read_csv("data/processed/S&P 500.csv")
sp['date'] = pd.to_datetime(sp['Date'], format='%m/%d/%Y')
sp['sp500_price'] = sp['Price'].str.replace(',', '').astype(float)
sp = sp[['date', 'sp500_price']]

# Inflation (FRED - CPIAUCSL)
inf = pd.read_csv("data/processed/inflation_2015_2025.csv")
inf['date'] = pd.to_datetime(inf['date'])

df = btc.merge(gold, on='date').merge(sp, on='date').merge(inf, on='date')
df = df.sort_values('date').reset_index(drop=True)

print("Shape:", df.shape)
print("Coloane:", list(df.columns))
print("Perioada:", df['date'].min(), "->", df['date'].max())
print("\nValori lipsa:\n", df.isnull().sum())

df.to_csv("data/raw/dataset_final.csv", index=False)
print("\nDataset salvat in: dataset_final.csv")