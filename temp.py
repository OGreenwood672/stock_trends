import pandas as pd

df = pd.read_csv('./data/constituents.csv')

df['Symbol'].to_csv('./data/stocks.csv', index=False)
