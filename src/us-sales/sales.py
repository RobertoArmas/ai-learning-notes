import pandas as pd

df = pd.read_csv('./src/us-sales/Sales.csv')

print(df.head())
print(f"Number of rows: {len(df)}")