import pandas as pd

df = pd.read_csv("data/processed/hist_response/gpt-5_wiki_20250815_152120.csv")
print(df.head())
print(df['flagged'].value_counts())