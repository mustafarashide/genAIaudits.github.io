import pandas as pd

df = pd.read_csv('data/processed/hist_response/gpt-5_wiki_temp.csv')
df_1 = df[df['flagged']== -1]
print(df_1['model_response'].tolist()[0])