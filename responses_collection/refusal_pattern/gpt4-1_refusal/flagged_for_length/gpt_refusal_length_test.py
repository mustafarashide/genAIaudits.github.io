from front_end_pipeline.src.data import load_data

df_gpt41_wiki = load_data("openai-gpt", "wiki")
print(df_gpt41_wiki.columns)
