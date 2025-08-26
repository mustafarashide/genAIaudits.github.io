from front_end_pipeline.src.data import load_data

ds_en_responses = load_data("deepseek", "wiki")
print(ds_en_responses.shape)
print(ds_en_responses.columns)
print(ds_en_responses['source'].value_counts())
selected_sources = ["House church (China)", "Geopolitics"]
ds_en_filtered = ds_en_responses[ds_en_responses['source'].isin(selected_sources)]
print(ds_en_filtered.shape)
ds_en_filtered.to_csv("responses_collection/refusal_pattern/deepseek_refusal/ds_en_hardcoded_refusal.csv", index=False)