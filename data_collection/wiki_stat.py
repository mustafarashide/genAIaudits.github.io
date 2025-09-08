from responses_collection.data_processor import get_wiki_content
from front_end_pipeline.src.data import _filter_not_relevant_wiki

wiki_df = get_wiki_content()
print(wiki_df.columns)
print(wiki_df['token_length'].describe())

#identify number of entries below 5th percentile
token_length_5th = wiki_df['token_length'].quantile(0.05)
print(f"5th percentile of token length: {token_length_5th}")

below_5th = wiki_df[wiki_df['token_length'] < token_length_5th]
print(f"Number of entries below 5th percentile: {below_5th.shape[0]}")

# export below 5th percentile for manual examination 
# below_5th.to_csv("data_collection/below_5th_percentile.csv", index=False)

filtered_wiki_df = _filter_not_relevant_wiki(wiki_df)
print(f"Number of entries after filtering: {filtered_wiki_df.shape[0]}")
print(filtered_wiki_df.columns)
print(filtered_wiki_df['content_id'].nunique())
print(filtered_wiki_df['subcategory'].nunique())
print(filtered_wiki_df['category'].nunique())

