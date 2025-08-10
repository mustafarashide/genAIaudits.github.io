import pandas as pd

from pipeline_orchestrator import Pipeline
from data_processor import get_wiki_content
from config import config

refusal_patterns = ["I'm sorry, but", 
                    "I’m sorry, but", 
                    "I can't repeat", 
                    "I can't comply with", 
                    "I cannot repeat"]
length_phrase = ["long", "length"]

past_responses = pd.read_csv("data/processed/hist_response/gpt-4.1_wiki_20250728_233227.csv")
flagged_content = past_responses[past_responses['flagged'] == 1]
lengthy_refusals = flagged_content[flagged_content['model_response'].str.contains('|'.join(length_phrase), case=False)]
lengthy_content_ids = set(lengthy_refusals['content_id'])

# Fix the flagged column update logic
mask = past_responses['content_id'].isin(lengthy_content_ids)
past_responses.loc[mask, 'flagged'] = 2

# Save the updated DataFrame
# past_responses.to_csv("data/processed/hist_response/gpt-4.1_wiki_temp.csv", index=False)

# Read new dataset 
new_responses_raw = pd.read_csv("data/processed/hist_response/gpt-4.1_wiki_20250804_172010_not-flagged.csv")
new_responses = new_responses_raw[new_responses_raw['content_id'].isin(lengthy_content_ids)]

# Check new responses containing lengthy refusals
new_refusals = new_responses[new_responses['model_response'].str.contains('|'.join(refusal_patterns), case=False)]
print(f"New refusals found: {len(new_refusals)}")
new_lengthy_refusals = new_refusals[new_refusals['model_response'].str.contains('|'.join(length_phrase), case=False)]
# new_lengthy_refusals.to_csv("responses_collection/refusal_pattern/gpt-4.1_wiki_20250804_172010_lengthy_refusals.csv", index=False)
print(f"New lengthy refusals found: {len(new_lengthy_refusals)}")
new_lengthy_content_ids = set(new_lengthy_refusals['content_id'])

wiki_dataset = get_wiki_content()
new_lengthy_refusal_wiki_dataset = wiki_dataset[wiki_dataset['content_id'].isin(new_lengthy_content_ids)]
print(new_lengthy_refusal_wiki_dataset.columns)

temp_df = pd.merge(new_lengthy_refusal_wiki_dataset, new_lengthy_refusals, on='content_id', how='inner')
temp_df = temp_df[['content_id', 'permanent_link', 'model_response']]
# temp_df.to_csv("responses_collection/refusal_pattern/gpt-4.1_wiki_20250804_lengthy_refusal_wiki_dataset.csv", index=False)

temp_df_1 = pd.merge(new_responses, wiki_dataset, on='content_id', how='left')
temp_df_1 = temp_df_1[['content_id', 'permanent_link', 'model_response', 'flagged']]
# temp_df_1.to_csv("responses_collection/refusal_pattern/temp1.csv", index=False)
print(new_responses_raw['flagged'].value_counts())