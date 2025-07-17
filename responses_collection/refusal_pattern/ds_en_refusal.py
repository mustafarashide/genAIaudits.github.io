import pandas as pd
from responses_collection.data_processor import get_all_content

# Load the dataset
ds_en_responses = pd.read_csv("data/processed/hist_response/deepseek-chat_all_20250630_152359_not-flagged.csv")
print(ds_en_responses.shape)
print(ds_en_responses.columns)

# potential flag phrases
potential_refusal_phrases = ["sorry", "apologize", 
                   "can't comply", "beyond my current scope", "can't assist", "can't help",
                   "content_filter", "beyond my current scope",
                   "content exists risk"]

# Filter potential refusal with any above
ds_en_potential_flagged_content = ds_en_responses[ds_en_responses['model_response'].str.contains('|'.join(potential_refusal_phrases), case=False)]
# ds_en_potential_flagged_content.to_csv("responses_collection/refusal_pattern/ds_en_potential_flagged.csv", index=False)

# Randomly sample 100 responses for manual review
sampled_responses = ds_en_responses.sample(n=100, random_state=42)
# sampled_responses.to_csv("responses_collection/refusal_pattern/ds_en_sampled_responses.csv", index=False)
# Identify key phrases: I can't comply with, content_filter, content exists risk (api refusal), beyond my current scope (the last one is from online chat)

# Ensure our original dataset does not contain any I can't comply with phrases
all_content = get_all_content()
assert not any(all_content['content'].str.contains("I can't comply with", case=False)), "Found 'I can't comply with' in original content"
assert not any(all_content['content'].str.contains("beyond my current scope", case=False)), "Found 'beyond my current scope' in original content"
assert not any(all_content['content'].str.contains("content_filter", case=False)), "Found 'content_filter' in original content"
assert not any(all_content['content'].str.contains("content exists risk", case=False)), "Found 'content exists risk' in original content"

# verify every response with key flagged phrases should be flagged
key_flagged_phrases = ["I can't comply with", "beyond my current scope", "content_filter", "content exists risk"]
ds_en_flagged_responses = ds_en_responses[ds_en_responses['model_response'].str.contains('|'.join(key_flagged_phrases), case=False)]
# ds_en_flagged_responses.to_csv("responses_collection/refusal_pattern/ds_en_flagged_responses.csv", index=False)
# Manually review the flagged responses to confirm they are indeed refusals

# Revise the flagging pattern in the original dataset
# ds_en_responses['flagged'] = ds_en_responses['model_response'].str.contains('|'.join(key_flagged_phrases), case=False).astype(int)
# print(ds_en_responses['flagged'].value_counts())
# ds_en_responses.to_csv("data/processed/hist_response/deepseek-chat_all_20250630_152359.csv", index=False)
# ds_en_wiki = pd.read_csv("data/processed/hist_response/deepseek-chat_wiki_20250708_184537.csv")
# ds_en_wiki['flagged'] = ds_en_wiki['model_response'].str.contains('|'.join(key_flagged_phrases), case=False).astype(int)
# print(ds_en_wiki['flagged'].value_counts())
# ds_en_wiki.to_csv("data/processed/hist_response/deepseek-chat_wiki_20250708_184537.csv", index=False)
