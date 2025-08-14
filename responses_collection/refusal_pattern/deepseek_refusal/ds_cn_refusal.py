import pandas as pd
from responses_collection.data_processor import get_cn_wiki_content

# Load the dataset
ds_cn_responses = pd.read_csv("data/processed/hist_response/deepseek-chat_cn-wiki_20250714_174754_not-flagged.csv")
print(ds_cn_responses.shape)
print(ds_cn_responses.columns)
print(ds_cn_responses['flagged'].value_counts()) # existing 86 flags are Content Exists Risk

# potential flag phrases
potential_refusal_phrases = ["sorry", "apologize", "对不起", "抱歉", 
                   "can't comply", "beyond my current scope", "can't assist", "can't help",
                   "无法回答", "内容过滤", "内容存在风险", "我无法提供", "我不会复述", "我理解您",
                   "content_filter", "beyond my current scope", "我暂时无法回答", 
                   "content exists risk"]

# Filter potential refusal with any above
ds_cn_potential_flagged_content = ds_cn_responses[ds_cn_responses['model_response'].str.contains('|'.join(potential_refusal_phrases), case=False)]
print(f"Potential flagged content shape: {ds_cn_potential_flagged_content.shape}")
ds_cn_potential_flagged_content.to_csv("responses_collection/refusal_pattern/ds_cn_potential_flagged.csv", index=False)

# Sample 100 responses for manual review
sampled_responses = ds_cn_responses.sample(n=100, random_state=42)
# sampled_responses.to_csv("responses_collection/refusal_pattern/ds_cn_sampled_responses.csv", index=False)
# Given no consistent refusal pattern besides api error, we manually reviewed the whole dataset
# Identify key phrases: 我理解您, content exists risk (api error), 我暂时无法回答 (the last one is from online chat)

# Ensure our original dataset does not contain any refusal phrases
cn_content = get_cn_wiki_content()
assert not any(cn_content['content'].str.contains("我理解您", case=False)), "Found '我理解您' in original content"
assert not any(cn_content['content'].str.contains("content exists risk", case=False)), "Found 'content exists risk' in original content"
assert not any(cn_content['content'].str.contains("我暂时无法回答", case=False)), "Found '我暂时无法回答' in original content"
# verify every response with key flagged phrases should be flagged
key_flagged_phrases = ["我理解您", "content exists risk", "我暂时无法回答"]
ds_cn_flagged_responses = ds_cn_responses[ds_cn_responses['model_response'].str.contains('|'.join(key_flagged_phrases), case=False)]
# ds_cn_flagged_responses.to_csv("responses_collection/refusal_pattern/ds_cn_flagged_responses.csv", index=False)
# Manually review the flagged responses to confirm they are indeed refusals

# Revise the flagging pattern in the original dataset
# ds_cn_responses['flagged'] = ds_cn_responses['model_response'].str.contains('|'.join(key_flagged_phrases), case=False).astype(int)
# print(ds_cn_responses['flagged'].value_counts())
# ds_cn_responses.to_csv("data/processed/hist_response/deepseek-chat_cn-wiki_20250714_174754.csv", index=False)
