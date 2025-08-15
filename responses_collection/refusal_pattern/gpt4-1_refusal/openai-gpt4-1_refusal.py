import pandas as pd
from front_end_pipeline.src.data import _extract_openai_gpt_response

# Load the dataset
openai_gpt_responses = pd.read_csv("data/processed/hist_response/gpt-4.1_wiki_20250723_093726_not-flagged.csv")
print(openai_gpt_responses.shape)
print(openai_gpt_responses.columns)

# potential flag phrases
potential_refusal_phrases = ["sorry", "apologize", 
                   "can't comply", "beyond my current scope", "can't assist", "can't help",
                   "content_filter", "beyond my current scope",
                   "content exists risk"]

# Filter potential refusal with any above
openai_gpt_responses['model_response'] = openai_gpt_responses['model_response'].apply(_extract_openai_gpt_response)
gpt_potential_flagged_content = openai_gpt_responses[openai_gpt_responses['model_response'].str.lower().str.contains('|'.join(potential_refusal_phrases), case=False)]
gpt_potential_flagged_content.to_csv("responses_collection/refusal_pattern/gpt4-1_refusal/gpt_potential_flagged.csv", index=False)

# Randomly sample 100 responses for manual review
sampled_responses = openai_gpt_responses.sample(n=100, random_state=42)
sampled_responses.to_csv("responses_collection/refusal_pattern/gpt4-1_refusal/gpt_sampled_responses.csv", index=False)
# Identify key phrases: "I'm sorry, but"; "I’m sorry, but"

# verify every response with key flagged phrases should be flagged
key_flagged_phrases = ["I'm sorry, but", "I’m sorry, but", 
                       "I can't repeat", "I can't provide", "I can’t repeat",
                       "I can't comply with", "I cannot repeat"]

# Union with gpt 5 refusal patterns
refusal_patterns = ["I'm sorry, but", "I’m sorry, but", 
                       "I can't repeat", "I can’t repeat", "I cannot repeat",
                       "I can’t simply repeat", "I can’t verbatim repeat",
                       "I’m not able to repeat", "I am not able to repeat",
                       "I can't comply with", "I can't provide"]

gpt_flagged_responses = openai_gpt_responses[openai_gpt_responses['model_response'].str.contains('|'.join(refusal_patterns), case=False)]
# gpt_flagged_responses.to_csv("responses_collection/refusal_pattern/gpt4-1_refusal/gpt_flagged_responses.csv", index=False)
# Manually review the flagged responses to confirm they are indeed refusals

# Revise the flagging pattern in the original dataset
'''openai_gpt_responses['flagged'] = openai_gpt_responses['model_response'].str.contains('|'.join(key_flagged_phrases), case=False).astype(int)
print(openai_gpt_responses['flagged'].value_counts())
openai_gpt_responses.to_csv("data/processed/hist_response/gpt-4.1_wiki_temp.csv", index=False)'''
# gpt_wiki = pd.read_csv("data/processed/hist_response/deepseek-chat_wiki_20250708_184537.csv")
# gpt_wiki['flagged'] = gpt_wiki['model_response'].str.contains('|'.join(key_flagged_phrases), case=False).astype(int)
# print(gpt_wiki['flagged'].value_counts())
# gpt_wiki.to_csv("data/processed/hist_response/deepseek-chat_wiki_20250708_184537.csv", index=False)
