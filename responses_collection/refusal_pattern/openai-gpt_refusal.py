import pandas as pd
from responses_collection.data_processor import get_all_content

# Load the dataset
openai_gpt_responses = pd.read_csv("data/processed/hist_response/gpt-4.1_wiki_temp_not-flagged.csv")
print(openai_gpt_responses.shape)
print(openai_gpt_responses.columns)

# potential flag phrases
potential_refusal_phrases = ["sorry", "apologize", 
                   "can't comply", "beyond my current scope", "can't assist", "can't help",
                   "content_filter", "beyond my current scope",
                   "content exists risk"]

# Filter potential refusal with any above
gpt_potential_flagged_content = openai_gpt_responses[openai_gpt_responses['model_response'].str.lower().str.contains('|'.join(potential_refusal_phrases), case=False)]
# gpt_potential_flagged_content.to_csv("responses_collection/refusal_pattern/gpt_potential_flagged.csv", index=False)

# Randomly sample 100 responses for manual review
sampled_responses = openai_gpt_responses.sample(n=100, random_state=42)
# sampled_responses.to_csv("responses_collection/refusal_pattern/gpt_sampled_responses.csv", index=False)
# Identify key phrases: "I'm sorry, but"; "I’m sorry, but"

# Ensure our original dataset does not contain any I can't comply with phrases
all_content = get_all_content()
assert not any(all_content['content'].str.contains("I can't comply with", case=False)), "Found 'I can't comply with' in original content"

# There is one sentence in the original content that contains I\'m sorry, but ... wow
try :
    assert not any(all_content['content'].str.contains("I'm sorry, but", case=False)), f"Found 'I'm sorry, but' in original content: {all_content[all_content['content'].str.contains("I'm sorry, but", case=False)]['content'].values[0]}"
except AssertionError as e:
    if "I\'m sorry, but ... wow" in str(e):
        pass  # This is an expected case and we will handle it in gpt batch client 
    else:
        raise e

assert not any(all_content['content'].str.contains("I’m sorry, but", case=False)), "Found 'I’m sorry, but' in original content"
assert not any(all_content['content'].str.contains("I can't repeat", case=False)), "Found 'I can't repeat' in original content"
assert not any(all_content['content'].str.contains("I cannot repeat", case=False)), "Found 'I cannot repeat' in original content"

# verify every response with key flagged phrases should be flagged
key_flagged_phrases = ["I'm sorry, but", "I’m sorry, but", "I can't repeat", "I can't comply with", "I cannot repeat"]
gpt_flagged_responses = openai_gpt_responses[openai_gpt_responses['model_response'].str.contains('|'.join(key_flagged_phrases), case=False)]
# gpt_flagged_responses.to_csv("responses_collection/refusal_pattern/gpt_flagged_responses.csv", index=False)
# Manually review the flagged responses to confirm they are indeed refusals

# Revise the flagging pattern in the original dataset
'''openai_gpt_responses['flagged'] = openai_gpt_responses['model_response'].str.contains('|'.join(key_flagged_phrases), case=False).astype(int)
print(openai_gpt_responses['flagged'].value_counts())
openai_gpt_responses.to_csv("data/processed/hist_response/gpt-4.1_wiki_temp.csv", index=False)'''
# gpt_wiki = pd.read_csv("data/processed/hist_response/deepseek-chat_wiki_20250708_184537.csv")
# gpt_wiki['flagged'] = gpt_wiki['model_response'].str.contains('|'.join(key_flagged_phrases), case=False).astype(int)
# print(gpt_wiki['flagged'].value_counts())
# gpt_wiki.to_csv("data/processed/hist_response/deepseek-chat_wiki_20250708_184537.csv", index=False)
