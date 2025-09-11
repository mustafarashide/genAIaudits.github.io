import pandas as pd
from responses_collection.openai_batch_client import OpenAIBatchClient
import json

with open("responses_collection/api_config.json", "r") as f:
    api_config = json.load(f)
    openai_api_key = api_config["openai_sorelle"]

# Prepare dataset
abortion_df = pd.read_csv("data/processed/wiki_content/Abortion.csv")
duplicated_df = pd.concat([abortion_df]*100, ignore_index=True)

# Copy content_id to content_id_copy
duplicated_df['content_id_copy'] = duplicated_df['content_id']
# Revise content id to be unique
duplicated_df['content_id'] = duplicated_df.index + 1

# Initialize clients 
gpt4_1_client = OpenAIBatchClient(api_key=openai_api_key, model="gpt-4.1", endpoint="/v1/chat/completions")
gpt5_client = OpenAIBatchClient(api_key=openai_api_key, model="gpt-5", endpoint="/v1/chat/completions")

# gpt4_1_responses is a list of dictionaries
gpt4_1_responses = gpt4_1_client.process_dataset(duplicated_df)
gpt5_responses = gpt5_client.process_dataset(duplicated_df)

# Convert responses to DataFrames
gpt4_1_responses_df = pd.DataFrame(gpt4_1_responses)
gpt5_responses_df = pd.DataFrame(gpt5_responses)

# Save to CSV
gpt4_1_responses_df.to_csv("responses_collection/response_variance/gpt4_1_abortion_responses.csv", index=False)
gpt5_responses_df.to_csv("responses_collection/response_variance/gpt5_abortion_responses.csv", index=False)
