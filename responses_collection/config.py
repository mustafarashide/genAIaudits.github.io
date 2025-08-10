import os
import json

with open("responses_collection/api_config.json", "r") as f:
    api_config = json.load(f)
    openai_api_key = api_config["openai-me"]
    deepseek_api_key = api_config["deepseek"]
    openai_sorelle_api_key = api_config["openai_sorelle"]

openai_me_model = "omni-moderation-latest"
openai_gpt41 = "gpt-4.1"
openai_gpt5 = "gpt-5"
deepseek_model = "deepseek-chat"

config = {
    "openai-me": {
        "api_key": openai_api_key,
        "model": openai_me_model,
        "rate_limit": 4,
        "temp_file_template": os.path.join("data/processed/hist_response", "{model}_{dataset}_temp.csv"),
        "output_file": os.path.join("data/processed/hist_response", f"{openai_me_model}.csv"),
        "batch_size": 50
    },
    "openai-gpt4.1": {
        "api_key": openai_sorelle_api_key,
        "model": openai_gpt41,
        "endpoint": "/v1/chat/completions",
        "rate_limit": 4,
        "temp_file_template": os.path.join("data/processed/hist_response", "{model}_{dataset}_temp.csv"),
        "output_file": os.path.join("data/processed/hist_response", f"{openai_gpt41}.csv"),
        "max_tokens_per_day": 850000,  # 900k words per day for OpenAI GPT-4.1 (for safety, cap at 850k)
        "batch_size": 100,
        "lengthy_refusal_truncation": 30000  # Truncate to 30000 characters for lengthy refusals
    },
    "openai-gpt5": {
        "api_key": openai_sorelle_api_key,
        "model": openai_gpt5,
        "endpoint": "/v1/chat/completions",
        "rate_limit": 4,
        "temp_file_template": os.path.join("data/processed/hist_response", "{model}_{dataset}_temp.csv"),
        "output_file": os.path.join("data/processed/hist_response", f"{openai_gpt5}.csv"),
        "max_tokens_per_day": 850000,  # 900k words per day for OpenAI GPT-5 (for safety, cap at 850k)
        "batch_size": 100,
        "lengthy_refusal_truncation": 30000  # Truncate to 30000 characters for lengthy refusals
    },
    "deepseek": {
        "api_key": deepseek_api_key,
        "model": deepseek_model,
        "rate_limit": 10,
        "temp_file_template": os.path.join("data/processed/hist_response", "{model}_{dataset}_temp.csv"),
        "output_file": os.path.join("data/processed/hist_response", f"{deepseek_model}.csv"),
        "batch_size": 50
    }
}