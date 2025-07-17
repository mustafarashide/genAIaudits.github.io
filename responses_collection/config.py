import os
import json

with open("responses_collection/api_config.json", "r") as f:
    api_config = json.load(f)
    openai_api_key = api_config["openai-me"]
    deepseek_api_key = api_config["deepseek"]
    openai_sorelle_api_key = api_config["openai_sorelle"]

openai_me_model = "omni-moderation-latest"
openai_gpt_model = "gpt-4.1"
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
    "openai-gpt": {
        "api_key": openai_sorelle_api_key,
        "model": openai_gpt_model,
        "endpoint": "/v1/chat/completions",
        "rate_limit": 4,
        "temp_file_template": os.path.join("data/processed/hist_response", "{model}_{dataset}_temp.csv"),
        "output_file": os.path.join("data/processed/hist_response", f"{openai_gpt_model}.csv"),
        "max_tokens_per_day": 850000,  # 900k words per day for OpenAI GPT-4.1 (for safety, cap at 850k)
        "batch_size": 100
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