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
openai_gpt51 = "gpt-5.1"
openai_gpt52 = "gpt-5.2"
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
        "max_tokens_per_day": 90000000,  # keep at min for long context, tier 4, 100,000,000 tokens per batch
        "batch_size": 100,
        "lengthy_refusal_truncation": 19000  # Truncate to 19000 characters for lengthy refusals (median of wiki)
    },
    "openai-gpt5": {
        "api_key": openai_sorelle_api_key,
        "model": openai_gpt5,
        "endpoint": "/v1/chat/completions",
        "rate_limit": 4,
        "temp_file_template": os.path.join("data/processed/hist_response", "{model}_{dataset}_temp.csv"),
        "output_file": os.path.join("data/processed/hist_response", f"{openai_gpt5}.csv"),
        "max_tokens_per_day": 190000000,  # tier 4 has 200,000,000 tokens per day; refer to https://platform.openai.com/docs/models/gpt-5
        "batch_size": 100,
        "lengthy_refusal_truncation": 19000  # Truncate to 19000 characters for lengthy refusals (median of wiki)
    },
    "openai-gpt5.1": {
        "api_key": openai_sorelle_api_key,
        "model": openai_gpt51,
        "endpoint": "/v1/chat/completions",
        "rate_limit": 4,
        "temp_file_template": os.path.join("data/processed/hist_response", "{model}_{dataset}_temp.csv"),
        "output_file": os.path.join("data/processed/hist_response", f"{openai_gpt51}.csv"),
        "max_tokens_per_day": 190000000,  # tier 4 has 200,000,000 tokens per day; refer to https://platform.openai.com/docs/models/gpt-5.1
        "batch_size": 100,
        "lengthy_refusal_truncation": 19000  # Truncate to 19000 characters for lengthy refusals (median of wiki)
    },
    "openai-gpt5.2": {
        "api_key": openai_sorelle_api_key,
        "model": openai_gpt52,
        "endpoint": "/v1/chat/completions",
        "rate_limit": 4,
        "temp_file_template": os.path.join("data/processed/hist_response", "{model}_{dataset}_temp.csv"),
        "output_file": os.path.join("data/processed/hist_response", f"{openai_gpt52}.csv"),
        "max_tokens_per_day": 190000000,  # tier 4 has 200,000,000 tokens per day; refer to https://platform.openai.com/docs/models/gpt-5.2
        "batch_size": 100,
        "lengthy_refusal_truncation": 19000  # Truncate to 19000 characters for lengthy refusals (median of wiki)
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