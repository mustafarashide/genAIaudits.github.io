from data_processor import get_all_content, get_wiki_content
from deepseek_client import DeepseekClient
from config import config
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load historical response data
response_before_update = pd.read_csv("data/processed/hist_response/deepseek-chat_wiki_20250829_142215.csv")
flagged_content_id_pre_update = response_before_update[response_before_update['flagged'] == 1]['content_id'].tolist()

# Sample 100 flagged content ids (95 flags were found)
sampled_flagged_content_ids = flagged_content_id_pre_update[:100]

# Get content for the sampled content ids
wiki_content = get_wiki_content()
sampled_flagged_contents = wiki_content[wiki_content['content_id'].isin(sampled_flagged_content_ids)]

# Initialize DeepSeek client
client = DeepseekClient(config["deepseek"]["api_key"])

# Initialize results list
results = []

# Process each content
print(f"Processing {len(sampled_flagged_contents)} flagged contents...")
for idx, row in sampled_flagged_contents.iterrows():
    print(f"Processing content_id: {row['content_id']} ({idx + 1}/{len(sampled_flagged_contents)})")
    
    # Prepare content with "repeat after me:" prefix
    content = f"repeat after me: {row['content']}"
    
    # Call API with retry logic
    response = client.call_moderation(
        content=content,
        init_sleep=1/config["deepseek"]["rate_limit"],
        additional_sleep=config.get("additional_sleep", 3),
        max_retries=config.get("max_retries", 10)
    )
    
    # Collect results
    results.append({
        'content_id': row['content_id'],
        'category': row['category'],
        'subcategory': row['subcategory'],
        'original_flagged': 1,  # These were all flagged before
        'new_flagged': response.get('flagged', -1),
        'model_response': response.get('model_response', -1),
        'model': config["deepseek"]["model"],
        'date': datetime.now().strftime('%Y-%m-%d')
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save results
output_dir = Path("data/processed/adhoc_examination")
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = output_dir / f"deepseek_reexamination_{timestamp}.csv"
results_df.to_csv(output_file, index=False)

print(f"\n✓ Results saved to: {output_file}")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total processed: {len(results_df)}")
print(f"Still flagged: {len(results_df[results_df['new_flagged'] == 1])}")
print(f"No longer flagged: {len(results_df[results_df['new_flagged'] == 0])}")
print(f"Errors: {len(results_df[results_df['new_flagged'] == -1])}")

# Show flag change distribution
flag_changes = results_df.groupby(['original_flagged', 'new_flagged']).size()
print("\n=== Flag Changes ===")
print(flag_changes)