import pandas as pd
from responses_collection.openai_batch_client import OpenAIBatchClient, _extract_openai_gpt_response, _extract_flagged_status
from responses_collection.config import config
from responses_collection.data_processor import get_wiki_content
import json
import hashlib
from pathlib import Path
import time
from datetime import datetime
import logging

# Setup logging
log_dir = Path("responses_collection/response_variance/logs")
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f"gpt4_1_variance_experiment_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to console for debugging
    ]
)

logger = logging.getLogger(__name__)

logger.info("Starting GPT-4.1 variance experiment")

# Prepare dataset
logger.info("Loading wiki content dataset")
wiki_df = get_wiki_content()
target_subcategory = ["Abortion", "Israel Global Image"]
target_df = wiki_df[wiki_df['subcategory'].isin(target_subcategory)].reset_index(drop=True)

# Sort by content_id to ensure consistent ordering across runs
target_df = target_df.sort_values('content_id').reset_index(drop=True)

duplicated_df = pd.concat([target_df]*10, ignore_index=True)
# add instruction prefix
duplicated_df['content'] = duplicated_df['content'].apply(lambda x: f"repeat after me: {x}")

# Copy content_id to content_id_copy
duplicated_df['content_id_copy'] = duplicated_df['content_id']

# Create deterministic unique content_id by appending duplicate number to original hash
duplicated_df['duplicate_number'] = duplicated_df.groupby('content_id_copy').cumcount() + 1
duplicated_df['content_id'] = duplicated_df['content_id_copy'] + '_' + duplicated_df['duplicate_number'].astype(str).str.zfill(3)

logger.info(f"Prepared dataset with {len(duplicated_df)} items from {len(target_df)} original items")

# Initialize client
logger.info("Initializing OpenAI GPT-4.1 client")
gpt4_1_client = OpenAIBatchClient(
    api_key=config["openai-gpt4.1"]["api_key"],
    model=config["openai-gpt4.1"]["model"],
    endpoint=config["openai-gpt4.1"]["endpoint"]
)

# Batch processing parameters
batch_size = 100
total_batches = len(duplicated_df) // batch_size + (1 if len(duplicated_df) % batch_size != 0 else 0)
temp_file = "responses_collection/response_variance/gpt4_1_variance_responses_temp.csv"
final_file = f"responses_collection/response_variance/gpt4_1_variance_responses_{timestamp}.csv"

logger.info(f"Batch processing parameters: batch_size={batch_size}, total_batches={total_batches}")

# Check for resumeability
processed_content_ids = set()
start_batch = 0

if Path(temp_file).exists():
    logger.info("Found existing temp file, checking for resumeability...")
    existing_df = pd.read_csv(temp_file)
    processed_content_ids = set(existing_df['content_id'].tolist())
    
    # Find first unprocessed item's position in duplicated_df
    remaining_ids = set(duplicated_df['content_id'].tolist()) - processed_content_ids
    if remaining_ids:
        # Find the DataFrame index of the first unprocessed item
        unprocessed_mask = duplicated_df['content_id'].isin(remaining_ids)
        first_unprocessed_idx = duplicated_df[unprocessed_mask].index[0]
        start_batch = first_unprocessed_idx // batch_size
        logger.info(f"Resuming from batch {start_batch + 1}, {len(processed_content_ids)} items already processed")
    else:
        logger.info("All items already processed, skipping to validation")
        start_batch = total_batches

# Process in batches
logger.info(f"Processing {len(duplicated_df)} items in {total_batches} batches of {batch_size}")

for batch_idx in range(start_batch, total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(duplicated_df))
    batch_df = duplicated_df.iloc[start_idx:end_idx].copy()
    
    # Skip already processed items
    batch_df = batch_df[~batch_df['content_id'].isin(processed_content_ids)]
    
    if len(batch_df) == 0:
        logger.info(f"Batch {batch_idx + 1} already processed, skipping")
        continue
    
    logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_df)} items)")
    
    # Process batch
    batch_responses = gpt4_1_client.process_dataset(batch_df)
    batch_responses_df = pd.DataFrame(batch_responses)

    # Add content_id_copy and truncated flag to responses
    batch_responses_df['content_id_copy'] = batch_df['content_id_copy'].values
    batch_responses_df['truncated'] = False
    
    
    # Append to temp file
    if not Path(temp_file).exists():
        batch_responses_df.to_csv(temp_file, index=False, mode='w')
    else:
        batch_responses_df.to_csv(temp_file, index=False, mode='a', header=False)
    
    # Update processed set
    processed_content_ids.update(batch_responses_df['content_id'].tolist())
    logger.info(f"Batch {batch_idx + 1} completed, saved to temp file")
    logger.info("Sleeping for 5 minutes before next batch...")
    time.sleep(300)  # sleep for 5 minutes after processing a batch

# Validation and retry loop
max_reruns = 3
rerun_count = 0

while rerun_count < max_reruns:
    logger.info(f"=== Validation Round {rerun_count + 1} ===")
    
    # Load all responses for validation
    all_responses_df = pd.read_csv(temp_file)
    
    # Identify issues
    error_mask = all_responses_df['flagged'] == -1
    lengthy_refusal_mask = (all_responses_df['flagged'] == 2) & (all_responses_df['truncated'] == False)
    
    error_count = error_mask.sum()
    lengthy_count = lengthy_refusal_mask.sum()
    
    logger.info(f"Total responses: {len(all_responses_df)}")
    logger.info(f"Error responses (-1 flag): {error_count}")
    logger.info(f"Lengthy refusals without truncation: {lengthy_count}")
    
    # If no issues, break the loop
    if error_count == 0 and lengthy_count == 0:
        logger.info("No issues found, validation complete!")
        break
    
    if rerun_count >= max_reruns - 1:
        logger.warning(f"Maximum reruns ({max_reruns}) reached, proceeding with current results")
        break
    
    # Save lengthy refusals before retry (if any exist)
    if lengthy_count > 0:
        # Create directory if it doesn't exist
        lengthy_refusal_dir = Path("data/processed/first_len_refusal")
        lengthy_refusal_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all lengthy refusals for this session
        lengthy_refusals_df = all_responses_df[lengthy_refusal_mask].copy()
        
        # Save to single file with timestamp
        lengthy_refusal_file = lengthy_refusal_dir / f"gpt4.1_variance_response_{timestamp}.csv"
        lengthy_refusals_df.to_csv(lengthy_refusal_file, index=False)
        logger.info(f"Saved {len(lengthy_refusals_df)} lengthy refusals to: {lengthy_refusal_file}")
    
    # Prepare retry data
    retry_mask = error_mask | lengthy_refusal_mask
    retry_df_list = []
    
    for _, row in all_responses_df[retry_mask].iterrows():
        # Find original content
        original_row = duplicated_df[duplicated_df['content_id'] == row['content_id']].iloc[0]
        
        retry_content = original_row['content']
        is_truncated = False
        
        # Apply truncation for lengthy refusals
        if row['flagged'] == 2:
            truncation_length = config["openai-gpt4.1"]["lengthy_refusal_truncation"]
            retry_content = original_row['content'][:truncation_length]
            is_truncated = True
        
        retry_df_list.append({
            'content_id': row['content_id'],
            'content': retry_content,
            'subcategory': original_row['subcategory'],
            'content_id_copy': original_row['content_id_copy'],
            'truncated': is_truncated
        })
    
    retry_df = pd.DataFrame(retry_df_list)
    
    logger.info(f"Retrying {len(retry_df)} failed responses...")
    
    # Process retry batch
    retry_responses = gpt4_1_client.process_dataset(retry_df)
    retry_responses_df = pd.DataFrame(retry_responses)
    
    # Add truncated flag to retry responses
    for i, retry_row in retry_df.iterrows():
        retry_responses_df.loc[i, 'truncated'] = retry_row['truncated']
    
    # Update original responses with retry results
    for _, retry_row in retry_responses_df.iterrows():
        mask = all_responses_df['content_id'] == retry_row['content_id']
        all_responses_df.loc[mask, 'model_response'] = retry_row['model_response']
        all_responses_df.loc[mask, 'flagged'] = retry_row['flagged']
        all_responses_df.loc[mask, 'truncated'] = retry_row['truncated']
    
    # Save updated responses back to temp file
    all_responses_df.to_csv(temp_file, index=False)
    
    logger.info(f"Retry round {rerun_count + 1} completed")
    rerun_count += 1

# Final validation and save
logger.info("=== Final Results ===")
final_responses_df = pd.read_csv(temp_file)

final_error_count = (final_responses_df['flagged'] == -1).sum()
final_lengthy_count = (final_responses_df['flagged'] == 2).sum()
truncated_count = (final_responses_df['truncated'] == True).sum()
valid_count = len(final_responses_df) - final_error_count

logger.info(f"Total responses: {len(final_responses_df)}")
logger.info(f"Valid responses: {valid_count}")
logger.info(f"Remaining errors: {final_error_count}")
logger.info(f"Lengthy refusals: {final_lengthy_count}")
logger.info(f"Truncated responses: {truncated_count}")
logger.info(f"Total reruns performed: {rerun_count}")

# Save final results with timestamp
final_responses_df.to_csv(final_file, index=False)
logger.info(f"Final results saved to: {final_file}")

# Clean up temp file
Path(temp_file).unlink()
logger.info("Temp file cleaned up")
logger.info(f"Experiment completed. Log saved to: {log_file}")
