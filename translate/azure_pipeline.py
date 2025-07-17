import requests
import json
import uuid
import os
import pandas as pd
from tqdm import tqdm
from data_collection.wiki_extract import create_content_id
import time

def split_text_by_sentences(text, max_chars=49000):
    """
    Split text into chunks by sentences, respecting character limits
    """
    import re
    
    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split it by words
                words = sentence.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 > max_chars:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = word
                        else:
                            # Single word is too long, truncate it
                            chunks.append(word[:max_chars])
                            current_chunk = ""
                    else:
                        current_chunk += " " + word if current_chunk else word
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def translate_text_azure(text, api_key, endpoint, location="eastus", max_retries=3, delay=1):
    """
    Translate a single text using Azure Cognitive Services Translation API
    Handles large texts by splitting into sentences
    Returns: (translated_text, translation_failed_flag)
    """
    
    # Skip empty text
    if not text or not str(text).strip():
        return text, 0
    
    text = str(text).strip()
    
    # If text is within limits, translate directly
    if len(text) <= 49000:
        translated_chunks, failed_flags = translate_chunks_azure([text], api_key, endpoint, location, max_retries, delay)
        return translated_chunks[0], failed_flags[0]
    
    # Split large text into sentence-based chunks
    print(f"Text too large ({len(text)} chars), splitting into chunks...")
    chunks = split_text_by_sentences(text)
    print(f"Split into {len(chunks)} chunks")
    
    # Translate each chunk
    translated_chunks, failed_flags = translate_chunks_azure(chunks, api_key, endpoint, location, max_retries, delay)
    
    # If any chunk failed, mark the entire text as failed
    overall_failed = 1 if any(failed_flags) else 0
    
    # Combine translated chunks
    return " ".join(translated_chunks), overall_failed

def translate_chunks_azure(chunks, api_key, endpoint, location="eastus", max_retries=6, delay=2):
    """
    Translate a list of text chunks using Azure API
    Returns: (all_translations, failed_flags)
    """
    
    all_translations = []
    failed_flags = []
    
    for chunk in chunks:
        if not chunk or not chunk.strip():
            all_translations.append(chunk)
            failed_flags.append(0)
            continue
            
        translation_failed = 0
        
        for attempt in range(max_retries):
            try:
                path = '/translate'
                constructed_url = endpoint + path
                
                params = {
                    'api-version': '3.0',
                    'from': 'en',
                    'to': 'zh-Hans'  # Simplified Chinese
                }
                
                headers = {
                    "Ocp-Apim-Subscription-Key": api_key,
                    "Ocp-Apim-Subscription-Region": location,
                    "Content-type": 'application/json',
                    "X-ClientTraceId": str(uuid.uuid4())
                }
                
                body = [{'text': chunk}]
                
                response = requests.post(constructed_url, params=params, headers=headers, json=body)
                
                # Print response details for debugging on first attempt
                if attempt == 0 and response.status_code != 200:
                    print(f"Response status: {response.status_code}")
                    print(f"Response content: {response.text}")
                
                response.raise_for_status()
                
                response_data = response.json()
                
                # Extract translation
                if isinstance(response_data, list) and len(response_data) > 0:
                    if 'translations' in response_data[0] and len(response_data[0]['translations']) > 0:
                        all_translations.append(response_data[0]['translations'][0]['text'])
                        failed_flags.append(0)  # Success
                        break  # Success, move to next chunk
                    else:
                        print(f"Unexpected API response structure: {response_data}")
                        all_translations.append(chunk)  # Return original if failed
                        failed_flags.append(1)  # Mark as failed
                        translation_failed = 1
                        break
                else:
                    print(f"Unexpected API response structure: {response_data}")
                    raise ValueError("Invalid API response structure")
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    print(f"Rate limit hit on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        print(f"Waiting {wait_time} seconds for rate limit...")
                        time.sleep(wait_time)
                    continue
                elif e.response.status_code == 400:  # Bad Request
                    print(f"Bad request on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    continue
                else:
                    print(f"HTTP error on attempt {attempt + 1}: {e}")
            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        else:
            # If all attempts failed, return original chunk and mark as failed
            print(f"Failed to translate chunk after {max_retries} attempts: {chunk[:100]}...")
            all_translations.append(chunk)
            failed_flags.append(1)
        
        # Add small delay between chunks to respect rate limits
        time.sleep(0.5)
    
    return all_translations, failed_flags

def process_dataframe_azure(df, api_key, endpoint, location="eastus"):
    """
    Process a DataFrame by translating each row's content separately
    Respects Azure's 40M chars/hour rate limit (666,667 chars/minute)
    """
    print(f"Processing {len(df)} rows")
    
    # Calculate total characters
    total_chars = df['content'].str.len().sum()
    print(f"Total characters to translate: {total_chars:,}")
    
    # Rate limiting constants
    CHARS_PER_HOUR_LIMIT = 40_000_000  # 40 million characters per hour
    CHARS_PER_MINUTE_LIMIT = CHARS_PER_HOUR_LIMIT / 60  # ~666,667 chars per minute
    
    # Translate each row's content and track failures
    translations = []
    failed_flags = []
    
    # Rate limiting variables
    start_time = time.time()
    chars_processed_this_minute = 0
    minute_start_time = start_time
    
    for idx, content in tqdm(df['content'].items(), desc="Translating content", total=len(df)):
        content_chars = len(str(content))
        current_time = time.time()
        
        # Check if we've been processing for more than a minute
        if current_time - minute_start_time >= 60:
            # Reset the minute counter
            chars_processed_this_minute = 0
            minute_start_time = current_time
        
        # Check if adding this content would exceed the per-minute limit
        if chars_processed_this_minute + content_chars > CHARS_PER_MINUTE_LIMIT:
            # Calculate how long to wait until the next minute
            time_elapsed_this_minute = current_time - minute_start_time
            time_to_wait = 60 - time_elapsed_this_minute
            
            if time_to_wait > 0:
                print(f"\nRate limit protection: Processed {chars_processed_this_minute:,} chars this minute.")
                print(f"Waiting {time_to_wait:.1f} seconds to respect 40M chars/hour limit...")
                time.sleep(time_to_wait)
                
                # Reset counters
                chars_processed_this_minute = 0
                minute_start_time = time.time()
        
        # Translate the content
        translated_text, failed = translate_text_azure(content, api_key, endpoint, location)
        translations.append(translated_text)
        failed_flags.append(failed)
        
        # Update character count for this minute
        chars_processed_this_minute += content_chars
    
    # Update the dataframe
    df['content'] = translations
    df['translation_failed'] = failed_flags
    
    # Report processing time and rate
    total_time = time.time() - start_time
    chars_per_hour_actual = (total_chars / total_time) * 3600 if total_time > 0 else 0
    
    print(f"\nProcessing completed:")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Actual rate: {chars_per_hour_actual:,.0f} chars/hour")
    print(f"  Rate limit: {CHARS_PER_HOUR_LIMIT:,} chars/hour")
    
    return df

# Load configuration
with open("translate/config.json", "r") as f:
    config = json.load(f)

api_key = config['Azure_Translate_API_Key']
endpoint = "https://api.cognitive.microsofttranslator.com"
location = "eastus"

# Test with a simple translation first
print("Testing Azure Translation API...")
test_text = "Hello, world!"
result, failed = translate_text_azure(test_text, api_key, endpoint, location)
print(f"Test translation: '{test_text}' -> '{result}' (failed: {failed})")

if result == test_text or failed == 1:
    print("Translation test failed. Please check your API key and configuration.")
    exit(1)

# Set up paths
wiki_en_path = "data/processed/wiki_content"
wiki_cn_path = "data/processed/wiki_content_cn_azure"

if not os.path.exists(wiki_cn_path):
    os.makedirs(wiki_cn_path, exist_ok=True)

# Get and sort CSV files
csv_files = sorted([f for f in os.listdir(wiki_en_path) if f.endswith('.csv')])
first_n_files = 52  # Limit to total 52 files
print(f"Found {len(csv_files)} CSV files to process")
print("Files to translate:", csv_files[:first_n_files])  # Show first n files

test_counter = 0


# Translate the first CSV file only for testing purposes
for csv_file in csv_files[:first_n_files]:  # Only process first n files
    # Check if the file has already been translated
    output_path = os.path.join(wiki_cn_path, csv_file)
    if os.path.exists(output_path):
        print(f"File {csv_file} already exists in {wiki_cn_path}, skipping translation.")
        continue
    
    print(f"Processing file: {csv_file}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(os.path.join(wiki_en_path, csv_file))
        print(f"Loaded {len(df)} rows from {csv_file}")
        
        # Translate content using row-by-row processing
        df_translated = process_dataframe_azure(df, api_key, endpoint, location)
        
        # Update DataFrame with new content IDs
        df_translated.rename(columns={'content_id': 'content_id_en'}, inplace=True)
        df_translated['content_id_cn'] = df_translated['content'].apply(create_content_id)

        # Save translated file
        df_translated.to_csv(output_path, index=False)
        print(f"Translated {csv_file} and saved to {wiki_cn_path}")
        
        # Report translation statistics
        total_rows = len(df_translated)
        failed_rows = df_translated['translation_failed'].sum()
        success_rate = ((total_rows - failed_rows) / total_rows) * 100
        
        print(f"Translation Summary for {csv_file}:")
        print(f"  Total rows: {total_rows}")
        print(f"  Successfully translated: {total_rows - failed_rows}")
        print(f"  Failed translations: {failed_rows}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        if failed_rows > 0:
            print(f"\nNote: {failed_rows} rows failed translation and are marked with translation_failed=1")
            print("Use azure_validation.py to review and retry failed translations.")
                
        test_counter += 1
        
    except Exception as e:
        print(f"Error during translation: {e}")
        raise

print(f"Translated {test_counter} file successfully.")

# Calculate and display estimated cost
if test_counter > 0:
    print("\n" + "="*50)
    print("COST ESTIMATION")
    print("="*50)
    
    total_chars_processed = 0
    total_rows_processed = 0
    total_failed = 0
    
    # Calculate total characters processed across all files
    for csv_file in csv_files[:first_n_files]:  # Only process first n files
        output_path = os.path.join(wiki_cn_path, csv_file)
        if os.path.exists(output_path):
            df_result = pd.read_csv(output_path)
            # Use original content length for cost calculation (what was sent to API)
            original_file = os.path.join(wiki_en_path, csv_file)
            if os.path.exists(original_file):
                df_original = pd.read_csv(original_file)
                file_chars = df_original['content'].str.len().sum()
                total_chars_processed += file_chars
                total_rows_processed += len(df_result)
                total_failed += df_result['translation_failed'].sum()
    
    # Calculate cost at $10 per 1M characters
    cost_per_million = 10.0  # USD
    estimated_cost = (total_chars_processed / 1_000_000) * cost_per_million
    
    print(f"Total characters sent to Azure API: {total_chars_processed:,}")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Total failed translations: {total_failed:,}")
    print(f"Estimated cost: ${estimated_cost:.4f} USD")
    print(f"Cost per character: ${cost_per_million/1_000_000:.8f} USD")
    
    if total_chars_processed > 0:
        success_rate = ((total_rows_processed - total_failed) / total_rows_processed) * 100
        print(f"Overall success rate: {success_rate:.1f}%")
    
    print("\nNote: Cost is estimated based on Azure Translator pricing of $10 per 1M characters.")
    print("Actual costs may vary. Check your Azure billing for precise charges.") 