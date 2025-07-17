import requests
import json
import uuid
import os
import pandas as pd
from tqdm import tqdm
import time
import re

def split_text_by_sentences(text, max_chars=49000):
    """
    Split text into chunks by sentences, respecting character limits
    """
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

def translate_text_azure_retry(text, api_key, endpoint, location="eastus", max_retries=6, delay=2):
    """
    Retry translation for failed texts with more aggressive parameters
    """
    
    # Skip empty text
    if not text or not str(text).strip():
        return text, 0
    
    text = str(text).strip()
    
    # If text is within limits, translate directly
    if len(text) <= 49000:
        translated_chunks, failed_flags = translate_chunks_azure_retry([text], api_key, endpoint, location, max_retries, delay)
        return translated_chunks[0], failed_flags[0]
    
    # Split large text into sentence-based chunks
    print(f"Text too large ({len(text)} chars), splitting into chunks...")
    chunks = split_text_by_sentences(text)
    print(f"Split into {len(chunks)} chunks")
    
    # Translate each chunk
    translated_chunks, failed_flags = translate_chunks_azure_retry(chunks, api_key, endpoint, location, max_retries, delay)
    
    # If any chunk failed, mark the entire text as failed
    overall_failed = 1 if any(failed_flags) else 0
    
    # Combine translated chunks
    return " ".join(translated_chunks), overall_failed

def translate_chunks_azure_retry(chunks, api_key, endpoint, location="eastus", max_retries=5, delay=2):
    """
    Retry translation for chunks with more conservative rate limiting
    """
    
    all_translations = []
    failed_flags = []
    
    for chunk in chunks:
        if not chunk or not chunk.strip():
            all_translations.append(chunk)
            failed_flags.append(0)
            continue
            
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
                
                # More conservative delay for retry attempts
                time.sleep(1.5)
                
                response = requests.post(constructed_url, params=params, headers=headers, json=body)
                
                if response.status_code != 200:
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
                        break
                else:
                    print(f"Unexpected API response structure: {response_data}")
                    raise ValueError("Invalid API response structure")
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    print(f"Rate limit hit on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        wait_time = delay * (3 ** attempt)  # More aggressive backoff
                        print(f"Waiting {wait_time} seconds for rate limit...")
                        time.sleep(wait_time)
                    continue
                elif e.response.status_code == 400:  # Bad Request
                    print(f"Bad request on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delay * 2)
                    continue
                else:
                    print(f"HTTP error on attempt {attempt + 1}: {e}")
            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {delay * 2} seconds...")
                time.sleep(delay * 2)
        else:
            # If all attempts failed, return original chunk and mark as failed
            print(f"Failed to translate chunk after {max_retries} attempts: {chunk[:100]}...")
            all_translations.append(chunk)
            failed_flags.append(1)
    
    return all_translations, failed_flags

def detect_translation_issues(df):
    """
    Detect various types of translation issues in the DataFrame
    """
    issues = []
    
    for idx, row in df.iterrows():
        content = str(row['content'])
        issue_types = []
        
        # Check if translation_failed flag is set
        if 'translation_failed' in df.columns and row['translation_failed'] == 1:
            issue_types.append("marked_as_failed")
        
        # Check for very short content that might indicate failure
        if len(content.strip()) < 10:
            issue_types.append("too_short")
        
        # Check for repeated patterns that might indicate API errors
        if len(set(content.split())) < len(content.split()) * 0.3:  # Too much repetition
            issue_types.append("repetitive_content")
        
        if issue_types:
            issues.append({
                'index': idx,
                'content_id_en': row.get('content_id_en', 'unknown'),
                'content_preview': content[:100] + '...' if len(content) > 100 else content,
                'issue_types': issue_types,
                'content_length': len(content)
            })
    
    return issues

def validate_and_retry_translations(file_path, api_key, endpoint, location="eastus"):
    """
    Load a translated file, validate translations, and retry failed ones
    """
    
    print(f"Loading file: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df)} rows")
    
    # Detect issues
    print("Detecting translation issues...")
    issues = detect_translation_issues(df)
    
    print(f"Found {len(issues)} rows with potential issues:")
    for issue in issues[:10]:  # Show first 10 issues
        print(f"  Row {issue['index']}: {', '.join(issue['issue_types'])} - {issue['content_preview']}")
    
    if len(issues) > 10:
        print(f"  ... and {len(issues) - 10} more")
    
    # Ask user if they want to retry
    if not issues:
        print("No issues found!")
        return df
    
    retry_choice = input(f"\nFound {len(issues)} potential issues. Retry translation for these rows? (y/n): ")
    
    if retry_choice.lower() != 'y':
        print("Skipping retry. Returning original DataFrame.")
        return df
    
    # Load original English content for retry
    original_file = file_path.replace('_cn_azure', '')  # Get original file path
    if not os.path.exists(original_file):
        print(f"Warning: Original English file not found at {original_file}")
        print("Cannot retry translation without original content.")
        return df
    
    print(f"Loading original English content from: {original_file}")
    df_original = pd.read_csv(original_file)
    
    # Retry translations for problematic rows
    retry_count = 0
    for issue in tqdm(issues, desc="Retrying translations"):
        idx = issue['index']
        
        if idx >= len(df_original):
            print(f"Warning: Row {idx} not found in original file, skipping...")
            continue
        
        original_content = df_original.loc[idx, 'content']
        print(f"Retrying translation for row {idx}...")
        
        # Retry translation
        new_translation, still_failed = translate_text_azure_retry(
            original_content, api_key, endpoint, location
        )
        
        # Update the DataFrame
        df.loc[idx, 'content'] = new_translation
        if 'translation_failed' in df.columns:
            df.loc[idx, 'translation_failed'] = still_failed
        
        if not still_failed:
            retry_count += 1
            print(f"  Success: {new_translation[:50]}...")
        else:
            print(f"  Still failed: {new_translation[:50]}...")
    
    print(f"Successfully retranslated {retry_count} out of {len(issues)} problematic rows")
    
    return df

def main():
    """
    Main validation script
    """
    
    # Load configuration
    with open("translate/config.json", "r") as f:
        config = json.load(f)
    
    api_key = config['Azure_Translate_API_Key']
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = "eastus"
    
    # Set up paths
    wiki_cn_path = "data/processed/wiki_content_cn_azure"
    
    if not os.path.exists(wiki_cn_path):
        print(f"Directory {wiki_cn_path} does not exist!")
        return
    
    # Get all translated CSV files (excluding backup files)
    csv_files = sorted([f for f in os.listdir(wiki_cn_path) 
                       if f.endswith('.csv') and not f.endswith('_backup.csv')])
    
    if not csv_files:
        print("No translated CSV files found!")
        return
    
    print(f"Found {len(csv_files)} translated files:")
    for i, file in enumerate(csv_files):
        print(f"  {i+1}. {file}")
    
    # Let user choose which file to validate
    choice = input(f"\nEnter file number to validate (1-{len(csv_files)}) or 'all' for all files: ")
    
    if choice.lower() == 'all':
        files_to_process = csv_files
    else:
        try:
            file_idx = int(choice) - 1
            if 0 <= file_idx < len(csv_files):
                files_to_process = [csv_files[file_idx]]
            else:
                print("Invalid choice!")
                return
        except ValueError:
            print("Invalid choice!")
            return
    
    # Process selected files
    for csv_file in files_to_process:
        file_path = os.path.join(wiki_cn_path, csv_file)
        print(f"\n{'='*60}")
        print(f"Validating: {csv_file}")
        print(f"{'='*60}")
        
        # Validate and retry
        df_updated = validate_and_retry_translations(file_path, api_key, endpoint, location)
        
        # Save updated file
        backup_path = file_path.replace('.csv', '_backup.csv')
        print(f"Creating backup: {backup_path}")
        df_original = pd.read_csv(file_path)
        df_original.to_csv(backup_path, index=False)
        
        print(f"Saving updated file: {file_path}")
        df_updated.to_csv(file_path, index=False)
        
        print(f"Validation completed for {csv_file}")

if __name__ == "__main__":
    main()
