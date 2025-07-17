import pandas as pd
from pathlib import Path
from data_processor import get_all_content, get_wiki_content
from openai_client import OpenAIClient
from config import config
from tqdm import tqdm
import argparse

def find_missing_content_ids(historical_file_path: str, current_content: pd.DataFrame) -> set:
    """
    Find content IDs that exist in current content but are missing from historical responses.
    
    Args:
        historical_file_path: Path to the historical response CSV file
        current_content: Current content DataFrame from data_processor
        
    Returns:
        Set of missing content IDs
    """
    # Read historical responses
    historical_df = pd.read_csv(historical_file_path)
    
    # Get content IDs from both datasets
    historical_ids = set(historical_df['content_id'].unique())
    current_ids = set(current_content['content_id'].unique())
    
    # Find missing IDs (in current but not in historical)
    missing_ids = current_ids - historical_ids
    
    print(f"Historical responses: {len(historical_ids)} content IDs")
    print(f"Current content: {len(current_ids)} content IDs")
    print(f"Missing responses: {len(missing_ids)} content IDs")
    
    return missing_ids

def run_adhoc_collection(missing_ids: set, content_df: pd.DataFrame, api_name: str = "openai"):
    """
    Run ad-hoc response collection for missing content IDs.
    
    Args:
        missing_ids: Set of content IDs to process
        content_df: Full content DataFrame
        api_name: API to use for collection
    """
    # Filter content to only missing IDs
    missing_content = content_df[content_df['content_id'].isin(missing_ids)].copy()
    
    if missing_content.empty:
        print("No missing content found to process.")
        return
    
    print(f"Processing {len(missing_content)} missing content items...")
    
    # Initialize API client
    if api_name == "openai":
        client = OpenAIClient(config["openai"]["api_key"])
        init_sleep = 1 / config["openai"]["rate_limit"]
    else:
        raise ValueError(f"API {api_name} not supported in this ad-hoc script")
    
    # Collect responses
    results = []
    for _, row in tqdm(missing_content.iterrows(), total=len(missing_content)):
        try:
            response = client.call_moderation(
                content=row['content'],
                init_sleep=init_sleep,
                additional_sleep=2,
                max_retries=10
            )
            
            results.append({
                'content_id': row['content_id'],
                'flagged': response.get('flagged', -1),
                'model_response': response.get('model_response', -1),
                'model': config[api_name]['model'],
                'date': pd.Timestamp.now().strftime('%Y-%m-%d')
            })
            
        except Exception as e:
            print(f"Error processing content_id {row['content_id']}: {str(e)}")
            results.append({
                'content_id': row['content_id'],
                'flagged': -1,
                'model_response': str(e),
                'model': config[api_name]['model'],
                'date': pd.Timestamp.now().strftime('%Y-%m-%d')
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"data/processed/hist_response/adhoc_missing_{api_name}_{timestamp}.csv"
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    print(f"✓ Ad-hoc results saved to: {output_path}")
    
    return results_df

def merge_with_historical(adhoc_results_path: str, historical_file_path: str):
    """
    Merge ad-hoc results with historical file to create complete dataset.
    
    Args:
        adhoc_results_path: Path to ad-hoc results CSV
        historical_file_path: Path to historical response CSV
    """
    # Read both files
    adhoc_df = pd.read_csv(adhoc_results_path)
    historical_df = pd.read_csv(historical_file_path)
    
    # Combine and remove any duplicates (keeping adhoc version if duplicate)
    combined_df = pd.concat([historical_df, adhoc_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['content_id'], keep='last')
    
    # Save merged file
    base_path = Path(historical_file_path)
    merged_path = base_path.parent / f"{base_path.stem}_complete{base_path.suffix}"
    combined_df.to_csv(merged_path, index=False)
    
    print(f"✓ Merged results saved to: {merged_path}")
    print(f"  - Original: {len(historical_df)} rows")
    print(f"  - Ad-hoc: {len(adhoc_df)} rows") 
    print(f"  - Combined: {len(combined_df)} rows")

def main():
    parser = argparse.ArgumentParser(description="Run ad-hoc response collection for missing content IDs")
    parser.add_argument("--historical-file", 
                       required=True,
                       help="Path to historical response CSV file")
    parser.add_argument("--api",
                       choices=["openai"],
                       default="openai", 
                       help="API to use for missing responses")
    parser.add_argument("--merge-only",
                       action="store_true",
                       help="Only merge existing ad-hoc results with historical file")
    parser.add_argument("--adhoc-results",
                       help="Path to existing ad-hoc results file (for merge-only mode)")
    
    args = parser.parse_args()
    
    if args.merge_only:
        if not args.adhoc_results:
            print("--adhoc-results is required when using --merge-only")
            return
        merge_with_historical(args.adhoc_results, args.historical_file)
        return
    
    try:
        # Load current content
        print("Loading current content...")
        current_content = get_wiki_content()
        
        # Find missing content IDs
        print("Finding missing content IDs...")
        missing_ids = find_missing_content_ids(args.historical_file, current_content)
        
        if not missing_ids:
            print("No missing content IDs found. All content already has responses.")
            return
        
        # Run ad-hoc collection
        print(f"Running ad-hoc collection for {len(missing_ids)} missing content IDs...")
        adhoc_results = run_adhoc_collection(missing_ids, current_content, args.api)
        
        # Optionally merge immediately
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        adhoc_path = f"data/processed/hist_response/adhoc_missing_{args.api}_{timestamp}.csv"
        
        merge_choice = input("Merge with historical file now? (y/n): ").lower().strip()
        if merge_choice == 'y':
            merge_with_historical(adhoc_path, args.historical_file)
        
        print("✓ Ad-hoc collection completed!")
        
    except Exception as e:
        print(f"❌ Error in ad-hoc collection: {str(e)}")
        raise

if __name__ == "__main__":
    main()