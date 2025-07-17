import pandas as pd
from pathlib import Path
import os
import hashlib
from typing import List, Tuple

def create_content_id(content: str) -> str:
    """
    Create a unique ID for content using SHA-256 hash.
    
    Args:
        content: The content text to hash
        
    Returns:
        str: First 12 characters of the hash
    """
    if pd.isna(content):
        content = ""
    content_hash = hashlib.sha256(str(content).encode('utf-8')).hexdigest()
    return content_hash[:12]

def print_csv_columns(file_path: str, file_name: str) -> None:
    """
    Print the column names of a CSV file.
    
    Args:
        file_path: Path to the CSV file
        file_name: Name of the CSV file for display purposes
    """
    try:
        df = pd.read_csv(file_path)
        print(f"\nColumns in {file_name}:")
        for col in df.columns:
            print(f"- {col}")
        print(f"Total number of columns: {len(df.columns)}")
        print(f"Total number of rows: {len(df)}\n")
    except Exception as e:
        print(f"Error reading {file_name}: {str(e)}")

def preprocess_tmdb_data(file_path: str) -> pd.DataFrame:
    """
    Preprocess TMDB movie data.
    """
    df = pd.read_csv(file_path)
    processed_df = pd.DataFrame()
    
    processed_df['subcategory'] = df['rating']
    processed_df['category'] = 'Movie'
    processed_df['source'] = df['Title']
    processed_df['content'] = df['plots']
    processed_df['token_length'] = df['plots'].str.split().str.len()
    processed_df['revision_id'] = -1
    processed_df['permanent_link'] = -1
    processed_df['content_id'] = processed_df['content'].apply(create_content_id)
    
    return processed_df

def preprocess_tv_data(file_path: str, dataset_type: str) -> pd.DataFrame:
    """
    Preprocess TV show data.
    """
    print(f"Processing {dataset_type} TV data...")
    df = pd.read_csv(file_path)
    
    processed_df = pd.DataFrame(index=range(len(df)))
    processed_df['category'] = ['TV Shows'] * len(df)
    processed_df['subcategory'] = df['age_rating']
    
    processed_df['source'] = (df['show_name'].fillna('') + ' - ' + 
                            df['episode_title'].fillna('') + 
                            f' ({dataset_type})')
    
    content_col = {
        'short': 'episode-overview',
        'mid': 'wiki_descs',
        'long': 'synopsis_with_character_names'
    }[dataset_type]
    
    processed_df['content'] = df[content_col]
    processed_df['token_length'] = df[content_col].str.split().str.len()
    processed_df['revision_id'] = -1
    processed_df['permanent_link'] = -1
    processed_df['content_id'] = processed_df['content'].apply(create_content_id)
    
    return processed_df

def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate data based on content_id while keeping the first occurrence.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Deduplicated DataFrame
    """
    # Check for duplicates before deduplication
    duplicate_mask = df.duplicated(subset=['category','subcategory','source','content_id'], keep=False)
    if duplicate_mask.any():
        n_duplicates = duplicate_mask.sum()
        print(f"\nFound {n_duplicates} rows with duplicate content")
        print("Sample of duplicates before deduplication:")
        print(df[duplicate_mask][['category','subcategory','source','content_id']].head())
    
    # Keep first occurrence of each content_id
    deduped_df = df.drop_duplicates(subset=['category','subcategory','source','content_id'], keep='first')
    
    # Print deduplication results
    n_removed = len(df) - len(deduped_df)
    print(f"\nRemoved {n_removed} duplicate rows")
    print(f"Rows before deduplication: {len(df)}")
    print(f"Rows after deduplication: {len(deduped_df)}")
    
    return deduped_df

def main():
    # Define base paths
    base_dir = Path("data/raw_data")
    tv_movie_dir = base_dir / "tv_movie_data"
    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize list to store all dataframes
    all_dfs = []
    
    # Process TV/Movie data
    tv_movie_files = [
        ("short_TMDB_with_ME.csv", "short"),
        ("mid_wiki_with_ME.csv", "mid"),
        ("long_IMDB_with_ME.csv", "long"),
        ("TMDB_with_ME.csv", None)  # Movie data
    ]
    
    for file_name, dataset_type in tv_movie_files:
        file_path = tv_movie_dir / file_name
        if file_path.exists():
            if dataset_type:  # TV show data
                df = preprocess_tv_data(str(file_path), dataset_type)
            else:  # Movie data
                df = preprocess_tmdb_data(str(file_path))
            all_dfs.append(df)
            print(f"Processed {file_name}: {len(df)} rows")
    
    # Merge all dataframes
    print("\nMerging all datasets...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # Deduplicate data
    print("\nDeduplicating data...")
    deduped_df = deduplicate_data(merged_df)
    
    # Save processed and deduplicated data
    output_path = output_dir / "movie_tv_content.csv"
    deduped_df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")

if __name__ == "__main__":
    main()