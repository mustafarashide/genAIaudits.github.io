import pandas as pd
from pathlib import Path

def concat_wiki_content():
    """
    Read and concatenate all CSV files from the wiki_content directory.
    
    Returns:
        pandas.DataFrame: Combined DataFrame containing all wiki content
    """
    # Define the path to wiki_content directory
    wiki_content_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'wiki_content'
    
    # Get list of all CSV files
    csv_files = list(wiki_content_dir.glob('*.csv'))
    
    # Read and concatenate all CSV files
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
    
    # Concatenate all dataframes with invariant checking 
    df_output = pd.concat(df_list, ignore_index=True)
    assert df_output.shape[0] == 4210, "The combined DataFrame does not have the expected number of rows."
    assert df_output.topic.nunique() == 421, "The combined DataFrame does not have the expected number of unique topics."
    return df_output

def concat_cn_wiki_content():
    """
    Read and concatenate all CSV files from the wiki_content directory.
    
    Returns:
        pandas.DataFrame: Combined DataFrame containing all wiki content
    """
    # Define the path to wiki_content directory
    cn_wiki_content_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'wiki_content_cn_azure'
    
    # Get list of all CSV files
    csv_files = list(cn_wiki_content_dir.glob('*.csv'))
    
    # Read and concatenate all CSV files
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
    
    # Concatenate all dataframes with invariant checking 
    df_output = pd.concat(df_list, ignore_index=True)
    assert df_output.shape[0] == 4210, "The combined DataFrame does not have the expected number of rows."
    assert df_output.topic.nunique() == 421, "The combined DataFrame does not have the expected number of unique topics."
    return df_output

def get_wiki_content():
    """
    Read and concatenate all CSV files for llm evaluation
    
    Returns:
        pandas.DataFrame: a processed version for llm evaluation
    """   
    # Get list of all CSV files
    df_wiki = concat_wiki_content()

    # Using a simple approximation - splitting on whitespace
    df_wiki['token_length'] = df_wiki['content'].str.split().str.len()
    
    # Rename wiki columns to match movie columns
    df_wiki = df_wiki.rename(columns={
        'topic': 'subcategory',
        'wiki_title': 'source'
    })
    
    # Select and reorder columns to match movie structure
    df_wiki = df_wiki[[
        'category',
        'subcategory',
        'source',
        'content',
        'token_length',
        'revision_id',
        'permanent_link',
        'content_id'
    ]]    
    assert df_wiki.shape[0] == 4210, "The combined DataFrame does not have the expected number of rows."
    assert df_wiki.subcategory.nunique() == 421, "The combined DataFrame does not have the expected number of unique topics."
    return df_wiki

def get_tv_movie_content():
    """
    Read and process TV/movie content for LLM evaluation.
    
    Returns:
        pandas.DataFrame: Processed TV/movie content DataFrame
    """
    df_movie = pd.read_csv(Path(__file__).parent.parent / 'data' / 'processed' / 'movie_tv_content.csv')
    
    # Add token length calculation if not present
    if 'token_length' not in df_movie.columns:
        df_movie['token_length'] = df_movie['content'].str.split().str.len()
    
    # Validate data integrity
    assert not df_movie.empty, "Movie/TV content DataFrame is empty."
    assert 'content' in df_movie.columns, "Movie/TV content DataFrame missing 'content' column."
    
    return df_movie

def get_all_content():
    """
    Combine wiki and movie content with aligned column names and additional processing.
    
    Returns:
        pandas.DataFrame: Combined DataFrame with standardized columns
    """
    # Get wiki content and movie content
    df_wiki = get_wiki_content()
    df_movie = get_tv_movie_content()
    
    # Concatenate dataframes
    df_combined = pd.concat([df_movie, df_wiki], ignore_index=True)
    
    # Check for duplicates
    assert df_combined.shape[0] == df_movie.shape[0] + df_wiki.shape[0], "The combined DataFrame does not have the expected number of rows."
    assert df_combined.duplicated(subset=['category', 'subcategory', 'source', 'content_id']).sum() == 0, \
        f"Found {df_combined.duplicated(subset=['category', 'subcategory', 'source', 'content_id']).sum()} duplicates in the combined dataset based on category, subcategory, and source"
    
    return df_combined

def get_cn_wiki_content():
    """
    Read and concatenate all CSV files for llm evaluation
    
    Returns:
        pandas.DataFrame: a processed version for llm evaluation
    """   
    # Get list of all CSV files
    df_wiki = concat_cn_wiki_content()

    # Using a simple approximation - splitting on whitespace
    df_wiki['token_length'] = df_wiki['content'].str.split().str.len()
    
    # Rename wiki columns to match movie columns
    df_wiki = df_wiki.rename(columns={
        'topic': 'subcategory',
        'wiki_title': 'source',
        'content_id_en': 'content_id'
    })
    
    # Select and reorder columns to match movie structure
    df_wiki = df_wiki[[
        'category',
        'subcategory',
        'source',
        'content',
        'token_length',
        'revision_id',
        'permanent_link',
        'content_id',
        'content_id_cn'
    ]]    
    assert df_wiki.shape[0] == 4210, "The combined DataFrame does not have the expected number of rows."
    assert df_wiki.subcategory.nunique() == 421, "The combined DataFrame does not have the expected number of unique topics."
    return df_wiki

if __name__ == "__main__":
    # Test all functions
    print("Testing get_cn_wiki_content()...")
    df_cn_wiki = get_cn_wiki_content()
    print(f"Chinese Wiki rows: {len(df_cn_wiki)}")
    print(df_cn_wiki.columns)
