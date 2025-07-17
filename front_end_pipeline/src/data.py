"""Data loading and validation module."""
import pandas as pd
import json
from pathlib import Path
from typing import List
import glob
from responses_collection.data_processor import get_wiki_content, get_cn_wiki_content


def load_data(API: str, dataset_type: str, data_path: str = "data/processed/hist_response") -> pd.DataFrame:
    """
    Main function: Load and validate data for a specific API and dataset type.
    
    Args:
        API: API name, has to be one of ["openai-me", "openai-gpt", "deepseek"]
        dataset_type: Type of dataset ["wiki", "cn-wiki"]
        data_path: Base path to data directory
    
    Returns:
        Validated and concatenated DataFrame
    """
    # Validate API and dataset_type
    valid_APIs = ["openai-me", "openai-gpt", "deepseek"]
    if API not in valid_APIs:
        raise ValueError(f"Invalid API: {API}. Must be one of {valid_APIs}")
    
    valid_dataset_types = ["wiki", "cn-wiki"]
    if dataset_type not in valid_dataset_types:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be one of {valid_dataset_types}")
    
    # Load all CSV files for the API/dataset combination
    dataframes = _load_csv_files(data_path, API, dataset_type)
    
    # Validate and concatenate
    return _validate_and_concatenate(dataframes)


def _load_csv_files(data_path: str, API: str, dataset_type: str) -> List[pd.DataFrame]:
    """
    Helper function: Load all CSV files matching API and dataset type.
    
    Args:
        data_path: Base path to data directory
        API: API name
        dataset_type: Dataset type
    
    Returns:
        List of DataFrames from matching CSV files (empty list if no files found)
    """
    # Map API to model names
    api_to_models = {
        "openai-me": ["omni-moderation-latest", "text-moderation-007"],
        "openai-gpt": ["gpt-4.1"],
        "deepseek": ["deepseek-chat"]
    }
    
    if API not in api_to_models:
        raise ValueError(f"Unknown API: {API}")
    
    model_names = api_to_models[API]
    all_csv_files = []
    
    # Determine which dataset types to search for
    search_dataset_types = [dataset_type]
    if dataset_type == "wiki":
        search_dataset_types.append("all")
    
    # Find CSV files for each model and dataset type combination
    for model_name in model_names:
        for search_type in search_dataset_types:
            # Find all files matching the pattern (without wildcard for timestamp)
            potential_files = glob.glob(f"{data_path}/{model_name}_{search_type}_*.csv")
            
            # Filter out temp files and validate timestamp format
            for file_path in potential_files:
                filename = Path(file_path).name
                if filename.endswith('_temp.csv') or filename.endswith('not-flagged.csv') or filename.endswith('ex-cn.csv'):
                    continue  # Skip temp and not-flagged files
                else :
                    all_csv_files.append(file_path)
    
    if not all_csv_files:
        print(f"Warning: No CSV files found for API '{API}' with dataset type '{dataset_type}' in {data_path}")
        return []  # Return empty list instead of raising error
    
    dataframes = []
    for file_path in all_csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # If dataset_type is wiki, join with content data
            if dataset_type == "wiki":
                content_df = get_wiki_content()
                df = content_df.merge(df, on='content_id', how='left')  
                # Select and rename columns to match expected structure
                df = df[[
                    'category',
                    'subcategory',
                    'source',
                    'permanent_link',
                    'content',
                    'model',
                    'date',
                    'flagged',
                    'model_response']]
                
                # Handle special case for text-moderation-007
                if "text-moderation-007" in file_path:
                    df['date'] = '2024-07-21'  # Set a fixed date for this legacy model after examination
                
            elif dataset_type == "cn-wiki":
                content_df = get_cn_wiki_content()
                df = content_df.merge(df, on='content_id', how='left')
                # Select and rename columns to match expected structure
                df = df[[
                    'category',
                    'subcategory',
                    'source',
                    'permanent_link',
                    'content',
                    'model',
                    'date',
                    'flagged',
                    'model_response']]
            
            # Extract useful model response information
            if API == "openai-me":
                df['model_response'] = df['model_response'].apply(_extract_openaiME_response)
            elif API == "deepseek":
                df['model_response'] = df['model_response'].apply(_extract_deepseek_response)

            # Immediate validate DataFrame shape and NaN values
            assert df.shape[0] == 4210, f"DataFrame shape mismatch for {file_path}: expected 4210 rows, got {df.shape[0]}"
            assert df.isna().sum().sum() == 0, f"DataFrame contains NaN values for {file_path}"

            dataframes.append(df)
            
        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {str(e)}")
    
    return dataframes


def _validate_and_concatenate(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Helper function: Validate DataFrames and concatenate them.
    
    Args:
        dataframes: List of DataFrames to validate and concatenate
    
    Returns:
        Concatenated and validated DataFrame (empty if no dataframes)
    
    Raises:
        ValueError: If validation fails
    """
    if not dataframes:
        # Return empty DataFrame with expected columns
        empty_df = pd.DataFrame(columns=[
            'category', 'subcategory', 'source', 'permanent_link', 
            'content', 'model', 'date', 'flagged', 'model_response'
        ])
        empty_df['date'] = pd.to_datetime(empty_df['date'])
        return empty_df
    
    # Required columns
    required_columns = ['category', 'subcategory', 'source', 'model', 'date', 'flagged', 'model_response']
    
    # Validate each dataframe has required columns
    for i, df in enumerate(dataframes):
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame {i} missing required columns: {missing_cols}")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Validate flagged column contains only 0/1
    if not combined_df['flagged'].isin([0, 1]).all():
        raise ValueError("'flagged' column must contain only 0 or 1 values")
    
    # Convert date column to datetime
    try:
        combined_df['date'] = pd.to_datetime(combined_df['date'])
    except Exception as e:
        raise ValueError(f"Error converting 'date' column to datetime: {str(e)}")
    
    return combined_df

def _extract_openaiME_response(x):
    # Function to parse a single response
    def parse_single_response(response):
        try:
            # If response is already a dictionary, use it directly
            if isinstance(response, dict):
                response_dict = response
            else:
                # Safely evaluate the string as a Python literal
                response_dict = eval(response)
            
            categories = response_dict['results'][0]['categories']
            
            # Get flags that are True
            true_flags = [key for key, value in categories.items() if value]
            
            return true_flags
        except Exception as e:
            print(f"\nError parsing response: {e}")
            return []

    # Define the standardization map
    flag_standardization_map = {
        # Harassment related
        'harassment/threatening': 'harassment_threatening',
        
        # Hate related
        'hate/threatening': 'hate_threatening',
        
        # Illicit related
        'illicit/violent': 'illicit_violent',
        
        # Self-harm related
        'self-harm': 'self_harm',
        'self-harm/instructions': 'self_harm_instructions',
        'self-harm/intent': 'self_harm_intent',
        
        # Sexual related
        'sexual/minors': 'sexual_minors',
        
        # Violence related
        'violence/graphic': 'violence_graphic'
    }

    # Function to standardize flags in a list
    def standardize_flags(flag_list):
        return [flag_standardization_map.get(flag, flag) for flag in flag_list]
    try:
        flags = parse_single_response(x)
        # flag_response = standardize_flags(flags)
        return flags if flags else "Not Flagged"  # Return original if no flags found
    except Exception as e:
        print(f"Error extracting flags from response: {e}")
        return x

def _extract_deepseek_response(x):
    # Convert to string first to check for error patterns
    str_response = str(x).lower()
    
    if "error code" in str_response or "error" in str_response:
        return str(x)  # Return error as-is
    
    # If it's already a dict, try to extract
    if isinstance(x, dict):
        try:
            return x['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Failed to extract Deepseek response from dict: {e}. Response: {x}")
    
    # If it's a string, try multiple parsing methods
    if isinstance(x, str):
        # Method 1: Try JSON parsing first
        try:
            x_dict = json.loads(x)
            return x_dict['choices'][0]['message']['content']
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass  # Try next method
        
        # Method 2: Try eval() for Python dict-like strings (safer with ast.literal_eval)
        try:
            import ast
            x_dict = ast.literal_eval(x)
            return x_dict['choices'][0]['message']['content']
        except (ValueError, SyntaxError, KeyError, IndexError, TypeError):
            pass  # Try next method
        
        # Method 3: Try replacing single quotes with double quotes and parse as JSON
        try:
            # Replace single quotes with double quotes for JSON compatibility
            json_str = x.replace("'", '"').replace('False', 'false').replace('True', 'true').replace('None', 'null')
            x_dict = json.loads(json_str)
            return x_dict['choices'][0]['message']['content']
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            return str(x)    
    # If it's neither dict nor string, raise error
    raise ValueError(f"Deepseek response must be dict or string, got {type(x)}: {x}")
    
if __name__ == "__main__":
    test_df = load_data("openai-gpt", "cn-wiki")
    print(f"Loaded files: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    print(test_df.columns)