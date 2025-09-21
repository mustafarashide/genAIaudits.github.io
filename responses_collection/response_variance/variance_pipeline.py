import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime
import logging
from responses_collection.openai_client import OpenAIClient
from responses_collection.deepseek_client import DeepseekClient
from responses_collection.openai_batch_client import OpenAIBatchClient
from responses_collection.data_processor import get_wiki_content, get_cn_wiki_content
import time

class VariancePipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logs_dir = Path("responses_collection/response_variance/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self, api_name: str, run_id: str) -> logging.Logger:
        """Setup logger for specific API and run"""
        logger_name = f"variance_{api_name}_{run_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        log_filename = f"variance_{api_name}_{run_id}.log"
        log_path = self.logs_dir / log_filename
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def validate_subcategories(self, subcategories: List[str], is_cn: bool = False) -> List[str]:
        """Validate subcategories against available data"""
        if is_cn:
            available_data = get_cn_wiki_content()
        else:
            available_data = get_wiki_content()
        
        available_subcategories = set(available_data['subcategory'].unique())
        
        # Check if all requested subcategories exist
        invalid_subcategories = [sub for sub in subcategories if sub not in available_subcategories]
        
        if invalid_subcategories:
            raise ValueError(f"Invalid subcategories: {invalid_subcategories}. "
                           f"Available subcategories: {sorted(available_subcategories)}")
        
        return subcategories
    
    def create_duplicated_dataset(self, base_df: pd.DataFrame, num_duplicates: int) -> pd.DataFrame:
        """Create duplicated dataset with duplicate_id instead of renaming content_id"""
        duplicated_dfs = []
        
        for duplicate_id in range(num_duplicates):
            df_copy = base_df.copy()
            df_copy['duplicate_id'] = duplicate_id
            duplicated_dfs.append(df_copy)
        
        duplicated_df = pd.concat(duplicated_dfs, ignore_index=True)
        return duplicated_df
    
    def _get_temp_file_path(self, api_name: str, dataset_name: str, run_id: str) -> Path:
        """Generate unique temp file path for each API and run"""
        return Path(f"responses_collection/response_variance/temp_{api_name}_{dataset_name}_{run_id}.csv")
    
    def _prepare_data(self, dataset: pd.DataFrame, api_name: str, is_cn: bool = False) -> pd.DataFrame:
        """Prepare data based on API requirements"""
        prepared_df = dataset.copy()
        
        if api_name in ["openai-gpt4.1", "openai-gpt5"]:
            if is_cn:
                prepared_df['content'] = prepared_df['content'].apply(lambda x: f"跟我说: {x}")
            else:
                prepared_df['content'] = prepared_df['content'].apply(lambda x: f"repeat after me: {x}")
        elif api_name == "deepseek":
            if is_cn:
                prepared_df['content'] = prepared_df['content'].apply(lambda x: f"跟我说: {x}")
            else:
                prepared_df['content'] = prepared_df['content'].apply(lambda x: f"repeat after me: {x}")
        # openai-me doesn't need modification
        
        return prepared_df
    
    def _get_client(self, api_name: str):
        """Initialize appropriate client"""
        if api_name == "openai-me":
            return OpenAIClient(self.config["openai-me"]["api_key"])
        elif api_name == "deepseek":
            return DeepseekClient(self.config["deepseek"]["api_key"])
        elif api_name == "openai-gpt4.1":
            return OpenAIBatchClient(
                api_key=self.config["openai-gpt4.1"]["api_key"],
                model=self.config["openai-gpt4.1"]["model"],
                endpoint=self.config["openai-gpt4.1"]["endpoint"]
            )
        elif api_name == "openai-gpt5":
            return OpenAIBatchClient(
                api_key=self.config["openai-gpt5"]["api_key"],
                model=self.config["openai-gpt5"]["model"],
                endpoint=self.config["openai-gpt5"]["endpoint"]
            )
        else:
            raise ValueError(f"Unknown API: {api_name}")
    
    def run_variance_experiment(
        self, 
        subcategories: List[str],
        api_names: List[str], 
        num_duplicates: int = 100,
        dataset_name: str = "variance",
        is_cn: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Run variance experiment for multiple APIs"""
        
        # Validate subcategories
        valid_subcategories = self.validate_subcategories(subcategories, is_cn)
        
        # Load appropriate dataset
        if is_cn:
            full_dataset = get_cn_wiki_content()
        else:
            full_dataset = get_wiki_content()
        
        # Filter by subcategories
        base_dataset = full_dataset[full_dataset['subcategory'].isin(valid_subcategories)]
        
        if base_dataset.empty:
            raise ValueError(f"No data found for subcategories: {subcategories}")
        
        print(f"Found {len(base_dataset)} items for subcategories: {valid_subcategories}")
        
        # Create duplicated dataset
        duplicated_df = self.create_duplicated_dataset(base_dataset, num_duplicates)
        
        results = {}
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for api_name in api_names:
            logger = self._setup_logger(api_name, run_id)
            logger.info(f"Starting variance experiment for {api_name}")
            
            try:
                # Prepare data for this API
                prepared_data = self._prepare_data(duplicated_df, api_name, is_cn)
                
                # Get client
                client = self._get_client(api_name)
                
                # Process with unique temp file
                temp_file = self._get_temp_file_path(api_name, dataset_name, run_id)
                
                if api_name in ["openai-gpt4.1", "openai-gpt5"]:
                    # Use batch processing with temp file for resume capability
                    results_df = self._process_batch_api(
                        client, prepared_data, temp_file, api_name, logger
                    )
                else:
                    # Use individual processing with temp file for resume capability
                    results_df = self._process_individual_api(
                        client, prepared_data, temp_file, api_name, logger
                    )
                
                # Add metadata
                results_df['model'] = self.config[api_name]['model']
                results_df['run_id'] = run_id
                results_df['is_cn'] = is_cn
                
                results[api_name] = results_df
                
                # Save results with timestamp when task finishes
                finish_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"responses_collection/response_variance/{api_name}_{dataset_name}_variance_{finish_timestamp}.csv"
                results_df.to_csv(output_file, index=False)
                
                logger.info(f"Completed {api_name} - saved {len(results_df)} responses")
                
                # Cleanup temp file
                if temp_file.exists():
                    temp_file.unlink()
                    
            except Exception as e:
                logger.error(f"Error processing {api_name}: {str(e)}")
                raise
        
        return results
    
    def _process_batch_api(
        self, client, dataset: pd.DataFrame, temp_file: Path, 
        api_name: str, logger: logging.Logger
    ) -> pd.DataFrame:
        """Process batch API calls with resume capability"""
        
        # Check for existing progress using content_id + duplicate_id
        processed_pairs = set()
        if temp_file.exists():
            existing_data = pd.read_csv(temp_file)
            valid_processed = existing_data[existing_data['flagged'] != -1]
            processed_pairs = set(zip(valid_processed['content_id'], valid_processed['duplicate_id']))
            logger.info(f"Found {len(processed_pairs)} previously processed items")
        
        # Get remaining items
        remaining_df = dataset[~dataset.apply(lambda row: (row['content_id'], row['duplicate_id']) in processed_pairs, axis=1)]
        
        batch_size = 100  # Use 100 queries per batch as requested
        
        # Process in batches
        for i in range(0, len(remaining_df), batch_size):
            batch = remaining_df.iloc[i:i + batch_size].copy()
            
            try:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(remaining_df) + batch_size - 1)//batch_size}")
                
                # Call batch API
                responses = client.process_dataset(batch)
                batch_results_df = pd.DataFrame(responses)
                
                # Save batch progress
                batch_results_df.to_csv(temp_file, mode='a', header=not temp_file.exists(), index=False)
                
                logger.info(f"Completed batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Return combined results
        if temp_file.exists():
            return pd.read_csv(temp_file)
        else:
            return pd.DataFrame()
    
    def _process_individual_api(
        self, client, dataset: pd.DataFrame, temp_file: Path, 
        api_name: str, logger: logging.Logger
    ) -> pd.DataFrame:
        """Process individual API calls with resume capability"""
        
        # Check for existing progress using content_id + duplicate_id
        processed_pairs = set()
        if temp_file.exists():
            existing_data = pd.read_csv(temp_file)
            valid_processed = existing_data[existing_data['flagged'] != -1]
            processed_pairs = set(zip(valid_processed['content_id'], valid_processed['duplicate_id']))
            logger.info(f"Found {len(processed_pairs)} previously processed items")
        
        # Get remaining items
        remaining_df = dataset[~dataset.apply(lambda row: (row['content_id'], row['duplicate_id']) in processed_pairs, axis=1)]
        
        results = []
        rate_limit = 1 / self.config[api_name]["rate_limit"]
        
        for idx, (_, row) in enumerate(remaining_df.iterrows()):
            try:
                response = client.call_moderation(
                    content=row['content'],
                    init_sleep=rate_limit,
                    additional_sleep=3,
                    max_retries=10
                )
                
                result = {
                    'content_id': row['content_id'],
                    'duplicate_id': row['duplicate_id'],
                    'flagged': response.get('flagged', -1),
                    'model_response': response.get('model_response', -1)
                }
                results.append(result)
                
                # Save progress periodically
                if len(results) % 10 == 0:
                    batch_df = pd.DataFrame(results[-10:])
                    batch_df.to_csv(temp_file, mode='a', header=not temp_file.exists(), index=False)
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{len(remaining_df)} items")
                    
            except Exception as e:
                logger.error(f"Error processing content_id {row['content_id']}, duplicate_id {row['duplicate_id']}: {str(e)}")
                continue
        
        # Save final batch
        if results:
            final_batch = pd.DataFrame(results[-(len(results) % 10):]) if len(results) % 10 != 0 else pd.DataFrame()
            if not final_batch.empty:
                final_batch.to_csv(temp_file, mode='a', header=not temp_file.exists(), index=False)
        
        # Return combined results
        if temp_file.exists():
            return pd.read_csv(temp_file)
        else:
            return pd.DataFrame(results)