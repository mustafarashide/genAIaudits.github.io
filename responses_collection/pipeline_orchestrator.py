from pathlib import Path
import pandas as pd
from typing import Dict, Any
from openai_client import OpenAIClient
from deepseek_client import DeepseekClient
import datetime
import time
from openai_batch_client import OpenAIBatchClient
import logging

class Pipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.additional_sleep = config.get("additional_sleep", 3)  # Default additional sleep time
        self.max_retries = config.get("max_retries", 10)  # Default max retries
        self.lengthy_refusal_truncation = config["openai-gpt4.1"].get("lengthy_refusal_truncation", 30000)  # Default truncation length in characters

        # API-specific initial sleep times based on rate limits
        self.init_sleep_times = {
            "openai-me": 1/ config["openai-me"]["rate_limit"],  # OpenAI ME rate limit: 4 requests/second
            "openai-gpt4.1": 1/ config["openai-gpt4.1"]["rate_limit"],  # OpenAI GPT-4.1 rate limit: 60 requests/second
            "deepseek": 1/ config["deepseek"]["rate_limit"]  # Deepseek rate limit: 10 requests/second
        }

        # API-specific batch sizes 
        self.batch_size = {
            "openai-me": config["openai-me"]["batch_size"],
            "openai-gpt4.1": config["openai-gpt4.1"]["batch_size"],
            "deepseek": config["deepseek"]["batch_size"]
        }
        
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("responses_collection/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logger(self, api_name: str, dataset_type: str) -> logging.Logger:
        """Setup logger for specific API and dataset combination"""
        # Create unique logger name
        logger_name = f"pipeline_{api_name}_{dataset_type}"
        
        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"{api_name}_{dataset_type}_{timestamp}.log"
        log_path = self.logs_dir / log_filename
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        return logger
        
    def _log_and_print(self, message: str, logger: logging.Logger) -> None:
        """Log message and print it"""
        print(message)
        logger.info(message)
        
    def run(self, dataset: pd.DataFrame, api_name: str, dataset_type: str) -> pd.DataFrame:
        """Main pipeline execution flow"""
        # Setup logger for this run
        logger = self._setup_logger(api_name, dataset_type)
        
        if api_name not in ["openai-me", "deepseek", "openai-gpt4.1", "openai-gpt5"]:
            error_msg = f"Unknown API: {api_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Log pipeline start
        start_msg = f"Starting pipeline for API: {api_name}, Dataset: {dataset_type}, Rows: {len(dataset)}"
        self._log_and_print(start_msg, logger)
            
        # 1. Data preparation stage
        prepared_data = self._prepare_data(dataset, api_name, dataset_type)
        msg = f"✓ Data prepared for {api_name} processing"
        self._log_and_print(msg, logger)
        
        # 2. Process in batches with resume capability
        temp_file_template = self.config[api_name]["temp_file_template"]
        temp_file = Path(temp_file_template.format(
            model=self.config[api_name]["model"], 
            dataset=dataset_type
        ))
        response_data = self._process_in_batches(
            dataset=prepared_data,
            temp_file=temp_file,
            api_name=api_name,
            init_sleep=self.init_sleep_times[api_name],
            additional_sleep=self.additional_sleep,
            max_retries=self.max_retries,
            logger=logger
        )
        msg = f"✓ Batch processing completed for {api_name}"
        self._log_and_print(msg, logger)
        
        # 3. Validate and clean responses
        is_rerun = False
        validated_data, exit_code = self._validate_responses(response_data, dataset, logger, is_rerun)

        # If validation fails and we need to rerun for lengthy refusals
        if exit_code == 1 and validated_data is None:
            response_data = self._process_in_batches(
                dataset=prepared_data,
                temp_file=temp_file,
                api_name=api_name,
                init_sleep=self.init_sleep_times[api_name],
                additional_sleep=self.additional_sleep,
                max_retries=self.max_retries,
                logger=logger
            )
            is_rerun = True
            validated_data, exit_code = self._validate_responses(response_data, dataset, logger, is_rerun)
        msg = f"✓ Response validation completed for {api_name}"
        self._log_and_print(msg, logger)
        
        # 4. Format final output
        final_data = self._format_output(validated_data, api_name)
        msg = f"✓ Output formatting completed for {api_name}"
        self._log_and_print(msg, logger)
        
        # Log pipeline completion
        completion_msg = f"Pipeline completed successfully for API: {api_name}, Dataset: {dataset_type}, Final rows: {len(final_data)}"
        self._log_and_print(completion_msg, logger)
        
        return final_data

    def _prepare_data(self, dataset: pd.DataFrame, api_name: str, dataset_type: str) -> pd.DataFrame:
        """Prepare data based on API requirements"""
        if api_name == "openai-me":
            return dataset  # OpenAI-ME does not require special formatting
        elif api_name == "openai-gpt4.1" or api_name == "openai-gpt5":
            # OpenAI GPT-4.1 and GPT-5 requires content to be prefixed with "repeat after me:"
            dataset['content'] = dataset['content'].apply(lambda x: f"repeat after me: {x}")
            return dataset
        elif api_name == "deepseek": 
            # Check if the dataset_type is 'cn-wiki' and modify content accordingly
            if dataset_type == "cn-wiki":
                dataset['content'] = dataset['content'].apply(lambda x: f"跟我说：{x}")
            elif dataset_type == "wiki":
                dataset['content'] = dataset['content'].apply(lambda x: f"repeat after me: {x}")
            return dataset # DeepSeek requires to repeat content
        raise ValueError(f"Unknown API: {api_name}")
    
    def _process_in_batches(
        self, 
        dataset: pd.DataFrame, 
        temp_file: Path,
        api_name: str,
        init_sleep: float,
        additional_sleep: float,
        max_retries: int,
        logger: logging.Logger
    ) -> pd.DataFrame:
        """Process data in batches with resume capability using content_ids"""
        # Get all unique content IDs sorted by category, subcategory, content_id
        dataset_sorted = dataset.drop_duplicates(subset=['content_id']).sort_values(['category', 'subcategory', 'content_id'])
        all_content_ids = dataset_sorted['content_id'].tolist()
        
        # Check for existing progress
        processed_ids = set()
        lengthy_refusals_ids = set()
        if temp_file.exists():
            existing_data = pd.read_csv(temp_file)
            # Only consider IDs where flagged is not -1 or 2 (for lengthy refusals)
            valid_processed = existing_data[(existing_data['flagged'] != -1) & (existing_data['flagged'] != 2)]
            lengthy_refusals = existing_data[existing_data['flagged'] == 2]
            lengthy_refusals_ids = set(lengthy_refusals['content_id'])
            processed_ids = set(valid_processed['content_id'])
            msg = f"Found {len(processed_ids)} previously processed items"
            self._log_and_print(msg, logger)

            # drop -1 flagged items from existing data
            existing_data = existing_data[existing_data['flagged'] != -1]
            # Save existing data back to temp file without failed items
            existing_data.to_csv(temp_file, mode='w', index=False)
        
        # Get remaining content IDs to process (maintaining sort order)
        remaining_ids = [content_id for content_id in all_content_ids if content_id not in processed_ids]
        msg = f"Total items to process: {len(remaining_ids)}"
        self._log_and_print(msg, logger)

        batch_size = self.batch_size.get(api_name, 50)  # Default to 50 if not specified

        # Process in batches by content_id
        for i in range(0, len(remaining_ids), batch_size):
            batch_ids = remaining_ids[i:i + batch_size]
            batch = dataset[dataset['content_id'].isin(batch_ids)].copy().drop_duplicates(subset=['content_id'])

            # Check if batch contains any lengthy refusals, if so, shorten it 
            mask = batch['content_id'].isin(lengthy_refusals_ids)
            batch.loc[mask, 'content'] = batch.loc[mask, 'content'].str[:self.lengthy_refusal_truncation]
            msg = f"found {len(batch[mask])} lengthy refusals in batch, shortening content to {self.lengthy_refusal_truncation} chars"
            self._log_and_print(msg, logger)

            try:
                # Process batch through API with retry parameters
                processed_batch = self._call_api(
                    batch=batch,
                    api_name=api_name,
                    init_sleep=init_sleep,
                    additional_sleep=additional_sleep,
                    max_retries=max_retries
                )
                
                # Save progress
                self._save_batch(processed_batch, temp_file)
                
                msg = f"✓ Processed batch of {len(batch)} items ({len(remaining_ids) - i - len(batch_ids)} remaining)"
                self._log_and_print(msg, logger)
                
            except Exception as e:
                error_msg = f"❌ Error processing batch starting with content_id {batch_ids[0]}: {str(e)}"
                self._log_and_print(error_msg, logger)
                logger.error(error_msg)
                raise
    
        # Return combined results
        final_df = pd.read_csv(temp_file)  
        return final_df
    
    def _call_api(
        self, 
        batch: pd.DataFrame,
        api_name: str,
        init_sleep: float,
        additional_sleep: float,
        max_retries: int
    ) -> pd.DataFrame:
        """Make API calls and collect responses"""
        # Initialize API client based on api_name
        client = None
        if api_name == "openai-me":
            client = OpenAIClient(self.config["openai-me"]["api_key"])
        elif api_name == "deepseek":
            client = DeepseekClient(self.config["deepseek"]["api_key"])
        elif api_name == "openai-gpt4.1":
            # Use OpenAI GPT-4.1 batch client for processing
            client = OpenAIBatchClient(
                api_key=self.config["openai-gpt4.1"]["api_key"],
                model=self.config["openai-gpt4.1"]["model"],
                endpoint=self.config["openai-gpt4.1"]["endpoint"]
            )
        elif api_name == "openai-gpt5":
            # Use OpenAI GPT-5 batch client for processing
            client = OpenAIBatchClient(
                api_key=self.config["openai-gpt5"]["api_key"],
                model=self.config["openai-gpt5"]["model"],
                endpoint=self.config["openai-gpt5"]["endpoint"]
            )
        else:
            raise ValueError(f"Unknown API: {api_name}")
        
        if not client:
            raise ValueError(f"Unknown API: {api_name}")
        
        results = []

        # If using OpenAI GPT batch client, process the entire batch at once
        if api_name == "openai-gpt4.1" or api_name == "openai-gpt5":
            results = client.process_dataset(batch)
            time.sleep(300)  # sleep for 5 minutes after processing a batch
            return pd.DataFrame(results)
        
        # Process each row in the batch
        for _, row in batch.iterrows():
            # Call API with retry logic handled by client
            response = client.call_moderation(
                content=row['content'],
                init_sleep=init_sleep,
                additional_sleep=additional_sleep,
                max_retries=max_retries
            )

            # print(f"Processed content_id {row['content_id']} with response: {response}")
            
            # Collect results
            results.append({
                'content_id': row['content_id'],
                'flagged': response.get('flagged', -1),
                'model_response': response.get('model_response', -1)
            })
        
        # Convert results to DataFrame
        return pd.DataFrame(results)
    
    def _save_batch(self, batch: pd.DataFrame, temp_file: Path) -> None:
        """Save batch results with header only if file doesn't exist"""
        batch.to_csv(temp_file, 
                    mode='a', 
                    header=not temp_file.exists(), 
                    index=False)

    def _validate_responses(self, df: pd.DataFrame, content_dataset: pd.DataFrame, logger: logging.Logger, is_rerun: bool) -> pd.DataFrame:
        """
        Validate API responses against original dataset.
        
        Args:
            df: Response data from API calls
            content_dataset: Original input dataset
            logger: Logger instance
            is_rerun: Flag indicating if this is a rerun

        Returns:
            Validated and deduplicated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Check for empty response data
            if df.empty:
                error_msg = "Response data is empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check invalid responses
            if (df['flagged'] == -1).any():
                error_msg = "Some responses have invalid 'flagged' status (-1)"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check for lengthy refusals when rerun is False
            if (df['flagged'] == 2).any() and not is_rerun:
                # Run the pipeline again for lengthy refusals
                exit_code = 1
                return None, exit_code

            # Get content ID sets
            expected_ids = set(content_dataset['content_id'])
            actual_ids = set(df['content_id'])
            
            # Check for missing or extra content IDs
            missing_ids = expected_ids - actual_ids
            extra_ids = actual_ids - expected_ids
            
            if missing_ids:
                error_msg = f"Missing responses for {len(missing_ids)} content_ids"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if extra_ids:
                error_msg = f"Found {len(extra_ids)} unexpected content_ids in responses"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Remove duplicates keeping latest version
            deduped_df = df.drop_duplicates(subset=['content_id'], keep='last')         
            msg = f"✓ Validation passed: {len(deduped_df)} unique responses"
            exit_code = 0
            self._log_and_print(msg, logger)
            return deduped_df, exit_code
            
        except Exception as e:
            error_msg = f"❌ Validation failed: {str(e)}"
            self._log_and_print(error_msg, logger)
            logger.error(error_msg)
            raise
    
    def _format_output(self, df: pd.DataFrame, api_name: str) -> pd.DataFrame:
        """Format final output with model and date metadata"""
        # Create a copy to avoid modifying input DataFrame
        output_df = df.copy()
        
        # Get current date
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Add metadata columns
        output_df['model'] = self.config[api_name]['model']
        output_df['date'] = current_date
        
        return output_df