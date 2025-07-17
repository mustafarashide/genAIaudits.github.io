from openai import OpenAI
import time
import sys
from typing import Dict, Any, List, Union
from config import config
import re

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = config['openai-me']['model']  # OpenAI moderation model
    
    def call_moderation(
        self,
        content: str,
        init_sleep: float,
        additional_sleep: float,
        max_retries: int
    ) -> Dict[str, Any]:
        """
        Call OpenAI moderation API with retry logic.
        Returns:
            Dict with:
                - flagged: 1 if content is flagged, 0 if not, -1 if error
                - model_response: Full processed API response or -1 if error
        """
        retry_count = 0
        current_sleep = init_sleep

        while retry_count <= max_retries:
            try:
                # Rate limiting
                time.sleep(current_sleep)
                
                # Make API call
                api_response = self.client.moderations.create(
                    model=self.model,
                    input=content
                )
                
                # Get raw response and process it
                response_dict = api_response.model_dump()
                processed_response = self._process_response(response_dict)
                
                # Extract flag status
                flag = self._extract_flag(processed_response)
                
                return {
                    'flagged': 1 if flag else 0,
                    'model_response': processed_response
                }
                
            except Exception as e:
                error_str = str(e).lower()
                error_code = self._get_error_code(error_str)
                
                # Handle different error types
                if error_code in ['401', '403']:
                    print(f"Critical API error {error_code}: {str(e)}")
                    sys.exit(1)
                    
                elif error_code == '429':
                    retry_count += 1
                    if retry_count <= max_retries:
                        # Arithmetic progression for sleep time
                        current_sleep = init_sleep + (retry_count * additional_sleep)
                        # print(f"Rate limit reached. Retry {retry_count}/{max_retries} "
                        #       f"with {current_sleep}s delay...")
                        continue
                        
                elif error_code in ['500', '503']:
                    print(f"Server error {error_code}. Waiting 30 minutes...")
                    time.sleep(1800)  # 30 minutes
                    retry_count += 1
                    continue
                    
                print(f"Max retries ({max_retries}) exceeded or unknown error: {str(e)}")
                break
                
        return {'flagged': -1, 'model_response': -1}
    
    def _get_error_code(self, error_str: str) -> str:
        """
        Extract error code from error message using regex pattern matching.
        
        Args:
            error_str: Error message string to parse
            
        Returns:
            str: Error code ('401', '429', etc.) or 'unknown' if not found
        """
        # Pattern to match common API error codes
        error_pattern = r'(?:error code:|status code:)?\s*([45][0-9]{2})'
        
        # Try to find error code in the message
        match = re.search(error_pattern, error_str)
        if match:
            return match.group(1)
        
        # Check for specific error keywords
        error_keywords = {
            'rate limit': '429',
            'unauthorized': '401',
            'forbidden': '403',
            'server error': '500',
            'service unavailable': '503'
        }
        
        error_str_lower = error_str.lower()
        for keyword, code in error_keywords.items():
            if keyword in error_str_lower:
                return code
                
        return 'unknown'
    
    def _extract_flag(self, response: Dict) -> bool:
        """
        Extract flag status from API response.
        Raises an error if flag can't be extracted.
        """
        if isinstance(response, dict):
            results = response.get("results", [])
            if results and isinstance(results[0], dict):
                flag = results[0].get("flagged")
                if flag is not None:
                    return flag
    
        # If we reach here, we couldn't extract a valid flag
        raise Exception("Error code: 429 - Unable to extract flag from API response")
    
    def _process_response(self, data: Union[Dict, List, Any]) -> Any:
        """
        Process API response data by sorting keys and formatting floats.
        """
        if isinstance(data, dict):
            return {k: self._process_response(v) for k, v in sorted(data.items())}
        elif isinstance(data, list):
            return [self._process_response(item) for item in data]
        elif isinstance(data, float):
            return f"{data:.10f}"
        return data