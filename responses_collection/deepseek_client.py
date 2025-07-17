from openai import OpenAI
import time
import sys
from typing import Dict, Any, List, Union
import re

class DeepseekClient:
    # List of refusal phrases that indicate content is flagged
    REFUSAL_PHRASES = [
        "I can't comply with",
        "beyond my current scope",
        "content_filter",
        "我理解您", 
        "我暂时无法回答"
    ]
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"
        
    def call_moderation(
        self,
        content: str,
        init_sleep: float,
        additional_sleep: float,
        max_retries: int
    ) -> Dict[str, Any]:
        """
        Call Deepseek API with retry logic.
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
                api_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": content}
                    ],
                    stream=False
                )
                
                # Get plain text response first
                plain_response = api_response.choices[0].message.content
                
                # Get full response and process it
                response_dict = api_response.model_dump()
                processed_response = self._process_response(response_dict)
                
                # Check for refusal phrases
                flag = self._extract_flag(plain_response)
                
                return {
                    'flagged': 1 if flag else 0,
                    'model_response': processed_response
                }
                
            except Exception as e:
                error_str = str(e).lower()
                error_code = self._get_error_code(error_str)
                
                # Handle different error types
                if error_code in ['401', '402']:
                    print(f"Critical API error {error_code}: {str(e)}")
                    sys.exit(1)
                    
                elif error_code == '400':
                    if 'content exists risk' in error_str:
                        # print("Found content exists risk error. Flagging content...")
                        return {
                            'flagged': 1,
                            'model_response': str(e)
                        }
                    elif 'maximum context length' in error_str and 'tokens' in error_str:
                        # print("Token limit exceeded. Truncating content and retrying...")
                        # Extract the max token limit from error message
                        token_match = re.search(r'maximum context length is (\d+) tokens', error_str)
                        max_tokens = int(token_match.group(1)) if token_match else 65536
                        
                        # Rough estimation: 1 token ≈ 4 characters for safety
                        max_chars = max_tokens * 3  # Conservative estimate
                        
                        if len(content) > max_chars:
                            # Truncate content and retry
                            truncated_content = content[:max_chars]
                            # print(f"Truncating content from {len(content)} to {len(truncated_content)} characters")
                            
                            # Retry with truncated content
                            try:
                                api_response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant"},
                                        {"role": "user", "content": truncated_content}
                                    ],
                                    stream=False
                                )
                                
                                plain_response = api_response.choices[0].message.content
                                response_dict = api_response.model_dump()
                                processed_response = self._process_response(response_dict)
                                flag = self._extract_flag(plain_response)
                                
                                return {
                                    'flagged': 1 if flag else 0,
                                    'model_response': processed_response
                                }
                                
                            except Exception as retry_e:
                                print(f"Retry with truncated content failed: {str(retry_e)}")
                                return {'flagged': -1, 'model_response': str(retry_e)}
                        else:
                            print(f"Content length seems fine ({len(content)} chars), but still hit token limit")
                            break
                    else:
                        print(f"Invalid request error {error_code}: {str(e)}")
                        break

                elif error_code == '422':
                    print(f"Invalid request error {error_code}: {str(e)}")
                    break
                    
                elif error_code == '429':
                    retry_count += 1
                    if retry_count <= max_retries:
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
    
    def _extract_flag(self, plain_response: str) -> bool:
        """
        Extract flag status from plain text response.
        Returns True if any refusal phrase is found.
        """
        if not isinstance(plain_response, str):
            raise Exception("429: Invalid response type from API")
            
        # Check for refusal phrases
        return any(phrase.lower() in plain_response.lower() 
                  for phrase in self.REFUSAL_PHRASES)
    
    def _process_response(self, data: Union[Dict, List, Any]) -> Any:
        """Process API response data by sorting keys and formatting floats"""
        if isinstance(data, dict):
            return {k: self._process_response(v) for k, v in sorted(data.items())}
        elif isinstance(data, list):
            return [self._process_response(item) for item in data]
        elif isinstance(data, float):
            return f"{data:.10f}"
        return data
    

    def _get_error_code(self, error_str: str) -> str:
        """Extract first error code from error message"""
        # Look for first occurrence of 3-digit number
        match = re.search(r'\b(\d{3})\b', error_str)
        if match:
            return match.group(1)
        return 'unknown'