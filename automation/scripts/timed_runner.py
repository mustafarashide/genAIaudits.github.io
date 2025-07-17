import subprocess
import time
import signal
import threading
from typing import Optional
from pathlib import Path
from datetime import datetime, timezone

class PipelineRunner:
    def __init__(self, api_type: str):
        self.api_type = api_type
        self.project_root = Path(__file__).parent.parent.parent
        self.log_dir = self.project_root / "automation" / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.current_process = None
    
    def _log_with_timestamp(self, message: str, level: str = "INFO"):
        """Print message with timestamp for cron logging"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}", flush=True)
        
    def _is_deepseek_discount_time(self) -> bool:
        """Check if current UTC time is within DeepSeek discount hours (16:30 - 00:30 UTC)"""
        if self.api_type.lower() != 'deepseek':
            return True  # No time restriction for other APIs
            
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.time()
        
        # DeepSeek discount hours: 16:30 - 00:30 UTC
        start_time = datetime.strptime("16:30", "%H:%M").time()
        end_time = datetime.strptime("00:30", "%H:%M").time()
        
        # Handle the case where discount period crosses midnight
        if current_time >= start_time or current_time <= end_time:
            return True
        else:
            return False
    
    def _wait_for_discount_time(self):
        """Wait until DeepSeek discount time begins"""
        if self.api_type.lower() != 'deepseek':
            return  # No waiting for other APIs
            
        while not self._is_deepseek_discount_time():
            now_utc = datetime.now(timezone.utc)
            self._log_with_timestamp(f"⏰ Waiting for DeepSeek discount hours. Current UTC time: {now_utc.strftime('%H:%M')}. Discount hours: 16:30-00:30 UTC", "INFO")
            time.sleep(1800)  # Check every 30 minutes

    def _monitor_discount_time(self):
        """Monitor discount time and kill process if it ends"""
        if self.api_type.lower() != 'deepseek':
            return
            
        while self.current_process and self.current_process.poll() is None:
            if not self._is_deepseek_discount_time():
                self._log_with_timestamp("⏰ DeepSeek discount time ended. Terminating subprocess...", "WARNING")
                try:
                    self.current_process.terminate()
                    # Wait a bit for graceful termination
                    time.sleep(10)
                    if self.current_process.poll() is None:
                        self._log_with_timestamp("Process didn't terminate gracefully, force killing...", "WARNING")
                        self.current_process.kill()
                except Exception as e:
                    self._log_with_timestamp(f"Error terminating process: {e}", "ERROR")
                break
            time.sleep(1800)  # Check every 30 minutes during execution

    def _check_output_for_validation_error(self, output: str) -> bool:
        """Check if the error output contains validation-specific errors"""
        if output is None:
            return False  # Can't determine from None output
        
        validation_errors = [
            "Some responses have invalid 'flagged' status (-1)",
            "Missing responses for",
        ]
        return any(error in output for error in validation_errors)

    def run_pipeline(self, max_retries: int = 3, retry_delay: int = 300) -> Optional[bool]:
        # For DeepSeek, wait until discount time before starting
        self._wait_for_discount_time()
        
        attempt = 0
        first_dataset_completed = False
        
        while attempt < max_retries - 1:
            try:
                # For DeepSeek, check if we're still in discount time before each attempt
                if self.api_type.lower() == 'deepseek' and not self._is_deepseek_discount_time():
                    self._log_with_timestamp(f"⏰ DeepSeek discount time ended. Waiting for next discount period...", "INFO")
                    self._wait_for_discount_time()
                
                self._log_with_timestamp(f"🚀 Running pipeline attempt {attempt + 1}/{max_retries} for {self.api_type}")
                
                # Determine datasets to run
                datasets = ['cn-wiki', 'wiki'] if self.api_type.lower() == 'deepseek' else ['wiki']
                if first_dataset_completed and self.api_type.lower() == 'deepseek':
                    datasets = ['wiki']  # Skip cn-wiki for deepseek if already completed

                for dataset in datasets:
                    self._log_with_timestamp(f"📊 Running {dataset} dataset for {self.api_type}")
                    
                    # Start the subprocess
                    self.current_process = subprocess.Popen(
                        ['python', '/home/h302/llm-speech-monitor-core/responses_collection/main.py', '--api', self.api_type, '--dataset', dataset],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Start monitoring thread for DeepSeek
                    monitor_thread = None
                    if self.api_type.lower() == 'deepseek':
                        monitor_thread = threading.Thread(target=self._monitor_discount_time)
                        monitor_thread.daemon = True
                        monitor_thread.start()
                    
                    # Wait for process to complete
                    stdout, stderr = self.current_process.communicate()
                    combined_output = (stdout or "") + (stderr or "")
                    return_code = self.current_process.returncode

                    
                    # Clean up
                    self.current_process = None
                    
                    # Successful run
                    if return_code == 0:
                        self._log_with_timestamp(f"✅ {dataset} dataset completed successfully for {self.api_type}")
                        # Print the output
                        if combined_output:
                            self._log_with_timestamp(f"{dataset} dataset output:")
                            print(combined_output, flush=True)

                        if self.api_type.lower() != 'deepseek':
                            return True  # Skip further attempts for other APIs
                        
                        # If DeepSeek cn-wiki is completed, proceed to wiki
                        first_dataset_completed = True
                        continue
                    
                    # Check if it was terminated due to discount time ending for DeepSeek
                    elif self.api_type.lower() == 'deepseek' and (return_code == -15 or return_code == -9):  # SIGTERM or SIGKILL
                        self._log_with_timestamp("Process was terminated due to discount time ending. Will retry when discount time resumes.", "WARNING")
                        # Wait for next discount period and retry (don't increment attempt counter or change first_dataset_completed)
                        if self.api_type.lower() == 'deepseek':
                            self._wait_for_discount_time()
                            continue
                    
                    # Check if it was a validation error
                    elif self._check_output_for_validation_error(combined_output):
                        self._log_with_timestamp(f"🔄 Validation error detected for {dataset}. Retrying in {retry_delay} seconds...", "WARNING")
                        time.sleep(retry_delay)
                        attempt += 1
                        continue

                    else:
                        # Process failed or was terminated
                        self._log_with_timestamp(f"❌ Non-validation error occurred for {dataset} dataset on attempt {attempt + 1}: (return code: {return_code})", "ERROR")

                        # Print combined output for debugging
                        if combined_output:
                            self._log_with_timestamp("Combined output captured:", "DEBUG")
                            print(combined_output, flush=True)
                        
                        return False
                            
            except Exception as e:
                self._log_with_timestamp(f"❌ Exception during pipeline execution: {e}", "ERROR")
                if self.current_process:
                    self.current_process.terminate()
                    self.current_process = None
                return False

        # Exhausted all attempts
        self._log_with_timestamp(f"❌ All {max_retries} attempts failed for {self.api_type}", "ERROR")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python timed_runner.py <api_type>", flush=True)
        sys.exit(1)
    
    api_type = sys.argv[1]
    runner = PipelineRunner(api_type)
    
    try:
        runner._log_with_timestamp(f"Starting {api_type} pipeline runner")
        result = runner.run_pipeline()
        if result:
            runner._log_with_timestamp(f"{api_type} pipeline completed successfully")
        else:
            runner._log_with_timestamp(f"{api_type} pipeline failed", "ERROR")
            sys.exit(1)
    except Exception as e:
        runner._log_with_timestamp(f"Fatal error: {e}", "ERROR")
        sys.exit(1)