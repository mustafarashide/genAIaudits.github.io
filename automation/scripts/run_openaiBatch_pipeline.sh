#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")/../.."

# Activate uv environment
source .venv/bin/activate

# Run the pipeline
python3 automation/scripts/timed_runner.py openai-gpt