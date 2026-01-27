#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")/../.."

# Activate uv environment
source .venv/bin/activate

# Check if current week number is even (runs every other week)
WEEK_NUM=$(date +%V)
if [ $((WEEK_NUM % 2)) -eq 0 ]; then
    echo "Week $WEEK_NUM is even - running DeepSeek pipeline (dummy)"
    # python3 -m automation.scripts.timed_runner deepseek
else
    echo "Week $WEEK_NUM is odd - skipping DeepSeek pipeline"
fi