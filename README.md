Disclaimer: This repository is created with AI coding tools 

# Basic Tasks

## API response collection

To start API data collection running manually, choose the APITYPE and run:
`response_collection/main.py --api APITYPE --dataset wiki`

Alternatively, start the cron job running using `crontab -e` with a command like this one for DeepSeek:

`30 12 * * 1 llm-speech-monitor-core/automation/scripts/run_deepseek_pipeline.sh >> llm-speech-monitor-core/automation/logs/cron_deepseek.log 2>&1`

This means that it will run once a week on Monday at 12:30pm. The prefix is ordered as: minute, hour, day of month, month, day of week. The every-other-week logic is handled by the shell script only proceeding on even given weeks.

To terminate the main process if you want to stop this job do:

`ps aux | grep timed_runner`

to find the pid to `kill`.

## Update the website

TODO: add description here

# Repository Structure

## automation

This directory includes the scripts to run the cron jobs. There is one script per model (OpenAI ME, OpenAI GPT-4.1, and DeepSeek) as well as one script to update the website. These should all be run individually, to allow them to potentially run on different servers, although we run them all on the same machine.

## data_collection

The code to gather Wikipedia sites as part of the Issues dataset. This should only be run once.

## data

processed: this subdirectory houses the processed datasets, including collected data.
- `hist_response`: processed collected data labeled with the model name, dataset, and date collected. Any dataset currently in process of collection has a `_temp` suffix which will be removed once collection has completed (this may take days or weeks). There are a few files needed for legacy compatibility - these have the suffix `not-flagged` or `ex-cn`.
- `movie_tv_content.csv` is the fully processed and combined TV and Movie content data
- `wiki_content`: the collected Wikipedia pages as of a specific date, including the full Wikipedia content, for the Issues dataset listed by category.
- `wiki_content_cn_azure`: the Chinese translation, processed via Azure's Translation API, of the Issues dataset
raw_tv_movie_data: original TV and Movie data from the paper on identity-related speech suppression as well as the scripts to create the processed version of the TV and Movie Content
response_data: ad hoc experimentation with DeepSeek and refusal messages

## front_end_pipeline

`pipeline.py` is the main script; it creates the visualizations based on the data in hist_response and saves output to `front_end_pipeline/html_output`. It should be run from the top directory as:

`python3 -m front_end_pipeline.pipeline`

Note that this command takes awhile to run since it is handling the large dataset.

## responses_collection

Handles response collection for all models we consider (OpenAI's ME, OpenAI's GPT-4.1, and DeepSeek). The files in this directory will be run via a cron job. The usual way to run this collection is:
`responses_collection/main.py --api openai-me --dataset all`
which will collect responses for the Issues dataset (in English only) and the TV and Movies dataset from OpenAI's moderation endpoint (ME) API. The collected responses will be put in data/processed/hist_response
and labeled with the model name, dataset option, and a timestamp.

## translate

Handles the translation to Chinese for the Issues dataset.
