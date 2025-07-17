# Azure Translation Pipeline with Validation

This pipeline provides robust Azure translation with automatic error detection and retry capabilities.

## Files

1. **azure_pipeline.py** - Main translation pipeline
2. **azure_validation.py** - Validation and retry script

## Features

### Main Pipeline (azure_pipeline.py)
- Translates content row-by-row for better error isolation
- Handles large texts by splitting into sentence-based chunks
- Tracks failed translations with `translation_failed` column (1 = failed, 0 = success)
- Provides detailed success/failure statistics

### Validation Script (azure_validation.py)
- Detects multiple types of translation issues:
  - Rows marked as failed (`translation_failed = 1`)
  - Content that appears mostly untranslated (English)
  - Very short content that might indicate errors
  - Repetitive content patterns
- Allows selective retry of problematic translations
- Creates backups before modifying files
- Uses more conservative retry parameters

## Usage

### Step 1: Run Main Translation
```bash
python translate/azure_pipeline.py
```

This will:
- Translate the first CSV file for testing
- Create output in `data/processed/wiki_content_cn_azure/`
- Show translation statistics including failure rate

### Step 2: Validate and Retry (if needed)
```bash
python translate/azure_validation.py
```

This will:
- Scan translated files for potential issues
- Show a report of problematic rows
- Allow you to retry failed translations
- Create backups before making changes

## Example Output

### Main Pipeline
```
Translation Summary:
  Total rows: 150
  Successfully translated: 142
  Failed translations: 8
  Success rate: 94.7%

Note: 8 rows failed translation and are marked with translation_failed=1
Use azure_validation.py to review and retry failed translations.
```

### Validation Script
```
Found 8 rows with potential issues:
  Row 23: marked_as_failed, likely_untranslated - Machine learning (ML) is a field of study...
  Row 45: marked_as_failed - In computer science and operations research...
  ...

Found 8 potential issues. Retry translation for these rows? (y/n): y
Successfully retranslated 6 out of 8 problematic rows
```

## Error Detection Logic

The validation script detects issues using multiple heuristics:

1. **Marked Failures**: Rows with `translation_failed = 1`
2. **Untranslated Content**: Text with high ratio of English words
3. **Short Content**: Very brief text that might indicate truncation
4. **Repetitive Patterns**: Content with excessive repetition

## Configuration

Both scripts use the same `translate/config.json` file:
```json
{
  "Azure_Translate_API_Key": "your-api-key-here"
}
```

## Rate Limiting

- Main pipeline: Standard rate limiting with exponential backoff
- Validation retry: More conservative delays to avoid repeated failures

## File Structure

Input: `data/processed/wiki_content/`
Output: `data/processed/wiki_content_cn_azure/`
Backups: Created automatically during validation as `*_backup.csv`
