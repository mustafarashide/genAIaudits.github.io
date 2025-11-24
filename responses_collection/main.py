from pathlib import Path
from pipeline_orchestrator import Pipeline
from config import config
import argparse
from data_processor import get_wiki_content, get_all_content, get_tv_movie_content, get_cn_wiki_content
import pandas as pd
from datetime import datetime
import sys

def save_results(df: pd.DataFrame, output_path: str, dataset_type: str) -> None:
    """Save results with timestamp and version control"""
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Add dataset type and timestamp to filename
    base_path = Path(output_path)
    timestamped_path = base_path.parent / f"{base_path.stem}_{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{base_path.suffix}"
    
    # Save with validation
    if df.empty:
        raise ValueError("No results to save!")
    
    df.to_csv(timestamped_path, index=False)
    print(f"✓ Results saved to: {timestamped_path}")

def cleanup_temp_files(api_name: str, config: dict, dataset_type: str) -> None:
    """Remove temporary files"""
    temp_file_template = config[api_name]["temp_file_template"]
    temp_file = Path(temp_file_template.format(model=config[api_name]["model"], dataset=dataset_type))
    if temp_file.exists():
        temp_file.unlink()
        print(f"✓ Cleaned up temporary file: {temp_file}")

def run_pipeline(dataset: pd.DataFrame, api_name: str, config: dict, dataset_type: str) -> None:
    """Run pipeline for specific API and handle results"""
    try:
        print(f"\n📋 Starting {api_name} pipeline...")
        
        # Initialize and run pipeline
        pipeline = Pipeline(config)
        results = pipeline.run(dataset, api_name, dataset_type)
        
        # Save results
        output_file = config[api_name]["output_file"]
        save_results(results, output_file, dataset_type)
        
        # Cleanup
        cleanup_temp_files(api_name, config, dataset_type)
        
        print(f"✓ {api_name} pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in {api_name} pipeline: {str(e)}")
        raise

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run content moderation pipelines")
    parser.add_argument("--api", 
                       choices=["openai-me", "openai-gpt4.1", "deepseek", "openai-gpt5", "openai-gpt5.1"],
                       required=True,
                       help="Which API pipeline to run: openai-me, openai-gpt4.1, deepseek, openai-gpt5, or openai-gpt5.1")
    parser.add_argument("--dataset", 
                       choices=["wiki", "tv-movie", "all", "cn-wiki"],
                       required=True,
                       help="Which dataset to use: wiki, tv-movie, cn-wiki, or all content")
    args = parser.parse_args()
    
    try:
        # Load dataset based on argument
        if args.dataset == "wiki":
            dataset = get_wiki_content()
            dataset_type = "wiki"
        elif args.dataset == "tv-movie":
            dataset = get_tv_movie_content()
            dataset_type = "tv-movie"
        elif args.dataset == "cn-wiki":
            dataset = get_cn_wiki_content()
            dataset_type = "cn-wiki"
        else:
            dataset = get_all_content() # means wiki + tv-movie
            dataset_type = "all"
            
        print(f"📊 Loaded {dataset_type} dataset with {len(dataset)} rows")
        
        # Run requested pipelines
        if args.api in ["openai-me"]:
            run_pipeline(dataset, "openai-me", config, dataset_type)

        if args.api in ["openai-gpt4.1"]:
            run_pipeline(dataset, "openai-gpt4.1", config, dataset_type)

        if args.api in ["deepseek"]:
            run_pipeline(dataset, "deepseek", config, dataset_type)

        if args.api in ["openai-gpt5"]:
            run_pipeline(dataset, "openai-gpt5", config, dataset_type)
        
        if args.api in ["openai-gpt5.1"]:
            run_pipeline(dataset, "openai-gpt5.1", config, dataset_type)

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()