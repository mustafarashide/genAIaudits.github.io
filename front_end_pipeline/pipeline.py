"""Main pipeline orchestrator."""
import os
from datetime import datetime
from front_end_pipeline.src.data import load_data
from front_end_pipeline.src.charts import create_trends_chart
from front_end_pipeline.src.html import generate_dashboard_html

def run_pipeline():
    # Load data
    df_openaiME_wiki = load_data("openai-me", "wiki")
    df_openaiGPT_wiki = load_data("openai-gpt", "wiki")
    df_deepseek_wiki = load_data("deepseek", "wiki")
    df_deepseek_cn_wiki = load_data("deepseek", "cn-wiki")
    df_gpt5_wiki = load_data("openai-gpt-5", "wiki")

    # Create charts
    fig_openaiME_wiki = create_trends_chart(df_openaiME_wiki)
    fig_openaiGPT_wiki = create_trends_chart(df_openaiGPT_wiki)
    fig_deepseek_wiki = create_trends_chart(df_deepseek_wiki)
    fig_deepseek_cn_wiki = create_trends_chart(df_deepseek_cn_wiki)
    fig_gpt5_wiki = create_trends_chart(df_gpt5_wiki)

    # Name titles for plots
    title_openaiME_wiki = "OpenAI Moderation Endpoint Wikipedia Moderation Trends"
    title_openaiGPT_wiki = "OpenAI GPT4.1 Wikipedia Moderation Trends"
    title_deepseek_wiki = "DeepSeek Chat Wikipedia Moderation Trends"
    title_deepseek_cn_wiki = "DeepSeek Chat Chinese Translated Wikipedia Moderation Trends"
    title_gpt5_wiki = "OpenAI GPT-5 Wikipedia Moderation Trends"

    # Generate HTML output
    fig_input_dict = {
        title_openaiME_wiki: [fig_openaiME_wiki, df_openaiME_wiki],
        # title_openaiGPT_wiki: [fig_openaiGPT_wiki, df_openaiGPT_wiki],
        title_gpt5_wiki: [fig_gpt5_wiki, df_gpt5_wiki],
        title_deepseek_wiki: [fig_deepseek_wiki, df_deepseek_wiki],
        title_deepseek_cn_wiki: [fig_deepseek_cn_wiki, df_deepseek_cn_wiki],
    }

    html_output = generate_dashboard_html(fig_input_dict)
    
    # Save HTML to file
    output_path = "index.html"
    
    # if there is index.html, rename it to index_old_timestamp.html
    try:
        with open(output_path, 'r') as f:
            old_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            old_output_path = f"index_old_{old_timestamp}.html"
            f.close()
            os.rename(output_path, old_output_path)
            print(f"Renamed existing index.html to {old_output_path}")
    except FileNotFoundError:
        pass  # No existing file to rename

    with open(output_path, 'w') as f:
        f.write(html_output)
    print(f"Dashboard saved to {output_path}")

if __name__ == "__main__":
    run_pipeline()