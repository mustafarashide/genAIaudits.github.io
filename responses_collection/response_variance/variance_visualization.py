import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import re
from responses_collection.data_processor import get_wiki_content

# List of CSV files
csv_files = [
    "responses_collection/response_variance/gpt4_1_variance_responses_20251007_222951.csv",
    "responses_collection/response_variance/gpt4_1_variance_responses_20251020_100145.csv",
    "responses_collection/response_variance/gpt4_1_variance_responses_20251103_124300.csv",
    "responses_collection/response_variance/gpt4_1_variance_responses_20251117_101819.csv"
]

def extract_date(filepath):
    """Extract date from filepath like '20251007_222951'"""
    match = re.search(r'(\d{8})_\d{6}', filepath)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d')
    return None

def calculate_ci_by_category(df, wiki_df):
    """Calculate mean and bootstrap CI for each category"""
    # Join df with wiki_df to get both category and subcategory
    df['group'] = df['content_id'].apply(lambda x: x.split('_')[1])
    df['content_id'] = df['content_id'].apply(lambda x: x.split('_')[0])
    df['flagged_bool'] = df['flagged'].apply(lambda x: True if x >= 1 else False)

    df = df.merge(wiki_df[['content_id','category', 'subcategory']], 
                  left_on='content_id', 
                  right_on='content_id', 
                  how='left')
    
    results_by_category = {}
    
    for category in df['category'].unique():
        category_df = df[df['category'] == category].copy()
        
        # Get the subcategory for this category
        subcategory = category_df['subcategory'].iloc[0]
        
        # Calculate mean for each group
        group_means = category_df.groupby('group')['flagged_bool'].mean().values
        
        # Bootstrap 95% confidence interval
        n_bootstrap = 10000
        np.random.seed(42)
        bootstrap_means = np.array([np.random.choice(group_means, size=len(group_means), replace=True).mean() 
                                   for _ in range(n_bootstrap)])
        
        mean = np.mean(group_means)
        ci = np.percentile(bootstrap_means, [2.5, 97.5])
        
        results_by_category[category] = {
            'mean': mean,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'subcategory': subcategory
        }
    
    return results_by_category

# Get wiki dataframe
raw_wiki_df = get_wiki_content()
target_subcategory = ["Abortion", "Israel Global Image"]
filter_wiki_df = raw_wiki_df[raw_wiki_df['subcategory'].isin(target_subcategory)].reset_index(drop=True)
print(f"Wiki categories: {filter_wiki_df['category'].unique()}")

# Process all CSV files
results = []
for filepath in csv_files:
    df = pd.read_csv(filepath)
    date = extract_date(filepath)
    category_results = calculate_ci_by_category(df, filter_wiki_df)
    
    for category, stats in category_results.items():
        results.append({
            'date': date,
            'category': category,
            'subcategory': stats['subcategory'],
            'mean': stats['mean'],
            'ci_lower': stats['ci_lower'],
            'ci_upper': stats['ci_upper']
        })
        
        print(f"File: {filepath}")
        print(f"Date: {date}")
        print(f"Category: {category}")
        print(f"Subcategory: {stats['subcategory']}")
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Bootstrap 95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")

# Convert to DataFrame for easier manipulation
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('date')

# Create Plotly visualization
fig = go.Figure()

# Define colors for each category
colors = ['rgb(0,100,80)', 'rgb(220,20,60)']
category_colors = dict(zip(results_df['category'].unique(), colors))

# Plot each category as a separate line with shaded CI
for category in results_df['category'].unique():
    category_data = results_df[results_df['category'] == category].sort_values('date')
    
    dates = category_data['date'].tolist()
    means = category_data['mean'].tolist()
    ci_upper = category_data['ci_upper'].tolist()
    ci_lower = category_data['ci_lower'].tolist()
    subcategory = category_data['subcategory'].iloc[0]
    
    # Create custom hover text showing mean and CI bounds
    hover_text = [f"Mean: {mean:.4f}<br>95% CI: [{lower:.4f}, {upper:.4f}]" 
                  for mean, lower, upper in zip(means, ci_lower, ci_upper)]
    
    color = category_colors[category]
    
    # Add the main line with subcategory as name
    fig.add_trace(go.Scatter(
        x=dates,
        y=means,
        mode='markers+lines',
        name=subcategory,
        line=dict(color=color, width=2),
        marker=dict(size=10),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_text
    ))
    
    # Add shaded confidence interval
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],  # dates, then dates reversed
        y=ci_upper + ci_lower[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor=color.replace('rgb', 'rgba').replace(')', ',0.2)'),
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

fig.update_layout(
    title="Mean Proportion Flagged Over Time by Subcategory with Bootstrap 95% CI",
    xaxis_title="Date",
    yaxis_title="Mean Proportion Flagged",
    showlegend=True,
    yaxis=dict(range=[0, 1]),
    legend=dict(title="Subcategory")
)

# Save the plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"responses_collection/response_variance/variance_by_category_{timestamp}.html"
fig.write_html(output_path)
print(f"Plot saved to: {output_path}")