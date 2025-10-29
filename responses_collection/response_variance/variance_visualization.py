import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import re
from responses_collection.data_processor import get_wiki_content

# List of CSV files
csv_files = [
    "responses_collection/response_variance/gpt4_1_variance_responses_20251007_222951.csv",
    "responses_collection/response_variance/gpt4_1_variance_responses_20251020_100145.csv"
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
    # Join df with wiki_df
    df = df.merge(wiki_df[['content_id', 'category']], 
                  left_on='content_id_copy', 
                  right_on='content_id', 
                  how='left')
    
    results_by_category = {}
    
    for category in df['category'].unique():
        category_df = df[df['category'] == category].copy()
        
        # Sort by content_id_copy first, then assign groups
        category_df = category_df.sort_values('content_id_copy').reset_index(drop=True)
        
        # Assign group based on the position within each content_id_copy's duplicates
        category_df['group'] = category_df.groupby('content_id_copy').cumcount()
        
        # Calculate mean for each group
        group_means = category_df.groupby('group')['flagged'].mean().values
        
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
            'ci_upper': ci[1]
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
            'mean': stats['mean'],
            'ci_lower': stats['ci_lower'],
            'ci_upper': stats['ci_upper']
        })
        
        print(f"File: {filepath}")
        print(f"Date: {date}")
        print(f"Category: {category}")
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Bootstrap 95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")

# Convert to DataFrame for easier manipulation
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('date')

# Create Plotly visualization
fig = go.Figure()

# Plot each category as a separate line
for category in results_df['category'].unique():
    category_data = results_df[results_df['category'] == category]
    
    dates = category_data['date'].tolist()
    means = category_data['mean'].tolist()
    ci_upper = (category_data['ci_upper'] - category_data['mean']).tolist()
    ci_lower = (category_data['mean'] - category_data['ci_lower']).tolist()
    
    # Add points with error bars
    fig.add_trace(go.Scatter(
        x=dates,
        y=means,
        mode='markers+lines',
        name=category,
        marker=dict(size=12),
        line=dict(width=2),
        error_y=dict(
            type='data',
            symmetric=False,
            array=ci_upper,
            arrayminus=ci_lower,
            thickness=2,
            width=10
        )
    ))

fig.update_layout(
    title="Mean Proportion Flagged Over Time by Category with Bootstrap 95% CI",
    xaxis_title="Date",
    yaxis_title="Mean Proportion Flagged",
    showlegend=True,
    yaxis=dict(range=[0, 1]),
    legend=dict(title="Category")
)

# Save the plot
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"responses_collection/response_variance/variance_by_category_{timestamp}.html"
fig.write_html(output_path)
print(f"Plot saved to: {output_path}")