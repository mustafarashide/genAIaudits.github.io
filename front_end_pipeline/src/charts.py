"""Chart generation module using Plotly."""
import plotly.graph_objects as go
import pandas as pd
from front_end_pipeline.src.data import load_data, load_synthetic_data


def create_trends_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create main trends line chart showing flagged rates over time by category.
    
    Args:
        df: DataFrame with columns ['date', 'category', 'flagged', 'model']
    
    Returns:
        Plotly Figure object
    """
    # Handle empty DataFrame
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="",
            xaxis_title="Date",
            yaxis_title="Flagged Rate (%)",
            font=dict(family="Arial, sans-serif", size=12),
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        # Remove x-axis ticks
        fig.update_xaxes(showticklabels=False)
        return fig
    
    # Prepare data for charting
    chart_data = _prepare_chart_data(df)
    
    # Calculate average flagging rate per category and sort by descending order
    avg_flagging_rates = (
        chart_data
        .groupby('category')['flagging_rate']
        .mean()
        .sort_values(ascending=False)
    )
    categories = avg_flagging_rates.index.tolist()  # Categories sorted by avg flagging rate
    
    # Create consistent color mapping based on category names
    all_categories = sorted(chart_data['category'].unique())  # Sort alphabetically for consistency
    colors = [
    '#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', 
    '#19d3f3', '#ff6692', '#b6e880', '#ff97ff', '#fecb52',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
    '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
    '#c44569', '#f8b500', '#6c5ce7', '#a29bfe', '#fd79a8',
    '#e84393', '#00b894', '#00cec9', '#0984e3', '#74b9ff',
    '#fd5068', '#fdcb6e', '#e17055', '#81ecec', '#55a3ff',
    '#ff7675', '#fab1a0', '#00b8d4', '#26de81', '#fed330',
    '#3742fa', '#7bed9f', '#70a1ff'
    ]
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(all_categories)}    
    # Create figure
    fig = go.Figure()
    
    # Add a line for each category (in sorted order)
    for category in categories:
        category_data = chart_data[chart_data['category'] == category]
        
        fig.add_trace(go.Scatter(
            x=category_data['date'],
            y=category_data['flagging_rate'],
            mode='lines+markers',
            name=category,
            line=dict(color=color_map[category], width=2),
            marker=dict(size=8),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "Date: %{x}<br>" +
                "Flagged Rate: %{y:.1f}%<br>" +
                "<br><b>Topic Breakdown:</b><br>" +
                "%{customdata}<br>" +
                "<extra></extra>"
            ),
            customdata=category_data['hovertext']
        ))
    
    # Apply styling
    fig.update_layout(
        title="", # Set title in the main pipeline through static html 
        xaxis_title="Date",
        yaxis_title="Flagged Rate (%)",
        font=dict(family="Arial, sans-serif", size=12),
        hovermode='closest',
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".1f")

    # Handle unique date values for x-axis
    unique_dates = chart_data['date'].nunique()

    # Handle x-axis formatting based on number of dates
    if unique_dates == 1:
        # For single date, show just the date without time
        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            tickmode='array',
            tickvals=chart_data['date'].unique(),
            ticktext=[d.strftime('%Y-%m-%d') for d in chart_data['date'].unique()]
        )
    
    return fig


def create_synthetic_trends_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create trends chart for synthetic data showing GPT-4.1 to GPT-5 transition.
    
    Args:
        df: DataFrame with columns ['date', 'category', 'flagged', 'model']
    
    Returns:
        Plotly Figure object with transition line on August 7th
    """
    # Handle empty DataFrame
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="",
            xaxis_title="Date",
            yaxis_title="Flagged Rate (%)",
            font=dict(family="Arial, sans-serif", size=12),
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        fig.update_xaxes(showticklabels=False)
        return fig
    
    # Prepare data for charting
    chart_data = _prepare_chart_data(df)
    
    # Calculate average flagging rate per category and sort by descending order
    avg_flagging_rates = (
        chart_data
        .groupby('category')['flagging_rate']
        .mean()
        .sort_values(ascending=False)
    )
    categories = avg_flagging_rates.index.tolist()
    
    # Create consistent color mapping
    all_categories = sorted(chart_data['category'].unique())
    colors = [
        '#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', 
        '#19d3f3', '#ff6692', '#b6e880', '#ff97ff', '#fecb52',
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
        '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
        '#c44569', '#f8b500', '#6c5ce7', '#a29bfe', '#fd79a8',
        '#e84393', '#00b894', '#00cec9', '#0984e3', '#74b9ff',
        '#fd5068', '#fdcb6e', '#e17055', '#81ecec', '#55a3ff',
        '#ff7675', '#fab1a0', '#00b8d4', '#26de81', '#fed330',
        '#3742fa', '#7bed9f', '#70a1ff'
    ]
    color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(all_categories)}
    
    # Create figure
    fig = go.Figure()
    
    # Add a line for each category (in sorted order)
    for category in categories:
        category_data = chart_data[chart_data['category'] == category]
        
        fig.add_trace(go.Scatter(
            x=category_data['date'],
            y=category_data['flagging_rate'],
            mode='lines+markers',
            name=category,
            line=dict(color=color_map[category], width=2),
            marker=dict(size=8),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "Date: %{x}<br>" +
                "Flagged Rate: %{y:.1f}%<br>" +
                "<br><b>Topic Breakdown:</b><br>" +
                "%{customdata}<br>" +
                "<extra></extra>"
            ),
            customdata=category_data['hovertext']
        ))
    
    # Add vertical line at August 7th to show transition using add_shape instead of add_vline
    transition_date = pd.to_datetime('2025-08-07')
    
    fig.add_shape(
        type="line",
        x0=transition_date,
        x1=transition_date,
        y0=0,
        y1=1,
        yref="paper",  # Use paper coordinates for y-axis (0 to 1)
        line=dict(
            color="gray",
            width=2,
            dash="dash"
        )
    )
    
    # Add annotations for model regions
    if len(chart_data) > 0:
        y_range = chart_data['flagging_rate'].max() - chart_data['flagging_rate'].min()
        y_pos = chart_data['flagging_rate'].max() - 0.1 * y_range
        
        # Get date range for positioning annotations
        min_date = chart_data['date'].min()
        max_date = chart_data['date'].max()
        
        # Calculate positions for annotations
        left_pos = min_date + (transition_date - min_date) / 2
        right_pos = transition_date + (max_date - transition_date) / 2
        
        # Left side annotation (GPT-4.1)
        fig.add_annotation(
            x=left_pos,
            y=y_pos,
            text="GPT-4.1",
            showarrow=False,
            font=dict(size=14, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        # Right side annotation (GPT-5)
        fig.add_annotation(
            x=right_pos,
            y=y_pos,
            text="GPT-5",
            showarrow=False,
            font=dict(size=14, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    
    # Apply styling
    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="Flagged Rate (%)",
        font=dict(family="Arial, sans-serif", size=12),
        hovermode='closest',
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    
    # Format y-axis as percentage
    fig.update_yaxes(tickformat=".1f")
    
    # Handle unique date values for x-axis
    unique_dates = chart_data['date'].nunique()
    
    if unique_dates == 1:
        fig.update_xaxes(
            tickformat='%Y-%m-%d',
            tickmode='array',
            tickvals=chart_data['date'].unique(),
            ticktext=[d.strftime('%Y-%m-%d') for d in chart_data['date'].unique()]
        )
    
    return fig


def _prepare_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for charting with proper aggregation."""
    
    # 1. Compute category-level flagged rates
    # Step 1: deduplicate to remove overcounting for category level flagging rate
    df_deduplicated = df.drop_duplicates(subset=['date', 'category', 'source'], keep='last')

    # Step 2: Aggregate the deduplicated data
    df_cat_rates = (
        df_deduplicated
        .groupby(['date', 'category'], as_index=False)
        .agg({
            'flagged': ['count', lambda x: (x == 1).sum()]
        })
    )
    df_cat_rates.columns = ['date', 'category', 'total_count', 'flagged_count']
    df_cat_rates['flagging_rate'] = (df_cat_rates['flagged_count'] / df_cat_rates['total_count']) * 100
    
    # 2. Compute subcategory-level flagged rates
    df_sub_rates = (
        df
        .groupby(['date', 'category', 'subcategory'], as_index=False)
        .agg({
            'flagged': ['count', lambda x: (x == 1).sum()]
        })
    )
    df_sub_rates.columns = ['date', 'category', 'subcategory', 'total_count', 'flagged_count']
    df_sub_rates['flag_rate'] = (df_sub_rates['flagged_count'] / df_sub_rates['total_count']) * 100

    # 3. Prepare hover text
    df_sub_rates = df_sub_rates.sort_values(['date', 'category', 'flag_rate'], ascending=[True, True, False])
    df_hover = (
        df_sub_rates
        .groupby(['date', 'category'], as_index=False)
        .agg({
            'subcategory': lambda lst: list(lst),
            'flag_rate': lambda lst: list(lst),
            'total_count': lambda lst: list(lst)
        })
    )
    df_hover['hovertext'] = df_hover.apply(
        lambda row: '<br>'.join(
            f"{sc}: {rate:.1f}% ({count} pages)"
            for sc, rate, count in zip(row['subcategory'], row['flag_rate'], row['total_count'])
        ),
        axis=1
    )
    df_hover = df_hover[['date', 'category', 'hovertext']]
    
    # 4. Merge category rates with hover text
    result_df = df_cat_rates.merge(df_hover, on=['date', 'category'], how='left')
    
    # 5. Select final columns
    result_df = result_df[['date', 'category', 'flagging_rate', 'hovertext']]
    
    return result_df

if __name__ == "__main__":
    chat_gpt_data = load_synthetic_data()
    visual = create_synthetic_trends_chart(chat_gpt_data)
    visual.show()
