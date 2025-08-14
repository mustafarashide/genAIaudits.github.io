"""Chart generation module using Plotly."""
import plotly.graph_objects as go
import pandas as pd
from front_end_pipeline.src.data import load_data


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
    
    # Create figure
    fig = go.Figure()
    
    # Add a line for each category (in sorted order)
    colors = ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A']
    
    for i, category in enumerate(categories):
        category_data = chart_data[chart_data['category'] == category]
        
        fig.add_trace(go.Scatter(
            x=category_data['date'],
            y=category_data['flagging_rate'],
            mode='lines+markers',
            name=category,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>" +
                "Date: %{x}<br>" +
                "Flagged Rate: %{y:.1f}%<br>" +
                "<br><b>Subcategory Breakdown:</b><br>" +
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


def _prepare_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for charting with proper aggregation."""
    
    # 1. Compute category-level flagged rates
    df_cat_rates = (
        df
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
            f"{sc}: {rate:.1f}% ({count} wiki pages)" 
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
    me_data = load_data("openai-gpt-5", "wiki")
    print(me_data['date'].value_counts())
    fig = create_trends_chart(me_data)
    fig.show()
