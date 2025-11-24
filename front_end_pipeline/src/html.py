"""HTML dashboard generation module."""
from jinja2 import Environment, DictLoader
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly
import json
from datetime import datetime
import os

def export_chart_data(fig_input_dict: Dict[str, List], output_dir: str = "data/charts", max_size_mb: int = 90) -> None:
    """
    Export chart data to separate JSON files, splitting large files into chunks.
    
    Args:
        fig_input_dict: Dict with title as key, [figure, dataframe] as value
        output_dir: Directory to save JSON files
        max_size_mb: Maximum file size in MB before splitting
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    chart_data = {}
    
    for title, (fig, df) in fig_input_dict.items():
        # Prepare dataframe for JavaScript (same logic as before)
        df_for_js = df.copy()
        
        if df_for_js.empty:
            chart_data[title] = []
        else:
            df_for_js['date'] = df_for_js['date'].dt.strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(df_for_js['date']) else df_for_js['date']
            
            df_for_js['content'] = df_for_js.apply(
                lambda row: 'TRUNCATED TO FIRST 19,000 CHARS: ' + row['content'][:300] + '...' if row['flagged'] == 2 
                else row['content'][:300] + '...' if len(row['content']) > 300 else row['content'],
                axis=1
            )
            
            df_for_js['model_response_full'] = df_for_js.apply(
                lambda row: row['model_response'] if row['flagged'] != 0 
                else (str(row['model_response'])[:300] + '...' if len(str(row['model_response'])) > 300 else str(row['model_response'])),
                axis=1
            )
            
            df_for_js['model_response_needs_expand'] = df_for_js['model_response_full'].str.len() > 150
            
            df_for_js['sort_order'] = df_for_js['flagged'].map({1: 1, 2: 2, 0: 3})
            df_for_js = df_for_js.sort_values(by=['sort_order', 'date'], ascending=[True, False])
            df_for_js = df_for_js.drop(columns=['sort_order'])

            if 'model_response' in df_for_js.columns:
                df_for_js = df_for_js.drop(columns=['model_response'])
            
            chart_data[title] = df_for_js.to_dict('records')
    
    # Save individual chart data files with chunking
    manifest_files = {}
    max_bytes = max_size_mb * 1024 * 1024
    
    for title, data in chart_data.items():
        safe_filename = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')
        
        # Test size of full data
        test_json = json.dumps({
            'title': title,
            'data': data,
            'generated_at': datetime.now().isoformat(),
            'count': len(data)
        })
        
        if len(test_json.encode('utf-8')) > max_bytes:
            # Split into chunks
            chunk_size = len(data) // ((len(test_json.encode('utf-8')) // max_bytes) + 1)
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            
            chunk_files = []
            for idx, chunk in enumerate(chunks):
                chunk_filename = f"{safe_filename}_part{idx + 1}.json"
                file_path = Path(output_dir) / chunk_filename
                
                with open(file_path, 'w') as f:
                    json.dump({
                        'title': title,
                        'part': idx + 1,
                        'total_parts': len(chunks),
                        'data': chunk,
                        'generated_at': datetime.now().isoformat(),
                        'count': len(chunk),
                        'total_count': len(data)
                    }, f, indent=2)
                
                chunk_files.append(chunk_filename)
            
            manifest_files[title] = chunk_files
        else:
            # Single file
            filename = safe_filename + '.json'
            file_path = Path(output_dir) / filename
            
            with open(file_path, 'w') as f:
                json.dump({
                    'title': title,
                    'data': data,
                    'generated_at': datetime.now().isoformat(),
                    'count': len(data)
                }, f, indent=2)
            
            manifest_files[title] = [filename]
    
    # Create manifest file
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'charts': list(chart_data.keys()),
        'files': manifest_files
    }
    
    with open(Path(output_dir) / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

def generate_dashboard_html(fig_input_dict: Dict[str, List]) -> str:
    """
    Generate complete HTML dashboard with 4 charts and click-based filtering.

    Args:
        fig_input_dict: Dict with title as key, [figure, dataframe] as value

    Returns:
        Complete HTML string for the dashboard
    """
    # Export chart data to JSON files
    export_chart_data(fig_input_dict)
    
    # Convert figures to HTML (same logic as before)
    chart_htmls = {}

    # First, determine the y-axis range across all figures
    y_max = 0
    for title, (fig, df) in fig_input_dict.items():
        for trace in fig.data:
            if hasattr(trace, 'y') and trace.y is not None:
                y_max = max(y_max, max(trace.y))

    # Add some padding to the max value
    y_range = [0, y_max * 1.1]

    for i, (title, (fig, df)) in enumerate(fig_input_dict.items()):
        # Create unique chart ID
        chart_id = f"chart_{i}"

        # Modify figure properties
        fig.update_layout(
            # showlegend=(i == 0),  # Only show legend for first chart
            yaxis=dict(
                range=y_range,  # Use consistent y-axis range
                tickformat=".1f"
            ),
            margin=dict(l=50, r=30, t=50, b=50)  # Tighter margins for smaller charts
        )

        # Convert Plotly figure to HTML
        if i == 0:
            # First chart includes full plotlyjs
            chart_html = fig.to_html(include_plotlyjs='inline', div_id=chart_id)
        else:
            # Subsequent charts don't include plotlyjs
            chart_html = fig.to_html(include_plotlyjs=False, div_id=chart_id)

        # Extract just the div part (remove html/head/body tags)
        start_div = chart_html.find('<div')
        end_div = chart_html.rfind('</div>') + 6
        chart_div = chart_html[start_div:end_div]

        chart_htmls[title] = {
            'html': chart_div,
            'id': chart_id
        }

    # Template variables (removed chart_data)
    template_vars = {
        'chart_htmls': chart_htmls,
        'chart_titles': list(fig_input_dict.keys()),  # Pass chart titles for data loading
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'css_content': _get_css_content(),
        'js_content': _get_js_content()
    }

    # Render and return HTML
    return _render_template(template_vars)


def _render_template(template_vars: Dict) -> str:
    """Render HTML template with 6 charts in 3x2 grid and click-based filtering."""
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>AI Watchman</title>
    <style>{{ css_content }}</style>
</head>
<body>
    <header>
        <div class="container">
           <div class="item">
        <h1>AI Watchman</h1>
        <center><img src="site_images/logo.jpg"
             alt="Logo generated by Pixel Studio using the prompt: image of lightouse with light shining down on a robot. make the lighthouse big and classical and the robot small to the side below."
             title="Logo generated by Pixel Studio using the prompt: image of lightouse with light shining down on a robot. make the lighthouse big and classical and the robot small to the side below."
             width=80px
             height=80px
        /></center>
        <h3 align="center" style="color:gray;"><i>Quis custodiet ipsos custodes?</i></h3>
        <div class="metadata">
            <span>🕒 Last updated: {{ last_updated }}</span>
        </div>
           </div>
           <div class = "item">
              <h2 align="center">What information will AI platforms let you see?</h2>
              <p>Large language models won't generate everything you ask them to. The AI
              companies make decisions to flag some content so that it's not shown to users.
              They might do this so you don't see overly violent or sexual content, or for
              other reasons including based on legal or political considerations. Based on
              <a href="https://www.pewresearch.org/topics/">Pew Research Center topics</a>
              and augmented with
              <a
              href="https://www.cambridge.org/core/journals/perspectives-on-politics/article/censoring-the-intellectual-public-space-in-china-what-topics-are-not-allowed-and-who-gets-blacklisted/B5774AC7925D68814C989326EC3AE36B">Chinese
              Sensitive Topics</a>, the below
              charts let you explore what social issues are flagged by OpenAI and DeepSeek.
              The data and code behind these charts is available on our
              <a href="https://github.com/genAIaudits/genAIaudits.github.io">GitHub</a> page.</p>
              <p>For similar exploration of what TV and movie synopses are filtered, see
              <a href="tv_movie_05-25.html">this page</a>.</p>
              <h3>We use Wikipedia page data to see what encyclopedic content about social issues
              is flagged as inappropriate and filtered out by the AI platforms.</h3>
              <p>The Wikipedia page content is linked below so you can see for yourself what
              content is allowed and what is not.</p>
            </div>
        </div>
        <div class="instructions">
            <p>💡 Click on any category point in the charts OR click on legend items to view detailed data for that category</p>
        </div>
    </header>

    <main>
        <!-- Loading indicator -->
        <div id="loadingIndicator" style="text-align: center; padding: 20px;">
            <p>Loading chart data...</p>
        </div>

        <!-- Six Charts Grid (3x2) -->
        <div class="charts-grid" id="chartsGrid" style="display: none;">
            {% for title, chart_info in chart_htmls.items() %}
                <div class="chart-container" data-chart-title="{{ title }}">
                    <h2>{{ title }}</h2>
                    {{ chart_info.html | safe }}
                </div>
            {% endfor %}
        </div>

        <!-- Dynamic Table Section -->
        <section id="tableSection" class="table-section" style="display: none;">
            <div class="table-header">
                <h2 id="tableTitle">Category Data</h2>
                <button id="clearTable" class="clear-btn">Clear Table</button>
            </div>
            <div id="tableContainer"></div>
        </section>
    </main>

    <!-- Pass chart titles to JavaScript -->
    <script>
        window.chartTitles = {{ chart_titles | tojson }};
        window.chartData = {}; // Will be populated by data loading
    </script>
    <script>{{ js_content }}</script>
</body>
</html>
"""
    
    env = Environment(loader=DictLoader({'dashboard': template_str}))
    template = env.get_template('dashboard')
    return template.render(**template_vars)


def _get_css_content() -> str:
    """CSS for 3x2 chart grid layout and click-based interaction."""
    return """
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 20px; 
            line-height: 1.6; 
            background-color: #f8f9fa;
        }
        
        header h1 { 
            color: #2c3e50; 
            margin-bottom: 2px; 
            text-align: center;
        }
        
        .metadata { 
            color: #666; 
            font-size: 12px; 
            text-align: center;
            margin-bottom: 10px; 
        }
        
        .instructions {
            text-align: center;
            margin-bottom: 30px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 6px;
            color: #1565c0;
        }
        .container {
            display: flex;
        }
        .item {
            padding: 10px;
            border: 0px;
            flex-grow: 1;
        }
        
        /* 3x2 Chart Grid Layout (3 rows, 2 columns) */
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 40px;
            min-height: 1200px;
        }
        
        .chart-container {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            cursor: pointer;
        }
        
        .chart-container:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .chart-container h2 {
            margin-top: 0;
            margin-bottom: 15px;
            color: #34495e;
            font-size: 16px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }
        
        /* Table Section */
        .table-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        
        .table-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .table-header h2 {
            margin: 0;
            color: #2c3e50;
        }
        
        .clear-btn {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .clear-btn:hover {
            background: #c0392b;
        }
        
        /* Data Table */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: white;
        }
        
        .data-table th,
        .data-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        .data-table th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
            position: sticky;
            top: 0;
        }
        
        .data-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        .data-table tr:hover {
            background-color: #e3f2fd;
        }
        
        .flagged-1 {
            background-color: #ffebee !important;
        }
        
        .flagged-2 {
            background-color: #fff3e0 !important;
        }
        
        .flagged-0 {
            background-color: #e8f5e8 !important;
        }
        
        .response-cell {
            max-width: 250px;
            word-wrap: break-word;
        }
        
        .expand-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 2px 6px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            margin-top: 5px;
            display: block;
        }
        
        .expand-btn:hover {
            background: #2980b9;
        }
        
        .full-text {
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        
        .short-text {
            word-wrap: break-word;
        }
        
        .table-info {
            margin-top: 10px;
            color: #666;
            font-size: 12px;
            font-style: italic;
        }
        
        /* Responsive Design */
        @media (max-width: 1200px) {
            .charts-grid {
                grid-template-columns: 1fr;
                grid-template-rows: repeat(6, 700px); /* Increased from 350px to 700px */
                min-height: auto;
                gap: 15px;
            }
            
            .chart-container {
                height: 660px; /* Ensure container height matches */
                padding: 10px;
            }
            
            .chart-container .plotly-graph-div {
                height: 620px !important; 
                min-height: 620px;
            }
        }
    """


def _get_js_content() -> str:
    """JavaScript for click-based category filtering with dynamic data loading."""
    return """
        document.addEventListener('DOMContentLoaded', async function() {
            const tableSection = document.getElementById('tableSection');
            const tableTitle = document.getElementById('tableTitle');
            const tableContainer = document.getElementById('tableContainer');
            const clearBtn = document.getElementById('clearTable');
            const loadingIndicator = document.getElementById('loadingIndicator');
            const chartsGrid = document.getElementById('chartsGrid');
            
            try {
                // Load chart data from JSON files
                await loadChartData();
                
                // Hide loading indicator and show charts
                loadingIndicator.style.display = 'none';
                chartsGrid.style.display = 'grid';
                
                // Initialize chart interactions after data is loaded
                initializeChartInteractions();
                
            } catch (error) {
                console.error('Failed to load chart data:', error);
                loadingIndicator.innerHTML = '<p style="color: red;">Failed to load chart data. Please refresh the page.</p>';
            }
            
            async function loadChartData() {
                try {
                    // First, try to load manifest
                    const manifestResponse = await fetch('data/charts/manifest.json');
                    const manifest = await manifestResponse.json();
                    
                    // Load individual chart data files based on manifest
                    const dataPromises = window.chartTitles.map(async (title) => {
                        const filenames = manifest.files[title];
                        if (!filenames || filenames.length === 0) {
                            console.warn(`No file mapping for chart: ${title}`);
                            return { title, data: [] };
                        }
                        
                        // Load all chunks for this chart
                        const chunkPromises = filenames.map(async (filename) => {
                            const response = await fetch(`data/charts/${filename}`);
                            if (!response.ok) {
                                throw new Error(`Failed to load ${filename}: ${response.statusText}`);
                            }
                            return await response.json();
                        });
                        
                        const chunks = await Promise.all(chunkPromises);
                        
                        // Combine all chunks
                        const combinedData = chunks.flatMap(chunk => chunk.data);
                        
                        return { title, data: combinedData };
                    });
                    
                    const results = await Promise.all(dataPromises);
                    
                    // Populate window.chartData
                    results.forEach(({ title, data }) => {
                        window.chartData[title] = data;
                    });
                    
                } catch (error) {
                    console.error('Failed to load from manifest:', error);
                    throw error;
                }
                
                console.log('Chart data loaded:', Object.keys(window.chartData));
            }
            
            function initializeChartInteractions() {
                // Add click listeners to all charts
                Object.keys(window.chartData).forEach((chartTitle, index) => {
                    const chartDiv = document.getElementById(`chart_${index}`);
                    if (chartDiv) {
                        // Point click event
                        chartDiv.on('plotly_click', function(data) {
                            console.log('Click data:', data);
                            if (data.points && data.points.length > 0) {
                                const clickedCategory = data.points[0].data.name;
                                console.log('Clicked category from point:', clickedCategory);
                                showCategoryData(chartTitle, clickedCategory);
                            }
                        });
                        
                        // Custom legend click handling
                        let clickTimeout;
                        let clickCount = 0;
                        
                        chartDiv.on('plotly_legendclick', function(data) {
                            clickCount++;
                            
                            if (clickCount === 1) {
                                // Set timeout for single click
                                clickTimeout = setTimeout(() => {
                                    // Single click action
                                    const plotElement = document.getElementById(`chart_${index}`);
                                    const plotData = plotElement.data;
                                    
                                    if (plotData && plotData[data.curveNumber]) {
                                        const clickedCategory = plotData[data.curveNumber].name;
                                        console.log('Single clicked category from legend:', clickedCategory);
                                        showCategoryData(chartTitle, clickedCategory);
                                    }
                                    
                                    clickCount = 0;
                                }, 800); // 800ms delay to detect double click
                            } else if (clickCount === 2) {
                                // Double click detected - clear timeout and allow default behavior
                                clearTimeout(clickTimeout);
                                clickCount = 0;
                                return; // Let plotly handle the double-click (toggle visibility)
                            }
                            
                            // Prevent default single click behavior (but allow double-click)
                            return false;
                        });
                    }
                });
                
                // Clear table button
                clearBtn.addEventListener('click', function() {
                    tableSection.style.display = 'none';
                });
                
                // Handle expand/collapse clicks
                tableContainer.addEventListener('click', function(e) {
                    if (e.target.classList.contains('expand-btn')) {
                        const responseCell = e.target.closest('.response-cell');
                        const shortText = responseCell.querySelector('.short-text');
                        const fullText = responseCell.querySelector('.full-text');
                        const expandBtn = responseCell.querySelector('.expand-btn');
                        
                        if (fullText.style.display === 'none' || !fullText.style.display) {
                            // Show full text
                            shortText.style.display = 'none';
                            fullText.style.display = 'block';
                            expandBtn.textContent = 'Show Less';
                        } else {
                            // Show short text
                            shortText.style.display = 'block';
                            fullText.style.display = 'none';
                            expandBtn.textContent = 'Show More';
                        }
                    }
                });
            }
            
            function showCategoryData(chartTitle, category) {
                console.log('Showing data for:', chartTitle, category);
                
                // Get data for the clicked chart
                const chartData = window.chartData[chartTitle];
                
                if (!chartData) {
                    alert(`No data loaded for chart: ${chartTitle}`);
                    return;
                }
                
                // Filter data for the clicked category
                const categoryData = chartData.filter(row => row.category === category);
                
                console.log('Filtered data:', categoryData);
                
                if (categoryData.length === 0) {
                    alert(`No data found for category: ${category} in chart: ${chartTitle}`);
                    return;
                }
                
                // Update table title
                tableTitle.textContent = `${chartTitle} - ${category} Category Data`;
                
                // Generate table HTML
                const tableHTML = generateTableHTML(categoryData);
                tableContainer.innerHTML = tableHTML;
                
                // Show table section
                tableSection.style.display = 'block';
                
                // Scroll to table
                tableSection.scrollIntoView({ behavior: 'smooth' });
            }
            
            function generateTableHTML(data) {
                const flaggedCount = data.filter(row => row.flagged === 1).length;
                const lengthFlaggedCount = data.filter(row => row.flagged === 2).length;
                const notFlaggedCount = data.filter(row => row.flagged === 0).length;
                
                let html = `
                    <div class="table-info">
                        Total: ${data.length} items | 
                        Flagged: ${flaggedCount} | 
                        Length Flagged: ${lengthFlaggedCount} |
                        Not Flagged: ${notFlaggedCount}
                    </div>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Category</th>
                                <th>Topic</th>
                                <th>Source & Content</th>
                                <th>Model</th>
                                <th>Date</th>
                                <th>Flagged</th>
                                <th>Response / Reasons</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                data.forEach(row => {
                    const flaggedClass = row.flagged === 1 ? 'flagged-1' : 
                                       row.flagged === 2 ? 'flagged-2' : 'flagged-0';
                    
                    // Create source cell with permanent link and content preview
                    let sourceContent;
                    if (row.permanent_link) {
                        const linkText = row.source || 'Link';
                        let contentPreview = '';
                        
                        if (row.content && row.content.trim() !== '') {
                            // Show first 100 chars of content
                            const content = row.content.substring(0, 100);
                            contentPreview = `<br><small class="content-preview">${content}${row.content.length > 100 ? '...' : ''}</small>`;
                        } else {
                            // Show "see link" if no content
                            contentPreview = '<br><small class="see-link">see link</small>';
                        }
                        
                        sourceContent = `
                            <a href="${row.permanent_link}" target="_blank" class="source-link">${linkText}</a>
                            ${contentPreview}
                        `;
                    } else {
                        sourceContent = row.source || 'N/A';
                    }
                    
                    // Create expandable response cell
                    let responseContent;
                    if (row.model_response_needs_expand) {
                        const shortText = row.model_response_full.substring(0, 100) + '...';
                        responseContent = `
                            <div class="short-text">${shortText}</div>
                            <div class="full-text" style="display: none;">${row.model_response_full}</div>
                            <button class="expand-btn" type="button">Show More</button>
                        `;
                    } else {
                        responseContent = row.model_response_full || 'N/A';
                    }
                    
                    // Update flagged display
                    let flaggedDisplay;
                    if (row.flagged === 1) {
                        flaggedDisplay = '🚩 Yes';
                    } else if (row.flagged === 2) {
                        flaggedDisplay = '⚠️ Length';
                    } else {
                        flaggedDisplay = '✅ No';
                    }
                    
                    html += `
                        <tr class="${flaggedClass}">
                            <td>${row.category || 'N/A'}</td>
                            <td>${row.subcategory || 'N/A'}</td>
                            <td class="source-cell">${sourceContent}</td>
                            <td>${row.model || 'N/A'}</td>
                            <td>${row.date || 'N/A'}</td>
                            <td>${flaggedDisplay}</td>
                            <td class="response-cell">${responseContent}</td>
                        </tr>
                    `;
                });
                
                html += `
                        </tbody>
                    </table>
                `;
                
                return html;
            }
        });
    """