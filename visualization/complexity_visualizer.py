"""Complexity visualization for ToM-SWE."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime

logger = logging.getLogger(__name__)


def plot_complexity_over_time(analyses: List[Dict[str, Any]]) -> None:
    """Plot code complexity over time.
    
    Args:
        analyses: List of analysis results.
    """
    logger.info("Plotting complexity over time")
    
    if not analyses:
        logger.warning("No analyses to plot")
        return
    
    # Extract timestamps and complexity values
    timestamps = []
    complexity_values = []
    
    for analysis in analyses:
        timestamp_str = analysis.get("timestamp", "")
        complexity = analysis.get("analysis", {}).get("cyclomatic_complexity", 1)
        
        if timestamp_str:
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                timestamps.append(timestamp)
                complexity_values.append(complexity)
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp format: {timestamp_str}")
    
    if not timestamps:
        logger.warning("No valid timestamps found in analyses")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(mdates.date2num(timestamps), complexity_values, 'b-', marker='o')
    plt.title("Code Complexity Over Time")
    plt.xlabel("Time")
    plt.ylabel("Cyclomatic Complexity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_complexity_distribution(analyses: List[Dict[str, Any]]) -> None:
    """Plot distribution of code complexity.
    
    Args:
        analyses: List of analysis results.
    """
    logger.info("Plotting complexity distribution")
    
    if not analyses:
        logger.warning("No analyses to plot")
        return
    
    # Extract complexity values
    complexity_values = [
        analysis.get("analysis", {}).get("cyclomatic_complexity", 1)
        for analysis in analyses
    ]
    
    if not complexity_values:
        logger.warning("No complexity values found in analyses")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(complexity_values, bins=10, alpha=0.7, color='blue')
    plt.title("Distribution of Code Complexity")
    plt.xlabel("Cyclomatic Complexity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_function_vs_class_count(analyses: List[Dict[str, Any]]) -> None:
    """Plot function count vs class count.
    
    Args:
        analyses: List of analysis results.
    """
    logger.info("Plotting function vs class count")
    
    if not analyses:
        logger.warning("No analyses to plot")
        return
    
    # Extract function and class counts
    function_counts = []
    class_counts = []
    
    for analysis in analyses:
        analysis_data = analysis.get("analysis", {})
        function_counts.append(analysis_data.get("functions", 0))
        class_counts.append(analysis_data.get("classes", 0))
    
    if not function_counts or not class_counts:
        logger.warning("No function or class counts found in analyses")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(function_counts, class_counts, alpha=0.7, s=100)
    plt.title("Function Count vs Class Count")
    plt.xlabel("Number of Functions")
    plt.ylabel("Number of Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_plots_to_html(analyses: List[Dict[str, Any]], output_path: str) -> None:
    """Save all plots to an HTML file.
    
    Args:
        analyses: List of analysis results.
        output_path: Path to save the HTML file.
    """
    logger.info(f"Saving plots to HTML: {output_path}")
    
    if not analyses:
        logger.warning("No analyses to plot")
        return
    
    # Extract data
    timestamps = []
    complexity_values = []
    function_counts = []
    class_counts = []
    
    for analysis in analyses:
        timestamp_str = analysis.get("timestamp", "")
        analysis_data = analysis.get("analysis", {})
        
        if timestamp_str:
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                complexity_values.append(analysis_data.get("cyclomatic_complexity", 1))
                function_counts.append(analysis_data.get("functions", 0))
                class_counts.append(analysis_data.get("classes", 0))
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp format: {timestamp_str}")
    
    if not timestamps:
        logger.warning("No valid timestamps found in analyses")
        return
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Code Analysis Visualizations</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .chart-container {{ height: 400px; margin-bottom: 30px; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Code Analysis Visualizations</h1>
        
        <div class="chart-container">
            <h2>Code Complexity Over Time</h2>
            <canvas id="complexityChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Distribution of Code Complexity</h2>
            <canvas id="distributionChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Function Count vs Class Count</h2>
            <canvas id="scatterChart"></canvas>
        </div>
    </div>

    <script>
        // Data
        const timestamps = {timestamps};
        const complexityValues = {complexity_values};
        const functionCounts = {function_counts};
        const classCounts = {class_counts};
        
        // Complexity over time chart
        const complexityCtx = document.getElementById('complexityChart').getContext('2d');
        new Chart(complexityCtx, {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'Cyclomatic Complexity',
                    data: complexityValues,
                    borderColor: 'blue',
                    backgroundColor: 'rgba(0, 0, 255, 0.1)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Complexity distribution chart
        const distributionCtx = document.getElementById('distributionChart').getContext('2d');
        new Chart(distributionCtx, {{
            type: 'bar',
            data: {{
                labels: Array.from(new Set(complexityValues)).sort((a, b) => a - b),
                datasets: [{{
                    label: 'Frequency',
                    data: Array.from(new Set(complexityValues)).sort((a, b) => a - b).map(
                        value => complexityValues.filter(v => v === value).length
                    ),
                    backgroundColor: 'rgba(0, 0, 255, 0.7)',
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Function vs class scatter chart
        const scatterCtx = document.getElementById('scatterChart').getContext('2d');
        new Chart(scatterCtx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Functions vs Classes',
                    data: functionCounts.map((f, i) => ({{ x: f, y: classCounts[i] }})),
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    pointRadius: 8,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Number of Functions'
                        }},
                        beginAtZero: true
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Number of Classes'
                        }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save to file
    with open(output_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Plots saved to {output_path}")


def create_complexity_report(analyses: List[Dict[str, Any]]) -> str:
    """Create a text report of complexity analysis.
    
    Args:
        analyses: List of analysis results.
        
    Returns:
        Text report.
    """
    logger.info("Creating complexity report")
    
    if not analyses:
        return "No analyses available for report."
    
    # Extract complexity values
    complexity_values = [
        analysis.get("analysis", {}).get("cyclomatic_complexity", 1)
        for analysis in analyses
    ]
    
    # Calculate statistics
    avg_complexity = sum(complexity_values) / len(complexity_values) if complexity_values else 0
    max_complexity = max(complexity_values) if complexity_values else 0
    min_complexity = min(complexity_values) if complexity_values else 0
    
    # Create report
    report = "Code Complexity Analysis Report\n"
    report += "==============================\n\n"
    report += f"Number of snapshots analyzed: {len(analyses)}\n"
    report += f"Average cyclomatic complexity: {avg_complexity:.2f}\n"
    report += f"Maximum complexity: {max_complexity}\n"
    report += f"Minimum complexity: {min_complexity}\n\n"
    
    # Complexity trend
    if len(complexity_values) > 1:
        first = complexity_values[0]
        last = complexity_values[-1]
        change = last - first
        percent_change = (change / first) * 100 if first != 0 else 0
        
        report += "Complexity Trend:\n"
        report += f"Initial complexity: {first}\n"
        report += f"Final complexity: {last}\n"
        report += f"Change: {change:+.2f} ({percent_change:+.2f}%)\n\n"
        
        if change > 0:
            report += "The code complexity has increased over time.\n"
        elif change < 0:
            report += "The code complexity has decreased over time.\n"
        else:
            report += "The code complexity has remained stable over time.\n"
    
    return report