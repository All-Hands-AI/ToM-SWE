"""Visualization tools for displaying code analysis trajectories."""

import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


def plot_complexity_over_time(analyses: List[Dict[str, Any]]) -> None:
    """
    Plot code complexity over time.

    Args:
        analyses: A list of code analysis results.
    """
    if not analyses:
        print("No analyses to plot")
        return
    
    # Extract timestamps and complexity values
    timestamps = []
    lines = []
    functions = []
    classes = []
    
    for analysis in analyses:
        timestamps.append(analysis.get("timestamp", ""))
        metrics = analysis.get("analysis", {})
        lines.append(metrics.get("lines", 0))
        functions.append(metrics.get("functions", 0))
        classes.append(metrics.get("classes", 0))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, lines, 'b-', marker='o')
    plt.title('Code Complexity Over Time')
    plt.ylabel('Lines of Code')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, functions, 'g-', marker='o')
    plt.ylabel('Number of Functions')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, classes, 'r-', marker='o')
    plt.ylabel('Number of Classes')
    plt.xlabel('Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_complexity_distribution(analyses: List[Dict[str, Any]]) -> None:
    """
    Plot distribution of code complexity.

    Args:
        analyses: A list of code analysis results.
    """
    if not analyses:
        print("No analyses to plot")
        return
    
    # Count complexity categories
    complexity_counts = {"low": 0, "medium": 0, "high": 0}
    
    for analysis in analyses:
        metrics = analysis.get("analysis", {})
        complexity = metrics.get("complexity", "low")
        complexity_counts[complexity] += 1
    
    # Create the plot
    categories = list(complexity_counts.keys())
    counts = list(complexity_counts.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color=['green', 'orange', 'red'])
    plt.title('Distribution of Code Complexity')
    plt.xlabel('Complexity Level')
    plt.ylabel('Number of Code Snippets')
    plt.grid(axis='y')
    plt.show()


def plot_function_vs_class_count(analyses: List[Dict[str, Any]]) -> None:
    """
    Plot function count vs class count.

    Args:
        analyses: A list of code analysis results.
    """
    if not analyses:
        print("No analyses to plot")
        return
    
    # Extract function and class counts
    functions = []
    classes = []
    
    for analysis in analyses:
        metrics = analysis.get("analysis", {})
        functions.append(metrics.get("functions", 0))
        classes.append(metrics.get("classes", 0))
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(functions, classes, alpha=0.7)
    plt.title('Function Count vs Class Count')
    plt.xlabel('Number of Functions')
    plt.ylabel('Number of Classes')
    plt.grid(True)
    plt.show()


def save_plots_to_html(analyses: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save visualization plots to an HTML file.

    Args:
        analyses: A list of code analysis results.
        output_path: Path to save the HTML file.
    """
    import io
    import base64
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    if not analyses:
        print("No analyses to plot")
        return
    
    # Function to convert plot to base64 image
    def plot_to_base64(fig):
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{img_data}'
    
    # Create plots
    # Plot 1: Complexity over time
    fig1 = Figure(figsize=(10, 6))
    
    # Extract data
    timestamps = []
    lines = []
    functions = []
    classes = []
    
    for analysis in analyses:
        timestamps.append(analysis.get("timestamp", ""))
        metrics = analysis.get("analysis", {})
        lines.append(metrics.get("lines", 0))
        functions.append(metrics.get("functions", 0))
        classes.append(metrics.get("classes", 0))
    
    # Create subplots
    ax1 = fig1.add_subplot(3, 1, 1)
    ax1.plot(timestamps, lines, 'b-', marker='o')
    ax1.set_title('Code Complexity Over Time')
    ax1.set_ylabel('Lines of Code')
    ax1.grid(True)
    
    ax2 = fig1.add_subplot(3, 1, 2)
    ax2.plot(timestamps, functions, 'g-', marker='o')
    ax2.set_ylabel('Number of Functions')
    ax2.grid(True)
    
    ax3 = fig1.add_subplot(3, 1, 3)
    ax3.plot(timestamps, classes, 'r-', marker='o')
    ax3.set_ylabel('Number of Classes')
    ax3.set_xlabel('Time')
    ax3.grid(True)
    
    fig1.tight_layout()
    img1 = plot_to_base64(fig1)
    
    # Plot 2: Complexity distribution
    fig2 = Figure(figsize=(8, 6))
    
    # Count complexity categories
    complexity_counts = {"low": 0, "medium": 0, "high": 0}
    
    for analysis in analyses:
        metrics = analysis.get("analysis", {})
        complexity = metrics.get("complexity", "low")
        complexity_counts[complexity] += 1
    
    categories = list(complexity_counts.keys())
    counts = list(complexity_counts.values())
    
    ax = fig2.add_subplot(1, 1, 1)
    ax.bar(categories, counts, color=['green', 'orange', 'red'])
    ax.set_title('Distribution of Code Complexity')
    ax.set_xlabel('Complexity Level')
    ax.set_ylabel('Number of Code Snippets')
    ax.grid(axis='y')
    
    img2 = plot_to_base64(fig2)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Code Analysis Visualizations</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .plot-container {{ margin-bottom: 30px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Code Analysis Visualizations</h1>
        
        <div class="plot-container">
            <h2>Code Complexity Over Time</h2>
            <img src="{img1}" alt="Code Complexity Over Time">
        </div>
        
        <div class="plot-container">
            <h2>Distribution of Code Complexity</h2>
            <img src="{img2}" alt="Distribution of Code Complexity">
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Visualizations saved to {output_path}")