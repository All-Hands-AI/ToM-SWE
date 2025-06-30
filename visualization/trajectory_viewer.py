"""Trajectory visualization for ToM-SWE."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import datetime

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, jsonify, request

logger = logging.getLogger(__name__)


def plot_trajectory(trajectory_data: List[Dict[str, Any]], user_id: str) -> None:
    """Plot code trajectory over time.
    
    Args:
        trajectory_data: List of analysis results with timestamps.
        user_id: ID of the user.
    """
    logger.info(f"Plotting trajectory for user {user_id}")
    
    if not trajectory_data:
        logger.warning("No trajectory data to plot")
        return
    
    # Extract timestamps and complexity values
    timestamps = []
    complexity_values = []
    function_counts = []
    class_counts = []
    
    for point in trajectory_data:
        timestamp = point.get("timestamp", "")
        analysis = point.get("analysis", {})
        
        if timestamp and analysis:
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                timestamps.append(dt)
                complexity_values.append(analysis.get("cyclomatic_complexity", 1))
                function_counts.append(analysis.get("functions", 0))
                class_counts.append(analysis.get("classes", 0))
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp format: {timestamp}")
    
    if not timestamps:
        logger.warning("No valid timestamps found in trajectory data")
        return
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot complexity over time
    ax1.plot(timestamps, complexity_values, 'b-', marker='o')
    ax1.set_title(f"Code Complexity Over Time - User {user_id}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Cyclomatic Complexity")
    ax1.grid(True)
    
    # Plot function and class counts
    ax2.plot(timestamps, function_counts, 'g-', marker='o', label="Functions")
    ax2.plot(timestamps, class_counts, 'r-', marker='s', label="Classes")
    ax2.set_title(f"Functions and Classes Over Time - User {user_id}")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"trajectory_{user_id}.png"
    plt.savefig(output_path)
    logger.info(f"Trajectory plot saved to {output_path}")
    
    plt.close(fig)


def create_interactive_visualization(trajectory_data: List[Dict[str, Any]], user_id: str) -> str:
    """Create an interactive HTML visualization of code trajectory.
    
    Args:
        trajectory_data: List of analysis results with timestamps.
        user_id: ID of the user.
        
    Returns:
        Path to the HTML file.
    """
    logger.info(f"Creating interactive visualization for user {user_id}")
    
    if not trajectory_data:
        logger.warning("No trajectory data to visualize")
        return ""
    
    # Prepare data for visualization
    vis_data = []
    
    for point in trajectory_data:
        timestamp = point.get("timestamp", "")
        analysis = point.get("analysis", {})
        
        if timestamp and analysis:
            vis_data.append({
                "timestamp": timestamp,
                "complexity": analysis.get("cyclomatic_complexity", 1),
                "functions": analysis.get("functions", 0),
                "classes": analysis.get("classes", 0),
                "lines": analysis.get("lines", 0),
                "intent": analysis.get("intent_analysis", {}).get("inferred_intent", "")
            })
    
    if not vis_data:
        logger.warning("No valid data points found for visualization")
        return ""
    
    # Create HTML file
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"visualization_{user_id}.html"
    
    html_content = _generate_html_visualization(vis_data, user_id)
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Interactive visualization saved to {output_path}")
    return str(output_path)


def _generate_html_visualization(data: List[Dict[str, Any]], user_id: str) -> str:
    """Generate HTML content for visualization.
    
    Args:
        data: Visualization data.
        user_id: ID of the user.
        
    Returns:
        HTML content.
    """
    # Convert data to JSON for JavaScript
    json_data = json.dumps(data)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Code Trajectory - User {user_id}</title>
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
        <h1>Code Trajectory Analysis - User {user_id}</h1>
        
        <div class="chart-container">
            <h2>Complexity Over Time</h2>
            <canvas id="complexityChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Functions and Classes</h2>
            <canvas id="structureChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Code Size (Lines)</h2>
            <canvas id="sizeChart"></canvas>
        </div>
    </div>

    <script>
        // Data from Python
        const trajectoryData = {json_data};
        
        // Extract data for charts
        const timestamps = trajectoryData.map(d => d.timestamp);
        const complexity = trajectoryData.map(d => d.complexity);
        const functions = trajectoryData.map(d => d.functions);
        const classes = trajectoryData.map(d => d.classes);
        const lines = trajectoryData.map(d => d.lines);
        
        // Complexity chart
        const complexityCtx = document.getElementById('complexityChart').getContext('2d');
        new Chart(complexityCtx, {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'Cyclomatic Complexity',
                    data: complexity,
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
        
        // Structure chart
        const structureCtx = document.getElementById('structureChart').getContext('2d');
        new Chart(structureCtx, {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [
                    {{
                        label: 'Functions',
                        data: functions,
                        borderColor: 'green',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        tension: 0.1
                    }},
                    {{
                        label: 'Classes',
                        data: classes,
                        borderColor: 'red',
                        backgroundColor: 'rgba(255, 0, 0, 0.1)',
                        tension: 0.1
                    }}
                ]
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
        
        // Size chart
        const sizeCtx = document.getElementById('sizeChart').getContext('2d');
        new Chart(sizeCtx, {{
            type: 'line',
            data: {{
                labels: timestamps,
                datasets: [{{
                    label: 'Lines of Code',
                    data: lines,
                    borderColor: 'purple',
                    backgroundColor: 'rgba(128, 0, 128, 0.1)',
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
    </script>
</body>
</html>
"""
    return html


class TrajectoryViewer:
    """Web-based trajectory viewer."""
    
    def __init__(self, data_dir: str = "data") -> None:
        """Initialize the trajectory viewer.
        
        Args:
            data_dir: Directory containing analysis data.
        """
        self.app = Flask(__name__, template_folder="../templates")
        self.data_dir = Path(data_dir)
        self.setup_routes()
        logger.info("TrajectoryViewer initialized")
    
    def setup_routes(self) -> None:
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            """Render the index page."""
            # Get list of available users
            user_files = list(self.data_dir.glob("analysis_*.json"))
            users = [f.stem.replace("analysis_", "") for f in user_files]
            return render_template('index.html', users=users)
        
        @self.app.route('/user/<user_id>')
        def user_trajectory(user_id):
            """Render trajectory for a specific user."""
            analysis_file = self.data_dir / f"analysis_{user_id}.json"
            
            if not analysis_file.exists():
                return render_template('error.html', message=f"No data found for user {user_id}")
            
            with open(analysis_file, "r") as f:
                analysis_data = json.load(f)
            
            # Create visualization
            vis_path = create_interactive_visualization(
                analysis_data.get("results", []), user_id
            )
            
            return render_template(
                'trajectory.html',
                user_id=user_id,
                visualization_path=vis_path,
                analysis_data=analysis_data
            )
        
        @self.app.route('/api/users')
        def api_users():
            """API endpoint for user list."""
            user_files = list(self.data_dir.glob("analysis_*.json"))
            users = [f.stem.replace("analysis_", "") for f in user_files]
            return jsonify(users)
        
        @self.app.route('/api/user/<user_id>')
        def api_user_data(user_id):
            """API endpoint for user data."""
            analysis_file = self.data_dir / f"analysis_{user_id}.json"
            
            if not analysis_file.exists():
                return jsonify({"error": f"No data found for user {user_id}"})
            
            with open(analysis_file, "r") as f:
                analysis_data = json.load(f)
            
            return jsonify(analysis_data)
    
    def run(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        """Run the Flask application.
        
        Args:
            host: Host to run the server on.
            port: Port to run the server on.
        """
        logger.info(f"Starting TrajectoryViewer on {host}:{port}")
        self.app.run(host=host, port=port, debug=True)