"""Web-based viewer for code analysis trajectories."""

import os
import json
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify


class TrajectoryViewer:
    """Web-based viewer for code analysis trajectories."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        """
        Initialize the trajectory viewer.

        Args:
            host: The host to run the server on.
            port: The port to run the server on.
        """
        self.host = host
        self.port = port
        self.app = Flask(__name__, 
                         template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))
        self.analyses = []
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Set up the Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/analyses')
        def get_analyses():
            return jsonify(self.analyses)
        
        @self.app.route('/analysis/<int:analysis_id>')
        def get_analysis(analysis_id):
            if 0 <= analysis_id < len(self.analyses):
                return jsonify(self.analyses[analysis_id])
            return jsonify({"error": "Analysis not found"}), 404
        
        @self.app.route('/visualize')
        def visualize():
            return render_template('visualization.html')
    
    def load_analyses(self, analyses: List[Dict[str, Any]]) -> None:
        """
        Load analyses data.

        Args:
            analyses: A list of code analysis results.
        """
        self.analyses = analyses
    
    def load_analyses_from_file(self, file_path: str) -> None:
        """
        Load analyses data from a JSON file.

        Args:
            file_path: Path to the JSON file.
        """
        try:
            with open(file_path, 'r') as f:
                self.analyses = json.load(f)
        except Exception as e:
            print(f"Error loading analyses from file: {e}")
    
    def run(self) -> None:
        """Run the web server."""
        print(f"Starting trajectory viewer at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=True)


def create_sample_data(output_path: str) -> None:
    """
    Create sample data for the trajectory viewer.

    Args:
        output_path: Path to save the sample data.
    """
    import datetime
    import random
    
    # Generate sample code snippets
    code_snippets = [
        """def hello_world():
    print("Hello, world!")""",
        
        """class Calculator:
    def __init__(self):
        self.result = 0
        
    def add(self, a, b):
        self.result = a + b
        return self.result""",
        
        """import numpy as np
import matplotlib.pyplot as plt

def plot_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title('Data Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()"""
    ]
    
    # Generate sample analyses
    analyses = []
    
    for i in range(10):
        # Select a code snippet or generate a variation
        if i < len(code_snippets):
            code = code_snippets[i]
        else:
            base_snippet = random.choice(code_snippets)
            # Add a comment to make it different
            code = base_snippet + f"\n# Variation {i}"
        
        # Generate timestamp
        timestamp = (datetime.datetime.now() - 
                     datetime.timedelta(days=10-i, 
                                        hours=random.randint(0, 23), 
                                        minutes=random.randint(0, 59))).isoformat()
        
        # Generate analysis
        lines = code.count('\n') + 1
        functions = code.count('def ')
        classes = code.count('class ')
        
        if lines > 10 or functions > 2 or classes > 1:
            complexity = "medium"
        elif lines > 20 or functions > 5 or classes > 2:
            complexity = "high"
        else:
            complexity = "low"
        
        analysis = {
            "id": i,
            "code_snippet": code,
            "analysis": {
                "lines": lines,
                "functions": functions,
                "classes": classes,
                "complexity": complexity
            },
            "timestamp": timestamp
        }
        
        analyses.append(analysis)
    
    # Save to file
    try:
        with open(output_path, 'w') as f:
            json.dump(analyses, f, indent=2)
        print(f"Sample data saved to {output_path}")
    except Exception as e:
        print(f"Error saving sample data: {e}")


def main() -> None:
    """Run the trajectory viewer with sample data."""
    sample_data_path = "sample_analyses.json"
    
    # Create sample data if it doesn't exist
    if not os.path.exists(sample_data_path):
        create_sample_data(sample_data_path)
    
    # Create and run the viewer
    viewer = TrajectoryViewer()
    viewer.load_analyses_from_file(sample_data_path)
    viewer.run()


if __name__ == "__main__":
    main()