# ToM-SWE: Theory of Mind for Software Engineering

This module implements Theory of Mind principles for software engineering tasks.

## Components

- `tom_module.py`: Main module implementing Theory of Mind functionality
- `database.py`: Database functionality for storing and retrieving analyses
- `utils/`: Utility functions for code analysis and LLM configuration
- `visualization/`: Tools for visualizing code analysis results
- `templates/`: HTML templates for the web interface

## Usage

```python
from tom_swe import ToMModule

# Initialize the module
tom = ToMModule()

# Analyze code
code = """
def hello_world():
    print("Hello, world!")
"""
analysis = tom.analyze_code(code)

# Understand intent
intent = tom.understand_intent(code)

# Explain code
explanation = tom.explain_code(code, audience="beginner")

# Suggest improvements
suggestions = tom.suggest_improvements(code)
```

## Visualization

The module includes visualization tools for displaying code analysis results:

```python
from tom_swe.visualization.display_trajectory import plot_complexity_over_time

# Plot complexity metrics over time
plot_complexity_over_time(analyses)
```

## Web Interface

A web interface is available for interacting with the module:

```python
from tom_swe.visualization.web_trajectory_viewer import TrajectoryViewer

# Create and run the viewer
viewer = TrajectoryViewer()
viewer.load_analyses(analyses)
viewer.run()
```