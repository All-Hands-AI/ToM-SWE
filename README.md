# ToM-SWE

Theory of Mind for Software Engineering

This repository contains code for the ToM-SWE project, which applies Theory of Mind principles to software engineering tasks.

## Overview

ToM-SWE helps developers understand code better through advanced analysis and visualization. It provides tools for:

- Code analysis and complexity metrics
- Understanding developer intent
- Visualizing code complexity and development trajectories
- Tracking user interactions with code

## Installation

```bash
pip install -e .
```

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

## Components

- `tom_swe/`: Main package
  - `core.py`: Core functionality
  - `tom_module.py`: Theory of Mind module
  - `database.py`: Database functionality
  - `utils/`: Utility functions
  - `visualization/`: Visualization tools
  - `templates/`: HTML templates for web interface
- `tests/`: Test suite

## Web Interface

The package includes a web interface for interacting with the ToM-SWE functionality:

```python
from tom_swe.visualization.web_trajectory_viewer import TrajectoryViewer

# Create and run the viewer
viewer = TrajectoryViewer()
viewer.run()
```

## License

MIT