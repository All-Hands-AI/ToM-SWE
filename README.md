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
from tom_module.analyzer import CodeAnalyzer
from visualization.trajectory_viewer import plot_trajectory

# Initialize the analyzer
analyzer = CodeAnalyzer()

# Analyze code
code = """
def hello_world():
    print("Hello, world!")
"""
analysis = analyzer.analyze(code)

# Understand intent
intent = analyzer.understand_intent(code)

# Visualize trajectory
trajectory_data = [
    {"timestamp": "2023-06-15T10:00:00", "analysis": analysis},
    # More snapshots...
]
plot_trajectory(trajectory_data, "user_123")
```

## Project Structure

- `data/`: Contains user data and analysis results
- `templates/`: HTML templates for the web interface
- `test_typing/`: Type checking tests
- `tests/`: Unit and integration tests
- `tom_module/`: Core Theory of Mind functionality
- `utils/`: Utility functions and helpers
- `visualization/`: Tools for visualizing code trajectories

## Experiments

See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for details on our experiments and findings.

## RAG Agent

We've implemented a Retrieval-Augmented Generation agent to enhance code understanding. See [RAG_AGENT_SUMMARY.md](RAG_AGENT_SUMMARY.md) for details.

## License

MIT