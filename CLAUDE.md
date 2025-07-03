# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies with uv (recommended)
uv sync

# Install with development dependencies (for linting, testing, etc.)
uv sync --extra dev

# Interactive LLM configuration setup
uv run tom-config
```

**Note**: This project uses `uv` for dependency management. Always use `uv` commands instead of `pip` for installing dependencies to ensure consistency with the project's lock file.

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_process_and_save_steps.py

# Run tests with verbose output
pytest -v
```

### Linting and Code Quality
```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Format code with black
black .

# Sort imports with isort
isort .

# Run ruff linter
ruff check .

# Run mypy type checker
mypy .
```

### Main Application Commands
```bash
# Basic user analysis
uv run user-analysis

# User analysis with options
uv run user-analysis --user-id <user_id> --output-dir <path> --generate-viz

# Analyze all users (with sampling)
uv run user-analysis --all-users --sample-size 100

# Theory of Mind analysis (test on sample users)
uv run tom-test

# Run full ToM analysis
uv run tom-analyze

# RAG agent for document analysis
uv run rag-agent
```

### Visualization Tools
```bash
# Start web trajectory viewer
python visualization/web_trajectory_viewer.py --user-id <user_id>

# Web viewer with custom host/port
python visualization/web_trajectory_viewer.py --user-id <user_id> --host 0.0.0.0 --port 8080

# Terminal trajectory viewer
python visualization/display_trajectory.py --user-id <user_id>
```

### Data Processing
```bash
# Combine events to trajectory (utility script)
python utils/combine_events_to_trajectory.py

# Aggregate user behavior analysis
python utils/aggregate_user_behavior.py

# Pull data from Google Cloud
python utils/pull_google_cloud_data.py
```

## Architecture Overview

### Core Components

**Theory of Mind Module (`tom_swe/`)**
- `tom_module.py`: Main psychological analysis engine using LLM for user mental state modeling
- `database.py`: Pydantic data models for user profiles, session summaries, and message analyses
- `rag_module.py`: RAG-based document analysis and question answering
- `utils.py`: Utility functions for structured LLM output parsing

**User Analysis Pipeline**
- `user_interaction_analysis.py`: Comprehensive user behavior analysis with n-gram patterns and metrics
- `analyze_users.py`: Individual user analysis with session statistics
- `analyze_all_users.py`: Batch analysis across multiple users with sampling

**Visualization System**
- `web_trajectory_viewer.py`: Flask-based web interface for interactive trajectory viewing
- `display_trajectory.py`: Rich-based terminal interface for trajectory data
- `user_behavior_visualizer.py`: Statistical visualizations and charts
- `complexity_visualizer.py`: Code complexity analysis visualization

**Data Processing Utils (`utils/`)**
- `llm_client.py`: Abstracted LLM client supporting multiple providers
- `data_utils.py`: Data loading and preprocessing utilities
- `configure_llm.py`: Interactive LLM configuration and setup

### Data Flow Architecture

1. **Raw Data**: User session data in `data/sessions/` and CSV metadata in `data/studio_results_*.csv`
2. **Processing**: `combine_events_to_trajectory.py` converts raw sessions to structured trajectory data
3. **Analysis**:
   - User interaction analysis generates metrics and HTML reports
   - ToM module creates psychological profiles and predictions
4. **Visualization**: Web and terminal viewers provide interactive data exploration
5. **Output**: Results saved to `data/user_analysis/` and `data/user_model/`

### Key Design Patterns

**Async Processing**: Heavy use of asyncio for concurrent LLM calls and data processing
**Caching**: Multi-level caching in web viewer for performance (metadata, individual conversations)
**Modular Analysis**: Separate concerns between statistical analysis and psychological modeling
**Pydantic Models**: Structured data validation throughout the pipeline
**Graceful Degradation**: Fallback to rule-based analysis when LLM unavailable

### LLM Integration

The system uses litellm proxy for Claude models:
- Primary model: `litellm_proxy/claude-sonnet-4-20250514`
- Fallback models: Claude 3.5 Sonnet, Claude Haiku for testing
- Configuration via `.env` file with `LITELLM_API_KEY` and `LITELLM_BASE_URL`
- Structured output parsing with custom Pydantic parsers

### Testing Strategy

- `pytest` with async support for concurrent testing
- Mock fixtures for LLM responses to avoid API calls during testing
- Temporary directories for file I/O testing
- Comprehensive test coverage for data processing pipeline
