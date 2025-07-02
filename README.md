# User Research Analysis Project

A comprehensive analysis toolkit for understanding user behavior and mental states in coding assistant interactions.

## Components

### 1. User Interaction Analysis (`user_interaction_analysis.py`)
- Analyzes user session data and message patterns
- Generates comprehensive metrics and n-gram analysis
- Creates interactive HTML reports
- Supports individual user analysis and comprehensive analysis across all users
- Includes studio results integration for cost and token usage analysis

### 2. Theory of Mind Module (`tom_module/`)
- LLM-powered psychological analysis of user behavior
- Predicts user intentions and next actions
- Generates mental state profiles and recommendations

### 3. Visualization Tools
#### Web Trajectory Viewer (`visualization/web_trajectory_viewer.py`)
- Modern, interactive web interface for viewing trajectory data
- User selection and conversation browsing
- Detailed conversation view with filtering options
- Keyboard shortcuts for navigation
- Responsive design for desktop and mobile

#### Terminal Trajectory Viewer (`visualization/display_trajectory.py`)
- Rich-based CLI interface for viewing trajectory data
- Fast navigation and data exploration
- Works in any terminal environment
- Compact display of information

## Installation

Install with uv (recommended):
```bash
uv sync
```

## LLM Configuration

This project uses the All Hands AI LLM proxy for sophisticated mental state analysis. You need an API key from Xingyao.

### Quick Setup
```bash
uv run tom-config
```

This interactive script will help you:
- Set up your API key and proxy endpoint
- Choose the best model for your needs
- Create a `.env` file with proper configuration

### Manual Setup
Create a `.env` file in the project root:
```bash
# LLM API Configuration
LITELLM_API_KEY=your_api_key_here
LITELLM_BASE_URL=https://your-proxy-endpoint.com
DEFAULT_LLM_MODEL=litellm_proxy/claude-3-5-sonnet-20241022
```

### Available Models
All models require the `litellm_proxy/` prefix:
- `litellm_proxy/claude-sonnet-4-20250514` (Most capable)
- `litellm_proxy/claude-opus-4-20250514` (Most capable)
- `litellm_proxy/claude-3-5-sonnet-20241022` (Recommended)
- `litellm_proxy/claude-3-5-haiku-20241022` (Fastest, good for testing)

## Usage

### Basic User Analysis
```bash
uv run user-analysis
```

Options:
- `--user-id`: Analyze specific user ID
- `--all-users`: Analyze all users
- `--sample-size`: Sample size for all users analysis
- `--output-dir`: Output directory for results
- `--generate-viz`: Generate visualizations
- `--studio-analysis`: Include studio results analysis

### Theory of Mind Analysis
Test on sample users:
```bash
uv run tom-test
```

Run full ToM analysis:
```bash
uv run tom-analyze
```

### Web Trajectory Viewer
Start the web server:
```bash
python visualization/web_trajectory_viewer.py
```

Options:
- `--host HOST`: Host to run the server on (default: 127.0.0.1)
- `--port PORT`: Port to run the server on (default: 5000)
- `--debug`: Run in debug mode

### Terminal Trajectory Viewer
View trajectory data in terminal:
```bash
python visualization/display_trajectory.py --user-id=<user_id>
```

## Data Structure

The project expects the following data structure:
```
data/
├── processed_data/
│   ├── {user_id}.json    # Trajectory data for each user
├── power_user_2025_05.csv
|── sessions/
|   |── {user_id}    # A folder containing all the conversation data for the user pull directly from the google cloud bucket.
└── studio_results_20250604_1645.csv
```
One can obtain the processed data for each user through running the `combine_events_to_trajectory.py` script in the `utils` directory.

## Output

- `data/user_analysis/`: User interaction statistics and reports
- `data/user_model/`: Theory of Mind psychological profiles and predictions

## Analysis Metrics

### User Engagement Levels
- **High**: 50+ conversations
- **Medium**: 10-49 conversations
- **Low**: 1-9 conversations

### Key Metrics Analyzed
- Total conversations per user
- Conversation duration patterns
- User-to-agent message ratios
- Event type distributions
- Cost and token usage (from studio results)
- Repository engagement patterns
- Time-based usage patterns

## Requirements

- Python 3.8+
- LLM API key from Xingyao for All Hands AI proxy (for ToM analysis)
- uv package manager (recommended)

See `pyproject.toml` for full dependency list.

## Troubleshooting

### LLM API Issues
If you get LLM-related errors:
1. Run `uv run tom-config` to check your configuration
2. Ensure your API key is valid and has sufficient credits
3. Try using a different model (e.g., haiku for testing)

### Import Errors
If you get import errors when testing:
```bash
uv sync  # Ensure all dependencies are installed
```

### Web Viewer Issues
- **Port Already in Use**: Try a different port with `--port` option
- **Data Not Found**: Ensure trajectory data files are in `./data/processed_data/`
- **Permission Errors**: Use `--host 127.0.0.1 --port 8080` on restricted systems

The module will gracefully fallback to rule-based analysis if LLM access fails.
