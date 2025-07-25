# ToM-SWE

A package for doing theory of mind on users and help SWE agents understand users better.

## Quick Start

To see the ToM Agent's `propose_instructions` function in action:

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up your LLM API credentials in `.env` file:
   ```bash
   # Create .env file in project root
   LITELLM_API_KEY=your_api_key_here
   LITELLM_BASE_URL=your_proxy_endpoint
   DEFAULT_LLM_MODEL=litellm_proxy/claude-sonnet-4-20250514
   ```

3. Run the example:
   ```bash
   uv run python example.py
   ```

The example shows how the ToM Agent analyzes user context and improves vague instructions like "Fix my code" with personalized suggestions and confidence scores.


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
