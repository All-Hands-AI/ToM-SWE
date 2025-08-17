# Theory of Mind (ToM) Module

An LLM-powered analysis system for understanding user mental states and predicting their next actions based on interaction patterns with coding assistants.

## Overview

This module analyzes user interaction data from processed conversation logs to:

- **Extract user intents** from typed messages (debugging, code generation, learning, etc.)
- **Identify emotional states** (frustrated, confident, exploratory, etc.)
- **Generate psychological profiles** including cognitive style, learning preferences, and expertise assessment
- **Predict next actions** with confidence scores and reasoning
- **Provide interaction recommendations** for optimal assistant behavior

## Features

- **LLM-based Analysis**: Uses Large Language Models (via litellm) for nuanced understanding of user behavior
- **Async Processing**: Full async/await support for concurrent processing and better performance
- **Comprehensive Profiling**: Multi-dimensional psychological analysis covering cognitive, emotional, and behavioral patterns
- **Mental State Prediction**: Forecasts user intentions and next likely actions
- **Scalable Processing**: Handles large datasets of user interaction logs with concurrent execution
- **Rich Output**: Generates both machine-readable JSON and human-readable summaries

## Installation

Using uv (recommended):

```bash
cd tom_module
uv sync
```

Using pip:

```bash
cd tom_module
pip install -e .
```

## Usage

### Command Line

Test the module on a few users:
```bash
tom-test
```

Run full analysis on all users:
```bash
tom-analyze
```

### Python API

#### Synchronous Usage (Legacy)

```python
from tom_module import ToMAnalyzer

# Initialize analyzer
analyzer = ToMAnalyzer(
    processed_data_dir="./data/processed_data",
    model="gpt-4o-mini"
)

# Analyze a specific user session
result = analyzer.analyze_user_mental_state("user_id", "session_id")
if result:
    analyses, messages = result
    print(f"Analyzed {len(analyses)} messages")
```

#### Async Usage (Recommended)

```python
import asyncio
from tom_module.tom_module import ToMAnalyzer

async def analyze_user():
    # Initialize analyzer
    analyzer = ToMAnalyzer(
        processed_data_dir="./data/processed_data",
        model="gpt-4o-mini"
    )

    # Analyze a single message
    analysis = await analyzer.analyze_user_message("Help me debug this code")
    print(f"Intent: {analysis.intent}, Emotions: {analysis.emotions}")

    # Analyze a user session
    result = await analyzer.analyze_user_mental_state("user_id", "session_id")
    if result:
        analyses, messages = result
        print(f"Analyzed {len(analyses)} messages concurrently")

    # Process all sessions for a user (concurrent processing)
    await analyzer.process_all_user_sessions("user_id", "./data/user_model")

# Run async function
asyncio.run(analyze_user())
```

#### Convenience Functions

```python
import asyncio
from tom_module.tom_module import (
    analyze_user_async,
    process_user_session_async,
    process_all_sessions_async
)

async def quick_analysis():
    # Analyze a single session
    result = await analyze_user_async("user_id", "session_id")

    # Process a single session
    await process_user_session_async("user_id", "session_id")

    # Process all sessions for a user
    await process_all_sessions_async("user_id")

asyncio.run(quick_analysis())
```

### Batch Analysis

#### Async Batch Processing (Recommended)

```python
import asyncio
from tom_module.tom_module import ToMAnalyzer

async def process_multiple_users():
    analyzer = ToMAnalyzer("./data/processed_data")

    user_ids = ["user1", "user2", "user3"]

    # Process all users concurrently
    tasks = [
        analyzer.process_all_user_sessions(user_id, "./data/user_model")
        for user_id in user_ids
    ]

    await asyncio.gather(*tasks)
    print("All users processed concurrently!")

asyncio.run(process_multiple_users())
```

#### Legacy Batch Processing

```python
from tom_module import analyze_all_users

# Process all users and generate models
analyze_all_users(
    processed_data_dir="./data/processed_data",
    output_dir="./data/user_model"
)
```

## Performance Benefits

The async implementation provides significant performance improvements:

### Concurrent Processing
- **Multiple Messages**: Process all messages in a session concurrently
- **Multiple Sessions**: Process all sessions for a user concurrently
- **Multiple Users**: Process multiple users concurrently

### Performance Comparison
```python
# Sequential: ~10 seconds for 10 messages
for message in messages:
    await analyzer.analyze_user_message(message)

# Concurrent: ~2 seconds for 10 messages (5x faster)
tasks = [analyzer.analyze_user_message(msg) for msg in messages]
await asyncio.gather(*tasks)
```

### Scalability
- **I/O Bound Operations**: Non-blocking API calls to LLM services
- **Memory Efficient**: Processes data streams without loading everything into memory
- **Resource Utilization**: Better CPU and network utilization

## Data Requirements

The module expects user data in JSON format with the following structure:

```json
{
  "session_id": {
    "convo_start": "2025-05-20T21:52:07.071005",
    "convo_end": "2025-05-20T21:58:39.528188",
    "convo_events": [
      {
        "id": "processed_0001",
        "source": "user",
        "content": "User message content here"
      }
    ]
  }
}
```

## Output

For each user, the module generates:

- `{user_id}_mental_state.json`: Complete psychological analysis with predictions
- `{user_id}_summary.txt`: Human-readable summary report
- `aggregate_mental_state_analysis.json`: Cross-user statistics
- `aggregate_analysis_report.txt`: Summary across all users

## Configuration

### LLM Models

The module supports any model available through litellm:

```python
analyzer = ToMAnalyzer(model="gpt-4")  # OpenAI
analyzer = ToMAnalyzer(model="claude-3-sonnet")  # Anthropic
analyzer = ToMAnalyzer(model="gemini-pro")  # Google
```

### Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Mental State Analysis

The module provides comprehensive psychological profiling including:

### Behavioral Categories
- **Intent Classification**: debugging, code_generation, learning, optimization, etc.
- **Emotional States**: frustrated, confident, exploratory, focused, etc.
- **Cognitive Styles**: systematic vs. exploratory problem-solving approaches
- **Communication Patterns**: concise vs. detailed interaction styles

### Predictions
- **Next Intent**: Most likely user goal with confidence score
- **Predicted Actions**: Specific actions the user might take
- **Mental State**: Current psychological state and trajectory
- **Recommendations**: Suggestions for optimal assistant interaction

## Example Output

```
COMPREHENSIVE USER MENTAL STATE ANALYSIS
========================================

User ID: ff416668-1ea9-4a8c-b8b7-1c980d9df576
Analysis Date: 2025-01-04T15:30:00

INTERACTION STATISTICS:
- Total Messages Analyzed: 127
- Active Sessions: 23
- Avg Messages per Session: 5.5
- Intent Diversity: 6 different types

BEHAVIORAL PROFILE:
- Primary Intent: Code_Generation
- Dominant Emotion: Focused
- Focus Level: 0.78/1.0
- Expertise Level: 0.65/1.0

PSYCHOLOGICAL PROFILE:
• Cognitive Style: Systematic problem-solver who breaks down complex tasks
• Emotional Patterns: Generally calm and focused, shows frustration with configuration issues
• Learning Style: Prefers examples and hands-on experimentation
• Work Approach: Methodical, asks clarifying questions before proceeding

PREDICTIONS FOR NEXT INTERACTION:
- Most Likely Intent: Debugging
- Confidence: 0.73/1.0
- Mental State: User appears ready to tackle implementation challenges
- Predicted Actions:
  • Request specific code examples for current task
  • Ask for debugging help with implementation
  • Seek clarification on best practices

ASSISTANT INTERACTION RECOMMENDATIONS:
- Provide step-by-step guidance with code examples
- Anticipate follow-up questions about edge cases
- Offer alternative approaches when initial solution doesn't work
```

## Development

### Setup Development Environment

```bash
cd tom_module
uv sync --all-extras
```

### Run Tests

```bash
uv run tom-test
```

### Code Formatting

```bash
uv run black .
uv run isort .
```

### Type Checking

```bash
uv run mypy .
```

## License

MIT License - see LICENSE file for details.
