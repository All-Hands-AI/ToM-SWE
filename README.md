# ToM-SWE

Theory of Mind package for SWE agents. Provides personalized consultation and guidance for SWE agents through user behavior analysis and a three-tier memory system.

## Quick Start

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up LLM API credentials:
   ```bash
   # Create .env file
   LITELLM_API_KEY=your_api_key_here
   LITELLM_BASE_URL=your_proxy_endpoint
   DEFAULT_LLM_MODEL=litellm_proxy/claude-sonnet-4-20250514
   ```

3. Run example or configure LLM:
   ```bash
   uv run python example.py           # See consultation functionality in action
   uv run tom-config                  # Interactive LLM setup
   ```

## Core Features

- **Three-Tier Memory**: Cleaned sessions → Session analyses → User profiles
- **Agent Consultation**: Provides personalized guidance and recommendations for SWE agents
- **User Behavior Analysis**: LLM-powered psychological insights and preferences
- **OpenHands Integration**: Use `TomCodeActAgent` for automatic instruction enhancement

## Main Commands

```bash
# User analysis
uv run user-analysis --user-id <user_id>

# Theory of Mind analysis
uv run tom-test                      # Test on sample users
uv run tom-analyze                   # Full analysis

# RAG document analysis
uv run rag-agent
```

## OpenHands Integration

Configure OpenHands to use Tom-enhanced agent:
```toml
default_agent = "TomCodeActAgent"
```

The agent automatically:
1. Provides consultation and personalized guidance to SWE agents
2. Processes user sessions for better understanding
3. Shows progress during analysis

## Prompts

The prompts are stored in `tom_swe/prompts/registry.py`.
You could also find some prompts in `tom_swe/generation/dataclass.py`


## Requirements

- Python 3.8+
- uv package manager
- LLM API key (contact All Hands AI for access)
