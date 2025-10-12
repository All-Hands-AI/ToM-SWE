<img width="987" height="397" alt="Screenshot 2025-10-12 at 10 34 46 AM" src="https://github.com/user-attachments/assets/bfa44483-85a4-4614-b4b6-b887e00d7274" />


<h1 align="center">TOM-SWE: User Mental Modeling For Software Engineering Agents</h1>

<div align="center">

[![pypi](https://img.shields.io/pypi/v/tom-swe.svg)](https://pypi.python.org/pypi/tom-swe)
[![versions](https://img.shields.io/pypi/pyversions/tom-swe.svg)](https://github.com/XuhuiZhou/ToM-SWE)
[![CI](https://img.shields.io/github/actions/workflow/status/XuhuiZhou/ToM-SWE/tests.yml?branch=main&logo=github&label=CI)](https://github.com/XuhuiZhou/ToM-SWE/actions?query=branch%3Amain)

</div>

## News

* [10/2024] Beta testing program launched - join us in testing ToM-enhanced agents!

## Introduction

ToM-SWE is a Theory of Mind package designed to enhance Software Engineering agents with personalized user understanding and adaptive behavior.

ToM-SWE integrates seamlessly with OpenHands and other SWE agent frameworks, providing consultation capabilities that help agents understand user intent, preferences, and working styles for improved task performance.

## Join the Beta Test

```bash
pip install uv

uvx --python 3.12 --from git+https://github.com/XuhuiZhou/OpenHands@feature/tom-codeact-agent openhands
```

For details, please refer to the [Google doc](https://docs.google.com/document/d/1P8b1SSF_HYgahK6eO7qSHbOcTv3o3z6SWMH_osyR3_w/edit?usp=sharing)

## Get Started

### Install Locally

We recommend using a virtual environment with uv:

```bash
pip install uv
uv sync
```

> [!NOTE]
> You can use any other package manager to install dependencies (e.g. pip, conda), but we strongly recommend using uv for the best development experience.

### Set up LLM API Credentials

Create a `.env` file with your credentials:

```bash
LITELLM_API_KEY=your_api_key_here
LITELLM_BASE_URL=your_proxy_endpoint
DEFAULT_LLM_MODEL=litellm_proxy/claude-sonnet-4-20250514
```

### Easy Sample Demo

You can test the consultation functionality with a simple example:

```python
from tom_swe.tom_module import TomModule
import asyncio

async def demo():
    tom = TomModule()

    # Consult on user preferences
    consultation = await tom.consult(
        user_id="demo_user",
        current_context="User wants to implement a new feature"
    )
    print(consultation)

asyncio.run(demo())
```

Or run the included example:

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
uv run user-analysis --all-users --sample-size 100

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

The prompts are stored in [tom_swe/prompts/registry.py](tom_swe/prompts/registry.py).
You can also find some prompts in [tom_swe/generation/dataclass.py](tom_swe/generation/dataclass.py)

## Requirements

- Python 3.8+
- uv package manager
- LLM API key (contact All Hands AI for access)

## Citation

If you use ToM-SWE in your research, please cite:

```bibtex
@software{zhou2024tomswe,
  title = {ToM-SWE: Theory of Mind for Software Engineering Agents},
  author = {Zhou, Xuhui and others},
  year = {2024},
  url = {https://github.com/XuhuiZhou/ToM-SWE}
}
```
