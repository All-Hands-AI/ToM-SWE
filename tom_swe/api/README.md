# ToM Agent REST API

This directory contains the RESTful API implementation for the Theory of Mind (ToM) Agent. The API provides endpoints for interacting with the ToM agent's capabilities, including instruction improvement, next action suggestions, and comprehensive personalized guidance.

## Features

The API provides three main endpoints as requested in issue #17:

1. **`/api/v1/suggest_next_actions`** - API for using the `suggest_next_actions` functionality
2. **`/api/v1/propose_instructions`** - API for using the `propose_instructions` functionality  
3. **`/api/v1/send_message`** - API for sending messages to the ToM agent and getting comprehensive guidance

## Quick Start

### Installation

The API dependencies are included in the main project dependencies. Make sure you have FastAPI and uvicorn installed:

```bash
# Install the project with API dependencies
pip install -e .
```

### Running the Server

You can start the API server using the provided CLI script:

```bash
# Using the tom-api command
tom-api --host 0.0.0.0 --port 8000

# Or directly with Python
python -m tom_swe.api.server --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn tom_swe.api.main:app --host 0.0.0.0 --port 8000
```

### Configuration

The API server can be configured using command-line arguments or environment variables:

- `--host` / `API_HOST`: Host to bind the server to (default: 0.0.0.0)
- `--port` / `API_PORT`: Port to bind the server to (default: 8000)
- `--reload` / `API_RELOAD`: Enable auto-reload for development (default: false)
- `--processed-data-dir` / `TOM_PROCESSED_DATA_DIR`: Directory containing processed user data
- `--user-model-dir` / `TOM_USER_MODEL_DIR`: Directory containing user model data

## API Endpoints

### Health Check

**GET** `/health`

Check the health status of the API server and ToM agent.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "tom_agent_ready": true
}
```

### Suggest Next Actions

**POST** `/api/v1/suggest_next_actions`

Get personalized next action suggestions for a user based on their context and current task.

**Request:**
```json
{
  "user_id": "user_123",
  "current_task_context": "Debugging Python application"
}
```

**Response:**
```json
{
  "user_id": "user_123",
  "suggestions": [
    {
      "action_description": "Add logging statements to identify the error location",
      "priority": "high",
      "reasoning": "Based on user's systematic debugging approach",
      "expected_outcome": "Clear identification of where the error occurs",
      "user_preference_alignment": 0.9
    }
  ],
  "success": true,
  "message": "Generated 2 action suggestions"
}
```

### Propose Instructions

**POST** `/api/v1/propose_instructions`

Get improved, personalized instructions based on user context and preferences.

**Request:**
```json
{
  "user_id": "user_123",
  "original_instruction": "Debug the function that's causing errors",
  "domain_context": "Python web development"
}
```

**Response:**
```json
{
  "user_id": "user_123",
  "recommendations": [
    {
      "original_instruction": "Debug the function that's causing errors",
      "improved_instruction": "Debug the function by systematically adding logging statements and checking each variable step-by-step",
      "reasoning": "Personalized for user's preference for detailed, systematic approaches",
      "confidence_score": 0.85,
      "personalization_factors": ["detailed_explanations", "systematic_approach"]
    }
  ],
  "success": true,
  "message": "Generated 1 instruction recommendations"
}
```

### Send Message

**POST** `/api/v1/send_message`

Send a message to the ToM agent and get comprehensive personalized guidance.

**Request:**
```json
{
  "user_id": "user_123",
  "message": "I need help debugging my Python application",
  "instruction": "Debug the function that's causing errors",
  "current_task": "Debugging Python web application",
  "domain_context": "Web development"
}
```

**Response:**
```json
{
  "user_id": "user_123",
  "guidance": {
    "user_context": {
      "user_id": "user_123",
      "user_profile": { ... },
      "recent_sessions": [ ... ],
      "preferences": ["detailed_explanations", "step_by_step_help"],
      "mental_state_summary": "A focused developer who prefers detailed guidance"
    },
    "instruction_recommendations": [ ... ],
    "next_action_suggestions": [ ... ],
    "overall_guidance": "Based on your systematic approach, I've provided detailed debugging steps and next actions.",
    "confidence_score": 0.85
  },
  "success": true,
  "message": "Generated personalized guidance successfully"
}
```

### Get User Context

**GET** `/api/v1/users/{user_id}/context`

Get the current context for a specific user.

**Response:**
```json
{
  "user_id": "user_123",
  "context": {
    "user_id": "user_123",
    "user_profile": { ... },
    "recent_sessions": [ ... ],
    "preferences": [ ... ],
    "mental_state_summary": "..."
  },
  "success": true
}
```

## Error Handling

The API provides consistent error responses:

```json
{
  "success": false,
  "error": "Error message",
  "detail": "Detailed error information (in debug mode)"
}
```

Common HTTP status codes:
- `200`: Success
- `422`: Validation error (invalid request data)
- `500`: Internal server error
- `503`: Service unavailable (ToM agent not ready)

## Development

### Running Tests

```bash
# Run API tests specifically
pytest tests/test_api.py -v

# Run all tests
pytest tests/ -v
```

### Development Mode

For development, you can run the server with auto-reload:

```bash
tom-api --reload --log-level debug
```

### CORS Configuration

The API includes CORS middleware configured to allow all origins for development. For production, update the `allow_origins` setting in `main.py`.

## Architecture

The API is built using:
- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server for running the application
- **AsyncIO**: Asynchronous programming for better performance

The API integrates with the existing ToM agent infrastructure:
- `ToMAgent`: Main agent class for personalized guidance
- `RAGAgent`: Retrieval-augmented generation for user behavior patterns
- `UserMentalStateAnalyzer`: User mental state analysis

## Files

- `main.py`: Main FastAPI application with all endpoints
- `models.py`: Pydantic models for API requests and responses
- `server.py`: CLI script for running the API server
- `README.md`: This documentation file