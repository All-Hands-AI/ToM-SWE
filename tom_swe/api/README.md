# ToM Agent REST API

This directory contains the RESTful API implementation for the Theory of Mind (ToM) Agent. The API provides endpoints for personalized instruction improvement and next action suggestions for SWE agents.

## Features

The API v2.0.0 provides these main endpoints:

1. **`POST /propose_instructions`** - Get improved, personalized instructions
2. **`POST /suggest_next_actions`** - Get personalized next action suggestions
3. **`GET /health`** - Health check endpoint
4. **`GET /conversation_status`** - Get conversation status for a user

## Quick Start

### Installation

The API dependencies are included in the main project dependencies. Make sure you have FastAPI and uvicorn installed:

```bash
# Install the project with API dependencies
uv sync --extra dev
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
- `TOM_PROCESSED_DATA_DIR`: Directory containing processed user data (default: ./data/processed_data)
- `TOM_USER_MODEL_DIR`: Directory containing user model data (default: ./data/user_model)
- `TOM_ENABLE_RAG`: Enable RAG functionality (default: true)

## API Endpoints

### Health Check

**GET** `/health`

Check the health status of the API server and ToM agent.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "tom_agent_ready": true,
  "active_conversations": 3
}
```

### Propose Instructions

**POST** `/propose_instructions`

Get improved, personalized instructions based on user context and preferences.

**Request:**
```json
{
  "user_id": "user_123",
  "original_instruction": "Debug the function that's causing errors",
  "context": "User message: I need help debugging my Python application. Previous conversation includes discussion about web development and systematic debugging approaches."
}
```

**Response:**
```json
{
  "user_id": "user_123",
  "original_instruction": "Debug the function that's causing errors",
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

### Suggest Next Actions

**POST** `/suggest_next_actions`

Get personalized next action suggestions based on user context and current situation.

**Request:**
```json
{
  "user_id": "user_123",
  "context": "Agent response: I've identified the error in the login function. The issue is with the password validation logic. Previous interactions show user prefers step-by-step debugging."
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
  "message": "Generated 1 next action suggestions"
}
```

### Get Conversation Status

**GET** `/conversation_status?user_id=user_123`

Get the current conversation status for a user.

**Response:**
```json
{
  "user_id": "user_123",
  "has_pending_instructions": true,
  "has_pending_next_actions": false,
  "last_activity": "2023-10-20T14:30:00Z",
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

### Debug Endpoints

Additional endpoints for debugging and monitoring:

- **GET** `/active_conversations` - Get information about all active conversations
- **DELETE** `/conversation/{user_id}` - Clear conversation state for a specific user

## Usage Flow

The API supports two main use cases:

### Instruction Improvement Flow
1. **SWE Agent** calls `POST /propose_instructions` with user message and original instruction
2. **ToM Agent** analyzes user context and generates improved instructions
3. **SWE Agent** receives personalized instruction recommendations

### Next Action Suggestions Flow
1. **SWE Agent** calls `POST /suggest_next_actions` with current conversation context
2. **ToM Agent** analyzes user behavior patterns and generates next action suggestions
3. **SWE Agent** receives personalized action recommendations

## Architecture

The API is built using:
- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server for running the application
- **AsyncIO**: Asynchronous programming for better performance

### Key Components

- **Direct Processing**: Each endpoint handles its specific functionality
- **User Context Analysis**: Analyzes user behavior patterns and preferences
- **Async Processing**: Non-blocking ToM agent operations
- **Error Handling**: Comprehensive error handling and logging

### Integration

The API integrates with the existing ToM agent infrastructure:
- `ToMAgent`: Main agent class for personalized guidance
- `RAGAgent`: Retrieval-augmented generation for user behavior patterns
- `ToMAnalyzer`: User mental state analysis

## Files

- `main.py`: Main FastAPI application with all endpoints
- `models.py`: Pydantic models for API requests and responses
- `server.py`: CLI script for running the API server
- `README.md`: This documentation file
