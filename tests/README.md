# Tests for ToM-SWE (Post-Cleanup)

This directory contains tests for the cleaned-up ToM-SWE system focused on **instruction improvement functionality only**.

## 🧪 Current Test Files

### ✅ Active Test Files

1. **`test_instruction_improvement.py`** - Core functionality tests
   - ToM Agent creation and configuration
   - `analyze_user_context()` method
   - `propose_instructions()` method
   - RAG integration with instruction improvement
   - Database model validation
   - Error handling and edge cases

2. **`test_api_instruction_improvement.py`** - API functionality tests
   - `/propose_instructions` endpoint
   - `/health` endpoint
   - `/conversation_status` endpoint
   - Request/response model validation
   - API documentation and OpenAPI schema
   - Integration tests

3. **`test_process_and_save_steps.py`** - ToM module tests
   - User mental state analysis pipeline
   - Session processing functionality
   - Data loading and validation
   - (Kept because tom_module.py functionality is still used)

### 🗑️ Removed Test Files

- ~~`test_api.py`~~ - **REMOVED** (tested removed next action endpoints)
- ~~`test_tom_agent.py`~~ - **REMOVED** (tested removed tom_agent methods)

### 📋 Supporting Files

- **`conftest.py`** - Test fixtures and mock data (still relevant)
- **`__init__.py`** - Package initialization

## 🚀 Running Tests

### Run All Tests
```bash
# Run all tests in the tests directory
uv run pytest

# Or run specific test files
uv run pytest tests/test_instruction_improvement.py tests/test_api_instruction_improvement.py tests/test_process_and_save_steps.py
```

### Run Specific Test Categories
```bash
# Core functionality only
uv run pytest tests/test_instruction_improvement.py -v

# API tests only
uv run pytest tests/test_api_instruction_improvement.py -v

# ToM module tests only
uv run pytest tests/test_process_and_save_steps.py -v

# Run with coverage
uv run pytest --cov=tom_swe --cov-report=html
```

## 🎯 Test Coverage

The test suite covers:

### Core Functionality
- ✅ User context analysis
- ✅ Instruction improvement generation
- ✅ RAG context retrieval
- ✅ LLM integration
- ✅ Error handling and fallbacks

### API Layer
- ✅ All remaining endpoints
- ✅ Request/response validation
- ✅ Error responses and status codes
- ✅ OpenAPI documentation

### Data Models
- ✅ Pydantic model validation
- ✅ Score constraints (0-1 range)
- ✅ Required field validation

### Cleanup Verification
- ✅ Confirms removed methods/models are gone
- ✅ Verifies remaining functionality works
- ✅ Tests that removed API endpoints return 404

## 📊 What's Tested vs. What's Removed

### ✅ Still Tested (Core Features)
- `ToMAgent.analyze_user_context()`
- `ToMAgent.propose_instructions()`
- `InstructionRecommendation` model
- `/propose_instructions` API endpoint
- RAG integration for context
- User mental state analysis (tom_module)

### ❌ No Longer Tested (Removed Features)
- ~~`ToMAgent.suggest_next_actions()`~~
- ~~`ToMAgent.get_personalized_guidance()`~~
- ~~`NextActionSuggestion` model~~
- ~~`PersonalizedGuidance` model~~
- ~~`/suggest_next_actions` API endpoint~~

## 🔧 Development

When adding new tests:

1. **For core functionality** → Add to `test_instruction_improvement.py`
2. **For API features** → Add to `test_api_instruction_improvement.py`
3. **For tom_module features** → Add to `test_process_and_save_steps.py`

## ✅ Test Status

All tests should pass after the cleanup. If any tests fail:

1. Check that removed functionality isn't being referenced
2. Verify that mocks are properly set up for remaining functionality
3. Ensure test data matches the cleaned data models

The test suite comprehensively verifies that the cleaned system works correctly with only instruction improvement functionality.
