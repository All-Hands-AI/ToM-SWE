# Tom Module Tests

This directory contains comprehensive tests for the tom_module's 3-tier user analysis system.

## Test Coverage

### **TestDataLoading**
- Tests session discovery (`get_user_session_ids`)
- Handles missing users, empty files, and valid users

### **TestStep1AnalyzeUserMentalState**
- Tests **Step 1** of `process_and_save_user_session`
- Covers LLM analysis and error handling
- Uses mocked LLM responses for consistency

### **TestStep2SaveSessionAnalysesToJsonl**
- Tests **Step 2** of `process_and_save_user_session`
- Validates JSONL file format and content
- Tests directory creation and file writing

### **TestStep3SummarizeSessionFromAnalyses**
- Tests **Step 3** of `process_and_save_user_session`
- Validates session summary aggregation logic
- Tests intent counting and emotion progression

### **TestStep4UpdateOverallUserAnalysis**
- Tests **Step 4** of `process_and_save_user_session`
- Validates user profile creation and updates
- Tests new users vs existing users

### **TestIntegrationProcessAndSaveUserSession**
- Tests the complete `process_and_save_user_session` pipeline
- End-to-end validation of all 3 tiers
- Tests `process_all_user_sessions` for multiple sessions

### **TestUtilityFunctions**
- Tests helper functions and directory creation
- Profile update logic and data structures

## Running Tests

### **Run All Tests**
```bash
# From project root
python tests/run_tests.py

# Or with pytest directly
pytest tests/ -v
```

### **Run Specific Test File**
```bash
# Using test runner
python tests/run_tests.py test_process_and_save_steps.py

# Or with pytest
pytest tests/test_process_and_save_steps.py -v
```

### **Run Specific Test Class**
```bash
pytest tests/test_process_and_save_steps.py::TestIntegrationProcessAndSaveUserSession -v
```

### **Run Specific Test Method**
```bash
pytest tests/test_process_and_save_steps.py::TestIntegrationProcessAndSaveUserSession::test_process_and_save_user_session_success -v
```

## Test Data

Tests use temporary directories and mock data:
- **`conftest.py`**: Contains fixtures for test data and mock objects
- **Mock LLM responses**: Prevents real API calls during testing
- **Temporary directories**: Isolated file operations
- **Sample user data**: Realistic session structures

## Key Testing Features

- ✅ **No external dependencies**: Tests use mocked LLM calls
- ✅ **Isolated file operations**: Each test uses temp directories
- ✅ **Comprehensive step coverage**: Tests each step in the pipeline
- ✅ **Error handling**: Tests missing files, empty data, etc.
- ✅ **Data validation**: Verifies JSONL format, JSON structure
- ✅ **Integration testing**: End-to-end pipeline validation

## Expected Output Structure

When tests run successfully, they validate the creation of:

```
temp_test_dir/user_model/
├── user_model_detailed/           # Tier 1: Per-message analysis
│   └── test_user_123/
│       ├── session_001.jsonl      # JSONL with message analyses
│       └── session_002.jsonl
└── user_model_overall/            # Tier 2 & 3: Session summaries + user profiles
    └── test_user_123.json         # Complete user analysis
```
