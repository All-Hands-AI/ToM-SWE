# ToM-SWE Tests

This directory contains tests for the ToM-SWE package.

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_core.py
```

To run tests with coverage:

```bash
pytest --cov=tom_swe
```

## Test Structure

- `test_core.py`: Tests for the core functionality
- `test_tom_module.py`: Tests for the Theory of Mind module
- `test_database.py`: Tests for the database functionality
- `test_utils.py`: Tests for utility functions
- `test_visualization.py`: Tests for visualization tools

## Adding Tests

When adding new functionality, please add corresponding tests. Follow these guidelines:

1. Create test functions with descriptive names
2. Use appropriate assertions to verify functionality
3. Include tests for edge cases and error conditions
4. Use fixtures for common setup and teardown
5. Keep tests independent of each other