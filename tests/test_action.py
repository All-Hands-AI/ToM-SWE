"""
Simple tests for the action module.
"""

import pytest
from unittest.mock import Mock
from tom_swe.generation.action import ActionExecutor
from tom_swe.generation.dataclass import SearchFileParams


def test_search_action_basic() -> None:
    """Test basic search action functionality."""
    # Create mock file store
    mock_file_store = Mock()
    mock_file_store.list.return_value = ["test_file.json"]
    mock_file_store.read.return_value = '{"content": "test content"}'

    # Create action executor
    executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

    # Create search parameters
    params = SearchFileParams(
        query="test", search_scope="session_analyses", search_method="string_match"
    )

    # Execute search
    result = executor._string_search(params)

    # Basic assertions
    assert "Found 1 files" in result
    assert "test_file.json" in result


def test_search_action_no_results() -> None:
    """Test search when no results found."""
    # Create mock file store
    mock_file_store = Mock()
    mock_file_store.list.return_value = ["test_file.json"]
    mock_file_store.read.return_value = '{"content": "different content"}'

    # Create action executor
    executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

    # Create search parameters
    params = SearchFileParams(
        query="nonexistent",
        search_scope="session_analyses",
        search_method="string_match",
    )

    # Execute search
    result = executor._string_search(params)

    # Basic assertions
    assert "No files found" in result


def test_bm25_search() -> None:
    """Test BM25 search functionality."""
    # Create mock file store with multiple documents
    mock_file_store = Mock()
    mock_file_store.list.return_value = ["file1.json", "file2.json", "file3.json"]
    mock_file_store.read.side_effect = [
        '{"content": "python programming guide"}',
        '{"content": "machine learning algorithms"}',
        '{"content": "data structures and algorithms"}',
    ]

    # Create action executor
    executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

    # Create search parameters
    params = SearchFileParams(
        query="python programming",
        search_scope="session_analyses",
        search_method="bm25",
        max_results=3,
    )

    # Execute BM25 search
    result = executor._action_search_file(params)

    # Debug: print the actual result to see the format
    print(f"\nBM25 Search Result:\n{result}")

    # Basic assertions
    assert "Found 3 files (BM25 ranked)" in result
    assert "file1.json" in result

    # Verify that file1.json is the first (highest ranked) result
    # The result format is: [Score: X.XX] filename: content
    lines = result.split("\n")
    # Find the first line that contains a filename
    first_file_line = None
    for line in lines:
        if "]" in line and ".json:" in line:
            first_file_line = line
            break

    assert first_file_line is not None, "No file line found in result"
    assert (
        "file1.json" in first_file_line
    ), f"file1.json should be first, but first line is: {first_file_line}"


if __name__ == "__main__":
    pytest.main([__file__])
