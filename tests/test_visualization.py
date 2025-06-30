"""Tests for visualization functionality."""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from visualization.complexity_visualizer import (
    plot_complexity_over_time,
    plot_complexity_distribution,
    plot_function_vs_class_count,
    save_plots_to_html
)
from visualization.user_behavior_visualizer import (
    analyze_user_edits,
    track_user_behavior,
    visualize_user_behavior,
    generate_behavior_report as generate_user_behavior_report
)


@pytest.fixture
def sample_analyses():
    """Create sample analyses for testing."""
    return [
        {
            "id": 1,
            "timestamp": "2023-01-01T12:00:00",
            "analysis": {
                "lines": 10,
                "functions": 2,
                "classes": 0,
                "complexity": "low"
            }
        },
        {
            "id": 2,
            "timestamp": "2023-01-02T12:00:00",
            "analysis": {
                "lines": 20,
                "functions": 4,
                "classes": 1,
                "complexity": "medium"
            }
        },
        {
            "id": 3,
            "timestamp": "2023-01-03T12:00:00",
            "analysis": {
                "lines": 30,
                "functions": 6,
                "classes": 2,
                "complexity": "high"
            }
        }
    ]


@pytest.fixture
def sample_user_actions():
    """Create sample user actions for testing."""
    return [
        {
            "type": "edit",
            "timestamp": 1000,
            "details": {"file": "test.py", "changes": 5}
        },
        {
            "type": "save",
            "timestamp": 1100,
            "details": {"file": "test.py"}
        },
        {
            "type": "run",
            "timestamp": 1200,
            "details": {"command": "python test.py"}
        },
        {
            "type": "edit",
            "timestamp": 1300,
            "details": {"file": "test.py", "changes": 3}
        },
        {
            "type": "save",
            "timestamp": 1400,
            "details": {"file": "test.py"}
        }
    ]


def test_analyze_user_edits():
    """Test user edit analysis functionality."""
    # Test with empty strings
    result = analyze_user_edits("", "")
    assert result["changes"] == 0
    assert result["lines_added"] == 0
    assert result["lines_removed"] == 0
    assert result["change_percentage"] == 0
    
    # Test with actual changes
    original = "line 1\nline 2\nline 3"
    edited = "line 1\nmodified line 2\nline 3\nnew line 4"
    
    result = analyze_user_edits(original, edited)
    assert result["changes"] > 0
    assert result["lines_added"] == 1
    assert result["lines_removed"] == 0
    assert result["change_percentage"] > 0


def test_track_user_behavior(sample_user_actions):
    """Test user behavior tracking functionality."""
    # Test with empty list
    result = track_user_behavior([])
    assert result["total_actions"] == 0
    assert result["action_types"] == {}
    assert result["average_time_between_actions"] == 0
    assert result["total_session_time"] == 0
    
    # Test with sample actions
    result = track_user_behavior(sample_user_actions)
    assert result["total_actions"] == 5
    assert result["action_types"]["edit"] == 2
    assert result["action_types"]["save"] == 2
    assert result["action_types"]["run"] == 1
    assert result["average_time_between_actions"] == 100  # (1100-1000 + 1200-1100 + 1300-1200 + 1400-1300) / 4
    assert result["total_session_time"] == 400  # 1400 - 1000


@patch('matplotlib.pyplot.show')
def test_plot_complexity_over_time(mock_show, sample_analyses):
    """Test complexity over time plotting functionality."""
    # This just tests that the function runs without errors
    plot_complexity_over_time(sample_analyses)
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_complexity_distribution(mock_show, sample_analyses):
    """Test complexity distribution plotting functionality."""
    # This just tests that the function runs without errors
    plot_complexity_distribution(sample_analyses)
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_function_vs_class_count(mock_show, sample_analyses):
    """Test function vs class count plotting functionality."""
    # This just tests that the function runs without errors
    plot_function_vs_class_count(sample_analyses)
    mock_show.assert_called_once()


def test_save_plots_to_html(sample_analyses):
    """Test saving plots to HTML functionality."""
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    
    try:
        # Test with sample analyses
        save_plots_to_html(sample_analyses, path)
        
        # Check that the file was created and contains expected content
        with open(path, 'r') as f:
            content = f.read()
            assert "<!DOCTYPE html>" in content
            assert "Code Analysis Visualizations" in content
            assert "Code Complexity Over Time" in content
            assert "Distribution of Code Complexity" in content
    finally:
        os.unlink(path)


@patch('matplotlib.pyplot.show')
def test_visualize_user_behavior(mock_show):
    """Test user behavior visualization functionality."""
    # Test with empty data
    visualize_user_behavior({})
    assert not mock_show.called
    
    # Test with actual data
    user_behavior = {
        "total_actions": 5,
        "action_types": {"edit": 2, "save": 2, "run": 1},
        "average_time_between_actions": 100,
        "total_session_time": 400
    }
    
    visualize_user_behavior(user_behavior)
    mock_show.assert_called_once()


def test_generate_user_behavior_report():
    """Test user behavior report generation functionality."""
    # Test with empty data
    report = generate_user_behavior_report({})
    assert "No user behavior data available" in report
    
    # Test with actual data
    user_behavior = {
        "total_actions": 5,
        "action_types": {"edit": 2, "save": 2, "run": 1},
        "average_time_between_actions": 30,
        "total_session_time": 120
    }
    
    report = generate_user_behavior_report(user_behavior)
    assert "User Behavior Analysis Report" in report
    assert "Total actions: 5" in report
    assert "edit: 2" in report
    assert "save: 2" in report
    assert "run: 1" in report