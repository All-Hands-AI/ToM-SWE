"""Tests for the database functionality."""

import os
import pytest
import tempfile
import json

from database import Database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    
    db = Database(db_path=path)
    db.connect()
    db.create_tables()
    
    yield db
    
    db.disconnect()
    os.unlink(path)


def test_database_initialization():
    """Test database initialization."""
    db = Database(db_path="test.db")
    assert db.db_path == "test.db"
    assert db.conn is None
    assert db.cursor is None


def test_database_connect_disconnect():
    """Test database connection and disconnection."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    
    try:
        db = Database(db_path=path)
        
        # Test connect
        db.connect()
        assert db.conn is not None
        assert db.cursor is not None
        
        # Test disconnect
        db.disconnect()
        assert db.conn is None
        assert db.cursor is None
    finally:
        os.unlink(path)


def test_create_tables(temp_db):
    """Test table creation."""
    # Tables should already be created by the fixture
    
    # Check if the table exists by trying to insert data
    temp_db.cursor.execute(
        "INSERT INTO code_analyses (code_snippet, analysis) VALUES (?, ?)",
        ("test code", "{}")
    )
    temp_db.conn.commit()
    
    # Verify the data was inserted
    temp_db.cursor.execute("SELECT COUNT(*) FROM code_analyses")
    count = temp_db.cursor.fetchone()[0]
    assert count == 1


def test_save_and_get_analysis(temp_db):
    """Test saving and retrieving analyses."""
    code_snippet = "def test(): pass"
    analysis = {"complexity": "low", "lines": 1}
    
    # Save analysis
    analysis_id = temp_db.save_analysis(code_snippet, analysis)
    assert analysis_id == 1  # First record should have ID 1
    
    # Retrieve analysis
    retrieved = temp_db.get_analysis(analysis_id)
    assert retrieved is not None
    assert retrieved["id"] == analysis_id
    assert retrieved["code_snippet"] == code_snippet
    assert retrieved["analysis"]["complexity"] == "low"
    assert retrieved["analysis"]["lines"] == 1


def test_get_nonexistent_analysis(temp_db):
    """Test retrieving a non-existent analysis."""
    retrieved = temp_db.get_analysis(999)  # ID that doesn't exist
    assert retrieved is None


def test_get_all_analyses(temp_db):
    """Test retrieving all analyses."""
    # Initially, there should be no analyses
    analyses = temp_db.get_all_analyses()
    assert len(analyses) == 0
    
    # Add some analyses
    temp_db.save_analysis("code1", {"complexity": "low"})
    temp_db.save_analysis("code2", {"complexity": "medium"})
    temp_db.save_analysis("code3", {"complexity": "high"})
    
    # Retrieve all analyses
    analyses = temp_db.get_all_analyses()
    assert len(analyses) == 3
    
    # Check that they're in the correct order (by ID)
    assert analyses[0]["code_snippet"] == "code1"
    assert analyses[1]["code_snippet"] == "code2"
    assert analyses[2]["code_snippet"] == "code3"