"""Simple database module for ToM-SWE."""

import sqlite3
import json
from typing import Dict, Any, List, Optional
from pathlib import Path


class Database:
    """Simple SQLite database wrapper."""
    
    def __init__(self, db_path: str = "tom_swe.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
    
    def connect(self) -> None:
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.cursor:
            self.cursor.close()
            self.cursor = None
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def create_tables(self) -> None:
        """Create necessary tables."""
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        # Create code_analyses table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_snippet TEXT NOT NULL,
                analysis TEXT NOT NULL
            )
        """)
        
        self.conn.commit()
    
    def save_analysis(self, code_snippet: str, analysis: Dict[str, Any]) -> int:
        """Save analysis data to database.
        
        Args:
            code_snippet: The code snippet being analyzed.
            analysis: Analysis data to save.
            
        Returns:
            The ID of the saved analysis.
        """
        if not self.conn:
            raise RuntimeError("Database not connected")
        
        self.cursor.execute(
            "INSERT INTO code_analyses (code_snippet, analysis) VALUES (?, ?)",
            (code_snippet, json.dumps(analysis))
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_analysis(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Get analysis data from database.
        
        Args:
            analysis_id: Analysis identifier.
            
        Returns:
            Analysis data dictionary or None if not found.
        """
        if not self.conn:
            return None
        
        self.cursor.execute(
            "SELECT id, code_snippet, analysis FROM code_analyses WHERE id = ?", 
            (analysis_id,)
        )
        row = self.cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "code_snippet": row["code_snippet"],
                "analysis": json.loads(row["analysis"])
            }
        return None
    
    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """Get all analysis data from database.
        
        Returns:
            List of analysis data dictionaries.
        """
        if not self.conn:
            return []
        
        self.cursor.execute(
            "SELECT id, code_snippet, analysis FROM code_analyses ORDER BY id"
        )
        rows = self.cursor.fetchall()
        return [
            {
                "id": row["id"],
                "code_snippet": row["code_snippet"],
                "analysis": json.loads(row["analysis"])
            }
            for row in rows
        ]