"""Database functionality for ToM-SWE."""

import sqlite3
from typing import Dict, List, Any, Optional


class Database:
    """Database class for ToM-SWE."""

    def __init__(self, db_path: str = "tom_swe.db"):
        """
        Initialize the database.

        Args:
            db_path: Path to the database file.
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def create_tables(self) -> None:
        """Create the necessary tables if they don't exist."""
        if not self.conn:
            self.connect()

        # Create table for storing code snippets and analyses
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code_snippet TEXT NOT NULL,
            analysis TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()

    def save_analysis(self, code_snippet: str, analysis: Dict[str, Any]) -> int:
        """
        Save a code analysis to the database.

        Args:
            code_snippet: The code snippet that was analyzed.
            analysis: The analysis results.

        Returns:
            The ID of the inserted record.
        """
        if not self.conn:
            self.connect()

        import json
        analysis_json = json.dumps(analysis)
        
        self.cursor.execute(
            "INSERT INTO code_analyses (code_snippet, analysis) VALUES (?, ?)",
            (code_snippet, analysis_json)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_analysis(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve an analysis from the database.

        Args:
            analysis_id: The ID of the analysis to retrieve.

        Returns:
            The analysis record as a dictionary, or None if not found.
        """
        if not self.conn:
            self.connect()

        self.cursor.execute(
            "SELECT id, code_snippet, analysis, timestamp FROM code_analyses WHERE id = ?",
            (analysis_id,)
        )
        row = self.cursor.fetchone()
        
        if not row:
            return None
            
        import json
        return {
            "id": row[0],
            "code_snippet": row[1],
            "analysis": json.loads(row[2]),
            "timestamp": row[3]
        }

    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """
        Retrieve all analyses from the database.

        Returns:
            A list of all analysis records as dictionaries.
        """
        if not self.conn:
            self.connect()

        self.cursor.execute(
            "SELECT id, code_snippet, analysis, timestamp FROM code_analyses"
        )
        rows = self.cursor.fetchall()
        
        import json
        return [
            {
                "id": row[0],
                "code_snippet": row[1],
                "analysis": json.loads(row[2]),
                "timestamp": row[3]
            }
            for row in rows
        ]