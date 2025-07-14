#!/usr/bin/env python3
import os
import shutil

# Remove old vector database to force regeneration with new format
db_path = "data/rag_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"Removed old database at {db_path}")
else:
    print(f"Database at {db_path} does not exist")
