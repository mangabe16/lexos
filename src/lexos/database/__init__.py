"""Database module for Lexos.

This module provides SQLite database integration capabilities for corpus storage,
full-text search, and efficient querying.
"""

from wasabi import msg

try:
    # Import database components
    from lexos.database.database_simple import CorpusDatabase, DatabaseRecord, DatabaseCorpus
    from lexos.database.corpus_db_integration import DatabaseEnabledCorpus, create_database_corpus
    
    database_available = True
    msg.info("Database integration available")
    
    __all__ = [
        "CorpusDatabase", 
        "DatabaseRecord", 
        "DatabaseCorpus",
        "DatabaseEnabledCorpus", 
        "create_database_corpus"
    ]
    
except Exception as e:
    msg.warn(f"Database integration not available: {e}")
    database_available = False
    __all__ = []