"""
Database connection and configuration for LanceDB.
Handles database initialization and connection management.
"""

import os
import logging
from typing import Optional
import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry

logger = logging.getLogger(__name__)

# Global database instance
_db_instance: Optional[lancedb.LanceDBConnection] = None


def get_database() -> lancedb.LanceDBConnection:
    """Get or create database connection."""
    global _db_instance
    
    if _db_instance is None:
        db_path = os.getenv("LANCE_DB_PATH", "../data/lancedb")
        os.makedirs(db_path, exist_ok=True)
        
        try:
            _db_instance = lancedb.connect(db_path)
            logger.info(f"Connected to LanceDB at {db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {str(e)}")
            raise
    
    return _db_instance


def get_embedding_registry() -> EmbeddingFunctionRegistry:
    """Get the embedding function registry."""
    return EmbeddingFunctionRegistry.get_instance()


def close_database():
    """Close database connection."""
    global _db_instance
    if _db_instance is not None:
        _db_instance = None
        logger.info("Database connection closed")
