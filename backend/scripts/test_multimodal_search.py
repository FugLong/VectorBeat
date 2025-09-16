#!/usr/bin/env python3
"""
Test script to demonstrate multimodal search capabilities.
Shows how text queries can match both text content and image content.
"""

import sys
import os
import asyncio
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.search_service import SearchService
from backend.database.connection import get_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_multimodal_search():
    """Test multimodal search with various queries."""
    try:
        search_service = SearchService(get_database())
        
        # Test queries that should match both text and image content
        test_queries = [
            # Visual queries
            "blue album cover",
            "man with guitar",
            "person on cover",
            "dark moody artwork",
            "colorful album art",
            
            # Text content queries
            "love song",
            "rock music",
            "sad lyrics",
            "upbeat song",
            
            # Mixed queries
            "blue love song",
            "rock album with guitar",
        ]
        
        for query in test_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing query: '{query}'")
            logger.info(f"{'='*60}")
            
            # Test semantic search (multimodal)
            results, total = await search_service.search(query, limit=5, mode='semantic')
            
            logger.info(f"Found {total} semantic results:")
            for i, result in enumerate(results, 1):
                logger.info(f"{i}. {result.track.title} by {result.track.artist}")
                logger.info(f"   Similarity: {result.similarity_score:.3f}")
                logger.info(f"   Match type: {result.match_type}")
                if result.track.album_art_url:
                    logger.info(f"   Album art: {result.track.album_art_url}")
                logger.info()
            
            # Also test combined search
            combined_results, combined_total = await search_service.search(query, limit=3, mode='combined')
            
            logger.info(f"Top 3 combined results:")
            for i, result in enumerate(combined_results, 1):
                logger.info(f"{i}. {result.track.title} by {result.track.artist}")
                logger.info(f"   Similarity: {result.similarity_score:.3f}")
                logger.info(f"   Match type: {result.match_type}")
            
            # Wait for user input to continue
            input("\nPress Enter to continue to next query...")
    
    except Exception as e:
        logger.error(f"Failed to test multimodal search: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(test_multimodal_search())
