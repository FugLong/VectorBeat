#!/usr/bin/env python3
"""
Test script to verify VectorBeat setup.
Tests all components to ensure everything is working correctly.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
from database.connection import get_database
from database.schema import create_tables, get_database_stats
from embeddings.text_embeddings import TextEmbeddingService
from embeddings.image_embeddings import ImageEmbeddingService
from embeddings.audio_embeddings import AudioEmbeddingService
from services.track_service import TrackService
from services.search_service import SearchService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SetupTester:
    """Test VectorBeat setup and components."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
    
    def test_passed(self, test_name: str):
        """Mark a test as passed."""
        self.tests_passed += 1
        logger.info(f"✅ {test_name}")
    
    def test_failed(self, test_name: str, error: str):
        """Mark a test as failed."""
        self.tests_failed += 1
        logger.error(f"❌ {test_name}: {error}")
    
    def test_environment(self):
        """Test environment setup."""
        try:
            # Check if .env file exists
            if not os.path.exists('.env'):
                raise FileNotFoundError(".env file not found")
            
            # Check if data directories exist
            required_dirs = ['data', 'data/lancedb', 'data/cache', 'data/models', 'logs']
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    raise FileNotFoundError(f"Directory {dir_path} not found")
            
            self.test_passed("Environment setup")
            
        except Exception as e:
            self.test_failed("Environment setup", str(e))
    
    def test_database(self):
        """Test database connection and operations."""
        try:
            # Test database connection
            db = get_database()
            if db is None:
                raise Exception("Failed to connect to database")
            
            # Test table creation
            asyncio.run(create_tables(db))
            
            # Test stats retrieval
            stats = asyncio.run(get_database_stats(db))
            if not isinstance(stats, dict):
                raise Exception("Invalid stats format")
            
            self.test_passed("Database operations")
            
        except Exception as e:
            self.test_failed("Database operations", str(e))
    
    def test_text_embeddings(self):
        """Test text embedding service."""
        try:
            service = TextEmbeddingService()
            
            # Test embedding generation
            test_text = "This is a test query for music search"
            embedding = service.embed_text(test_text)
            
            if not isinstance(embedding, list) or len(embedding) == 0:
                raise Exception("Invalid embedding format")
            
            # Test similarity calculation
            similarity = service.similarity(embedding, embedding)
            if not isinstance(similarity, float) or similarity < 0.9:
                raise Exception("Similarity calculation failed")
            
            self.test_passed("Text embeddings")
            
        except Exception as e:
            self.test_failed("Text embeddings", str(e))
    
    def test_image_embeddings(self):
        """Test image embedding service."""
        try:
            service = ImageEmbeddingService()
            
            # Test with placeholder image URL
            test_url = "https://via.placeholder.com/224x224"
            embedding = service.embed_image(test_url)
            
            if not isinstance(embedding, list) or len(embedding) == 0:
                raise Exception("Invalid image embedding format")
            
            self.test_passed("Image embeddings")
            
        except Exception as e:
            self.test_failed("Image embeddings", str(e))
    
    def test_audio_embeddings(self):
        """Test audio embedding service."""
        try:
            service = AudioEmbeddingService()
            
            # Test embedding dimension
            dimension = service.get_embedding_dimension()
            if not isinstance(dimension, int) or dimension <= 0:
                raise Exception("Invalid embedding dimension")
            
            self.test_passed("Audio embeddings")
            
        except Exception as e:
            self.test_failed("Audio embeddings", str(e))
    
    def test_services(self):
        """Test service layer."""
        try:
            db = get_database()
            track_service = TrackService(db)
            search_service = SearchService(db)
            
            # Test service initialization
            if track_service is None or search_service is None:
                raise Exception("Failed to initialize services")
            
            self.test_passed("Service layer")
            
        except Exception as e:
            self.test_failed("Service layer", str(e))
    
    def test_api_imports(self):
        """Test API imports."""
        try:
            # Test if we can import the main API module
            from backend.main import app
            if app is None:
                raise Exception("Failed to import FastAPI app")
            
            self.test_passed("API imports")
            
        except Exception as e:
            self.test_failed("API imports", str(e))
    
    def run_all_tests(self):
        """Run all tests."""
        logger.info("🧪 Running VectorBeat setup tests...")
        logger.info("=" * 50)
        
        self.test_environment()
        self.test_database()
        self.test_text_embeddings()
        self.test_image_embeddings()
        self.test_audio_embeddings()
        self.test_services()
        self.test_api_imports()
        
        logger.info("=" * 50)
        logger.info(f"Tests completed: {self.tests_passed} passed, {self.tests_failed} failed")
        
        if self.tests_failed == 0:
            logger.info("🎉 All tests passed! VectorBeat is ready to use.")
            return True
        else:
            logger.error("❌ Some tests failed. Please check the errors above.")
            return False


def main():
    """Main test function."""
    tester = SetupTester()
    success = tester.run_all_tests()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
