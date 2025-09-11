#!/usr/bin/env python3
"""
Download and setup embedding models for VectorBeat.
This script downloads the required models for text, image, and audio embeddings.
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from embeddings.text_embeddings import TextEmbeddingService
from embeddings.image_embeddings import ImageEmbeddingService
from embeddings.audio_embeddings import AudioEmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_models():
    """Download all required embedding models."""
    try:
        logger.info("Downloading text embedding model (sentence-transformers)...")
        text_service = TextEmbeddingService()
        logger.info("✅ Text embedding model ready")
        
        logger.info("Downloading image embedding model (CLIP)...")
        image_service = ImageEmbeddingService()
        logger.info("✅ Image embedding model ready")
        
        logger.info("Downloading audio embedding model (Whisper)...")
        audio_service = AudioEmbeddingService()
        logger.info("✅ Audio embedding model ready")
        
        logger.info("🎉 All models downloaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to download models: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    download_models()
