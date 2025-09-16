#!/usr/bin/env python3
"""
Script to regenerate embeddings using the new multimodal CLIP approach.
This will create unified embeddings that enable text-to-image search.
"""

import sys
import os
import asyncio
import logging
from typing import List, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.database.connection import get_database
from backend.database.schema import get_tracks_table, dict_to_track
from backend.embeddings.multimodal_embeddings import MultimodalEmbeddingService
from backend.models.track import Track

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def regenerate_multimodal_embeddings():
    """Regenerate all track embeddings using the multimodal approach."""
    try:
        # Initialize services
        db = get_database()
        embedding_service = MultimodalEmbeddingService()
        
        # Get all tracks
        table = get_tracks_table(db)
        df = table.to_pandas()
        
        if df.empty:
            logger.info("No tracks found in database")
            return
        
        logger.info(f"Found {len(df)} tracks to process")
        
        # Process each track
        updated_tracks = []
        processed_count = 0
        
        for _, row in df.iterrows():
            track_data = row.to_dict()
            track = dict_to_track(track_data)
            
            logger.info(f"Processing track {processed_count + 1}/{len(df)}: {track.title} by {track.artist}")
            
            # Generate multimodal embedding
            try:
                # Create combined text content
                text_parts = [track.title, track.artist]
                if track.album:
                    text_parts.append(track.album)
                if track.genre:
                    text_parts.append(track.genre)
                if track.tags:
                    text_parts.append(track.tags)
                if track.lyrics:
                    text_parts.append(track.lyrics)
                
                combined_text = " ".join(text_parts)
                
                # Generate multimodal embedding
                multimodal_embedding = embedding_service.embed_multimodal_content(
                    text=combined_text,
                    image_url=track.album_art_url
                )
                
                # Update track data
                track_data['multimodal_embedding'] = multimodal_embedding
                updated_tracks.append(track_data)
                
                processed_count += 1
                
                # Log progress every 10 tracks
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count}/{len(df)} tracks")
                
            except Exception as e:
                logger.error(f"Failed to process track {track.title}: {str(e)}")
                # Add track with empty embedding
                track_data['multimodal_embedding'] = [0.0] * embedding_service.get_embedding_dimension()
                updated_tracks.append(track_data)
                processed_count += 1
        
        # Update the database
        logger.info("Updating database with new embeddings...")
        
        # Convert to pandas DataFrame
        import pandas as pd
        updated_df = pd.DataFrame(updated_tracks)
        
        # Update the table
        table.overwrite(updated_df)
        
        logger.info(f"Successfully updated {processed_count} tracks with multimodal embeddings")
        
        # Test the embeddings
        logger.info("Testing multimodal search capabilities...")
        await test_multimodal_search(embedding_service, db)
        
    except Exception as e:
        logger.error(f"Failed to regenerate embeddings: {str(e)}")
        raise


async def test_multimodal_search(embedding_service: MultimodalEmbeddingService, db):
    """Test the multimodal search capabilities."""
    try:
        table = get_tracks_table(db)
        df = table.to_pandas()
        
        # Test queries
        test_queries = [
            "blue album cover",
            "man with guitar",
            "rock music",
            "sad song",
            "love song"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            
            # Generate query embedding
            query_embedding = embedding_service.embed_text(query)
            
            # Find most similar tracks
            similarities = []
            for _, row in df.iterrows():
                if row.get('multimodal_embedding') and len(row['multimodal_embedding']) > 0:
                    similarity = embedding_service.similarity(query_embedding, row['multimodal_embedding'])
                    similarities.append({
                        'title': row['title'],
                        'artist': row['artist'],
                        'album_art_url': row.get('album_art_url', ''),
                        'similarity': similarity
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Show top 3 results
            logger.info(f"Top 3 results for '{query}':")
            for i, result in enumerate(similarities[:3], 1):
                logger.info(f"  {i}. {result['title']} by {result['artist']} (similarity: {result['similarity']:.3f})")
                if result['album_art_url']:
                    logger.info(f"     Album art: {result['album_art_url']}")
    
    except Exception as e:
        logger.error(f"Failed to test multimodal search: {str(e)}")


if __name__ == "__main__":
    asyncio.run(regenerate_multimodal_embeddings())
