#!/usr/bin/env python3
"""
Regenerate embeddings for all existing tracks using the improved multimodal approach.
This script will:
1. Load all existing tracks from the database
2. Regenerate their embeddings using the improved image-focused approach
3. Update the database with the new embeddings
"""

import asyncio
import logging
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.database.connection import get_database
from backend.database.schema import get_tracks_table, dict_to_track
from backend.services.track_service import TrackService
from backend.embeddings.multimodal_embeddings import MultimodalEmbeddingService

logger = logging.getLogger(__name__)

async def regenerate_embeddings():
    """Regenerate embeddings for all tracks using improved approach."""
    try:
        # Initialize services
        db = get_database()
        track_service = TrackService(db)
        multimodal_service = MultimodalEmbeddingService()
        
        # Get all tracks
        table = get_tracks_table(db)
        df = table.to_pandas()
        
        if df.empty:
            print("No tracks found in database.")
            return
        
        print(f"Found {len(df)} tracks. Regenerating embeddings...")
        
        updated_count = 0
        failed_count = 0
        
        for index, row in df.iterrows():
            try:
                track_data = row.to_dict()
                track = dict_to_track(track_data)
                
                print(f"Processing: {track.title} by {track.artist}")
                
                # Combine text content
                text_content = f"{track.title} {track.artist} {track.album} {track.semantic_description}"
                if track.lyrics:
                    text_content += f" {track.lyrics}"
                
                # Generate embedding based on whether we have album art
                if track.album_art_url and track.album_art_url.strip():
                    # For tracks with album art, use image-focused embedding with emotional lyrics
                    # Extract key emotional lyrics for understanding mood/themes
                    key_lyrics = multimodal_service._extract_key_lyrics(track.lyrics or "", max_words=20)
                    
                    # Combine title, artist, and key emotional lyrics
                    balanced_text = f"{track.title} {track.artist}"
                    if key_lyrics:
                        balanced_text += f" {key_lyrics}"
                    
                    multimodal_embedding = multimodal_service.embed_image_focused_content(
                        image_url=track.album_art_url,
                        minimal_text=balanced_text
                    )
                    print(f"  Used image-focused embedding with emotional lyrics (has album art)")
                else:
                    # For tracks without album art, use text-focused embedding
                    multimodal_embedding = multimodal_service.embed_multimodal_content(
                        text=text_content,
                        image_url=None
                    )
                    print(f"  Used text-focused embedding (no album art)")
                
                # Update track with new embedding
                await track_service.update_track_embeddings(
                    track.track_id,
                    multimodal_embedding=multimodal_embedding
                )
                
                updated_count += 1
                print(f"  ✅ Updated successfully")
                
            except Exception as e:
                logger.error(f"Failed to regenerate embedding for {track.title}: {str(e)}")
                failed_count += 1
                print(f"  ❌ Failed: {str(e)}")
        
        print(f"\n🎉 Embedding regeneration complete!")
        print(f"✅ Successfully updated: {updated_count} tracks")
        print(f"❌ Failed: {failed_count} tracks")
        
    except Exception as e:
        logger.error(f"Failed to regenerate embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(regenerate_embeddings())
