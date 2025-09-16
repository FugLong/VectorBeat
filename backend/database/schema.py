"""
Database schema definitions and table creation for LanceDB.
Defines the structure of the tracks table and related operations.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import lancedb

from backend.models.track import Track, TrackCreate

logger = logging.getLogger(__name__)

# Table schema for tracks
TRACKS_TABLE_SCHEMA = {
    "track_id": "string",
    "title": "string", 
    "artist": "string",
    "album": "string",
    "year": "int32",
    "genre": "string",
    "tags": "string",
    "youtube_url": "string",
    "album_art_url": "string",
    "lyrics": "string",
    "semantic_description": "string",
    "mood": "string",
    "tempo": "string",
    "instruments": "string",
    "multimodal_embedding": "float32[512]",  # CLIP unified text+image embedding
    "audio_embedding": "float32[1024]",  # Whisper + custom audio embedding
    "created_at": "timestamp",
    "updated_at": "timestamp"
}

TABLE_NAME = "tracks"


async def create_tables(db: lancedb.LanceDBConnection) -> None:
    """Create database tables if they don't exist."""
    try:
        # Check if tracks table exists
        if TABLE_NAME not in db.table_names():
            # Create table with a dummy row (LanceDB requires at least one row)
            dummy_data = [{
                "track_id": "dummy",
                "title": "dummy",
                "artist": "dummy",
                "album": "dummy",
                "year": 2000,
                "genre": "dummy",
                "tags": "dummy",
                "youtube_url": "dummy",
                "album_art_url": "dummy",
                "lyrics": "dummy",
                "semantic_description": "dummy",
                "mood": "dummy",
                "tempo": "dummy",
                "instruments": "dummy",
                "multimodal_embedding": [0.0] * 512,
                "audio_embedding": [0.0] * 1024,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }]
            table = db.create_table(TABLE_NAME, dummy_data)
            # Remove the dummy row
            table.delete("track_id = 'dummy'")
            logger.info(f"Created table '{TABLE_NAME}' with schema")
        else:
            logger.info(f"Table '{TABLE_NAME}' already exists")
            
    except Exception as e:
        logger.error(f"Failed to create tables: {str(e)}")
        raise


def get_tracks_table(db: lancedb.LanceDBConnection):
    """Get the tracks table."""
    try:
        return db.open_table(TABLE_NAME)
    except Exception as e:
        logger.error(f"Failed to open tracks table: {str(e)}")
        raise


def track_to_dict(track: Track) -> Dict[str, Any]:
    """Convert Track model to dictionary for database storage."""
    return {
        "track_id": track.track_id,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "year": track.year,
        "genre": track.genre,
        "tags": track.tags,
        "youtube_url": track.youtube_url,
        "album_art_url": track.album_art_url,
        "lyrics": track.lyrics,
        "semantic_description": track.semantic_description,
        "mood": track.mood,
        "tempo": track.tempo,
        "instruments": track.instruments,
        "multimodal_embedding": track.multimodal_embedding,
        "audio_embedding": track.audio_embedding,
        "created_at": track.created_at,
        "updated_at": track.updated_at
    }


def dict_to_track(data: Dict[str, Any]) -> Track:
    """Convert dictionary from database to Track model."""
    return Track(
        track_id=data["track_id"],
        title=data["title"],
        artist=data["artist"],
        album=data.get("album"),
        year=data.get("year"),
        genre=data.get("genre"),
        tags=data.get("tags"),
        youtube_url=data.get("youtube_url"),
        album_art_url=data.get("album_art_url"),
        lyrics=data.get("lyrics"),
        semantic_description=data.get("semantic_description"),
        mood=data.get("mood"),
        tempo=data.get("tempo"),
        instruments=data.get("instruments"),
        multimodal_embedding=data.get("multimodal_embedding"),
        audio_embedding=data.get("audio_embedding"),
        created_at=data["created_at"],
        updated_at=data["updated_at"]
    )


async def reset_database(db: lancedb.LanceDBConnection) -> None:
    """Reset the database by dropping and recreating tables."""
    try:
        # Drop existing table if it exists
        if TABLE_NAME in db.table_names():
            db.drop_table(TABLE_NAME)
            logger.info(f"Dropped table '{TABLE_NAME}'")
        
        # Recreate tables
        await create_tables(db)
        logger.info("Database reset completed")
        
    except Exception as e:
        logger.error(f"Failed to reset database: {str(e)}")
        raise


async def get_database_stats(db: lancedb.LanceDBConnection) -> Dict[str, Any]:
    """Get database statistics."""
    try:
        if TABLE_NAME not in db.table_names():
            return {
                "total_tracks": 0,
                "total_artists": 0,
                "total_genres": 0,
                "year_range": None,
                "last_updated": None,
                "database_size_mb": 0.0
            }
        
        table = get_tracks_table(db)
        
        # Get basic counts
        total_tracks = len(table.to_pandas())
        
        # Get unique artists and genres
        df = table.to_pandas()
        total_artists = df["artist"].nunique() if not df.empty else 0
        total_genres = df["genre"].nunique() if not df.empty and "genre" in df.columns else 0
        
        # Get year range
        year_range = None
        if not df.empty and "year" in df.columns and df["year"].notna().any():
            year_range = {
                "min": int(df["year"].min()),
                "max": int(df["year"].max())
            }
        
        # Get last updated timestamp
        last_updated = None
        if not df.empty and "updated_at" in df.columns:
            last_updated = df["updated_at"].max()
        
        # Estimate database size (rough calculation)
        database_size_mb = 0.0
        try:
            import os
            # Use the same path logic as the database connection
            db_path = os.getenv("LANCE_DB_PATH", "../data/lancedb")
            # If running from project root, adjust the path
            if os.path.exists("./data/lancedb") and not os.path.exists(db_path):
                db_path = "./data/lancedb"
            
            if os.path.exists(db_path):
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(db_path)
                    for filename in filenames
                )
                database_size_mb = total_size / (1024 * 1024)
                logger.info(f"Database size calculation: {db_path} -> {database_size_mb:.2f} MB")
            else:
                logger.warning(f"Database path does not exist: {db_path}")
        except Exception as e:
            logger.error(f"Failed to calculate database size: {str(e)}")
            pass
        
        return {
            "total_tracks": total_tracks,
            "total_artists": total_artists,
            "total_genres": total_genres,
            "year_range": year_range,
            "last_updated": last_updated,
            "database_size_mb": round(database_size_mb, 2)
        }
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {str(e)}")
        raise
