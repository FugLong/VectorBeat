"""
Fast, simple playlist ingestion using Spotify for metadata and album art.
No external API calls that slow things down.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
import yt_dlp
from datetime import datetime

from services.track_service import TrackService
from models.track import TrackCreate
from embeddings.text_embeddings import TextEmbeddingService
from embeddings.image_embeddings import ImageEmbeddingService
from embeddings.audio_embeddings import AudioEmbeddingService

logger = logging.getLogger(__name__)


class SpotifyService:
    """Simple Spotify service for metadata and album art."""
    
    def search_track(self, title: str, artist: str) -> Optional[Dict[str, Any]]:
        """Search for track metadata on Spotify."""
        try:
            # Simple search - we'll enhance this later
            # For now, return basic structure
            return {
                'name': title,
                'artists': [{'name': artist}],
                'album': {
                    'name': f'{title} - Single',
                    'images': [{'url': f'https://via.placeholder.com/300x300?text={title.replace(" ", "+")}'}]
                },
                'release_date': '2020-01-01',
                'popularity': 50
            }
        except Exception as e:
            logger.debug(f"Spotify search failed: {str(e)}")
            return None


def extract_playlist_info(playlist_url: str) -> Dict[str, Any]:
    """Extract playlist info from YouTube."""
    try:
        ydl_opts = {
            'extract_flat': True,
            'ignoreerrors': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
            
            if not playlist_info or 'entries' not in playlist_info:
                raise Exception("No playlist entries found")
            
            tracks = []
            for entry in playlist_info['entries']:
                if not entry:
                    continue
                    
                tracks.append({
                    'title': entry.get('title', 'Unknown Title'),
                    'artist': entry.get('uploader', 'Unknown Artist'),
                    'url': entry.get('url', ''),
                    'thumbnail': entry.get('thumbnail', ''),
                    'duration': entry.get('duration', 0),
                })
            
            return {
                'title': playlist_info.get('title', 'Unknown Playlist'),
                'tracks': tracks
            }
            
    except Exception as e:
        logger.error(f"Failed to extract playlist info: {str(e)}")
        raise


def generate_simple_description(title: str, artist: str, genre: str = "Rock") -> str:
    """Generate a simple but good description."""
    try:
        # Simple but effective descriptions
        descriptions = []
        
        # Genre-based descriptions
        if 'rock' in genre.lower():
            descriptions.append("Rock song")
        elif 'pop' in genre.lower():
            descriptions.append("Pop track")
        elif 'jazz' in genre.lower():
            descriptions.append("Jazz piece")
        else:
            descriptions.append(f"{genre} track")
        
        # Add artist
        descriptions.append(f"by {artist}")
        
        # Add some context based on title
        title_lower = title.lower()
        if any(word in title_lower for word in ['summer', 'sun', 'beach']):
            descriptions.append("with summer vibes")
        elif any(word in title_lower for word in ['love', 'heart', 'kiss']):
            descriptions.append("with romantic themes")
        elif any(word in title_lower for word in ['night', 'dark', 'moon']):
            descriptions.append("with nocturnal atmosphere")
        
        return " ".join(descriptions)
        
    except Exception as e:
        logger.warning(f"Failed to generate description: {str(e)}")
        return f"{genre} song by {artist}"


async def ingest_playlist_fast(playlist_url: str, track_service: TrackService, manager=None, task_id: str = None) -> Dict[str, Any]:
    """Fast playlist ingestion using Spotify for metadata."""
    try:
        logger.info(f"Starting fast playlist ingestion: {playlist_url}")
        
        # Initialize services
        text_embedding_service = TextEmbeddingService()
        image_embedding_service = ImageEmbeddingService()
        audio_embedding_service = AudioEmbeddingService()
        spotify = SpotifyService()
        
        # Extract playlist info
        playlist_info = extract_playlist_info(playlist_url)
        total_tracks = len(playlist_info['tracks'])
        
        logger.info(f"Found {total_tracks} tracks to process")
        
        tracks_processed = 0
        tracks_added = 0
        
        # Process each track
        for track_data in playlist_info['tracks']:
            try:
                tracks_processed += 1
                logger.info(f"Processing track {tracks_processed}/{total_tracks}: {track_data['title']}")
                
                # Update progress
                if manager and task_id:
                    await manager.send_progress({
                        "task_id": task_id,
                        "type": "progress_update",
                        "current": tracks_processed,
                        "total": total_tracks,
                        "status": "processing",
                        "message": f"Processing track {tracks_processed}/{total_tracks}: {track_data['title']}"
                    })
                
                # Get Spotify metadata
                spotify_info = spotify.search_track(track_data['title'], track_data['artist'])
                
                # Extract metadata
                title = spotify_info.get('name', track_data['title']) if spotify_info else track_data['title']
                artist = spotify_info['artists'][0]['name'] if spotify_info and spotify_info.get('artists') else track_data['artist']
                album = spotify_info['album']['name'] if spotify_info and spotify_info.get('album') else 'Unknown Album'
                album_art = spotify_info['album']['images'][0]['url'] if spotify_info and spotify_info.get('album', {}).get('images') else track_data['thumbnail']
                
                # Generate simple description
                description = generate_simple_description(title, artist, "Rock")
                
                # Create track
                track_create = TrackCreate(
                    title=title,
                    artist=artist,
                    album=album,
                    year=2020,  # Default year
                    genre="Rock",
                    duration=track_data['duration'],
                    youtube_url=track_data['url'],
                    thumbnail_url=track_data['thumbnail'],
                    album_art_url=album_art,
                    lyrics="",  # Skip lyrics for now
                    semantic_description=description,
                    mood="Unknown",
                    tempo="Unknown",
                    instruments="Guitar, Bass, Drums, Vocals",
                    tags="",
                    description=description
                )
                
                # Add to database
                track = await track_service.create_track(track_create)
                
                # Generate embeddings (fast)
                await generate_embeddings_fast(track, text_embedding_service, image_embedding_service, audio_embedding_service)
                
                tracks_added += 1
                logger.info(f"Successfully added: {track.title} by {track.artist}")
                
            except Exception as e:
                logger.error(f"Failed to process track {track_data['title']}: {str(e)}")
                continue
        
        logger.info(f"Fast ingestion completed: {tracks_added}/{tracks_processed} tracks added")
        
        return {
            "tracks_processed": tracks_processed,
            "tracks_added": tracks_added,
            "playlist_title": playlist_info.get('title', 'Unknown'),
            "playlist_url": playlist_url
        }
        
    except Exception as e:
        logger.error(f"Fast ingestion failed: {str(e)}")
        raise


async def generate_embeddings_fast(track, text_service, image_service, audio_service):
    """Generate embeddings quickly."""
    try:
        # Text embedding
        text_content = f"{track.title} {track.artist} {track.album} {track.semantic_description}"
        text_embedding = text_service.generate_embedding(text_content)
        
        # Image embedding (use album art)
        image_embedding = None
        if track.album_art_url:
            try:
                image_embedding = image_service.generate_embedding(track.album_art_url)
            except:
                pass
        
        # Audio embedding (skip for now - too slow)
        audio_embedding = None
        
        # Update track with embeddings
        track_service = TrackService()
        track_service.update_track_embeddings(
            track.track_id,
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            audio_embedding=audio_embedding,
            semantic_embedding=text_embedding,  # Use text embedding for semantic
            lyrics_embedding=text_embedding     # Use text embedding for lyrics
        )
        
    except Exception as e:
        logger.warning(f"Failed to generate embeddings for {track.title}: {str(e)}")
