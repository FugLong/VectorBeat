"""
Spotify-based playlist ingestion with YouTube audio.
Gets perfect metadata from Spotify, audio from YouTube.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
import yt_dlp
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime

from backend.services.track_service import TrackService
from backend.services.audio_analysis_service import AudioAnalysisService
from backend.services.lyrics_service import LyricsService
from backend.models.track import TrackCreate
from backend.embeddings.multimodal_embeddings import MultimodalEmbeddingService
from backend.progress import update_progress

logger = logging.getLogger(__name__)


class SpotifyPlaylistService:
    """Service for extracting data from Spotify playlists."""
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        # Use provided credentials or fall back to environment variables
        if client_id and client_secret:
            import os
            os.environ['SPOTIPY_CLIENT_ID'] = client_id
            os.environ['SPOTIPY_CLIENT_SECRET'] = client_secret
        
        # Use Spotify API with credentials
        self.sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
        
        # Cache for artist genres to avoid duplicate API calls
        self.artist_genre_cache = {}
        
        # Test API connection
        try:
            test_search = self.sp.search('test', limit=1)
            logger.info("Spotify API connection successful")
        except Exception as e:
            logger.error(f"Spotify API connection failed: {str(e)}")
            raise
    
    async def extract_playlist_info(self, playlist_url: str, manager=None, task_id: str = None, lyrics_service=None) -> Dict[str, Any]:
        """Extract playlist and track info from Spotify."""
        try:
            # Extract playlist ID from URL
            playlist_id = self._extract_playlist_id(playlist_url)
            if not playlist_id:
                raise Exception("Invalid Spotify playlist URL")
            
            # Try to get playlist info first
            logger.info(f"Fetching playlist info for ID: {playlist_id}")
            try:
                playlist = self.sp.playlist(playlist_id)
                logger.info(f"Successfully fetched playlist: {playlist['name']}")
                
                # Update progress - playlist info fetched
                update_progress(
                    message=f"Fetched playlist '{playlist['name']}', loading tracks...",
                    currentTrack=""
                )
                
                # First, collect all raw track data
                raw_tracks = []
                results = self.sp.playlist_tracks(playlist_id)
                page_count = 0
                tracks_fetched = 0
                
                # Collect all tracks first
                while results:
                    page_count += 1
                    page_tracks = len(results['items'])
                    
                    # Update progress for this page
                    update_progress(
                        message=f"Fetching tracks from Spotify (page {page_count})...",
                        currentTrack=f"Loading {page_tracks} tracks from page {page_count}"
                    )
                    
                    for item in results['items']:
                        if item['track'] and item['track']['id']:  # Skip None tracks
                            raw_tracks.append(item['track'])
                            tracks_fetched += 1
                            
                            # Update progress every 10 tracks
                            if tracks_fetched % 10 == 0:
                                update_progress(
                                    message=f"Fetched {tracks_fetched} tracks from Spotify...",
                                    currentTrack=f"Latest: {item['track']['name']} by {', '.join([artist['name'] for artist in item['track']['artists']])}"
                                )
                    
                    # Get next page
                    if results['next']:
                        update_progress(
                            message=f"Fetched {tracks_fetched} tracks, loading next page...",
                            currentTrack=""
                        )
                    results = self.sp.next(results) if results['next'] else None
                
                # Pre-fetch artist genres for all unique artists
                update_progress(
                    message=f"Pre-fetching artist genres for {tracks_fetched} tracks...",
                    currentTrack=""
                )
                await self._prefetch_artist_genres(raw_tracks)
                
                # Pre-fetch lyrics for all tracks
                update_progress(
                    message=f"Fetching lyrics for {tracks_fetched} tracks...",
                    currentTrack=""
                )
                logger.info(f"Starting lyrics fetching for {tracks_fetched} tracks...")
                try:
                    # Create cancellation check function
                    def is_cancelled():
                        return manager and manager.is_cancelled(task_id or "current") if manager else False
                    
                    lyrics_results = lyrics_service.batch_get_lyrics(
                        raw_tracks, 
                        progress_callback=lambda current, total, title: update_progress(
                            message=f"Fetching lyrics... ({current}/{total})",
                            currentTrack=f"Latest: {title}"
                        ),
                        cancellation_check=is_cancelled
                    )
                    logger.info(f"Lyrics fetching completed. Results: {len(lyrics_results)} tracks with lyrics")
                except Exception as e:
                    logger.error(f"Lyrics fetching failed: {str(e)}")
                    lyrics_results = {}
                
                # Check for cancellation after lyrics fetching
                if manager and manager.is_cancelled(task_id or "current"):
                    logger.info("Ingestion cancelled by user after lyrics fetching")
                    update_progress(
                        isActive=False,
                        current=0,
                        total=len(raw_tracks),
                        message="Ingestion cancelled by user",
                        currentTrack=""
                    )
                    return {
                        "message": "Ingestion cancelled",
                        "playlist_url": playlist_url,
                        "tracks_processed": 0,
                        "tracks_added": 0
                    }
                
                # Now process tracks with cached genre data and lyrics
                tracks = []
                logger.info(f"Processing {len(raw_tracks)} tracks with lyrics_results: {len(lyrics_results)} entries")
                for track in raw_tracks:
                    track_lyrics = lyrics_results.get(track['id'], '')
                    if track_lyrics:
                        logger.info(f"Track {track['name']} has lyrics ({len(track_lyrics)} chars)")
                    else:
                        logger.debug(f"Track {track['name']} has no lyrics (ID: {track['id']})")
                    
                    track_info = {
                        'spotify_id': track['id'],
                        'title': track['name'],
                        'artist': ', '.join([artist['name'] for artist in track['artists']]),
                        'album': track['album']['name'] if track['album'] else 'Unknown Album',
                        'year': self._extract_year(track['album']['release_date']) if track['album'] else 2024,
                        'genre': self._extract_genre(track),
                        'album_art': self._get_album_art(track['album']) if track['album'] else None,
                        'duration_ms': track['duration_ms'],
                        'popularity': track['popularity'],
                        'explicit': track['explicit'],
                        'lyrics': track_lyrics  # Add fetched lyrics
                    }
                    tracks.append(track_info)
                
                return {
                    'title': playlist['name'],
                    'description': playlist['description'],
                    'tracks': tracks
                }
                
            except Exception as e:
                logger.error(f"Failed to access playlist: {str(e)}")
                raise Exception(f"Could not access this playlist. Please use a public, user-created playlist (not Spotify's official playlists like 'Top 50'). Error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Failed to extract Spotify playlist: {str(e)}")
            raise
    
    def _extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from Spotify URL."""
        try:
            logger.info(f"Extracting playlist ID from URL: {url}")
            
            # Handle different Spotify URL formats
            if 'spotify.com/playlist/' in url:
                playlist_id = url.split('spotify.com/playlist/')[1].split('?')[0].split('/')[0]
                logger.info(f"Extracted playlist ID: {playlist_id}")
                return playlist_id
            elif 'open.spotify.com/playlist/' in url:
                playlist_id = url.split('open.spotify.com/playlist/')[1].split('?')[0].split('/')[0]
                logger.info(f"Extracted playlist ID: {playlist_id}")
                return playlist_id
            
            logger.warning(f"Could not extract playlist ID from URL: {url}")
            return None
        except Exception as e:
            logger.error(f"Error extracting playlist ID: {str(e)}")
            return None
    
    
    def _extract_year(self, release_date: str) -> int:
        """Extract year from release date."""
        try:
            if release_date:
                return int(release_date.split('-')[0])
            return 2020
        except:
            return 2020
    
    async def _prefetch_artist_genres(self, tracks: List[Dict[str, Any]]) -> None:
        """Pre-fetch genres for all unique artists using batch requests (up to 50 per call)."""
        try:
            # Collect all unique artist IDs
            unique_artist_ids = set()
            for track in tracks:
                if track.get('artists'):
                    for artist in track['artists']:
                        if artist.get('id'):
                            unique_artist_ids.add(artist['id'])
            
            # Filter out already cached artists
            uncached_artist_ids = [aid for aid in unique_artist_ids if aid not in self.artist_genre_cache]
            
            if not uncached_artist_ids:
                logger.info("All artist genres already cached, skipping API calls")
                return
            
            logger.info(f"Pre-fetching genres for {len(uncached_artist_ids)} unique artists using batch requests...")
            
            # Process artists in batches of 50 (Spotify API limit)
            batch_size = 50
            total_batches = (len(uncached_artist_ids) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(uncached_artist_ids))
                batch_artist_ids = uncached_artist_ids[start_idx:end_idx]
                
                logger.info(f"Fetching batch {batch_num + 1}/{total_batches} ({len(batch_artist_ids)} artists)...")
                
                try:
                    # Make batch request for up to 50 artists
                    artists_response = self.sp.artists(batch_artist_ids)
                    
                    # Process each artist in the batch response
                    for artist in artists_response['artists']:
                        if artist:  # Check if artist data exists (some might be None)
                            artist_id = artist['id']
                            if artist.get('genres'):
                                self.artist_genre_cache[artist_id] = artist['genres'][0]
                            else:
                                self.artist_genre_cache[artist_id] = 'Pop'
                        else:
                            # Handle case where artist data is None
                            logger.warning(f"Received None artist data for one of the batch artists")
                    
                    logger.info(f"Successfully processed batch {batch_num + 1}/{total_batches}")
                    
                except Exception as e:
                    logger.error(f"Failed to fetch batch {batch_num + 1}: {str(e)}")
                    # Fallback: set default genre for all artists in this batch
                    for artist_id in batch_artist_ids:
                        self.artist_genre_cache[artist_id] = 'Pop'
            
            logger.info(f"Successfully cached genres for {len(self.artist_genre_cache)} total artists")
            logger.info(f"Optimization: Reduced API calls from {len(uncached_artist_ids)} individual calls to {total_batches} batch calls")
            
        except Exception as e:
            logger.error(f"Failed to pre-fetch artist genres: {str(e)}")

    def _extract_genre(self, track: Dict[str, Any]) -> str:
        """Extract genre from track info using cached artist data."""
        try:
            # Try to get genre from album first
            if track.get('album', {}).get('genres'):
                return track['album']['genres'][0]
            
            # Try to get genre from cached artist data
            if track.get('artists'):
                artist_id = track['artists'][0]['id']
                if artist_id in self.artist_genre_cache:
                    return self.artist_genre_cache[artist_id]
            
            return 'Pop'  # Default genre
        except:
            return 'Pop'
    
    def _get_album_art(self, album: Dict[str, Any]) -> Optional[str]:
        """Get highest quality album art URL."""
        try:
            if album.get('images'):
                # Return the largest image
                return album['images'][0]['url']
            return None
        except:
            return None

class YouTubeAudioService:
    """Service for finding and downloading audio from YouTube."""
    
    def search_youtube_video(self, title: str, artist: str) -> Optional[str]:
        """Search YouTube for a video using Spotify track info."""
        try:
            # Create search query
            query = f"{artist} - {title}"
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'default_search': 'ytsearch1',
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search for the video
                search_results = ydl.extract_info(f"ytsearch1:{query}", download=False)
                
                if search_results and 'entries' in search_results:
                    for entry in search_results['entries']:
                        if entry and entry.get('url'):
                            return entry['url']
            
            return None
            
        except Exception as e:
            logger.debug(f"YouTube search failed for {query}: {str(e)}")
            return None
    
    def download_audio_snippet(self, video_url: str, duration: int = 30) -> Optional[str]:
        """Download a 30-second audio snippet from YouTube."""
        try:
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_path,
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'external_downloader': 'ffmpeg',
                'external_downloader_args': ['-t', str(duration)],  # 30 seconds
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            return temp_path if os.path.exists(temp_path) else None
            
        except Exception as e:
            logger.debug(f"Audio download failed: {str(e)}")
            return None


def generate_smart_description(track_info: Dict[str, Any]) -> str:
    """Generate a smart description from Spotify metadata."""
    try:
        title = track_info['title']
        artist = track_info['artist']
        album = track_info['album']
        year = track_info['year']
        genre = track_info['genre']
        
        # Build description parts
        parts = []
        
        # Genre and style
        if 'rock' in genre.lower():
            parts.append("Rock song")
        elif 'pop' in genre.lower():
            parts.append("Pop track")
        elif 'jazz' in genre.lower():
            parts.append("Jazz piece")
        elif 'hip' in genre.lower() or 'rap' in genre.lower():
            parts.append("Hip-hop track")
        else:
            parts.append(f"{genre} song")
        
        # Add artist
        parts.append(f"by {artist}")
        
        # Add album context
        if album and album != 'Unknown Album':
            parts.append(f"from the album {album}")
        
        # Add year context
        if year and year != 2020:
            if year < 1990:
                parts.append("from the classic era")
            elif year < 2000:
                parts.append("from the 90s")
            elif year < 2010:
                parts.append("from the 2000s")
            else:
                parts.append("from the modern era")
        
        # Add mood based on title
        title_lower = title.lower()
        if any(word in title_lower for word in ['summer', 'sun', 'beach', 'happy']):
            parts.append("with upbeat vibes")
        elif any(word in title_lower for word in ['love', 'heart', 'kiss', 'romance']):
            parts.append("with romantic themes")
        elif any(word in title_lower for word in ['night', 'dark', 'moon', 'lonely']):
            parts.append("with nocturnal atmosphere")
        elif any(word in title_lower for word in ['party', 'dance', 'fun', 'celebration']):
            parts.append("with party energy")
        
        return " ".join(parts)
        
    except Exception as e:
        logger.warning(f"Failed to generate description: {str(e)}")
        return f"{track_info.get('genre', 'Music')} song by {track_info.get('artist', 'Unknown Artist')}"


async def ingest_spotify_playlist(playlist_url: str, track_service: TrackService, manager=None, task_id: str = None, spotify_client_id: str = None, spotify_client_secret: str = None) -> Dict[str, Any]:
    """Ingest a Spotify playlist with YouTube audio."""
    try:
        logger.info(f"Starting Spotify playlist ingestion: {playlist_url}")
        
        # Initialize services with embeddings (text + image only for now)
        multimodal_embedding_service = MultimodalEmbeddingService()
        lyrics_service = LyricsService()
        
        # Small delay to ensure frontend has cleared stale progress data
        import asyncio
        await asyncio.sleep(0.5)
        
        # Initialize Spotify service
        update_progress(
            isActive=True,
            current=0,
            total=0,
            message="Connecting to Spotify...",
            currentTrack=""
        )
        spotify_service = SpotifyPlaylistService(spotify_client_id, spotify_client_secret)
        
        # Extract playlist info from Spotify
        update_progress(
            message="Fetching playlist data from Spotify...",
            currentTrack=""
        )
        playlist_info = await spotify_service.extract_playlist_info(playlist_url, manager, task_id, lyrics_service)
        
        # Check if ingestion was cancelled during playlist extraction
        if playlist_info.get("message") == "Ingestion cancelled":
            logger.info("Ingestion cancelled during playlist extraction")
            return playlist_info
            
        total_tracks = len(playlist_info['tracks'])
        
        logger.info(f"Found {total_tracks} tracks from Spotify")
        
        # Update progress with total tracks
        update_progress(
            isActive=True,
            current=0,
            total=total_tracks,
            message=f"Found {total_tracks} tracks, preparing to process...",
            currentTrack=""
        )
        logger.info(f"Starting ingestion of {total_tracks} tracks...")
        
        # Update progress before starting track processing
        update_progress(
            current=0,
            total=total_tracks,
            message=f"Starting to process {total_tracks} tracks...",
            currentTrack=""
        )
        
        tracks_processed = 0
        tracks_added = 0
        
        # Process each track
        for track_info in playlist_info['tracks']:
            # Check for cancellation
            if manager and manager.is_cancelled(task_id or "current"):
                logger.info("Ingestion cancelled by user")
                update_progress(
                    isActive=False,
                    current=tracks_processed,
                    total=total_tracks,
                    message="Ingestion cancelled by user",
                    currentTrack=""
                )
                return {
                    "message": "Ingestion cancelled",
                    "playlist_url": playlist_url,
                    "tracks_processed": tracks_processed,
                    "tracks_added": tracks_added
                }
            
            try:
                tracks_processed += 1
                logger.info(f"Processing track {tracks_processed}/{total_tracks}: {track_info['title']} by {track_info['artist']}")
                
                # Update progress for each track
                update_progress(
                    current=tracks_processed,
                    total=total_tracks,
                    message=f"Processing track {tracks_processed}/{total_tracks}",
                    currentTrack=f"{track_info['title']} by {track_info['artist']}"
                )
                
                # Special message for first track
                if tracks_processed == 1:
                    update_progress(
                        current=tracks_processed,
                        total=total_tracks,
                        message=f"Processing first track of {total_tracks}...",
                        currentTrack=f"{track_info['title']} by {track_info['artist']}"
                    )
                
                # Log every 10 tracks
                if tracks_processed % 10 == 0:
                    logger.info(f"Processed {tracks_processed}/{total_tracks} tracks...")
                
                # Skip YouTube audio for now - focus on image search
                youtube_url = ""
                audio_path = None
                
                # Generate smart description
                description = generate_smart_description(track_info)
                
                # Skip audio analysis for now - focus on image search
                mood = "Unknown"
                tempo = "Unknown"
                instruments = "Guitar, Bass, Drums, Vocals"
                
                # Create track with perfect Spotify metadata and audio analysis
                lyrics_text = track_info.get('lyrics', '')
                if lyrics_text:
                    logger.info(f"Creating track with lyrics ({len(lyrics_text)} chars): {track_info['title']} by {track_info['artist']}")
                else:
                    logger.info(f"Creating track without lyrics: {track_info['title']} by {track_info['artist']}")
                
                track_create = TrackCreate(
                    title=track_info['title'],
                    artist=track_info['artist'],
                    album=track_info['album'],
                    year=track_info['year'],
                    genre=track_info['genre'],
                    youtube_url=youtube_url,
                    album_art_url=track_info['album_art'] or '',
                    lyrics=lyrics_text,  # Use fetched lyrics
                    semantic_description=description,
                    mood=mood,
                    tempo=tempo,
                    instruments=instruments,
                    tags=""
                )
                
                # Add to database
                track = await track_service.create_track(track_create)
                
                # Generate multimodal embeddings
                await generate_embeddings_from_spotify(track, track_info, multimodal_embedding_service)
                
                tracks_added += 1
                logger.info(f"Successfully added: {track.title} by {track.artist} ({track.year})")
                
                # Skip audio cleanup for now
                
            except Exception as e:
                logger.error(f"Failed to process track {track_info['title']}: {str(e)}")
                continue
        
        # Simple completion logging
        logger.info(f"Ingestion completed! Processed {tracks_processed} tracks, added {tracks_added}.")
        
        logger.info(f"Spotify ingestion completed: {tracks_added}/{tracks_processed} tracks added")
        
        return {
            "tracks_processed": tracks_processed,
            "tracks_added": tracks_added,
            "playlist_title": playlist_info.get('title', 'Unknown'),
            "playlist_url": playlist_url
        }
        
    except Exception as e:
        logger.error(f"Spotify ingestion failed: {str(e)}")
        raise


async def generate_embeddings_from_spotify(track, track_info, multimodal_service):
    """Generate multimodal embeddings from Spotify data."""
    try:
        # Combine all text content (this will be filtered in the embedding service)
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
        else:
            # For tracks without album art, use text-focused embedding
            multimodal_embedding = multimodal_service.embed_multimodal_content(
                text=text_content,
                image_url=None
            )
        
        # Update track with embeddings
        from backend.services.track_service import TrackService
        from backend.database.connection import get_database
        
        track_service = TrackService(get_database())
        await track_service.update_track_embeddings(
            track.track_id,
            multimodal_embedding=multimodal_embedding
        )
        
    except Exception as e:
        logger.warning(f"Failed to generate embeddings for {track.title}: {str(e)}")
