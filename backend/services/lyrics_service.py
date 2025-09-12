"""
Lyrics service for fetching song lyrics using free APIs.
Uses Lyrics.ovh API which requires no authentication.
"""

import logging
import requests
import time
import re
from typing import Optional, Dict, Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


class LyricsService:
    """Service for fetching song lyrics from free APIs."""
    
    def __init__(self):
        self.base_url = "https://api.lyrics.ovh/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'VectorBeat/1.0 (Music Search App)'
        })
        # Rate limiting: 10 requests per second (aggressive but safe for Lyrics.ovh)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests = 10 requests/second
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def get_lyrics(self, artist: str, title: str) -> Optional[str]:
        """
        Fetch lyrics for a song using Lyrics.ovh API.
        
        Args:
            artist: Artist name
            title: Song title
            
        Returns:
            Lyrics text if found, None otherwise
        """
        try:
            # Clean and encode artist and title
            clean_artist = self._clean_text(artist)
            clean_title = self._clean_text(title)
            
            if not clean_artist or not clean_title:
                logger.warning(f"Invalid artist or title: '{artist}' / '{title}'")
                return None
            
            # Apply rate limiting
            self._rate_limit()
            
            # Make API request
            url = f"{self.base_url}/{quote(clean_artist)}/{quote(clean_title)}"
            logger.debug(f"Fetching lyrics from: {url}")
            
            response = self.session.get(url, timeout=10)
            
            # Handle rate limiting gracefully
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 1))
                logger.warning(f"Rate limited, waiting {retry_after} seconds before retry")
                time.sleep(retry_after)
                # Retry once
                response = self.session.get(url, timeout=10)
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'lyrics' in data and data['lyrics']:
                lyrics = data['lyrics'].strip()
                if lyrics and lyrics != "Sorry, we don't have lyrics for this song yet.":
                    logger.info(f"Successfully fetched lyrics for '{title}' by '{artist}' ({len(lyrics)} characters)")
                    return lyrics
                else:
                    logger.debug(f"No lyrics available for '{title}' by '{artist}' - empty or placeholder response")
                    return None
            else:
                logger.debug(f"No lyrics found for '{title}' by '{artist}' - no lyrics field in response")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for '{title}' by '{artist}': {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching lyrics for '{title}' by '{artist}': {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text for API requests."""
        if not text:
            return ""
        
        # Remove common suffixes and clean up
        text = text.strip()
        
        # Remove common suffixes that might interfere with search
        suffixes_to_remove = [
            ' (feat.', ' (ft.', ' (featuring', ' (with',
            ' [feat.', ' [ft.', ' [featuring', ' [with',
            ' - feat.', ' - ft.', ' - featuring', ' - with'
        ]
        
        for suffix in suffixes_to_remove:
            if suffix in text.lower():
                text = text[:text.lower().find(suffix)].strip()
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        return text
    
    def _clean_artist_name(self, artist: str) -> str:
        """Clean artist name for better API matching - be conservative with multiple artists."""
        if not artist:
            return ""
        
        artist = artist.strip()
        
        # Only remove explicit feat/ft patterns, not comma-separated artists
        # This preserves legitimate collaborations and band names
        artist = re.sub(r'\s+feat\..*$', '', artist, flags=re.IGNORECASE)
        artist = re.sub(r'\s+ft\..*$', '', artist, flags=re.IGNORECASE)
        artist = re.sub(r'\s+featuring.*$', '', artist, flags=re.IGNORECASE)
        artist = re.sub(r'\s+with.*$', '', artist, flags=re.IGNORECASE)
        
        # For comma-separated artists, be very conservative
        # Only remove if there's an explicit "feat" or "ft" in the second part
        if ',' in artist:
            parts = [part.strip() for part in artist.split(',')]
            if len(parts) > 1:
                second_part = parts[1].lower()
                # Only remove if there's an explicit featured artist indicator
                explicit_featured = any(keyword in second_part for keyword in ['feat.', 'ft.', 'featuring', 'with'])
                
                if explicit_featured:
                    # Take only the first part (main artist)
                    artist = parts[0]
                else:
                    # Keep all parts - likely legitimate collaboration
                    # For multiple artists, try the first one first, then the full string
                    artist = f"{parts[0]}, {parts[1]}"
        
        artist = artist.strip()
        return artist
    
    def get_lyrics_for_track(self, track_info: Dict[str, Any]) -> Optional[str]:
        """
        Get lyrics for a track using track metadata.
        
        Args:
            track_info: Dictionary containing track metadata with 'title'/'name' and 'artist'/'artists' keys
            
        Returns:
            Lyrics text if found, None otherwise
        """
        # Handle both processed track info (title/artist) and raw Spotify data (name/artists)
        title = track_info.get('title') or track_info.get('name', '')
        
        # Handle artist field - could be string or list
        artist = track_info.get('artist', '')
        if not artist and track_info.get('artists'):
            # Convert artists list to string
            artists_list = track_info['artists']
            if isinstance(artists_list, list) and len(artists_list) > 0:
                if isinstance(artists_list[0], dict):
                    # List of artist objects
                    artist = ', '.join([a.get('name', '') for a in artists_list if a.get('name')])
                else:
                    # List of artist names
                    artist = ', '.join(artists_list)
        
        if not title or not artist:
            logger.warning(f"Missing title or artist in track info: {track_info}")
            return None
        
        # Try multiple artist and title variations to improve success rate
        artist_variations = self._get_artist_variations(artist)
        title_variations = self._get_title_variations(title)
        
        # Try all combinations of artist and title variations (limit to reasonable number)
        max_attempts = 12  # Increased to ensure we try more artist variations
        attempt_count = 0
        
        for i, clean_artist in enumerate(artist_variations):
            for j, clean_title in enumerate(title_variations):
                if attempt_count >= max_attempts:
                    logger.debug(f"Reached max attempts ({max_attempts}), stopping variations")
                    break
                    
                if i > 0 or j > 0:  # Log when trying variations
                    logger.debug(f"Trying variation {attempt_count + 1}: artist '{clean_artist}' + title '{clean_title}'")
                lyrics = self.get_lyrics(clean_artist, clean_title)
                if lyrics:
                    if i > 0 or j > 0:  # If we found lyrics with a variation, log it
                        logger.info(f"Found lyrics using variation {attempt_count + 1}: artist '{clean_artist}' + title '{clean_title}'")
                    return lyrics
                attempt_count += 1
            
            if attempt_count >= max_attempts:
                break
        
        return None
    
    def _get_artist_variations(self, artist: str) -> list[str]:
        """Get multiple artist name variations to try for lyrics search."""
        variations = []
        
        # 1. Clean version (removes explicit feat/ft)
        clean_artist = self._clean_artist_name(artist)
        variations.append(clean_artist)
        
        # 2. If there are multiple artists, try each individual artist
        if ',' in artist:
            parts = [part.strip() for part in artist.split(',')]
            for part in parts:
                if part and part not in variations:
                    variations.append(part)
        
        # 3. Special case: if we have "Tyler, The Creator" in the string, add it as a variation
        if 'Tyler, The Creator' in artist and 'Tyler, The Creator' not in variations:
            variations.append('Tyler, The Creator')
        
        # 4. Try the original artist name (in case cleaning was too aggressive)
        if artist not in variations:
            variations.append(artist)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var and var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations
    
    def _get_title_variations(self, title: str) -> list[str]:
        """Get multiple title variations to try for lyrics search."""
        variations = []
        
        # 1. Clean version (removes common suffixes)
        clean_title = self._clean_text(title)
        variations.append(clean_title)
        
        # 2. If title has slash, try just the first part
        if '/' in title:
            first_part = title.split('/')[0].strip()
            if first_part != clean_title:
                variations.append(first_part)
        
        # 3. If title has slash, try just the second part
        if '/' in title:
            parts = [part.strip() for part in title.split('/')]
            if len(parts) > 1:
                second_part = parts[1].strip()
                if second_part not in variations:
                    variations.append(second_part)
        
        # 4. If title has dash, try just the first part
        if ' - ' in title:
            first_part = title.split(' - ')[0].strip()
            if first_part not in variations:
                variations.append(first_part)
        
        # 5. Try the original title (in case cleaning was too aggressive)
        if title not in variations:
            variations.append(title)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var and var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations
    
    def batch_get_lyrics(self, tracks: list[Dict[str, Any]], progress_callback=None, cancellation_check=None) -> Dict[str, str]:
        """
        Fetch lyrics for multiple tracks with progress updates.
        
        Args:
            tracks: List of track dictionaries
            progress_callback: Optional callback function for progress updates
            cancellation_check: Optional function to check if operation should be cancelled
            
        Returns:
            Dictionary mapping track_id to lyrics text
        """
        lyrics_results = {}
        total_tracks = len(tracks)
        
        logger.info(f"Starting batch lyrics fetch for {total_tracks} tracks... (10 requests/second)")
        estimated_time = total_tracks * 0.1  # 100ms per track
        logger.info(f"Estimated completion time: {estimated_time:.1f} seconds")
        
        for i, track in enumerate(tracks):
            # Check for cancellation before processing each track
            if cancellation_check and cancellation_check():
                logger.info(f"Lyrics fetching cancelled by user at track {i+1}/{total_tracks}")
                break
                
            try:
                # Use 'id' for raw Spotify tracks, 'spotify_id' for processed tracks
                track_id = track.get('id') or track.get('spotify_id', f"track_{i}")
                lyrics = self.get_lyrics_for_track(track)
                
                if lyrics:
                    lyrics_results[track_id] = lyrics
                    logger.debug(f"Fetched lyrics for track {i+1}/{total_tracks}: {track.get('title', 'Unknown')}")
                else:
                    logger.debug(f"No lyrics found for track {i+1}/{total_tracks}: {track.get('title', 'Unknown')}")
                
                # Call progress callback if provided
                if progress_callback:
                    # Get track name from either 'name' (raw Spotify) or 'title' (processed)
                    track_name = track.get('name') or track.get('title', 'Unknown')
                    progress_callback(i + 1, total_tracks, track_name)
                    
            except Exception as e:
                logger.error(f"Error processing track {i+1}/{total_tracks}: {str(e)}")
                continue
        
        logger.info(f"Batch lyrics fetch completed: {len(lyrics_results)}/{total_tracks} tracks have lyrics")
        return lyrics_results
