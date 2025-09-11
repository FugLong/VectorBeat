"""
Search service for multi-modal semantic search.
Handles vector similarity search across text, image, and audio embeddings.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.database.connection import get_database
from backend.database.schema import get_tracks_table, dict_to_track
from backend.models.track import Track, TrackResponse, SearchResult, SearchResponse, TrackFilters
from backend.embeddings.text_embeddings import TextEmbeddingService
from backend.embeddings.image_embeddings import ImageEmbeddingService
from backend.embeddings.audio_embeddings import AudioEmbeddingService

logger = logging.getLogger(__name__)


class SearchService:
    """Service for multi-modal semantic search."""
    
    def __init__(self, db):
        """Initialize search service with database connection."""
        self.db = db
        self.text_embedding_service = TextEmbeddingService()
        self.image_embedding_service = ImageEmbeddingService()
        self.audio_embedding_service = AudioEmbeddingService()
    
    async def search(self, query: str, limit: int = 20, filters: Optional[TrackFilters] = None, mode: str = 'metadata') -> List[SearchResult]:
        """Perform metadata-based search (simplified version without embeddings)."""
        try:
            # Get all tracks from database
            table = get_tracks_table(self.db)
            df = table.to_pandas()
            
            if df.empty:
                return []
            
            # Apply filters if provided
            if filters:
                df = self._apply_filters(df, filters)
            
            if df.empty:
                return []
            
            # Calculate similarity scores based on mode
            results = []
            query_lower = query.lower()
            
            for _, row in df.iterrows():
                track_data = row.to_dict()
                track = dict_to_track(track_data)
                
                similarity_score = 0.0
                match_type = 'metadata'
                
                if mode == 'metadata':
                    # Traditional string search - highly effective
                    score = 0.0
                    query_words = query_lower.split()
                    
                    # Title matching (highest priority)
                    if track.title:
                        title_lower = track.title.lower()
                        # Exact phrase match
                        if query_lower in title_lower:
                            score += 0.9
                        # All words match
                        elif all(word in title_lower for word in query_words):
                            score += 0.8
                        # Partial word matches
                        else:
                            word_matches = sum(1 for word in query_words if word in title_lower)
                            score += (word_matches / len(query_words)) * 0.6
                    
                    # Artist matching (high priority)
                    if track.artist:
                        artist_lower = track.artist.lower()
                        if query_lower in artist_lower:
                            score += 0.8
                        elif all(word in artist_lower for word in query_words):
                            score += 0.7
                        else:
                            word_matches = sum(1 for word in query_words if word in artist_lower)
                            score += (word_matches / len(query_words)) * 0.5
                    
                    # Album matching (medium priority)
                    if track.album:
                        album_lower = track.album.lower()
                        if query_lower in album_lower:
                            score += 0.6
                        elif all(word in album_lower for word in query_words):
                            score += 0.5
                        else:
                            word_matches = sum(1 for word in query_words if word in album_lower)
                            score += (word_matches / len(query_words)) * 0.3
                    
                    # Genre matching (medium priority)
                    if track.genre:
                        genre_lower = track.genre.lower()
                        if query_lower in genre_lower:
                            score += 0.5
                        elif all(word in genre_lower for word in query_words):
                            score += 0.4
                        else:
                            word_matches = sum(1 for word in query_words if word in genre_lower)
                            score += (word_matches / len(query_words)) * 0.2
                    
                    # Description matching (low priority)
                    if track.semantic_description:
                        desc_lower = track.semantic_description.lower()
                        if query_lower in desc_lower:
                            score += 0.3
                        else:
                            word_matches = sum(1 for word in query_words if word in desc_lower)
                            score += (word_matches / len(query_words)) * 0.1
                    
                    # Year matching
                    if track.year and str(track.year) in query:
                        score += 0.2
                    
                    similarity_score = min(1.0, score)
                    match_type = 'metadata'
                
                elif mode == 'semantic':
                    # Multi-modal semantic search using embeddings
                    similarity_score = 0.0
                    match_type = 'semantic'
                    
                    # Generate query embedding for text-to-text search
                    query_embedding = self.text_embedding_service.embed_text(query)
                    
                    # Calculate similarities across all embedding types
                    text_similarity = 0.0
                    image_similarity = 0.0
                    audio_similarity = 0.0
                    
                    # Text similarity (title, artist, description, lyrics)
                    if track.text_embedding:
                        text_similarity = self._cosine_similarity(query_embedding, track.text_embedding)
                    
                    # Image similarity (album art) - for queries like "blue album art"
                    if track.image_embedding and self._is_image_query(query):
                        image_similarity = self.image_embedding_service.text_image_similarity(query, track.image_embedding)
                    
                    # Skip audio similarity for now - focus on image search
                    audio_similarity = 0.0
                    
                    # Combine similarities with weights (text + image only for now)
                    similarity_score = (
                        text_similarity * 0.7 +
                        image_similarity * 0.3
                    )
                    
                    # Determine match type based on highest similarity (text vs image only)
                    if image_similarity > text_similarity:
                        match_type = 'image'
                    else:
                        match_type = 'text'
                
                elif mode == 'combined':
                    # Combined metadata + semantic search
                    metadata_score = 0.0
                    query_words = query_lower.split()
                    
                    # Calculate metadata score (same as metadata mode)
                    if track.title:
                        title_lower = track.title.lower()
                        if query_lower in title_lower:
                            metadata_score += 0.9
                        elif all(word in title_lower for word in query_words):
                            metadata_score += 0.8
                        else:
                            word_matches = sum(1 for word in query_words if word in title_lower)
                            metadata_score += (word_matches / len(query_words)) * 0.6
                    
                    if track.artist:
                        artist_lower = track.artist.lower()
                        if query_lower in artist_lower:
                            metadata_score += 0.8
                        elif all(word in artist_lower for word in query_words):
                            metadata_score += 0.7
                        else:
                            word_matches = sum(1 for word in query_words if word in artist_lower)
                            metadata_score += (word_matches / len(query_words)) * 0.5
                    
                    if track.album:
                        album_lower = track.album.lower()
                        if query_lower in album_lower:
                            metadata_score += 0.6
                        elif all(word in album_lower for word in query_words):
                            metadata_score += 0.5
                        else:
                            word_matches = sum(1 for word in query_words if word in album_lower)
                            metadata_score += (word_matches / len(query_words)) * 0.3
                    
                    if track.genre:
                        genre_lower = track.genre.lower()
                        if query_lower in genre_lower:
                            metadata_score += 0.5
                        elif all(word in genre_lower for word in query_words):
                            metadata_score += 0.4
                        else:
                            word_matches = sum(1 for word in query_words if word in genre_lower)
                            metadata_score += (word_matches / len(query_words)) * 0.2
                    
                    if track.semantic_description:
                        desc_lower = track.semantic_description.lower()
                        if query_lower in desc_lower:
                            metadata_score += 0.3
                        else:
                            word_matches = sum(1 for word in query_words if word in desc_lower)
                            metadata_score += (word_matches / len(query_words)) * 0.1
                    
                    if track.year and str(track.year) in query:
                        metadata_score += 0.2
                    
                    # Calculate semantic score using embeddings
                    semantic_score = 0.0
                    if any([track.text_embedding, track.image_embedding, track.audio_embedding]):
                        query_embedding = self.text_embedding_service.embed_text(query)
                        
                        text_similarity = 0.0
                        image_similarity = 0.0
                        audio_similarity = 0.0
                        
                        if track.text_embedding:
                            text_similarity = self._cosine_similarity(query_embedding, track.text_embedding)
                        
                        if track.image_embedding and self._is_image_query(query):
                            image_similarity = self.image_embedding_service.text_image_similarity(query, track.image_embedding)
                        
                        # Skip audio similarity for now - focus on image search
                        audio_similarity = 0.0
                        
                        semantic_score = (
                            text_similarity * 0.7 +
                            image_similarity * 0.3
                        )
                    
                    # Combine metadata (60%) and semantic (40%) scores
                    similarity_score = min(1.0, (metadata_score * 0.6) + (semantic_score * 0.4))
                    match_type = 'combined'
                
                # Create search result
                track_response = TrackResponse.from_track(track)
                result = SearchResult(
                    track=track_response,
                    similarity_score=similarity_score,
                    match_type=match_type
                )
                results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top results
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to perform search: {str(e)}")
            raise
    
    def _apply_filters(self, df, filters: TrackFilters) -> 'pandas.DataFrame':
        """Apply filters to the dataframe."""
        try:
            if filters.genre:
                df = df[df['genre'].str.contains(filters.genre, case=False, na=False)]
            
            if filters.artist:
                df = df[df['artist'].str.contains(filters.artist, case=False, na=False)]
            
            if filters.year_min:
                df = df[df['year'] >= filters.year_min]
            
            if filters.year_max:
                df = df[df['year'] <= filters.year_max]
            
            if filters.tags:
                tag_conditions = []
                for tag in filters.tags:
                    tag_conditions.append(df['tags'].str.contains(tag, case=False, na=False))
                if tag_conditions:
                    df = df[np.logical_or.reduce(tag_conditions)]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to apply filters: {str(e)}")
            return df
    
    
    def _is_image_query(self, query: str) -> bool:
        """Check if query contains image-related terms."""
        image_keywords = [
            'cover', 'art', 'image', 'picture', 'photo', 'visual', 'album art',
            'front cover', 'back cover', 'artwork', 'design', 'illustration'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in image_keywords)
    
    def _is_audio_query(self, query: str) -> bool:
        """Check if query contains audio-related terms."""
        audio_keywords = [
            'sound', 'audio', 'music', 'melody', 'rhythm', 'beat', 'tone',
            'instrument', 'vocal', 'singing', 'acoustic', 'electric', 'jazz',
            'rock', 'pop', 'classical', 'electronic', 'upbeat', 'slow', 'fast'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in audio_keywords)
    
    def _calculate_image_similarity(self, query: str, image_embedding: List[float]) -> float:
        """Calculate image similarity using text-image matching."""
        try:
            return self.image_embedding_service.text_image_similarity(query, image_embedding)
        except Exception as e:
            logger.error(f"Failed to calculate image similarity: {str(e)}")
            return 0.0
    
    def _calculate_audio_similarity(self, query: str, audio_embedding: List[float]) -> float:
        """Calculate audio similarity using text-audio matching."""
        try:
            # For demo purposes, we'll use a simple keyword-based approach
            # since we can't directly compare text and audio embeddings of different dimensions
            
            # Check if query contains audio-related keywords
            audio_keywords = ['music', 'song', 'melody', 'rhythm', 'beat', 'sound', 'audio', 'instrumental', 'vocal', 'singing']
            query_lower = query.lower()
            
            # Calculate a simple score based on audio-related keywords
            keyword_matches = sum(1 for keyword in audio_keywords if keyword in query_lower)
            base_score = min(keyword_matches * 0.2, 1.0)
            
            # Add some randomness based on the audio embedding to make it more interesting
            # This is a demo approach - in production you'd use a proper audio-text model
            import random
            random.seed(hash(tuple(audio_embedding[:10])))  # Use first 10 dims as seed
            random_factor = random.uniform(0.1, 0.3)
            
            return min(base_score + random_factor, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate audio similarity: {str(e)}")
            return 0.0
    
    def _combine_similarities(self, text_sim: float, image_sim: float, audio_sim: float, query: str) -> float:
        """Combine different similarity scores with appropriate weights."""
        # Base weights
        text_weight = 0.6
        image_weight = 0.2
        audio_weight = 0.2
        
        # Adjust weights based on query content
        if self._is_image_query(query):
            image_weight = 0.4
            text_weight = 0.4
            audio_weight = 0.2
        elif self._is_audio_query(query):
            audio_weight = 0.4
            text_weight = 0.4
            image_weight = 0.2
        
        # Combine similarities
        combined = (
            text_weight * text_sim +
            image_weight * image_sim +
            audio_weight * audio_sim
        )
        
        return min(1.0, max(0.0, combined))  # Clamp to [0, 1]
    
    def _determine_match_type(self, text_sim: float, image_sim: float, audio_sim: float) -> str:
        """Determine the primary match type based on similarity scores."""
        similarities = {
            'text': text_sim,
            'image': image_sim,
            'audio': audio_sim
        }
        
        # Find the highest similarity
        max_type = max(similarities, key=similarities.get)
        max_score = similarities[max_type]
        
        # If multiple types have similar scores, return 'combined'
        if max_score > 0:
            other_scores = [v for k, v in similarities.items() if k != max_type]
            if other_scores and max(other_scores) > max_score * 0.7:
                return 'combined'
        
        return max_type if max_score > 0 else 'text'
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {str(e)}")
            return 0.0
    
    async def search_by_embedding(self, embedding: List[float], embedding_type: str, 
                                limit: int = 20, filters: Optional[TrackFilters] = None) -> List[SearchResult]:
        """Search using a pre-computed embedding."""
        try:
            table = get_tracks_table(self.db)
            df = table.to_pandas()
            
            if df.empty:
                return []
            
            # Apply filters if provided
            if filters:
                df = self._apply_filters(df, filters)
            
            if df.empty:
                return []
            
            # Calculate similarities
            results = []
            for _, row in df.iterrows():
                track_data = row.to_dict()
                track = dict_to_track(track_data)
                
                similarity = 0.0
                if embedding_type == 'text' and track.text_embedding:
                    similarity = self._cosine_similarity(embedding, track.text_embedding)
                elif embedding_type == 'image' and track.image_embedding:
                    similarity = self._cosine_similarity(embedding, track.image_embedding)
                elif embedding_type == 'audio' and track.audio_embedding:
                    similarity = self._cosine_similarity(embedding, track.audio_embedding)
                
                if similarity > 0:
                    track_response = TrackResponse.from_track(track)
                    result = SearchResult(
                        track=track_response,
                        similarity_score=similarity,
                        match_type=embedding_type
                    )
                    results.append(result)
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search by embedding: {str(e)}")
            raise
    
    def _is_image_query(self, query: str) -> bool:
        """Check if query is likely about visual/album art features."""
        image_keywords = [
            'album', 'art', 'cover', 'image', 'picture', 'photo', 'visual',
            'blue', 'red', 'green', 'yellow', 'dark', 'bright', 'color', 'colored',
            'abstract', 'geometric', 'vintage', 'retro', 'modern', 'classic',
            'person', 'people', 'face', 'portrait', 'landscape', 'nature',
            'moody', 'atmospheric', 'minimalist', 'complex', 'simple'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in image_keywords)
    
    def _is_audio_query(self, query: str) -> bool:
        """Check if query is likely about musical/audio features."""
        audio_keywords = [
            'tempo', 'beat', 'rhythm', 'fast', 'slow', 'upbeat', 'downbeat',
            'jazz', 'rock', 'pop', 'classical', 'electronic', 'acoustic',
            'guitar', 'piano', 'drums', 'bass', 'violin', 'saxophone', 'trumpet',
            'instrumental', 'vocal', 'singing', 'melody', 'harmony', 'chord',
            'loud', 'quiet', 'soft', 'aggressive', 'gentle', 'energetic',
            'mood', 'atmosphere', 'vibe', 'feeling', 'emotion'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in audio_keywords)
