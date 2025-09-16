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
from backend.embeddings.multimodal_embeddings import MultimodalEmbeddingService

logger = logging.getLogger(__name__)


class SearchService:
    """Service for multi-modal semantic search."""
    
    def __init__(self, db):
        """Initialize search service with database connection."""
        self.db = db
        self.multimodal_embedding_service = MultimodalEmbeddingService()
    
    async def search(self, query: str, limit: int = 20, page: int = 1, filters: Optional[TrackFilters] = None, mode: str = 'metadata') -> tuple[List[SearchResult], int]:
        """Perform search with improved performance for semantic search."""
        try:
            # Get all tracks from database
            table = get_tracks_table(self.db)
            df = table.to_pandas()
            
            if df.empty:
                return [], 0
            
            # Apply filters if provided
            if filters:
                df = self._apply_filters(df, filters)
            
            if df.empty:
                return [], 0
            
            # For semantic search, pre-compute query embeddings once for performance
            query_embedding = None
            query_phrase_embedding = None
            query_word_embeddings = None
            
            if mode in ['semantic', 'combined']:
                try:
                    
                    # Check if this is a year query (exact metadata should take priority)
                    import re
                    is_year_query = bool(re.match(r'^\d{4}$', query.strip())) or any(year in query for year in ['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'])
                    
                    # Generate phrase embedding (treats "yellow pants" as one concept)
                    query_phrase_embedding = self.multimodal_embedding_service.embed_text(query)
                    
                    # Generate individual word embeddings for fallback
                    query_words = query.lower().split()
                    if len(query_words) > 1:  # Only for multi-word queries
                        query_word_embeddings = []
                        for word in query_words:
                            if len(word) > 2:  # Skip very short words
                                word_embedding = self.multimodal_embedding_service.embed_text(word)
                                query_word_embeddings.append((word, word_embedding))
                    
                    # Use phrase embedding as primary
                    query_embedding = query_phrase_embedding
                    
                except Exception as e:
                    logger.warning(f"Failed to generate query embedding: {str(e)}")
                    # Fallback to metadata search if embedding fails
                    mode = 'metadata'
            
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
                    
                    # Filter out single letters and common words for better matching
                    all_query_words = query_lower.split()
                    meaningful_words = [word for word in all_query_words if len(word) > 1 and word not in ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
                    query_words = meaningful_words if meaningful_words else all_query_words
                    
                    # Check for exact phrase match first (highest priority)
                    # Only treat multi-word queries as phrases
                    phrase_matches = 0
                    is_multi_word_phrase = len(query_lower.split()) > 1
                    
                    if is_multi_word_phrase:
                        if query_lower in track.title.lower():
                            phrase_matches += 1
                        if query_lower in track.artist.lower():
                            phrase_matches += 1
                        if track.lyrics and query_lower in track.lyrics.lower():
                            phrase_matches += 1
                        
                        if phrase_matches > 0:
                            # Exact phrase matches get the highest score - longer phrases get bonus
                            phrase_length_bonus = len(query_lower.split()) * 0.1  # Bonus for longer phrases
                            score += 0.8 + (phrase_matches * 0.1) + phrase_length_bonus
                    
                    # Title matching (highest priority) - only if no phrase match
                    if track.title and phrase_matches == 0:
                        title_lower = track.title.lower()
                        if query_words:
                            # Check for whole word matches first (much higher priority)
                            whole_word_matches = 0
                            partial_word_matches = 0
                            
                            for word in query_words:
                                # Check for whole word match (word boundaries)
                                import re
                                whole_word_pattern = r'\b' + re.escape(word) + r'\b'
                                if re.search(whole_word_pattern, title_lower):
                                    whole_word_matches += title_lower.count(word)  # Count all occurrences
                                elif word in title_lower:
                                    partial_word_matches += title_lower.count(word)
                            
                            # For single-word searches, prioritize whole word matches
                            if len(query_words) == 1:
                                if whole_word_matches > 0:
                                    score += min(0.5, whole_word_matches * 0.25)  # Direct count-based scoring for whole words
                                elif partial_word_matches > 0:
                                    score += min(0.15, partial_word_matches * 0.08)  # Lower for partial matches
                            else:
                                # For multi-word searches, prioritize whole word matches
                                if whole_word_matches > 0:
                                    score += min(0.5, (whole_word_matches / len(query_words)) * 0.5)
                                elif partial_word_matches > 0:
                                    score += min(0.15, (partial_word_matches / len(query_words)) * 0.08)  # Lower for partial
                    
                    # Artist matching (third priority) - only if no phrase match
                    if track.artist and phrase_matches == 0:
                        artist_lower = track.artist.lower()
                        if query_words:
                            # Check for whole word matches first (much higher priority)
                            whole_word_matches = 0
                            partial_word_matches = 0
                            
                            for word in query_words:
                                # Check for whole word match (word boundaries)
                                import re
                                whole_word_pattern = r'\b' + re.escape(word) + r'\b'
                                if re.search(whole_word_pattern, artist_lower):
                                    whole_word_matches += artist_lower.count(word)  # Count all occurrences
                                elif word in artist_lower:
                                    partial_word_matches += artist_lower.count(word)
                            
                            # For single-word searches, prioritize whole word matches
                            if len(query_words) == 1:
                                if whole_word_matches > 0:
                                    score += min(0.4, whole_word_matches * 0.2)  # Lower than title
                                elif partial_word_matches > 0:
                                    score += min(0.12, partial_word_matches * 0.06)  # Lower than title partial
                            else:
                                # For multi-word searches, prioritize whole word matches
                                if whole_word_matches > 0:
                                    score += min(0.4, (whole_word_matches / len(query_words)) * 0.4)
                                elif partial_word_matches > 0:
                                    score += min(0.12, (partial_word_matches / len(query_words)) * 0.06)  # Lower than title partial
                    
                    # Lyrics matching (fourth priority) - only if no phrase match
                    if track.lyrics and phrase_matches == 0:
                        lyrics_lower = track.lyrics.lower()
                        if query_words:
                            # Check for whole word matches first (much higher priority)
                            whole_word_matches = 0
                            partial_word_matches = 0
                            
                            for word in query_words:
                                # Check for whole word match (word boundaries)
                                import re
                                whole_word_pattern = r'\b' + re.escape(word) + r'\b'
                                if re.search(whole_word_pattern, lyrics_lower):
                                    whole_word_matches += lyrics_lower.count(word)  # Count all occurrences
                                elif word in lyrics_lower:
                                    partial_word_matches += lyrics_lower.count(word)
                            
                            # For single-word searches, prioritize whole word matches
                            if len(query_words) == 1:
                                if whole_word_matches > 0:
                                    score += min(0.35, whole_word_matches * 0.15)  # Lower than artist
                                elif partial_word_matches > 0:
                                    score += min(0.1, partial_word_matches * 0.05)  # Lower than artist partial
                            else:
                                # For multi-word searches, prioritize whole word matches
                                if whole_word_matches > 0:
                                    score += min(0.35, (whole_word_matches / len(query_words)) * 0.35)
                                elif partial_word_matches > 0:
                                    score += min(0.1, (partial_word_matches / len(query_words)) * 0.05)  # Lower than artist partial
                    
                    # Album matching (second priority) - only if no phrase match
                    if track.album and phrase_matches == 0:
                        album_lower = track.album.lower()
                        if query_words:
                            # Check for whole word matches first (much higher priority)
                            whole_word_matches = 0
                            partial_word_matches = 0
                            
                            for word in query_words:
                                # Check for whole word match (word boundaries)
                                import re
                                whole_word_pattern = r'\b' + re.escape(word) + r'\b'
                                if re.search(whole_word_pattern, album_lower):
                                    whole_word_matches += album_lower.count(word)  # Count all occurrences
                                elif word in album_lower:
                                    partial_word_matches += album_lower.count(word)
                            
                            # For single-word searches, prioritize whole word matches
                            if len(query_words) == 1:
                                if whole_word_matches > 0:
                                    score += min(0.45, whole_word_matches * 0.22)  # Between title and artist
                                elif partial_word_matches > 0:
                                    score += min(0.13, partial_word_matches * 0.07)  # Between title and artist partial
                            else:
                                # For multi-word searches, prioritize whole word matches
                                if whole_word_matches > 0:
                                    score += min(0.45, (whole_word_matches / len(query_words)) * 0.45)
                                elif partial_word_matches > 0:
                                    score += min(0.13, (partial_word_matches / len(query_words)) * 0.07)  # Between title and artist partial
                    
                    # Genre matching (medium priority) - only if no phrase match
                    if track.genre and phrase_matches == 0:
                        genre_lower = track.genre.lower()
                        if query_words and all(word in genre_lower for word in query_words):
                            score += 0.4
                        elif query_words:
                            total_word_count = 0
                            for word in query_words:
                                total_word_count += genre_lower.count(word)
                            score += min(0.15, (total_word_count / len(query_words)) * 0.06)  # Cap per field and reduce multiplier
                    
                    # Description matching (low priority) - only if no phrase match
                    if track.semantic_description and phrase_matches == 0:
                        desc_lower = track.semantic_description.lower()
                        if query_words:
                            total_word_count = 0
                            for word in query_words:
                                total_word_count += desc_lower.count(word)
                            score += min(0.1, (total_word_count / len(query_words)) * 0.04)  # Cap per field and reduce multiplier
                    
                    # Year matching (boosted for exact matches)
                    if track.year and str(track.year) == query_lower:
                        score += 1.0  # Boosted for exact year matches
                    elif track.year and str(track.year) in query:
                        score += 0.4  # Boosted for partial year matches
                    
                    # Fallback: if no meaningful matches found, try single letters (very low priority)
                    if score == 0 and len(all_query_words) > 0:
                        single_letters = [word for word in all_query_words if len(word) == 1]
                        if single_letters:
                            # Very low score for single letter matches
                            single_letter_matches = 0
                            if track.title:
                                single_letter_matches += sum(1 for letter in single_letters if letter in track.title.lower())
                            if track.artist:
                                single_letter_matches += sum(1 for letter in single_letters if letter in track.artist.lower())
                            if track.lyrics:
                                single_letter_matches += sum(1 for letter in single_letters if letter in track.lyrics.lower())
                            
                            if single_letter_matches > 0:
                                score = min(0.1, single_letter_matches * 0.02)  # Very low score
                    
                    similarity_score = min(1.0, score)
                    match_type = 'metadata'
                
                elif mode == 'semantic':
                    # Pure semantic search using phrase-first approach
                    similarity_score = 0.0
                    match_type = 'semantic'
                    
                    if track.multimodal_embedding and query_embedding is not None:
                        # Primary: Phrase embedding (treats "yellow pants" as one concept)
                        phrase_similarity = self.multimodal_embedding_service.similarity(query_embedding, track.multimodal_embedding)
                        
                        # Secondary: Individual word similarities (weighted lower)
                        word_similarities = []
                        if query_word_embeddings:
                            for word, word_embedding in query_word_embeddings:
                                word_sim = self.multimodal_embedding_service.similarity(word_embedding, track.multimodal_embedding)
                                word_similarities.append(word_sim)
                        
                        # Combine phrase (primary) with individual words (secondary)
                        if word_similarities:
                            # Phrase gets 85% weight, individual words get 15% weight (balanced phrase priority)
                            avg_word_similarity = sum(word_similarities) / len(word_similarities)
                            similarity_score = (phrase_similarity * 0.85) + (avg_word_similarity * 0.15)
                        else:
                            # Single word or no word embeddings, use phrase only
                            similarity_score = phrase_similarity
                        
                        # Apply lyrics penalty for tracks without lyrics
                        if not track.lyrics or not track.lyrics.strip():
                            # Reduce score by 10% for tracks without lyrics (more balanced)
                            similarity_score *= 0.9
                        
                        
                        match_type = 'multimodal'
                    # Skip tracks without multimodal embeddings in semantic mode
                
                elif mode == 'combined':
                    # Combined metadata + semantic search
                    # Use the same improved metadata scoring logic as metadata mode
                    metadata_score = 0.0
                    
                    # Filter out single letters and common words for better matching
                    all_query_words = query_lower.split()
                    meaningful_words = [word for word in all_query_words if len(word) > 1 and word not in ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
                    query_words = meaningful_words if meaningful_words else all_query_words
                    
                    # Check for exact phrase match first (highest priority)
                    # Only treat multi-word queries as phrases
                    phrase_matches = 0
                    is_multi_word_phrase = len(query_lower.split()) > 1
                    
                    if is_multi_word_phrase:
                        if query_lower in track.title.lower():
                            phrase_matches += 1
                        if query_lower in track.artist.lower():
                            phrase_matches += 1
                        if track.lyrics and query_lower in track.lyrics.lower():
                            phrase_matches += 1
                        
                        if phrase_matches > 0:
                            # Exact phrase matches get the highest score - longer phrases get bonus
                            phrase_length_bonus = len(query_lower.split()) * 0.1  # Bonus for longer phrases
                            metadata_score += 0.8 + (phrase_matches * 0.1) + phrase_length_bonus
                    
                    # Title matching (highest priority) - only if no phrase match
                    if track.title and phrase_matches == 0:
                        title_lower = track.title.lower()
                        if query_words:
                            # Check for whole word matches first (much higher priority)
                            whole_word_matches = 0
                            partial_word_matches = 0
                            
                            for word in query_words:
                                # Check for whole word match (word boundaries)
                                import re
                                whole_word_pattern = r'\b' + re.escape(word) + r'\b'
                                if re.search(whole_word_pattern, title_lower):
                                    whole_word_matches += title_lower.count(word)  # Count all occurrences
                                elif word in title_lower:
                                    partial_word_matches += title_lower.count(word)
                            
                            # For single-word searches, prioritize whole word matches
                            if len(query_words) == 1:
                                if whole_word_matches > 0:
                                    metadata_score += min(0.5, whole_word_matches * 0.25)  # Direct count-based scoring for whole words
                                elif partial_word_matches > 0:
                                    metadata_score += min(0.25, partial_word_matches * 0.12)  # Higher for partial matches
                            else:
                                # For multi-word searches, prioritize whole word matches
                                if whole_word_matches > 0:
                                    metadata_score += min(0.5, (whole_word_matches / len(query_words)) * 0.5)
                                elif partial_word_matches > 0:
                                    metadata_score += min(0.25, (partial_word_matches / len(query_words)) * 0.12)  # Higher for partial
                    
                    # Artist matching (high priority) - only if no phrase match
                    if track.artist and phrase_matches == 0:
                        artist_lower = track.artist.lower()
                        if query_words:
                            # Check for whole word matches first (much higher priority)
                            whole_word_matches = 0
                            partial_word_matches = 0
                            
                            for word in query_words:
                                # Check for whole word match (word boundaries)
                                import re
                                whole_word_pattern = r'\b' + re.escape(word) + r'\b'
                                if re.search(whole_word_pattern, artist_lower):
                                    whole_word_matches += 1
                                elif word in artist_lower:
                                    partial_word_matches += artist_lower.count(word)
                            
                            # Whole word matches get much higher score
                            if whole_word_matches > 0:
                                metadata_score += min(0.4, whole_word_matches * 0.4)
                            # Partial word matches get very low score
                            elif partial_word_matches > 0:
                                metadata_score += min(0.05, (partial_word_matches / len(query_words)) * 0.02)  # Much lower multiplier
                    
                    # Lyrics matching (fourth priority) - only if no phrase match
                    if track.lyrics and phrase_matches == 0:
                        lyrics_lower = track.lyrics.lower()
                        if query_words:
                            # Check for whole word matches first (much higher priority)
                            whole_word_matches = 0
                            partial_word_matches = 0
                            
                            for word in query_words:
                                # Check for whole word match (word boundaries)
                                import re
                                whole_word_pattern = r'\b' + re.escape(word) + r'\b'
                                if re.search(whole_word_pattern, lyrics_lower):
                                    whole_word_matches += lyrics_lower.count(word)  # Count all occurrences
                                elif word in lyrics_lower:
                                    partial_word_matches += lyrics_lower.count(word)
                            
                            # For single-word searches, prioritize whole word matches
                            if len(query_words) == 1:
                                if whole_word_matches > 0:
                                    metadata_score += min(0.15, whole_word_matches * 0.05)  # Much lower than artist
                                elif partial_word_matches > 0:
                                    metadata_score += min(0.05, partial_word_matches * 0.02)  # Much lower than artist partial
                            else:
                                # For multi-word searches, prioritize whole word matches
                                if whole_word_matches > 0:
                                    metadata_score += min(0.15, (whole_word_matches / len(query_words)) * 0.15)
                                elif partial_word_matches > 0:
                                    metadata_score += min(0.05, (partial_word_matches / len(query_words)) * 0.02)  # Much lower than artist partial
                    
                    # Album matching (second priority) - only if no phrase match
                    if track.album and phrase_matches == 0:
                        album_lower = track.album.lower()
                        if query_words:
                            # Check for whole word matches first (much higher priority)
                            whole_word_matches = 0
                            partial_word_matches = 0
                            
                            for word in query_words:
                                # Check for whole word match (word boundaries)
                                import re
                                whole_word_pattern = r'\b' + re.escape(word) + r'\b'
                                if re.search(whole_word_pattern, album_lower):
                                    whole_word_matches += album_lower.count(word)  # Count all occurrences
                                elif word in album_lower:
                                    partial_word_matches += album_lower.count(word)
                            
                            # For single-word searches, prioritize whole word matches
                            if len(query_words) == 1:
                                if whole_word_matches > 0:
                                    metadata_score += min(0.45, whole_word_matches * 0.22)  # Between title and artist
                                elif partial_word_matches > 0:
                                    metadata_score += min(0.20, partial_word_matches * 0.10)  # Between title and artist partial
                            else:
                                # For multi-word searches, prioritize whole word matches
                                if whole_word_matches > 0:
                                    metadata_score += min(0.45, (whole_word_matches / len(query_words)) * 0.45)
                                elif partial_word_matches > 0:
                                    metadata_score += min(0.20, (partial_word_matches / len(query_words)) * 0.10)  # Between title and artist partial
                    
                    # Genre matching (medium priority) - only if no phrase match
                    if track.genre and phrase_matches == 0:
                        genre_lower = track.genre.lower()
                        if query_words and all(word in genre_lower for word in query_words):
                            metadata_score += 0.4
                        elif query_words:
                            total_word_count = 0
                            for word in query_words:
                                total_word_count += genre_lower.count(word)
                            metadata_score += min(0.15, (total_word_count / len(query_words)) * 0.06)  # Cap per field and reduce multiplier
                    
                    # Description matching (low priority) - only if no phrase match
                    if track.semantic_description and phrase_matches == 0:
                        desc_lower = track.semantic_description.lower()
                        if query_words:
                            total_word_count = 0
                            for word in query_words:
                                total_word_count += desc_lower.count(word)
                            metadata_score += min(0.1, (total_word_count / len(query_words)) * 0.04)  # Cap per field and reduce multiplier
                    
                    # Year matching (boosted for exact matches)
                    if track.year and str(track.year) == query_lower:
                        metadata_score += 1.0  # Boosted for exact year matches
                    elif track.year and str(track.year) in query:
                        metadata_score += 0.4  # Boosted for partial year matches
                    
                    # Calculate semantic score using phrase-first approach
                    semantic_score = 0.0
                    
                    if track.multimodal_embedding and query_embedding is not None:
                        # Primary: Phrase embedding (treats "yellow pants" as one concept)
                        phrase_similarity = self.multimodal_embedding_service.similarity(query_embedding, track.multimodal_embedding)
                        
                        # Secondary: Individual word similarities (weighted lower)
                        word_similarities = []
                        if query_word_embeddings:
                            for word, word_embedding in query_word_embeddings:
                                word_sim = self.multimodal_embedding_service.similarity(word_embedding, track.multimodal_embedding)
                                word_similarities.append(word_sim)
                        
                        # Combine phrase (primary) with individual words (secondary)
                        if word_similarities:
                            # Phrase gets 85% weight, individual words get 15% weight (balanced phrase priority)
                            avg_word_similarity = sum(word_similarities) / len(word_similarities)
                            semantic_score = (phrase_similarity * 0.85) + (avg_word_similarity * 0.15)
                        else:
                            # Single word or no word embeddings, use phrase only
                            semantic_score = phrase_similarity
                        
                        # Apply lyrics penalty for tracks without lyrics
                        if not track.lyrics or not track.lyrics.strip():
                            # Reduce score by 10% for tracks without lyrics (more balanced)
                            semantic_score *= 0.9
                        
                        # No semantic penalties - let metadata scoring handle the field hierarchy naturally
                        
                    
                    # Create a true hierarchy: Exact metadata first for specific queries, then semantic
                    # This ensures exact matches (like years) take priority over semantic understanding
                    
                    if is_year_query and metadata_score > 0:
                        # For year queries, prioritize exact metadata matches
                        similarity_score = min(1.0, metadata_score)  # Clamp to 1.0
                        match_type = 'metadata'
                    elif semantic_score > 0:
                        # If we have semantic understanding, use it as the primary score
                        # Add a moderate metadata boost (max 0.12) to balance semantic dominance
                        metadata_boost = min(0.12, metadata_score * 0.12)  # Moderate boost
                        similarity_score = max(0.0, min(1.0, semantic_score + metadata_boost))
                        match_type = 'combined'
                    else:
                        # Fallback to pure metadata if no semantic understanding
                        similarity_score = max(0.0, min(1.0, metadata_score))
                        match_type = 'combined'
                
                # Create search result
                track_response = TrackResponse.from_track(track)
                result = SearchResult(
                    track=track_response,
                    similarity_score=min(1.0, max(0.0, similarity_score)),  # Clamp to [0, 1]
                    match_type=match_type,
                    query=query
                )
                results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Filter out 0% matches
            filtered_results = [r for r in results if r.similarity_score > 0]
            
            # Calculate pagination based on filtered results
            total_results = len(filtered_results)
            start_index = (page - 1) * limit
            end_index = start_index + limit
            
            # Return paginated results and total count
            paginated_results = filtered_results[start_index:end_index]
            return paginated_results, total_results
            
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
    
    
    
    
    async def search_by_embedding(self, embedding: List[float], embedding_type: str, 
                                limit: int = 20, page: int = 1, filters: Optional[TrackFilters] = None) -> tuple[List[SearchResult], int]:
        """Search using a pre-computed multimodal embedding."""
        try:
            table = get_tracks_table(self.db)
            df = table.to_pandas()
            
            if df.empty:
                return [], 0
            
            # Apply filters if provided
            if filters:
                df = self._apply_filters(df, filters)
            
            if df.empty:
                return [], 0
            
            # Calculate similarities using multimodal embeddings only
            results = []
            for _, row in df.iterrows():
                track_data = row.to_dict()
                track = dict_to_track(track_data)
                
                similarity = 0.0
                if embedding_type == 'multimodal' and track.multimodal_embedding:
                    similarity = self.multimodal_embedding_service.similarity(embedding, track.multimodal_embedding)
                
                if similarity > 0:
                    track_response = TrackResponse.from_track(track)
                    result = SearchResult(
                        track=track_response,
                        similarity_score=similarity,
                        match_type='multimodal',
                        query=""  # Embedding search doesn't have a text query
                    )
                    results.append(result)
            
            # Sort by similarity and return paginated results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Filter out 0% matches
            filtered_results = [r for r in results if r.similarity_score > 0]
            
            # Calculate pagination based on filtered results
            total_results = len(filtered_results)
            start_index = (page - 1) * limit
            end_index = start_index + limit
            
            # Return paginated results and total count
            paginated_results = filtered_results[start_index:end_index]
            return paginated_results, total_results
            
        except Exception as e:
            logger.error(f"Failed to search by embedding: {str(e)}")
            raise
    
