"""
Track data models for VectorBeat API.
Defines Pydantic models for track data and API requests/responses.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class TrackFilters(BaseModel):
    """Filters for track search."""
    genre: Optional[str] = None
    artist: Optional[str] = None
    year_min: Optional[int] = Field(None, ge=1900, le=2030)
    year_max: Optional[int] = Field(None, ge=1900, le=2030)
    tags: Optional[List[str]] = None
    
    @validator('year_max')
    def validate_year_range(cls, v, values):
        if v is not None and 'year_min' in values and values['year_min'] is not None:
            if v < values['year_min']:
                raise ValueError('year_max must be greater than or equal to year_min')
        return v


class SearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[TrackFilters] = None
    mode: Optional[str] = Field('metadata', pattern='^(semantic|metadata|combined)$')


class TrackBase(BaseModel):
    """Base track model with common fields."""
    title: str = Field(..., min_length=1, max_length=200)
    artist: str = Field(..., min_length=1, max_length=200)
    album: Optional[str] = Field(None, max_length=200)
    year: Optional[int] = Field(None, ge=0, le=2030)
    genre: Optional[str] = Field(None, max_length=100)
    tags: Optional[str] = Field(None, max_length=500)
    youtube_url: Optional[str] = Field(None, max_length=500)
    album_art_url: Optional[str] = Field(None, max_length=500)
    lyrics: Optional[str] = Field(None, max_length=10000)
    semantic_description: Optional[str] = Field(None, max_length=1000)
    mood: Optional[str] = Field(None, max_length=100)
    tempo: Optional[str] = Field(None, max_length=50)
    instruments: Optional[str] = Field(None, max_length=200)


class TrackCreate(TrackBase):
    """Model for creating a new track."""
    pass


class Track(TrackBase):
    """Complete track model with all fields."""
    track_id: str = Field(..., min_length=1)
    created_at: datetime
    updated_at: datetime
    
    # Embedding vectors (not included in API responses)
    text_embedding: Optional[List[float]] = Field(None, exclude=True)
    image_embedding: Optional[List[float]] = Field(None, exclude=True)
    audio_embedding: Optional[List[float]] = Field(None, exclude=True)
    semantic_embedding: Optional[List[float]] = Field(None, exclude=True)
    lyrics_embedding: Optional[List[float]] = Field(None, exclude=True)
    
    class Config:
        from_attributes = True


class TrackResponse(TrackBase):
    """Track model for API responses (excludes embeddings)."""
    track_id: str
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def from_track(cls, track: Track) -> "TrackResponse":
        """Create TrackResponse from Track model."""
        return cls(
            track_id=track.track_id,
            title=track.title,
            artist=track.artist,
            album=track.album,
            year=track.year,
            genre=track.genre,
            tags=track.tags,
            youtube_url=track.youtube_url,
            album_art_url=track.album_art_url,
            lyrics=track.lyrics,
            semantic_description=track.semantic_description,
            mood=track.mood,
            tempo=track.tempo,
            instruments=track.instruments,
            created_at=track.created_at,
            updated_at=track.updated_at
        )


class SearchResult(BaseModel):
    """Search result model with similarity score."""
    track: TrackResponse
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    match_type: str = Field(..., description="Type of match: text, image, audio, or combined")


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[SearchResult]
    total: int
    query: Optional[str] = None
    filters: Optional[TrackFilters] = None


class DatabaseStats(BaseModel):
    """Database statistics model."""
    total_tracks: int
    total_artists: int
    total_genres: int
    year_range: Optional[Dict[str, int]] = None
    last_updated: Optional[datetime] = None
    database_size_mb: Optional[float] = None
