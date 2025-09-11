/**
 * TypeScript type definitions for VectorBeat frontend.
 */

export interface Track {
  track_id: string;
  title: string;
  artist: string;
  album?: string;
  year?: number;
  genre?: string;
  tags?: string;
  youtube_url?: string;
  album_art_url?: string;
  lyrics?: string;
  semantic_description?: string;
  mood?: string;
  tempo?: string;
  instruments?: string;
  created_at: string;
  updated_at: string;
}

export interface SearchResult {
  track: Track;
  similarity_score: number;
  match_type: 'text' | 'image' | 'audio' | 'combined';
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query?: string;
  filters?: TrackFilters;
}

export interface TrackFilters {
  genre?: string;
  artist?: string;
  year_min?: number;
  year_max?: number;
  tags?: string[];
}

export interface SearchQuery {
  query: string;
  filters?: TrackFilters;
  mode?: 'semantic' | 'metadata' | 'combined';
}

export interface DatabaseStats {
  total_tracks: number;
  total_artists: number;
  total_genres: number;
  year_range?: {
    min: number;
    max: number;
  };
  last_updated?: string;
  database_size_mb?: number;
}

export interface ApiError {
  detail: string;
  status_code: number;
}

export type SearchMode = 'semantic' | 'metadata' | 'combined';

export interface SearchState {
  query: string;
  filters: TrackFilters;
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  mode: SearchMode;
  page: number;
  hasMore: boolean;
}
