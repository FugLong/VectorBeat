/**
 * API service for communicating with VectorBeat backend.
 */

import axios, { AxiosResponse } from 'axios';
import { Track, SearchResponse, SearchQuery, DatabaseStats, ApiError } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for playlist ingestion
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Response Error:', error);
    
    if (error.response) {
      // Server responded with error status
      const apiError: ApiError = {
        detail: error.response.data?.detail || 'An error occurred',
        status_code: error.response.status,
      };
      return Promise.reject(apiError);
    } else if (error.request) {
      // Request was made but no response received
      const apiError: ApiError = {
        detail: 'Network error - please check your connection',
        status_code: 0,
      };
      return Promise.reject(apiError);
    } else {
      // Something else happened
      const apiError: ApiError = {
        detail: error.message || 'An unexpected error occurred',
        status_code: 0,
      };
      return Promise.reject(apiError);
    }
  }
);

export class ApiService {
  /**
   * Search for tracks using semantic search
   */
  static async searchTracks(query: SearchQuery, limit: number = 20): Promise<SearchResponse> {
    try {
      const response: AxiosResponse<SearchResponse> = await api.post('/api/search', {
        ...query,
        limit,
      });
      return response.data;
    } catch (error) {
      console.error('Search tracks error:', error);
      throw error;
    }
  }

  /**
   * Get all tracks with pagination
   */
  static async getTracks(skip: number = 0, limit: number = 50): Promise<Track[]> {
    try {
      const response: AxiosResponse<Track[]> = await api.get('/api/tracks', {
        params: { skip, limit },
      });
      return response.data;
    } catch (error) {
      console.error('Get tracks error:', error);
      throw error;
    }
  }

  /**
   * Get a specific track by ID
   */
  static async getTrack(trackId: string): Promise<Track> {
    try {
      const response: AxiosResponse<Track> = await api.get(`/api/tracks/${trackId}`);
      return response.data;
    } catch (error) {
      console.error('Get track error:', error);
      throw error;
    }
  }

  /**
   * Create a new track
   */
  static async createTrack(track: Omit<Track, 'track_id' | 'created_at' | 'updated_at'>): Promise<Track> {
    try {
      const response: AxiosResponse<Track> = await api.post('/api/tracks', track);
      return response.data;
    } catch (error) {
      console.error('Create track error:', error);
      throw error;
    }
  }

  /**
   * Delete a track by ID
   */
  static async deleteTrack(trackId: string): Promise<void> {
    try {
      await api.delete(`/api/tracks/${trackId}`);
    } catch (error) {
      console.error('Delete track error:', error);
      throw error;
    }
  }

  /**
   * Get database statistics
   */
  static async getStats(): Promise<DatabaseStats> {
    try {
      const response: AxiosResponse<DatabaseStats> = await api.get('/api/stats');
      return response.data;
    } catch (error) {
      console.error('Get stats error:', error);
      throw error;
    }
  }

  /**
   * Health check
   */
  static async healthCheck(): Promise<{ status: string; service: string }> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  /**
   * Clear the entire database
   */
  static async clearDatabase(): Promise<{ message: string }> {
    try {
      const response: AxiosResponse<{ message: string }> = await api.delete('/api/database/clear');
      return response.data;
    } catch (error) {
      console.error('Clear database error:', error);
      throw error;
    }
  }

  /**
   * Ingest a YouTube playlist
   */
  static async ingestPlaylist(playlistUrl: string, clientId: string, clientSecret: string): Promise<{
    message: string;
    playlist_url: string;
    tracks_processed: number;
    tracks_added: number;
  }> {
    try {
      const response: AxiosResponse<{
        message: string;
        playlist_url: string;
        tracks_processed: number;
        tracks_added: number;
      }> = await api.post('/api/playlists/ingest', {
        playlist_url: playlistUrl,
        spotify_client_id: clientId,
        spotify_client_secret: clientSecret,
      });
      return response.data;
    } catch (error) {
      console.error('Ingest playlist error:', error);
      throw error;
    }
  }

  /**
   * Cancel the current ingestion process
   */
  static async cancelIngestion(): Promise<{ message: string }> {
    try {
      const response: AxiosResponse<{ message: string }> = await api.post('/api/ingestion/cancel');
      return response.data;
    } catch (error) {
      console.error('Cancel ingestion error:', error);
      throw error;
    }
  }
}

export default ApiService;
