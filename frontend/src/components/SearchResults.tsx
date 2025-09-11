import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Chip,
  Alert,
  CircularProgress,
  Fade,
} from '@mui/material';
import { SearchResult, SearchMode } from '../types';
import TrackCard from './TrackCard';

interface SearchResultsProps {
  results: SearchResult[];
  loading: boolean;
  query: string;
  mode: SearchMode;
}

const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  loading,
  query,
  mode,
}) => {
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!results || results.length === 0) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        No results found for "{query}". Try adjusting your search terms or filters.
      </Alert>
    );
  }

  return (
    <Fade in={!loading} timeout={300}>
      <Box>
        {/* Results Header */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Found {results.length} result{results.length !== 1 ? 's' : ''} for "{query}"
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              label={`Mode: ${mode.charAt(0).toUpperCase() + mode.slice(1)}`}
              size="small"
              color="primary"
              variant="outlined"
            />
            {results.length > 0 && (
              <Chip
                label={`Avg. Similarity: ${(results.reduce((sum, r) => sum + r.similarity_score, 0) / results.length * 100).toFixed(1)}%`}
                size="small"
                color="secondary"
                variant="outlined"
              />
            )}
          </Box>
        </Box>

        {/* Results Grid */}
        <Grid container spacing={3}>
          {results.map((result, index) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={result.track.track_id}>
              <TrackCard
                result={result}
                index={index}
              />
            </Grid>
          ))}
        </Grid>

        {/* Results Footer */}
        {results.length > 0 && (
          <Box sx={{ mt: 4, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              Results are ranked by semantic similarity to your query
            </Typography>
          </Box>
        )}
      </Box>
    </Fade>
  );
};

export default SearchResults;
