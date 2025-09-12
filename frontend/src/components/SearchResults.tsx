import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Chip,
  Alert,
  CircularProgress,
  Fade,
  Pagination,
  IconButton,
} from '@mui/material';
import {
  ChevronLeft,
  ChevronRight,
} from '@mui/icons-material';
import { SearchResult, SearchMode } from '../types';
import TrackCard from './TrackCard';

interface SearchResultsProps {
  results: SearchResult[];
  loading: boolean;
  query: string;
  mode: SearchMode;
  page: number;
  totalPages: number;
  totalResults: number;
  hasMore: boolean;
  onPageChange: (page: number) => void;
  onNextPage: () => void;
  onPrevPage: () => void;
}

const SearchResults: React.FC<SearchResultsProps> = ({
  results,
  loading,
  query,
  mode,
  page,
  totalPages,
  totalResults,
  hasMore,
  onPageChange,
  onNextPage,
  onPrevPage,
}) => {
  // Results are already filtered on the backend (0% matches removed)
  const filteredResults = results;
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!filteredResults || filteredResults.length === 0) {
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
            Found {totalResults} result{totalResults !== 1 ? 's' : ''} for "{query}"
            {totalPages > 1 && ` (Page ${page} of ${totalPages})`}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              label={`Mode: ${mode.charAt(0).toUpperCase() + mode.slice(1)}`}
              size="small"
              color="primary"
              variant="outlined"
            />
            {filteredResults.length > 0 && (
              <Chip
                label={`Avg. Similarity: ${(filteredResults.reduce((sum, r) => sum + r.similarity_score, 0) / filteredResults.length * 100).toFixed(1)}%`}
                size="small"
                color="secondary"
                variant="outlined"
              />
            )}
          </Box>
        </Box>

        {/* Results Grid */}
        <Grid container spacing={3}>
          {filteredResults.map((result, index) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={result.track.track_id}>
              <TrackCard
                result={result}
                index={index}
              />
            </Grid>
          ))}
        </Grid>

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 2 }}>
            <IconButton
              onClick={onPrevPage}
              disabled={page <= 1}
              color="primary"
              size="large"
            >
              <ChevronLeft />
            </IconButton>
            
            <Pagination
              count={totalPages}
              page={page}
              onChange={(_, newPage) => onPageChange(newPage)}
              color="primary"
              size="large"
              showFirstButton
              showLastButton
            />
            
            <IconButton
              onClick={onNextPage}
              disabled={page >= totalPages}
              color="primary"
              size="large"
            >
              <ChevronRight />
            </IconButton>
          </Box>
        )}

      </Box>
    </Fade>
  );
};

export default SearchResults;
