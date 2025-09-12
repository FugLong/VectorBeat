import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Paper,
  TextField,
  Button,
  Grid,
  Typography,
  Chip,
  Slider,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList,
  Clear,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';

import { useSearch } from '../contexts/SearchContext';
import { ApiService } from '../services/api';
import { TrackFilters } from '../types';
import SearchResults from './SearchResults';

const SearchPage: React.FC = () => {
  const { state, dispatch } = useSearch();
  const [localQuery, setLocalQuery] = useState(state.query);
  const [localFilters, setLocalFilters] = useState<TrackFilters>(state.filters);
  const [showFilters, setShowFilters] = useState(false);

  // Search query - only enabled when there's actually a query
  const { data: searchData, isLoading, error } = useQuery({
    queryKey: ['search', state.query, state.filters, state.mode, state.page],
    queryFn: () => ApiService.searchTracks({
      query: state.query,
      filters: state.filters,
      mode: state.mode,
    }, 20, state.page),
    enabled: !!state.query && state.query.trim().length > 0,
    retry: false,
  });

  // Update results when search data changes
  useEffect(() => {
    if (searchData) {
      dispatch({ type: 'SET_RESULTS', payload: searchData.results });
      dispatch({ type: 'SET_LOADING', payload: false });
      
      // Calculate total pages based on total results
      const totalPages = Math.ceil((searchData.total || 0) / 20);
      dispatch({ type: 'SET_HAS_MORE', payload: state.page < totalPages });
    }
  }, [searchData, dispatch, state.page]);

  // Update loading state
  useEffect(() => {
    dispatch({ type: 'SET_LOADING', payload: isLoading });
  }, [isLoading, dispatch]);

  // Update error state
  useEffect(() => {
    dispatch({ type: 'SET_ERROR', payload: error?.message || null });
  }, [error, dispatch]);

  const handleSearch = () => {
    dispatch({ type: 'SET_QUERY', payload: localQuery });
    dispatch({ type: 'SET_FILTERS', payload: localFilters });
    dispatch({ type: 'CLEAR_RESULTS' });
    dispatch({ type: 'SET_PAGE', payload: 1 }); // Reset to first page
  };

  const handleClear = () => {
    setLocalQuery('');
    setLocalFilters({});
    dispatch({ type: 'RESET_SEARCH' });
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      handleSearch();
    }
  };

  const handleFilterChange = (key: keyof TrackFilters, value: any) => {
    setLocalFilters(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleModeChange = (mode: 'semantic' | 'metadata' | 'combined') => {
    dispatch({ type: 'SET_MODE', payload: mode });
  };

  const handlePageChange = (page: number) => {
    dispatch({ type: 'SET_PAGE', payload: page });
  };

  const handleNextPage = () => {
    if (state.hasMore) {
      dispatch({ type: 'SET_PAGE', payload: state.page + 1 });
    }
  };

  const handlePrevPage = () => {
    if (state.page > 1) {
      dispatch({ type: 'SET_PAGE', payload: state.page - 1 });
    }
  };

  // Calculate total pages and get total results
  const totalPages = searchData ? Math.ceil((searchData.total || 0) / 20) : 0;
  const totalResults = searchData?.total || 0;

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Multi-Modal Music Search
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 4 }}>
          Search across metadata, lyrics, audio, and album art using natural language
        </Typography>

        {/* Search Bar */}
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Search for music... (e.g., 'smooth jazz, male singer, blue album cover')"
              value={localQuery}
              onChange={(e) => setLocalQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              InputProps={{
                startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
              }}
            />
            <Button
              variant="contained"
              onClick={handleSearch}
              disabled={!localQuery.trim() || isLoading}
              sx={{ minWidth: 120 }}
            >
              {isLoading ? <CircularProgress size={20} /> : 'Search'}
            </Button>
            <Button
              variant="outlined"
              onClick={() => setShowFilters(!showFilters)}
              startIcon={<FilterList />}
            >
              Filters
            </Button>
            <Button
              variant="outlined"
              onClick={handleClear}
              startIcon={<Clear />}
            >
              Clear
            </Button>
          </Box>

          {/* Search Mode Selection */}
          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            <Chip
              label="Metadata"
              onClick={() => handleModeChange('metadata')}
              color={state.mode === 'metadata' ? 'primary' : 'default'}
              variant={state.mode === 'metadata' ? 'filled' : 'outlined'}
            />
            <Chip
              label="Semantic"
              onClick={() => handleModeChange('semantic')}
              color={state.mode === 'semantic' ? 'primary' : 'default'}
              variant={state.mode === 'semantic' ? 'filled' : 'outlined'}
            />
            <Chip
              label="Combined"
              onClick={() => handleModeChange('combined')}
              color={state.mode === 'combined' ? 'primary' : 'default'}
              variant={state.mode === 'combined' ? 'filled' : 'outlined'}
            />
          </Box>

          {/* Filters */}
          {showFilters && (
            <Box sx={{ mt: 2, p: 2, backgroundColor: 'background.paper', borderRadius: 1 }}>
              <Typography variant="h6" gutterBottom>
                Filters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Genre"
                    value={localFilters.genre || ''}
                    onChange={(e) => handleFilterChange('genre', e.target.value)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Artist"
                    value={localFilters.artist || ''}
                    onChange={(e) => handleFilterChange('artist', e.target.value)}
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography gutterBottom>Year Range</Typography>
                  <Slider
                    value={[localFilters.year_min || 1900, localFilters.year_max || 2024]}
                    onChange={(_, newValue) => {
                      const [min, max] = newValue as number[];
                      handleFilterChange('year_min', min);
                      handleFilterChange('year_max', max);
                    }}
                    min={1900}
                    max={2024}
                    valueLabelDisplay="auto"
                    size="small"
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    fullWidth
                    label="Tags"
                    value={localFilters.tags?.join(', ') || ''}
                    onChange={(e) => handleFilterChange('tags', e.target.value.split(',').map(t => t.trim()))}
                    size="small"
                    placeholder="tag1, tag2, tag3"
                  />
                </Grid>
              </Grid>
            </Box>
          )}
        </Paper>

        {/* Error Display */}
        {state.error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {state.error}
          </Alert>
        )}

        {/* Search Results */}
        {state.query && (
          <SearchResults
            results={state.results}
            loading={state.loading}
            query={state.query}
            mode={state.mode}
            page={state.page}
            totalPages={totalPages}
            totalResults={totalResults}
            hasMore={state.hasMore}
            onPageChange={handlePageChange}
            onNextPage={handleNextPage}
            onPrevPage={handlePrevPage}
          />
        )}

        {/* Example Queries */}
        {!state.query && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              Try these example queries:
            </Typography>
            <Grid container spacing={2}>
              {[
                "smooth jazz from the 80s",
                "songs about love with piano",
                "upbeat electronic music",
                "songs with blue album covers",
                "acoustic guitar, male singer",
                "classical music with strings",
              ].map((example, index) => (
                <Grid item key={index}>
                  <Chip
                    label={example}
                    onClick={() => {
                      setLocalQuery(example);
                      dispatch({ type: 'SET_QUERY', payload: example });
                      dispatch({ type: 'CLEAR_RESULTS' });
                    }}
                    variant="outlined"
                    sx={{ cursor: 'pointer' }}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </Box>
    </Container>
  );
};

export default SearchPage;
