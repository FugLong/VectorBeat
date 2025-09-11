import React from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  MusicNote,
  Person,
  Category,
  Storage,
  CalendarToday,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { ApiService } from '../services/api';
import { DatabaseStats } from '../types';

const StatsPage: React.FC = () => {
  const { data: stats, isLoading, error } = useQuery<DatabaseStats>({
    queryKey: ['stats'],
    queryFn: ApiService.getStats,
    refetchInterval: 30000, // Refetch every 30 seconds
    retry: 1, // Only retry once
  });

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">
          Failed to load statistics: {error.message}
        </Alert>
      </Container>
    );
  }

  if (!stats) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="info">
          No statistics available.
        </Alert>
      </Container>
    );
  }

  const statCards = [
    {
      title: 'Total Tracks',
      value: stats.total_tracks.toLocaleString(),
      icon: <MusicNote sx={{ fontSize: 40 }} />,
      color: '#bb86fc',
    },
    {
      title: 'Unique Artists',
      value: stats.total_artists.toLocaleString(),
      icon: <Person sx={{ fontSize: 40 }} />,
      color: '#03dac6',
    },
    {
      title: 'Genres',
      value: stats.total_genres.toLocaleString(),
      icon: <Category sx={{ fontSize: 40 }} />,
      color: '#ff6b6b',
    },
    {
      title: 'Database Size',
      value: `${stats.database_size_mb?.toFixed(1) || 0} MB`,
      icon: <Storage sx={{ fontSize: 40 }} />,
      color: '#4ecdc4',
    },
  ];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        Database Statistics
      </Typography>
      <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 4 }}>
        Overview of your music collection and search database
      </Typography>

      {/* Main Stats Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {statCards.map((stat, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card
              sx={{
                height: '100%',
                background: `linear-gradient(135deg, ${stat.color}20, ${stat.color}05)`,
                border: `1px solid ${stat.color}30`,
              }}
            >
              <CardContent sx={{ textAlign: 'center', py: 3 }}>
                <Box
                  sx={{
                    color: stat.color,
                    mb: 2,
                    display: 'flex',
                    justifyContent: 'center',
                  }}
                >
                  {stat.icon}
                </Box>
                <Typography variant="h4" component="div" sx={{ fontWeight: 600, mb: 1 }}>
                  {stat.value}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {stat.title}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Additional Information */}
      <Grid container spacing={3}>
        {/* Year Range */}
        {stats?.year_range && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CalendarToday sx={{ mr: 1, color: 'primary.main' }} />
                  <Typography variant="h6">
                    Year Range
                  </Typography>
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
                  {stats.year_range.min} - {stats.year_range.max}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Spanning {stats.year_range.max - stats.year_range.min + 1} years
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Last Updated */}
        {stats?.last_updated && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Storage sx={{ mr: 1, color: 'secondary.main' }} />
                  <Typography variant="h6">
                    Last Updated
                  </Typography>
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  {new Date(stats.last_updated).toLocaleDateString()}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {new Date(stats.last_updated).toLocaleTimeString()}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Database Health */}
      <Box sx={{ mt: 4 }}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Database Health
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">
                  Tracks per Artist
                </Typography>
                <Typography variant="h6">
                  {stats && stats.total_artists > 0 
                    ? (stats.total_tracks / stats.total_artists).toFixed(1)
                    : '0'
                  }
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">
                  Tracks per Genre
                </Typography>
                <Typography variant="h6">
                  {stats && stats.total_genres > 0 
                    ? (stats.total_tracks / stats.total_genres).toFixed(1)
                    : '0'
                  }
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body2" color="text.secondary">
                  Avg. Size per Track
                </Typography>
                <Typography variant="h6">
                  {stats && stats.total_tracks > 0 && stats.database_size_mb
                    ? `${(stats.database_size_mb / stats.total_tracks).toFixed(2)} MB`
                    : '0 MB'
                  }
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default StatsPage;
