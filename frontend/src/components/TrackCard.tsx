import React, { useState } from 'react';
import {
  Card,
  CardMedia,
  CardContent,
  CardActions,
  Typography,
  IconButton,
  Chip,
  Box,
  Tooltip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Grid,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  OpenInNew,
  Info,
  MusicNote,
  Image,
  Audiotrack,
  Psychology,
} from '@mui/icons-material';
import { SearchResult } from '../types';

interface TrackCardProps {
  result: SearchResult;
  index: number;
}

const TrackCard: React.FC<TrackCardProps> = ({ result, index }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const { track, similarity_score, match_type } = result;

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
    // TODO: Implement actual audio playback
  };

  const handleOpenYouTube = () => {
    if (track.youtube_url) {
      window.open(track.youtube_url, '_blank');
    }
  };

  const getMatchTypeIcon = (type: string) => {
    switch (type) {
      case 'text':
        return <Psychology />;
      case 'image':
        return <Image />;
      case 'audio':
        return <Audiotrack />;
      case 'combined':
        return <MusicNote />;
      default:
        return <MusicNote />;
    }
  };

  const getMatchTypeColor = (type: string) => {
    switch (type) {
      case 'text':
        return 'primary';
      case 'image':
        return 'secondary';
      case 'audio':
        return 'success';
      case 'combined':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <>
      <Card
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: 6,
          },
        }}
      >
        {/* Album Art */}
        <CardMedia
          component="img"
          height="200"
          image={
            track.album_art_url 
              ? (track.album_art_url.startsWith('http') 
                  ? track.album_art_url 
                  : `http://localhost:8000/static/${track.album_art_url.replace('data/cache/album_art/', '')}`)
              : '/placeholder-album.png'
          }
          alt={`${track.title} by ${track.artist}`}
          sx={{
            objectFit: 'cover',
            backgroundColor: 'background.paper',
          }}
          onError={(e) => {
            // Fallback to placeholder if image fails to load
            e.currentTarget.src = '/placeholder-album.png';
          }}
        />

        {/* Similarity Score */}
        <Box sx={{ position: 'relative' }}>
          <LinearProgress
            variant="determinate"
            value={similarity_score * 100}
            sx={{
              height: 4,
              backgroundColor: 'rgba(255,255,255,0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: similarity_score > 0.7 ? '#4caf50' : similarity_score > 0.4 ? '#ff9800' : '#f44336',
              },
            }}
          />
          <Typography
            variant="caption"
            sx={{
              position: 'absolute',
              right: 8,
              top: -20,
              backgroundColor: 'background.paper',
              px: 1,
              borderRadius: 1,
            }}
          >
            {Math.round(similarity_score * 100)}%
          </Typography>
        </Box>

        {/* Content */}
        <CardContent sx={{ flexGrow: 1, pb: 1 }}>
          <Typography
            variant="h6"
            component="h3"
            sx={{
              fontSize: '1rem',
              fontWeight: 600,
              lineHeight: 1.2,
              mb: 1,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {track.title}
          </Typography>

          <Typography
            variant="body2"
            color="text.secondary"
            sx={{
              mb: 1,
              display: '-webkit-box',
              WebkitLineClamp: 1,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {track.artist}
          </Typography>

          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
            {track.year && (
              <Chip
                label={track.year}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem', height: 20 }}
              />
            )}
            {track.genre && (
              <Chip
                label={track.genre}
                size="small"
                variant="outlined"
                sx={{ fontSize: '0.7rem', height: 20 }}
              />
            )}
            <Chip
              icon={getMatchTypeIcon(match_type)}
              label={match_type}
              size="small"
              color={getMatchTypeColor(match_type)}
              variant="outlined"
              sx={{ fontSize: '0.7rem', height: 20 }}
            />
          </Box>
        </CardContent>

        {/* Actions */}
        <CardActions sx={{ pt: 0, px: 2, pb: 2 }}>
          <Tooltip title={isPlaying ? 'Pause' : 'Play'}>
            <IconButton
              onClick={handlePlayPause}
              color="primary"
              sx={{ mr: 1 }}
            >
              {isPlaying ? <Pause /> : <PlayArrow />}
            </IconButton>
          </Tooltip>

          <Tooltip title="View Details">
            <IconButton
              onClick={() => setShowDetails(true)}
              color="inherit"
            >
              <Info />
            </IconButton>
          </Tooltip>

          {track.youtube_url && (
            <Tooltip title="Open in YouTube">
              <IconButton
                onClick={handleOpenYouTube}
                color="inherit"
                sx={{ ml: 'auto' }}
              >
                <OpenInNew />
              </IconButton>
            </Tooltip>
          )}
        </CardActions>
      </Card>

      {/* Details Dialog */}
      <Dialog
        open={showDetails}
        onClose={() => setShowDetails(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Typography variant="h6">
            {track.title} by {track.artist}
          </Typography>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Similarity Score: {Math.round(similarity_score * 100)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={similarity_score * 100}
              sx={{ mb: 2 }}
            />
          </Box>

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                Track Information
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Title:</strong> {track.title}
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Artist:</strong> {track.artist}
              </Typography>
              {track.album && track.album !== 'Unknown Album' && (
                <Typography variant="body2" paragraph>
                  <strong>Album:</strong> {track.album}
                </Typography>
              )}
              {track.year && track.year > 0 && (
                <Typography variant="body2" paragraph>
                  <strong>Year:</strong> {track.year}
                </Typography>
              )}
              {track.genre && track.genre !== 'Unknown' && (
                <Typography variant="body2" paragraph>
                  <strong>Genre:</strong> {track.genre}
                </Typography>
              )}
              {track.tags && track.tags !== '' && (
                <Typography variant="body2" paragraph>
                  <strong>Tags:</strong> {track.tags}
                </Typography>
              )}
              {track.youtube_url && (
                <Typography variant="body2" paragraph>
                  <strong>YouTube:</strong>{' '}
                  <Button
                    size="small"
                    onClick={handleOpenYouTube}
                    startIcon={<OpenInNew />}
                  >
                    Open
                  </Button>
                </Typography>
              )}
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                AI Analysis
              </Typography>
              {track.semantic_description && (
                <Typography variant="body2" paragraph>
                  <strong>Description:</strong> {track.semantic_description}
                </Typography>
              )}
              {track.mood && track.mood !== 'Unknown' && (
                <Typography variant="body2" paragraph>
                  <strong>Mood:</strong> {track.mood}
                </Typography>
              )}
              {track.tempo && track.tempo !== 'Unknown' && (
                <Typography variant="body2" paragraph>
                  <strong>Tempo:</strong> {track.tempo}
                </Typography>
              )}
              {track.instruments && track.instruments !== 'Unknown' && (
                <Typography variant="body2" paragraph>
                  <strong>Instruments:</strong> {track.instruments}
                </Typography>
              )}
            </Grid>
          </Grid>

          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Match Information
              </Typography>
              <Typography variant="body2" paragraph>
                <strong>Match Type:</strong> {match_type} | <strong>Similarity:</strong> {Math.round(similarity_score * 100)}%
              </Typography>
            </Grid>
          </Grid>

          {track.lyrics && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Lyrics
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  maxHeight: 200,
                  overflow: 'auto',
                  backgroundColor: 'background.paper',
                  p: 2,
                  borderRadius: 1,
                }}
              >
                {track.lyrics}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDetails(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default TrackCard;
