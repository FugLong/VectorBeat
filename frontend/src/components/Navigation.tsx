import React, { useState, useEffect, useRef } from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  Box, 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions, 
  TextField,
  Alert,
  Snackbar,
  LinearProgress,
  Chip
} from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import { MusicNote, Delete, PlaylistAdd, Settings } from '@mui/icons-material';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { ApiService } from '../services/api';

const Navigation: React.FC = () => {
  const location = useLocation();
  const queryClient = useQueryClient();
  
  // State for dialogs and notifications
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [playlistDialogOpen, setPlaylistDialogOpen] = useState(false);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [playlistUrl, setPlaylistUrl] = useState('');
  const [spotifyClientId, setSpotifyClientId] = useState(localStorage.getItem('spotify_client_id') || '');
  const [spotifyClientSecret, setSpotifyClientSecret] = useState(localStorage.getItem('spotify_client_secret') || '');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' as 'success' | 'error' });
  
  // Simple loading state for ingestion
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestionProgress, setIngestionProgress] = useState({
    current: 0,
    total: 0,
    message: "",
    currentTrack: ""
  });
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Poll for progress updates when ingesting
  useEffect(() => {
    if (isIngesting) {
      pollingIntervalRef.current = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:8000/api/ingestion/progress');
          if (response.ok) {
            const data = await response.json();
            setIngestionProgress({
              current: data.current || 0,
              total: data.total || 0,
              message: data.message || "",
              currentTrack: data.currentTrack || ""
            });
            
            // If ingestion is complete, stop polling and refresh data
            if (!data.isActive) {
              setIsIngesting(false);
              if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current);
                pollingIntervalRef.current = null;
              }
              
              // Refresh data when complete
              queryClient.invalidateQueries({ queryKey: ['tracks'] });
              queryClient.invalidateQueries({ queryKey: ['stats'] });
              
              // Show completion message and close dialog
              if (data.message && data.message.includes("completed")) {
                setSnackbar({ 
                  open: true, 
                  message: `Successfully processed ${data.current} tracks!`, 
                  severity: 'success' 
                });
                // Clear the URL and close the dialog
                setPlaylistUrl('');
                setPlaylistDialogOpen(false);
              }
            }
          }
        } catch (error) {
          console.error('Progress polling error:', error);
        }
      }, 1000); // Poll every second for real-time updates
    } else {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    }

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [isIngesting]);

  // Mutations
  const clearDatabaseMutation = useMutation({
    mutationFn: ApiService.clearDatabase,
    onSuccess: () => {
      setSnackbar({ open: true, message: 'Database cleared successfully!', severity: 'success' });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      queryClient.invalidateQueries({ queryKey: ['tracks'] });
    },
    onError: (error: any) => {
      setSnackbar({ open: true, message: `Failed to clear database: ${error.message}`, severity: 'error' });
    }
  });

  const ingestPlaylistMutation = useMutation({
    mutationFn: (data: {url: string, clientId: string, clientSecret: string}) => ApiService.ingestPlaylist(data.url, data.clientId, data.clientSecret),
    onSuccess: (response) => {
      console.log('Ingestion started:', response);
      // Don't set isIngesting to false here - let the polling handle it
      // Don't show success message here - let the progress polling handle it
      // Don't invalidate queries here - let the completion handler do it
    },
    onError: (error: any) => {
      console.error('Ingestion failed:', error);
      setIsIngesting(false);
      setSnackbar({ 
        open: true, 
        message: `Failed to ingest playlist: ${error.detail || error.message || 'Unknown error'}`, 
        severity: 'error' 
      });
    }
  });

  const handleClearDatabase = () => {
    clearDatabaseMutation.mutate();
    setClearDialogOpen(false);
  };

  const handleSaveCredentials = () => {
    localStorage.setItem('spotify_client_id', spotifyClientId);
    localStorage.setItem('spotify_client_secret', spotifyClientSecret);
    setSnackbar({ open: true, message: 'Spotify credentials saved!', severity: 'success' });
    setSettingsDialogOpen(false);
  };

  const handleIngestPlaylist = () => {
    if (playlistUrl.trim() && spotifyClientId.trim() && spotifyClientSecret.trim()) {
      setIsIngesting(true);
      ingestPlaylistMutation.mutate({
        url: playlistUrl.trim(),
        clientId: spotifyClientId.trim(),
        clientSecret: spotifyClientSecret.trim()
      });
    } else if (!spotifyClientId.trim() || !spotifyClientSecret.trim()) {
      setSnackbar({ open: true, message: 'Please set your Spotify credentials first', severity: 'error' });
      setSettingsDialogOpen(true);
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <>
      <AppBar position="static" sx={{ backgroundColor: 'primary.main' }}>
        <Toolbar>
          <MusicNote sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            VectorBeat
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              color="inherit"
              component={Link}
              to="/search"
              sx={{
                backgroundColor: location.pathname === '/search' || location.pathname === '/' ? 'rgba(255,255,255,0.1)' : 'transparent',
              }}
            >
              Search
            </Button>
            <Button
              color="inherit"
              component={Link}
              to="/stats"
              sx={{
                backgroundColor: location.pathname === '/stats' ? 'rgba(255,255,255,0.1)' : 'transparent',
              }}
            >
              Stats
            </Button>
            <Button
              color="inherit"
              startIcon={<Settings />}
              onClick={() => setSettingsDialogOpen(true)}
              sx={{ 
                ml: 2,
                backgroundColor: spotifyClientId && spotifyClientSecret ? 'rgba(30, 215, 96, 0.1)' : 'transparent',
                border: spotifyClientId && spotifyClientSecret ? '1px solid rgba(30, 215, 96, 0.3)' : '1px solid transparent'
              }}
            >
              Settings {spotifyClientId && spotifyClientSecret ? '✓' : ''}
            </Button>
            <Button
              color="inherit"
              startIcon={<PlaylistAdd />}
              onClick={() => setPlaylistDialogOpen(true)}
              sx={{ ml: 1 }}
            >
              Add Playlist
            </Button>
            <Button
              color="inherit"
              startIcon={<Delete />}
              onClick={() => setClearDialogOpen(true)}
              sx={{ color: 'error.light' }}
            >
              Clear DB
            </Button>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Clear Database Confirmation Dialog */}
      <Dialog open={clearDialogOpen} onClose={() => setClearDialogOpen(false)}>
        <DialogTitle>Clear Database</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to clear the entire database? This action cannot be undone.
            All tracks, embeddings, and search data will be permanently deleted.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleClearDatabase} 
            color="error" 
            variant="contained"
            disabled={clearDatabaseMutation.isPending}
          >
            {clearDatabaseMutation.isPending ? 'Clearing...' : 'Clear Database'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Settings Dialog */}
      <Dialog open={settingsDialogOpen} onClose={() => setSettingsDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Spotify Settings</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Spotify Client ID"
            placeholder="Your Spotify Client ID"
            fullWidth
            variant="outlined"
            value={spotifyClientId}
            onChange={(e) => setSpotifyClientId(e.target.value)}
            sx={{ mt: 2 }}
          />
          <TextField
            margin="dense"
            label="Spotify Client Secret"
            placeholder="Your Spotify Client Secret"
            fullWidth
            variant="outlined"
            value={spotifyClientSecret}
            onChange={(e) => setSpotifyClientSecret(e.target.value)}
            type="password"
            sx={{ mt: 2 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Get your Spotify API credentials from <a href="https://developer.spotify.com/dashboard" target="_blank" rel="noopener noreferrer">Spotify Developer Dashboard</a>.
            These will be saved locally and used for all playlist imports.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSaveCredentials} variant="contained">
            Save Credentials
          </Button>
        </DialogActions>
      </Dialog>

      {/* Add Playlist Dialog */}
      <Dialog open={playlistDialogOpen} onClose={() => setPlaylistDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Spotify Playlist</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Spotify Playlist URL"
            placeholder="https://open.spotify.com/playlist/..."
            fullWidth
            variant="outlined"
            value={playlistUrl}
            onChange={(e) => setPlaylistUrl(e.target.value)}
            disabled={isIngesting}
            sx={{ mt: 2 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Enter a Spotify playlist URL to ingest all tracks with perfect metadata.
            Make sure you've set your Spotify credentials in Settings first.
          </Typography>
          
          {/* Real-time Progress Display */}
          {isIngesting && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="primary" sx={{ mb: 1 }}>
                {ingestionProgress.message || "Processing playlist..."}
              </Typography>
              
              {ingestionProgress.total > 0 && (
                <Box sx={{ mb: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      {ingestionProgress.current}/{ingestionProgress.total} tracks
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {Math.round((ingestionProgress.current / ingestionProgress.total) * 100)}%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={(ingestionProgress.current / ingestionProgress.total) * 100}
                    sx={{ mb: 1 }}
                  />
                </Box>
              )}
              
              {ingestionProgress.currentTrack && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  Currently processing: {ingestionProgress.currentTrack}
                </Typography>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPlaylistDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleIngestPlaylist} 
            color="primary" 
            variant="contained"
            disabled={!playlistUrl.trim() || isIngesting}
          >
            {isIngesting ? 'Processing...' : 'Add Playlist'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
};

export default Navigation;
