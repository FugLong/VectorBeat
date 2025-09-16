"""
VectorBeat FastAPI Backend
Main application entry point for the multi-modal music search API.
"""

import os
import logging
import json
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Query, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.models.track import Track, TrackCreate, TrackResponse, SearchQuery, SearchResponse
from backend.services.search_service import SearchService
from backend.services.track_service import TrackService
from backend.database.connection import get_database
from backend.database.schema import create_tables

# WebSocket connection manager for progress tracking
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.ingestion_progress: Dict[str, Dict] = {}
        self.cancellation_flags: Dict[str, bool] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_progress(self, progress_data: Dict):
        """Send progress update to all connected clients"""
        print(f"DEBUG: Sending progress to {len(self.active_connections)} connections: {progress_data}")
        if self.active_connections:
            message = json.dumps(progress_data)
            # Send to all connected clients
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(message)
                    print(f"DEBUG: Progress sent successfully to connection")
                except Exception as e:
                    print(f"DEBUG: Failed to send progress to connection: {e}")
                    # Remove disconnected clients
                    self.active_connections.remove(connection)
        else:
            print("DEBUG: No active connections to send progress to")
    
    def set_cancellation_flag(self, task_id: str, cancelled: bool = True):
        """Set cancellation flag for a specific task"""
        self.cancellation_flags[task_id] = cancelled
    
    def is_cancelled(self, task_id: str) -> bool:
        """Check if a task should be cancelled"""
        return self.cancellation_flags.get(task_id, False)
    
    def clear_cancellation_flags(self):
        """Clear all cancellation flags."""
        self.cancellation_flags.clear()

    def update_progress(self, task_id: str, current: int, total: int, status: str, message: str = ""):
        """Update progress for a specific task"""
        self.ingestion_progress[task_id] = {
            "current": current,
            "total": total,
            "status": status,
            "message": message,
            "percentage": int((current / total) * 100) if total > 0 else 0
        }

# Global connection manager
manager = ConnectionManager()

# No progress tracking needed - simple synchronous API

# Configure logging
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / "vectorbeat.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown tasks."""
    # Startup
    logger.info("Starting VectorBeat backend...")
    
    # Ensure data directories exist
    os.makedirs("data/lancedb", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize database
    db = get_database()
    await create_tables(db)
    logger.info("Database initialized")
    
    # Initialize services
    app.state.search_service = SearchService(db)
    app.state.track_service = TrackService(db)
    logger.info("Services initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VectorBeat backend...")


# Create FastAPI app
app = FastAPI(
    title="VectorBeat API",
    description="Multi-modal music search and discovery API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for album art
app.mount("/static", StaticFiles(directory="data/cache/album_art"), name="static")


# Dependency injection
def get_search_service() -> SearchService:
    """Get search service instance."""
    return app.state.search_service


def get_track_service() -> TrackService:
    """Get track service instance."""
    return app.state.track_service


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "vectorbeat-api"}


# Search endpoints
@app.post("/api/search", response_model=SearchResponse)
async def search_tracks(
    query: SearchQuery,
    limit: int = Query(default=20, ge=1, le=100),
    page: int = Query(default=1, ge=1),
    search_service: SearchService = Depends(get_search_service)
):
    """Search tracks using multi-modal semantic search."""
    try:
        results, total = await search_service.search(
            query=query.query,
            limit=limit,
            page=page,
            filters=query.filters,
            mode=query.mode or 'metadata'
        )
        return SearchResponse(results=results, total=total)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/tracks", response_model=List[TrackResponse])
async def list_tracks(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    track_service: TrackService = Depends(get_track_service)
):
    """List all tracks with pagination."""
    try:
        tracks = await track_service.list_tracks(skip=skip, limit=limit)
        return [TrackResponse.from_track(track) for track in tracks]
    except Exception as e:
        logger.error(f"List tracks error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tracks: {str(e)}")


@app.get("/api/tracks/{track_id}", response_model=TrackResponse)
async def get_track(
    track_id: str,
    track_service: TrackService = Depends(get_track_service)
):
    """Get a specific track by ID."""
    try:
        track = await track_service.get_track(track_id)
        if not track:
            raise HTTPException(status_code=404, detail="Track not found")
        return TrackResponse.from_track(track)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get track error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get track: {str(e)}")


@app.get("/api/debug/lyrics")
async def debug_lyrics(
    track_service: TrackService = Depends(get_track_service)
):
    """Debug endpoint to check if tracks have lyrics."""
    try:
        tracks = await track_service.list_tracks(skip=0, limit=10)
        lyrics_info = []
        
        for track in tracks:
            has_lyrics = bool(track.lyrics and track.lyrics.strip())
            lyrics_info.append({
                "track_id": track.track_id,
                "title": track.title,
                "artist": track.artist,
                "has_lyrics": has_lyrics,
                "lyrics_length": len(track.lyrics) if track.lyrics else 0,
                "lyrics_preview": track.lyrics[:100] + "..." if track.lyrics and len(track.lyrics) > 100 else track.lyrics
            })
        
        return {
            "total_tracks": len(tracks),
            "tracks_with_lyrics": sum(1 for t in lyrics_info if t["has_lyrics"]),
            "tracks": lyrics_info
        }
    except Exception as e:
        logger.error(f"Debug lyrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to debug lyrics: {str(e)}")


@app.post("/api/tracks", response_model=TrackResponse)
async def create_track(
    track: TrackCreate,
    track_service: TrackService = Depends(get_track_service)
):
    """Add a new track to the database."""
    try:
        created_track = await track_service.create_track(track)
        return TrackResponse.from_track(created_track)
    except Exception as e:
        logger.error(f"Create track error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create track: {str(e)}")


@app.delete("/api/tracks/{track_id}")
async def delete_track(
    track_id: str,
    track_service: TrackService = Depends(get_track_service)
):
    """Delete a track from the database."""
    try:
        success = await track_service.delete_track(track_id)
        if not success:
            raise HTTPException(status_code=404, detail="Track not found")
        return {"message": "Track deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete track error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete track: {str(e)}")


# Statistics endpoints
@app.get("/api/stats")
async def get_stats(track_service: TrackService = Depends(get_track_service)):
    """Get database statistics."""
    try:
        stats = await track_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Get stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/api/stats/duplicates")
async def get_duplicate_stats(
    track_service: TrackService = Depends(get_track_service)
):
    """Get duplicate statistics."""
    try:
        duplicate_stats = await track_service.get_duplicate_stats()
        return duplicate_stats
    except Exception as e:
        logger.error(f"Duplicate stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get duplicate stats: {str(e)}")


# Simple progress tracking
from backend.progress import update_progress, get_progress, reset_progress

@app.get("/api/ingestion/progress")
async def get_ingestion_progress():
    """Get current ingestion progress."""
    return get_progress()

@app.post("/api/ingestion/cancel")
async def cancel_ingestion():
    """Cancel the current ingestion process."""
    try:
        # Set cancellation flag for the current task
        manager.set_cancellation_flag("current", True)
        
        # Reset progress to show cancellation
        update_progress(
            isActive=False,
            current=0,
            total=0,
            message="Ingestion cancelled by user",
            currentTrack=""
        )
        
        return {"message": "Ingestion cancelled successfully"}
    except Exception as e:
        logger.error(f"Failed to cancel ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel ingestion: {str(e)}")


# Database management endpoints
@app.delete("/api/database/clear")
async def clear_database(track_service: TrackService = Depends(get_track_service)):
    """Clear all tracks from the database."""
    try:
        await track_service.clear_database()
        return {"message": "Database cleared successfully"}
    except Exception as e:
        logger.error(f"Clear database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/playlists/ingest")
async def ingest_playlist(
    request: dict,
    track_service: TrackService = Depends(get_track_service)
):
    """Ingest a Spotify playlist into the database with progress tracking."""
    try:
        # Extract playlist URL and Spotify credentials from request body
        playlist_url = request.get("playlist_url")
        spotify_client_id = request.get("spotify_client_id")
        spotify_client_secret = request.get("spotify_client_secret")
        
        if not playlist_url:
            raise HTTPException(status_code=400, detail="playlist_url is required")
        if not spotify_client_id or not spotify_client_secret:
            raise HTTPException(status_code=400, detail="spotify_client_id and spotify_client_secret are required")
        
        # Initialize progress tracking and reset cancellation flag
        reset_progress()
        manager.clear_cancellation_flags()  # Clear all cancellation flags for new ingestion
        
        # Set initial progress state
        update_progress(
            isActive=False,  # Start as inactive to clear any stale data
            current=0,
            total=0,
            message="Preparing ingestion...",
            currentTrack=""
        )
        
        # Import here to avoid circular imports
        from backend.scripts.spotify_ingestion import ingest_spotify_playlist as ingest_func
        
        # Run ingestion in background task to allow progress updates
        import asyncio
        import threading
        
        def run_ingestion():
            try:
                # Run the async function in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    ingest_func(playlist_url, track_service, manager, "current", spotify_client_id, spotify_client_secret)
                )
                # Mark as completed
                update_progress(
                    isActive=False,
                    current=result.get("tracks_processed", 0),
                    total=result.get("tracks_processed", 0),
                    message="Ingestion completed!",
                    currentTrack=""
                )
                loop.close()
            except Exception as e:
                logger.error(f"Ingestion error: {str(e)}")
                update_progress(
                    isActive=False,
                    current=0,
                    total=0,
                    message=f"Ingestion failed: {str(e)}",
                    currentTrack=""
                )
        
        # Start ingestion in background thread
        thread = threading.Thread(target=run_ingestion)
        thread.daemon = True
        thread.start()
        
        # Return immediately so frontend can start polling
        return {
            "message": "Ingestion started",
            "playlist_url": playlist_url,
            "tracks_processed": 0,
            "tracks_added": 0,
            "tracks_skipped": 0
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Playlist ingestion error: {str(e)}")
        # Simple error handling - no progress tracking needed
        raise HTTPException(status_code=500, detail=f"Failed to ingest playlist: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True
    )
