"""
Simple progress tracking for ingestion.
"""

# Global progress state
_progress = {
    "isActive": False,
    "current": 0,
    "total": 0,
    "message": "",
    "currentTrack": ""
}

def update_progress(**kwargs):
    """Update the current progress state."""
    global _progress
    _progress.update(kwargs)

def get_progress():
    """Get the current progress state."""
    return _progress.copy()

def reset_progress():
    """Reset progress to idle state."""
    global _progress
    _progress = {
        "isActive": False,
        "current": 0,
        "total": 0,
        "message": "",
        "currentTrack": ""
    }
