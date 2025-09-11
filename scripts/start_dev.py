#!/usr/bin/env python3
"""
Development server startup script for VectorBeat.
Starts both backend and frontend servers for development.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting VectorBeat backend server...")
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    # Start backend
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "backend.main:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])
    
    return backend_process

def start_frontend():
    """Start the React frontend server."""
    print("🎨 Starting VectorBeat frontend server...")
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent.parent / "frontend"
    os.chdir(frontend_dir)
    
    # Start frontend
    frontend_process = subprocess.Popen([
        "npm", "start"
    ])
    
    return frontend_process

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Shutting down VectorBeat development servers...")
    sys.exit(0)

def main():
    """Main function to start both servers."""
    print("🎵 VectorBeat Development Environment")
    print("=" * 50)
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=start_backend)
        backend_thread.daemon = True
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Start frontend
        frontend_process = start_frontend()
        
        print("\n✅ VectorBeat development servers started!")
        print("📡 Backend API: http://localhost:8000")
        print("🌐 Frontend: http://localhost:3000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all servers")
        
        # Wait for frontend process
        frontend_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting servers: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
