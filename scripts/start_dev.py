#!/usr/bin/env python3
"""
VectorBeat Development Startup Script
Starts backend and frontend with hot reloading for development
"""

import os
import sys
import subprocess
import time
import signal
import threading
import requests
from pathlib import Path

# Global variables for process management
backend_process = None
frontend_process = None

def cleanup():
    """Cleanup function to stop all processes."""
    global backend_process, frontend_process
    print("")
    print("🛑 Stopping servers...")
    if backend_process:
        backend_process.terminate()
        backend_process.wait()
    if frontend_process:
        frontend_process.terminate()
        frontend_process.wait()
    print("✅ Servers stopped")
    sys.exit(0)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    cleanup()

def check_virtual_env():
    """Check if virtual environment exists."""
    venv_path = Path(__file__).parent.parent / "venv"
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run setup first.")
        sys.exit(1)

def check_node():
    """Check if node is available."""
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found. Please install Node.js first.")
        sys.exit(1)

def start_backend():
    """Start the FastAPI backend server with hot reload."""
    global backend_process
    
    print("🚀 Starting backend with hot reload...")
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Activate virtual environment and start backend
    if os.name == 'nt':  # Windows
        python_path = project_root / "venv" / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        python_path = project_root / "venv" / "bin" / "python"
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # Start backend with uvicorn and reload
    backend_process = subprocess.Popen([
        str(python_path), "-m", "uvicorn", 
        "backend.main:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ], env=env)
    
    return backend_process

def start_frontend():
    """Start the React frontend server."""
    global frontend_process
    
    print("🎨 Starting frontend...")
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent.parent / "frontend"
    os.chdir(frontend_dir)
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], check=True)
    
    # Start frontend
    env = os.environ.copy()
    env['BROWSER'] = 'none'
    frontend_process = subprocess.Popen([
        "npm", "start"
    ], env=env)
    
    return frontend_process

def wait_for_backend():
    """Wait for backend to start and check health."""
    print("⏳ Waiting for backend to start...")
    time.sleep(8)
    
    # Check if backend process is still running
    if backend_process.poll() is not None:
        print("❌ Backend failed to start.")
        sys.exit(1)
    
    # Test backend health
    print("🔍 Testing backend health...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend is healthy")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    
    print("❌ Backend is not responding.")
    sys.exit(1)

def wait_for_frontend():
    """Wait for frontend to start and check health."""
    print("⏳ Waiting for frontend to start...")
    time.sleep(10)
    
    # Check if frontend process is still running
    if frontend_process.poll() is not None:
        print("❌ Frontend failed to start.")
        cleanup()
    
    # Test frontend
    print("🔍 Testing frontend...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:3000", timeout=5)
            if "VectorBeat" in response.text:
                print("✅ Frontend is serving content")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    
    print("❌ Frontend not serving content properly.")
    cleanup()

def monitor_processes():
    """Monitor processes and restart if needed."""
    while True:
        time.sleep(30)
        
        # Check backend
        if backend_process and backend_process.poll() is not None:
            print("❌ Backend stopped unexpectedly")
            cleanup()
        
        # Check frontend
        if frontend_process and frontend_process.poll() is not None:
            print("❌ Frontend stopped unexpectedly")
            cleanup()

def main():
    """Main function to start both servers."""
    print("🎵 Starting VectorBeat Development Environment...")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Pre-flight checks
        check_virtual_env()
        check_node()
        
        # Start backend
        start_backend()
        wait_for_backend()
        
        # Start frontend
        start_frontend()
        wait_for_frontend()
        
        print("")
        print("✅ VectorBeat Development Environment is running!")
        print("📡 Backend: http://localhost:8000 (with hot reload)")
        print("🌐 Frontend: http://localhost:3000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("")
        print("🎵 VectorBeat is ready! Add a Spotify playlist to get started.")
        print("")
        print("📋 Logs:")
        print("   Backend: Check terminal output (hot reload enabled)")
        print("   Frontend: Check terminal output")
        print("")
        print("Press Ctrl+C to stop")
        
        # Monitor processes
        monitor_processes()
        
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"❌ Error starting servers: {e}")
        cleanup()

if __name__ == "__main__":
    main()
