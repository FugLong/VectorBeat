#!/bin/bash

# VectorBeat Setup Script
# This script sets up the VectorBeat project for development and production use.

set -e  # Exit on any error

echo "🎵 Setting up VectorBeat Multi-Modal Music Search..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.9+ is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.9+ is required but not installed"
        exit 1
    fi
}

# Check if Node.js is installed
check_node() {
    print_status "Checking Node.js installation..."
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js $NODE_VERSION found"
    else
        print_error "Node.js 18+ is required but not installed"
        exit 1
    fi
}

# Check if FFmpeg is installed
check_ffmpeg() {
    print_status "Checking FFmpeg installation..."
    if command -v ffmpeg &> /dev/null; then
        print_success "FFmpeg found"
    else
        print_warning "FFmpeg not found - audio processing may not work properly"
        print_warning "Install FFmpeg: https://ffmpeg.org/download.html"
    fi
}

# Setup Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements in the correct order to avoid conflicts
    print_status "Installing core dependencies..."
    # Use PyTorch with proper Apple Silicon support
    pip install torch torchaudio
    
    print_status "Installing ML libraries with compatible versions..."
    # Install huggingface_hub first with compatible version
    pip install "huggingface_hub>=0.16.0,<0.20.0"
    # Install sentence-transformers with compatible huggingface_hub
    pip install sentence-transformers==2.2.2
    pip install transformers==4.35.2
    pip install openai-whisper==20231117
    pip install open_clip_torch==2.20.0
    
    print_status "Installing remaining dependencies..."
    pip install fastapi==0.104.1
    pip install uvicorn[standard]==0.24.0
    pip install pydantic==2.5.0
    pip install python-multipart==0.0.6
    pip install lancedb==0.3.0
    pip install numpy==1.24.3
    pip install pandas==2.0.3
    pip install Pillow==10.1.0
    pip install librosa==0.10.1
    pip install soundfile==0.12.1
    pip install pydub==0.25.1
    pip install spotipy==2.23.0
    pip install yt-dlp==2023.11.16
    pip install httpx==0.25.2
    pip install aiohttp==3.9.1
    pip install requests==2.31.0
    pip install python-dotenv==1.0.0
    pip install tqdm==4.66.1
    pip install click==8.1.7
    pip install rich==13.7.0
    pip install scipy==1.11.4
    pip install scikit-learn==1.3.2
    
    print_success "Python dependencies installed"
}

# Setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Clean up any existing node_modules to avoid conflicts
    if [ -d "node_modules" ]; then
        print_status "Cleaning up existing node_modules..."
        rm -rf node_modules package-lock.json
    fi
    
    # Install dependencies with legacy peer deps to resolve conflicts
    # Skip audit to avoid breaking changes
    npm install --legacy-peer-deps --no-audit
    print_success "Frontend dependencies installed"
    
    cd ..
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/{lancedb,cache,models}
    mkdir -p logs
    mkdir -p frontend/public
    
    print_success "Directories created"
}

# Setup environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp env.example .env
        print_success "Environment file created from template"
        print_warning "YouTube API key is optional - you can use demo data instead"
        print_warning "To use real YouTube playlists, edit .env file with your API key"
    else
        print_status "Environment file already exists"
    fi
}

# Download models
download_models() {
    print_status "Downloading embedding models..."
    
    source venv/bin/activate
    python scripts/setup_models.py --models all
    
    print_success "Models downloaded and cached"
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    source venv/bin/activate
    python -c "
import asyncio
import sys
sys.path.append('backend')
from database.connection import get_database
from database.schema import create_tables

async def init():
    db = get_database()
    await create_tables(db)
    print('Database initialized')

asyncio.run(init())
"
    
    print_success "Database initialized"
}

# Main setup function
main() {
    echo "🚀 Starting VectorBeat setup..."
    echo
    
    # Check prerequisites
    check_python
    check_node
    check_ffmpeg
    echo
    
    # Setup components
    create_directories
    setup_env
    setup_python
    setup_frontend
    echo
    
    # Download models
    print_status "Downloading embedding models..."
    source venv/bin/activate
    python scripts/setup_models.py
    echo
    
    # Initialize database
    init_database
    echo
    
    # Final instructions
    print_success "VectorBeat setup completed!"
    echo
    echo "🎉 You're ready to go!"
    echo
    echo "Next steps:"
    echo "1. Start the application: ./start.sh"
    echo "2. Open http://localhost:3000 in your browser"
    echo "3. Add a Spotify playlist to get started!"
    echo
    echo "How to use:"
    echo "- Paste a Spotify playlist URL (e.g., https://open.spotify.com/playlist/...)"
    echo "- The system will extract perfect metadata from Spotify"
    echo "- Search using natural language: 'rock songs from the 90s', 'jazz with saxophone'"
    echo
    echo "For more information, see README.md"
}

# Run main function
main "$@"
