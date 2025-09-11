# VectorBeat: Multi-Modal Music Search

VectorBeat is a multi-modal music search system that lets you find music using natural language. Search by lyrics, mood, genre, or even describe what the album art looks like.

## 🎵 Features

- **Multi-Modal Search**: Search across text, audio, and image content
- **Natural Language Queries**: "smooth jazz from the 80s", "songs with blue album covers"
- **Spotify Integration**: Import playlists directly through the UI
- **Multiple Search Methods**: Traditional metadata, semantic vector search, or combined
- **Local Processing**: All models run locally, no cloud dependencies
- **Interactive Frontend**: React-based search interface with real-time updates
- **Dynamic Updates**: Add/remove songs with automatic embedding generation

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Spotify       │    │   Data Ingestion │    │   Embedding     │
│   Playlists     │───▶│   & Processing   │───▶│   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React         │    │   FastAPI        │    │   LanceDB       │
│   Frontend      │◀───│   Backend        │◀───│   Vector Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- macOS (currently only supported platform)
- Python 3.9+
- Node.js 18+
- FFmpeg (for audio processing)

### Setup & Run

```bash
# Clone and setup
git clone https://github.com/FugLong/VectorBeat.git
cd VectorBeat
chmod +x setup.sh
./setup.sh

# Start the app
./start.sh
```

### Add Your Music

1. **Configure Spotify**: Go to Settings in the UI and add your Spotify API credentials
2. **Import Playlist**: Click "Add Playlist" and paste a public Spotify playlist URL
3. **Wait for Processing**: The system will automatically download, process, and embed the music
4. **Start Searching**: Use the search interface to find your music!

## 📁 Project Structure

```
VectorBeat/
├── backend/           # FastAPI backend + ML services
├── frontend/          # React TypeScript UI
├── scripts/           # Setup and utility scripts
├── data/             # Vector database and cache
├── logs/             # Application logs
├── setup.sh          # Automated setup script
├── start.sh          # Quick start script
├── requirements.txt  # Python dependencies
└── .env.example      # Environment variables template
```

## 🔧 Configuration

### Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` with your settings. The Spotify API credentials are configured through the UI, but you can also set them in the `.env` file if preferred.

## 🎯 Usage

### Search Your Music

Use the web interface at `http://localhost:3000` with three search methods:

**Traditional Search**: Search by metadata (artist, album, title)
- "The Beatles"
- "Abbey Road"

**Semantic Search**: Natural language queries using AI embeddings
- "jazz songs from the 80s"
- "songs about love with piano"
- "upbeat electronic music"
- "songs with blue album covers"

**Combined Search**: Mix traditional and semantic search for best results

### Search Examples

- "smooth jazz from the 80s"
- "songs about love with piano"
- "upbeat electronic music"
- "songs with blue album covers"

## 🧠 Models Used

- **Text Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Image Embeddings**: `openai/clip-vit-base-patch32`
- **Audio Embeddings**: `openai/whisper-small` + custom audio encoder

## 📊 Access

- **Web UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LanceDB](https://lancedb.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [OpenAI CLIP](https://github.com/openai/CLIP) for image embeddings
- [Whisper](https://github.com/openai/whisper) for audio processing

