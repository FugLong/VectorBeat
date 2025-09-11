#!/bin/bash

# VectorBeat Startup Script
# Starts backend and frontend

echo "🎵 Starting VectorBeat..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Check if node is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start backend in background
echo "🚀 Starting backend..."
cd backend
PYTHONPATH=/Users/fuglong/Documents/Repos/VectorBeat python main.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 8

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check logs/backend.log for errors."
    cat logs/backend.log | tail -10
    exit 1
fi

# Test backend health
echo "🔍 Testing backend health..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Backend is not responding. Check logs/backend.log"
    cat logs/backend.log | tail -10
    exit 1
fi

# Start frontend
echo "🎨 Starting frontend..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start frontend
BROWSER=none npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 10

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start. Check logs/frontend.log for errors."
    cat logs/frontend.log | tail -10
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Test frontend
echo "🔍 Testing frontend..."
if ! curl -s http://localhost:3000 | grep -q "VectorBeat"; then
    echo "❌ Frontend not serving content properly. Check logs/frontend.log"
    cat logs/frontend.log | tail -10
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "✅ VectorBeat is running!"
echo "📡 Backend: http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "🎵 VectorBeat is ready! Add a Spotify playlist to get started."
echo ""
echo "📋 Logs:"
echo "   Backend: tail -f logs/backend.log"
echo "   Frontend: tail -f logs/frontend.log"
echo ""
echo "Press Ctrl+C to stop"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ Servers stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running and show status
while true; do
    sleep 30
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend stopped unexpectedly"
        cleanup
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ Frontend stopped unexpectedly"
        cleanup
    fi
done