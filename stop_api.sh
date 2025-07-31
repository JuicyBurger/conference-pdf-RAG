#!/bin/bash

# Stop Sinon-RAG API Server

SCREEN_NAME="sinon-api"

echo "🛑 Stopping Sinon-RAG API Server..."

# Check if screen session exists
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "📋 Found screen session '$SCREEN_NAME'"
    
    # Kill the screen session
    screen -S "$SCREEN_NAME" -X quit
    
    # Wait a moment
    sleep 1
    
    # Check if session was killed
    if screen -list | grep -q "$SCREEN_NAME"; then
        echo "⚠️  Failed to stop screen session, trying force kill..."
        screen -S "$SCREEN_NAME" -X kill
        sleep 1
    fi
    
    # Final check
    if screen -list | grep -q "$SCREEN_NAME"; then
        echo "❌ Failed to stop API server"
        echo "   Try manually: screen -S $SCREEN_NAME -X quit"
        exit 1
    else
        echo "✅ API server stopped successfully"
    fi
else
    echo "ℹ️  No screen session '$SCREEN_NAME' found"
    echo "   API server may not be running"
fi

# Also kill any remaining Python processes (backup)
echo "🧹 Cleaning up any remaining Python processes..."
pkill -f "python.*run_api.py" 2>/dev/null || true
pkill -f "python.*app.py" 2>/dev/null || true

echo "✅ Cleanup complete" 