#!/bin/bash

# View Sinon-RAG API Server Logs

SCREEN_NAME="sinon-api"

echo "📋 Viewing Sinon-RAG API Server Logs..."

# Check if screen session exists
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "✅ Found screen session '$SCREEN_NAME'"
    echo "📝 Attaching to screen session (Ctrl+C to exit)..."
    echo ""
    
    # Attach to screen session
    screen -r "$SCREEN_NAME"
else
    echo "❌ Screen session '$SCREEN_NAME' not found"
    echo ""
    echo "📋 Available options:"
    echo "   Start API:     ./start_api.sh"
    echo "   Check status:  screen -list"
    echo "   View logs:     tail -f logs/api_*.log"
    echo ""
    echo "📁 Recent log files:"
    ls -la logs/api_*.log 2>/dev/null || echo "   No log files found"
fi 