#!/bin/bash

# Start Sinon-RAG API Server in Screen Session

# Configuration
SCREEN_NAME="sinon-api"
API_DIR="/home/rtx4500ada/llm-project/sinon-RAG"
API_PORT="5000"
API_HOST="0.0.0.0"

echo "🚀 Starting Sinon-RAG API Server..."

# Check if screen session already exists
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "⚠️  Screen session '$SCREEN_NAME' already exists!"
    echo "   To view logs: screen -r $SCREEN_NAME"
    echo "   To stop: ./stop_api.sh"
    exit 1
fi

# Navigate to API directory
cd "$API_DIR"

# Create data directories
mkdir -p data/uploads
mkdir -p logs

# Start API server in screen session
screen -dmS "$SCREEN_NAME" bash -c "
    echo '🚀 Starting Sinon-RAG API Server...'
    echo '📁 Working directory: $(pwd)'
    echo '🌐 Server will be available at: http://$API_HOST:$API_PORT'
    echo ''
    echo '📋 Available endpoints:'
    echo '   GET  /health                    - Health check'
    echo '   POST /api/chat/message          - Send chat message'
    echo '   GET  /api/chat/history/{room}   - Get chat history'
    echo '   POST /api/upload/pdf            - Upload PDF files'
    echo '   GET  /api/upload/status/{task}  - Get upload progress'
    echo '   GET  /api/status/system         - System status'
    echo ''
    echo '🔧 Environment:'
    echo '   API_HOST: $API_HOST'
    echo '   API_PORT: $API_PORT'
    echo ''
    echo '📝 Logs will appear below:'
    echo '========================================'
    
    # Start the API server
    python -m src.API.run_api 2>&1 | tee logs/api_$(date +%Y%m%d_%H%M%S).log
"

# Wait a moment for screen to start
sleep 2

# Check if screen session was created successfully
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "✅ API server started successfully in screen session '$SCREEN_NAME'"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs:     screen -r $SCREEN_NAME"
    echo "   Detach:        Ctrl+A, then D"
    echo "   Stop server:   ./stop_api.sh"
    echo "   Health check:  curl http://$API_HOST:$API_PORT/health"
    echo ""
    echo "🌐 API will be available at: http://$API_HOST:$API_PORT"
else
    echo "❌ Failed to start API server"
    exit 1
fi 