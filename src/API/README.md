# Sinon-RAG API

A Flask-based REST API for real-time chat and PDF document processing using Qdrant vector database for both document storage and chat history.

## Features

- **Real-time Chat**: Chat with AI using RAG (Retrieval-Augmented Generation)
- **PDF Upload & Processing**: Upload PDF documents with real-time progress tracking
- **Qdrant Integration**: Uses Qdrant for both document vectors and chat history storage
- **Semantic Search**: Advanced semantic search with time decay for chat history
- **Progress Tracking**: Real-time progress updates for document ingestion
- **Async Processing**: Non-blocking async operations for optimal performance

## Quick Start

### 1. Install Dependencies

```bash
cd sinon-RAG
pip install -r requirements.txt
```

### 2. Configure Environment

Ensure your `.env` file (in sinon-RAG root) has:

```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=prod1
```

### 3. Start the Server

```bash
python run_api.py
```

Or directly:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
GET /
```

### Chat Endpoints

#### Send Message
```
POST /api/chat/message
Content-Type: application/json

{
  "room_id": "room-123",
  "message": "What is the revenue growth?",
  "user_id": "user-456"  // optional
}
```

#### Get Chat History
```
GET /api/chat/history/{room_id}?limit=50
```

#### Search Messages
```
POST /api/chat/search/{room_id}
Content-Type: application/json

{
  "query": "revenue growth",
  "limit": 20
}
```

#### Clear Chat History
```
DELETE /api/chat/clear/{room_id}
```

### Upload Endpoints

#### Upload PDF Files
```
POST /api/upload/pdf
Content-Type: multipart/form-data

files: [file1.pdf, file2.pdf, ...]
```

#### Get Upload Status
```
GET /api/upload/status/{task_id}
```

#### Get Upload History
```
GET /api/upload/history?limit=50
```

### Status Endpoints

#### System Status
```
GET /api/status/system
```

#### Task Progress
```
GET /api/status/progress/{task_id}
```

#### List All Tasks
```
GET /api/status/tasks?status=processing&limit=20
```

## Example Usage

### 1. Upload a PDF
```javascript
const formData = new FormData();
formData.append('files', pdfFile);

const response = await fetch('http://localhost:5000/api/upload/pdf', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Task ID:', result.data.task_id);
```

### 2. Monitor Progress
```javascript
const taskId = 'ingest-123';
const response = await fetch(`http://localhost:5000/api/upload/status/${taskId}`);
const progress = await response.json();

console.log(`Progress: ${progress.data.progress}%`);
console.log(`Status: ${progress.data.status}`);
```

### 3. Send Chat Message
```javascript
const response = await fetch('http://localhost:5000/api/chat/message', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    room_id: 'room-123',
    message: 'What is the revenue growth rate?'
  })
});

const result = await response.json();
console.log('AI Response:', result.data.ai_response);
```

### 4. Get Chat History
```javascript
const response = await fetch('http://localhost:5000/api/chat/history/room-123?limit=20');
const history = await response.json();

history.data.messages.forEach(msg => {
  console.log(`${msg.role}: ${msg.text}`);
});
```

## Response Format

All API responses follow this format:

### Success Response
```json
{
  "status": "success",
  "message": "Operation completed successfully",
  "data": { ... },
  "timestamp": "2024-01-01T10:00:00Z"
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Error description",
  "error_code": "ERROR_CODE",
  "details": { ... },
  "timestamp": "2024-01-01T10:00:00Z"
}
```

### Progress Response
```json
{
  "status": "progress",
  "task_id": "ingest-123",
  "data": {
    "progress": 65,
    "status": "processing",
    "current_file": "document.pdf",
    "message": "Processing page 15 of 23..."
  },
  "timestamp": "2024-01-01T10:00:00Z"
}
```

## Configuration

### Environment Variables

- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 5000)
- `API_DEBUG`: Debug mode (default: True)

### File Upload Limits

- Maximum file size: 50MB per file
- Maximum files per upload: 10 files
- Total upload size limit: 500MB

## Architecture

### Chat System
- **Qdrant Storage**: Chat messages stored as vectors in Qdrant
- **Recency Buffer**: In-memory buffer for fast recent message access
- **SQLite Fallback**: Backup storage for service restarts
- **Semantic Search**: Time-decay weighted semantic search

### Ingestion System
- **Async Processing**: Background PDF processing with progress tracking
- **Integration**: Uses existing RAG indexer components
- **Progress Tracking**: Real-time progress updates with ETA calculation

### RAG Integration
- **Document Retrieval**: Semantic search in document collection
- **Chat Context**: Combines document context with chat history
- **Response Generation**: Uses existing LLM generation pipeline

## Development

### Project Structure
```
API/
├── app.py                 # Main Flask application
├── run_api.py            # Launch script
├── requirements.txt      # Dependencies
├── services/
│   ├── chat_service.py   # Chat & Qdrant integration
│   └── ingestion.py      # PDF processing service
├── routes/
│   ├── chat.py          # Chat endpoints
│   ├── upload.py        # Upload endpoints
│   └── status.py        # Status endpoints
└── utils/
    ├── response.py      # Response formatting
    └── file_handler.py  # File validation
```

### Adding New Endpoints

1. Create route handler in appropriate file
2. Import and register blueprint in `app.py`
3. Add proper error handling and validation
4. Use standardized response utilities

## Error Handling

The API includes comprehensive error handling:

- **Validation Errors**: Invalid input data
- **File Errors**: Upload validation and processing errors
- **Service Errors**: Qdrant, embedding, or LLM failures
- **System Errors**: General server errors

All errors return standardized error responses with appropriate HTTP status codes.

## Security Considerations

- **CORS**: Configured for cross-origin requests
- **File Validation**: Strict PDF validation and size limits
- **Input Sanitization**: All inputs validated and sanitized
- **Error Information**: Sensitive information not exposed in error messages

## Monitoring

### Health Checks
- Basic health endpoint at `/health`
- System status with component health at `/api/status/system`

### Logging
- Structured logging to console and file
- Request/response logging
- Error tracking with stack traces

### Metrics
- Task completion statistics
- System resource usage via status endpoints
- Chat activity metrics 