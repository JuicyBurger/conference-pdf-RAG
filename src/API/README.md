# GraphRAG API Data Structure & Endpoint Document

This document organizes the data structures used by the application and provides a reference for the backend/Frontend teams to integrate with the API. Content can be adjusted based on actual needs.

## Table of Contents

- [Data Structures](#data-structures)
  - [Chat Room](#chat-room)
  - [Message](#message)
  - [User Settings](#user-settings-not-implemented)
- [API Endpoint Design](#api-endpoint-design)
  - [Chat-related APIs](#chat-related-apis)
  - [File-related APIs](#file-related-apis)
  - [Status APIs](#status-apis)
  - [Suggestions APIs](#suggestions-apis)
  - [Health Check](#health-check)
- [Notes](#notes)

## Data Structures

### Chat Room

Chat rooms represent conversation threads. If a request omits `room_id` (or passes "new"/empty), a new room is created automatically.

```javascript
{
  room_id: String,          // UUID
  room_title: String,       // Generated from first message
  createdAt: DateTime,      // ISO-8601
  updatedAt: DateTime,      // ISO-8601
  message_count: Number
}
```

### Message

Messages are individual dialogue items in a chat (user input or AI response).

```javascript
{
  msg_id: String,           // UUID
  role: String,             // "user" | "ai"
  content: String,
  timestamp: DateTime,      // ISO-8601
  files: Array<{            // Present when user uploads PDF with the message
    filename: String,
    uploaded_at: DateTime
  }>
}
```

### User Settings (Not implemented)

Will be implemented when a persistent user database is available.

```javascript
{
  userId: String,
  autoScroll: Boolean,
  notificationsEnabled: Boolean,
  language: String,
  theme: String,
  lastActiveProjectId: String
}
```

## API Endpoint Design

### Chat-related APIs

#### List Chat Rooms

```
GET /api/chat/rooms?limit=20
```

Success Response
```json
{
  "status": "success",
  "limit": 20,
  "total_count": 2,
  "data": [
    {
      "room_id": "a1b2c3...",
      "room_title": "文件分析討論",
      "createdAt": "2024-01-15T10:30:00Z",
      "updatedAt": "2024-01-15T10:35:00Z",
      "message_count": 3
    },
    {
      "room_id": "d4e5f6...",
      "room_title": "詢問技術問題",
      "createdAt": "2024-01-15T18:00:00Z",
      "updatedAt": "2024-01-15T18:23:00Z",
      "message_count": 1
    }
  ],
  "timestamp": "2024-01-15T10:36:00Z"
}
```

#### Get a Single Chat History Messages

```
GET /api/chat/histories/:room_id?limit=50
```

Success Response
```json
{
  "status": "success",
  "data": {
    "room_id": "a1b2c3...",
    "room_title": "文件分析討論",
    "createdAt": "2024-01-15T10:30:00Z",
    "updatedAt": "2024-01-15T10:35:00Z",
    "messages": [
      {
        "msg_id": "u-1642226",
        "role": "user",
        "content": "請總結關鍵重點",
        "timestamp": "2024-01-15T10:30:00Z",
        "files": [{ "filename": "report.pdf", "uploaded_at": "2024-01-15T10:29:50Z" }]
      },
      {
        "msg_id": "a-1642227",
        "role": "ai",
        "content": "以下是重點摘要...",
        "timestamp": "2024-01-15T10:31:00Z",
        "files": []
      }
    ]
  },
  "timestamp": "2024-01-15T10:36:00Z"
}
```

#### Send Message - Get AI Reply

```
POST /api/chat/message
```

Content-Type: multipart/form-data
```
content: String (Required)       // Message content
room_id: String|null (Optional)  // Omit or "new" to auto-create
user_id: String (Optional)
file: File (Optional)            // Single PDF
```

OR JSON
```json
{
  "room_id": "a1b2c3...", // Optional
  "content": "請幫我重點整理"
}
```

Success Response
```json
{
  "status": "success",
  "data": {
    "room_id": "a1b2c3...",
    "room_title": "文件分析討論",
    "createdAt": "2024-01-15T10:30:00Z",
    "updatedAt": "2024-01-15T10:31:05Z",
    "messages": [
      {
        "msg_id": "u-1642226",
        "role": "user",
        "content": "請幫我重點整理",
        "timestamp": "2024-01-15T10:31:00Z",
        "files": [{ "filename": "report.pdf", "uploaded_at": "2024-01-15T10:31:00Z" }]
      },
      {
        "msg_id": "a-1642227",
        "role": "ai",
        "content": "這份文件的重點包括...",
        "timestamp": "2024-01-15T10:31:05Z",
        "files": []
      }
    ]
  }
}
```

#### Search Messages

```
POST /api/chat/search/:room_id
Content-Type: application/json
```

Request Body
```json
{ "query": "摘要", "limit": 20 }
```

Success Response
```json
{
  "status": "success",
  "message": "Success",
  "data": {
    "room_id": "a1b2c3...",
    "query": "摘要",
    "results": [
      {
        "message": { "msg_id": "...", "role": "ai", "content": "以下是重點摘要...", "timestamp": "...", "files": [] },
        "score": 0.92,
        "semantic_score": 0.88,
        "time_score": 0.97
      }
    ],
    "result_count": 1
  },
  "timestamp": "2024-01-15T10:40:00Z"
}
```

#### Delete Single Chat Room

```
DELETE /api/chat/rooms/:room_id
```

Success Response
```json
{
  "status": "success",
  "data": {
    "room_id": "a1b2c3...",
    "room_title": "文件分析討論",
    "deleted": true
  },
  "timestamp": "2024-01-15T10:45:00Z"
}
```

### File-related APIs

#### Upload PDF Files (Training Ingestion)

```
POST /api/upload/pdf
Content-Type: multipart/form-data
```

Form fields
```
files[]: File (Required)         // 1-10 PDFs, each <= 50MB
training_room_id: String (Opt)
```

Success Response (202 Accepted)
```json
{
  "status": "success",
  "message": "PDF upload started successfully",
  "data": {
    "task_id": "ingest-123",
    "status": {
      "task_id": "ingest-123",
      "status": "pending",
      "progress": 0,
      "current_file": null,
      "files_processed": 0,
      "total_files": 2,
      "file_names": ["fileA.pdf", "fileB.pdf"],
      "message": "Initializing...",
      "chunks_indexed": 0
    },
    "files_uploaded": 2,
    "file_names": ["fileA.pdf", "fileB.pdf"]
  },
  "timestamp": "2024-01-15T11:00:00Z"
}
```

#### Get Upload Status

```
GET /api/upload/status/:task_id
```

Progress Response
```json
{
  "status": "progress",
  "task_id": "ingest-123",
  "data": {
    "task_id": "ingest-123",
    "status": "processing",
    "progress": 65,
    "current_file": "fileB.pdf",
    "files_processed": 1,
    "total_files": 2,
    "file_names": ["fileA.pdf", "fileB.pdf"],
    "message": "Completed fileA.pdf (120 chunks)",
    "chunks_indexed": 120
  },
  "timestamp": "2024-01-15T11:05:00Z"
}
```

#### Get Upload History

```
GET /api/upload/history?limit=50
```

Success Response
```json
{
  "status": "success",
  "message": "Retrieved 2 ingestion tasks",
  "data": {
    "tasks": [
      { "task_id": "ingest-123", "status": "completed", "progress": 100, "file_names": ["fileA.pdf", "fileB.pdf"] },
      { "task_id": "ingest-456", "status": "processing", "progress": 30, "file_names": ["fileC.pdf"] }
    ],
    "task_count": 2
  },
  "timestamp": "2024-01-15T11:10:00Z"
}
```

### Status APIs

#### System Status

```
GET /api/status/system
```

Success Response
```json
{
  "status": "success",
  "message": "System status retrieved successfully",
  "data": {
    "service": "sinon-rag-api",
    "status": "healthy",
    "qdrant": {
      "url": "http://localhost:6333",
      "status": "healthy",
      "document_collection": "prod1",
      "document_count": 1024,
      "chat_collection": "chat_messages",
      "chat_collection_status": "healthy",
      "chat_message_count": 50
    },
    "ingestion": {
      "active_tasks": 0,
      "recent_tasks": 2,
      "task_details": []
    },
    "chat": {
      "buffer_rooms": 1,
      "service_status": "running"
    }
  },
  "timestamp": "2024-01-15T11:15:00Z"
}
```

#### Task Progress (Generic)

```
GET /api/status/progress/:task_id
```

Success Response
```json
{
  "status": "success",
  "message": "Progress retrieved successfully",
  "data": {
    "task_id": "ingest-123",
    "status": "processing",
    "progress": 65,
    "current_file": "fileB.pdf",
    "files_processed": 1,
    "total_files": 2,
    "file_names": ["fileA.pdf", "fileB.pdf"],
    "message": "Completed fileA.pdf (120 chunks)",
    "chunks_indexed": 120
  },
  "timestamp": "2024-01-15T11:05:00Z"
}
```

#### List All Tasks

```
GET /api/status/tasks?status=processing&limit=20
```

Success Response
```json
{
  "status": "success",
  "message": "Retrieved 2 tasks",
  "data": {
    "tasks": [
      { "task_id": "ingest-123", "status": "completed" },
      { "task_id": "ingest-456", "status": "processing" }
    ],
    "statistics": {
      "total": 2,
      "pending": 0,
      "processing": 1,
      "completed": 1,
      "failed": 0
    },
    "filters": { "status": "processing", "limit": 20 }
  },
  "timestamp": "2024-01-15T11:20:00Z"
}
```

#### Component Health

```
GET /api/status/health
```

Success Response
```json
{
  "status": "success",
  "message": "Health status",
  "data": {
    "status": "healthy",
    "qdrant": true,
    "neo4j": true
  },
  "timestamp": "2024-01-15T11:25:00Z"
}
```

### Suggestions APIs

#### Get Question Suggestions

```
GET /api/v1/suggestions?room_id=room-123&k=5
GET /api/v1/suggestions?doc_id=document_123&k=5
GET /api/v1/suggestions?k=5               // Random when no room_id/doc_id
```

Rules: provide either `room_id` OR `doc_id`. `k` must be between 1 and 20.

Success Response
```json
{
  "status": "success",
  "message": "Success",
  "data": [
    { "id": "sug-1", "text": "該文件的主要結論是什麼？" },
    { "id": "sug-2", "text": "有無提及關鍵風險？" }
  ],
  "timestamp": "2024-01-15T11:30:00Z"
}
```

#### Generate Question Suggestions

```
POST /api/v1/suggestions
Content-Type: application/json | application/x-www-form-urlencoded
```

Request Body
```json
{
  "room_id": "room-123",          // OR use doc_id instead
  "num_questions": 8,
  "use_lightweight": true,
  "auto_init_collection": true
}
```

Success Response (room-based)
```json
{
  "status": "success",
  "message": "Success",
  "data": {
    "room_id": "room-123",
    "success": true,
    "num_questions": 8,
    "message": "Successfully generated 8 suggestions based on conversation context"
  },
  "timestamp": "2024-01-15T11:35:00Z"
}
```

Success Response (document-based)
```json
{
  "status": "success",
  "message": "Success",
  "data": {
    "doc_id": "document_123",
    "success": true,
    "num_questions": 8,
    "use_lightweight": true,
    "message": "Successfully generated 8 suggestions for document"
  },
  "timestamp": "2024-01-15T11:35:00Z"
}
```

#### List Documents with Suggestions

```
GET /api/v1/suggestions/docs
```

Success Response
```json
{
  "status": "success",
  "message": "Success",
  "data": {
    "doc_ids": ["doc1", "doc2", "doc3"],
    "total": 3
  },
  "timestamp": "2024-01-15T11:40:00Z"
}
```

### Health Check

```
GET /
GET /health
```

Success Response (root/health)
```json
{
  "status": "healthy",
  "service": "sinon-rag-api",
  "version": "1.0.0",
  "endpoints": [
    "/api/chat/message",
    "/api/chat/history/<room_id>",
    "/api/upload/pdf",
    "/api/upload/status/<task_id>",
    "/api/status/progress/<task_id>"
  ]
}
```

## Notes

1. All endpoints return standardized JSON envelopes. Validation errors use `error_code="VALIDATION_ERROR"`; not-found cases use `error_code="NOT_FOUND"`.
2. For chat with PDF, the API summarizes/extracts the content for immediate use and indexes it to Qdrant with `scope=chat` and `room_id` for room-scoped retrieval.
3. File constraints: max 10 files, 50MB per file, total request size up to 500MB.
4. Consider enabling streaming and/or SSE for enhanced UX (not included here).
5. CORS is enabled; sensitive deployments should add authentication/authorization.