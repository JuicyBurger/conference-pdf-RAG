"""
Chat Service with Qdrant Integration

Handles chat message storage, retrieval, and semantic search using Qdrant.
Implements recency buffer and time decay for optimal performance.
"""

import asyncio
import math
import sqlite3
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, VectorParams, Distance, PayloadSchemaType

# Import from our existing RAG system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.embedder import embed
from src.config import QDRANT_URL, QDRANT_API_KEY


class ChatBuffer:
    """Thread-safe recency buffer for chat messages"""
    
    def __init__(self, maxlen: int = 1024):
        self.buffers: Dict[str, deque] = {}
        self.lock = threading.Lock()
    
    def add_message(self, room_id: str, message: dict):
        """Add message to room buffer"""
        with self.lock:
            if room_id not in self.buffers:
                self.buffers[room_id] = deque(maxlen=1024)
            self.buffers[room_id].append(message)
    
    def get_recent_messages(self, room_id: str, k: int = 50) -> List[dict]:
        """Get k most recent messages for room"""
        with self.lock:
            if room_id not in self.buffers:
                return []
            messages = list(self.buffers[room_id])
            return messages[-k:]  # Last k messages in chronological order
    
    def clear_room(self, room_id: str):
        """Clear messages for a specific room"""
        with self.lock:
            if room_id in self.buffers:
                del self.buffers[room_id]


class ChatFallback:
    """SQLite fallback for when service restarts"""
    
    def __init__(self, db_path: str = "data/chat_fallback.db"):
        self.db_path = db_path
        self.initialized = False
        try:
            self.init_db()
        except Exception as e:
            print(f"⚠️ SQLite fallback disabled: {e}")
            self.initialized = False
    
    def init_db(self):
        """Initialize SQLite fallback database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                # Check if table exists and has correct schema
                cursor = conn.execute("PRAGMA table_info(messages)")
                columns = {row[1] for row in cursor.fetchall()}
                
                # If table doesn't exist or has wrong schema, recreate it
                if not columns or 'timestamp' not in columns:
                    conn.execute("DROP TABLE IF EXISTS messages")
                    conn.execute("""
                        CREATE TABLE messages (
                            msg_id TEXT PRIMARY KEY,
                            room_id TEXT,
                            role TEXT,
                            content TEXT,
                            timestamp TEXT,
                            files TEXT
                        )
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_room_timestamp ON messages(room_id, timestamp)")
                    print(f"✅ Created SQLite fallback table")
                else:
                    print(f"✅ SQLite fallback table exists")
                
                self.initialized = True
        except Exception as e:
            print(f"❌ Failed to initialize SQLite fallback: {e}")
            self.initialized = False
    
    def save_message(self, message: dict):
        """Save message to SQLite for fallback"""
        if not self.initialized:
            return
        
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO messages (msg_id, room_id, role, content, timestamp, files)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    message['msg_id'],
                    message['room_id'], 
                    message['role'],
                    message['content'],
                    message['timestamp'],
                    json.dumps(message.get('files', []))
                ))
        except Exception as e:
            print(f"⚠️ Failed to save message to SQLite fallback: {e}")
    
    def get_recent_messages_fallback(self, room_id: str, k: int = 50) -> List[dict]:
        """SQL fallback when buffer is cold"""
        if not self.initialized:
            return []
        
        try:
            import json
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                cursor = conn.execute("""
                    SELECT msg_id, room_id, role, content, timestamp, files FROM messages 
                    WHERE room_id = ? 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                """, (room_id, k))
                
                messages = []
                for row in cursor.fetchall():
                    message = dict(row)
                    message['files'] = json.loads(message['files']) if message['files'] else []
                    messages.append(message)
                return messages
        except Exception as e:
            print(f"⚠️ Failed to get messages from SQLite fallback: {e}")
            return []
    
    def delete_room_messages(self, room_id: str):
        """Delete all messages for a specific room from SQLite"""
        if not self.initialized:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM messages WHERE room_id = ?", (room_id,))
                deleted_count = cursor.rowcount
                conn.commit()
                print(f"🗑️ Deleted {deleted_count} messages from SQLite fallback for room {room_id[:8]}")
        except Exception as e:
            print(f"⚠️ Failed to delete room messages from SQLite: {e}")


def time_decay(ts: str, half_life_days: float = 3.0) -> float:
    """Exponential time decay with configurable half-life"""
    try:
        message_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        days_diff = (now - message_time).total_seconds() / 86400  # Convert to days
        return math.exp(-days_diff * math.log(2) / half_life_days)
    except:
        return 0.1  # Low score for invalid timestamps


class AsyncChatService:
    """Async chat service with Qdrant integration"""
    
    CHAT_COLLECTION = os.getenv("QDRANT_CHAT_DB", "chat_messages")
    ROOMS_COLLECTION = os.getenv("QDRANT_ROOMS_DB", "chat_rooms")
    VECTOR_SIZE = 768
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.buffer = ChatBuffer()
        self.fallback = ChatFallback()
        self.rooms = {}  # In-memory room metadata cache
        
        # Initialize Qdrant client with error handling
        try:
            self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            print(f"✅ Qdrant client connected")
            
            # Ensure all collections exist
            self._ensure_collections()
            self.ensure_document_collection()
            
            print(f"✅ Qdrant chat service initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Qdrant client: {e}")
            # Create a dummy client that will fail gracefully
            self.client = None
    
    def _ensure_collections(self):
        """Ensure chat and rooms collections exist in Qdrant with proper indexes"""
        if not self.client:
            return
            
        try:
            # Get existing collections
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            print(f"📋 Existing collections: {collection_names}")
            
            # Ensure chat messages collection
            if self.CHAT_COLLECTION not in collection_names:
                print(f"🆕 Creating chat collection: {self.CHAT_COLLECTION}")
                self.client.create_collection(
                    collection_name=self.CHAT_COLLECTION,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created chat collection: {self.CHAT_COLLECTION}")
            else:
                print(f"✅ Chat collection exists: {self.CHAT_COLLECTION}")
            
            # Create payload indexes for chat collection
            try:
                # Create index for room_id field (required for filtering)
                self.client.create_payload_index(
                    collection_name=self.CHAT_COLLECTION,
                    field_name="room_id",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                print(f"✅ Created room_id index for chat collection")
            except Exception as e:
                print(f"⚠️ Room_id index may already exist: {e}")
            
            # Ensure rooms collection
            if self.ROOMS_COLLECTION not in collection_names:
                print(f"🆕 Creating rooms collection: {self.ROOMS_COLLECTION}")
                self.client.create_collection(
                    collection_name=self.ROOMS_COLLECTION,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created rooms collection: {self.ROOMS_COLLECTION}")
            else:
                print(f"✅ Rooms collection exists: {self.ROOMS_COLLECTION}")
            
            # Validate collections after creation
            self._validate_collections()
                
        except Exception as e:
            print(f"⚠️ Error ensuring collections: {e}")
    
    def _validate_collections(self):
        """Validate that collections exist and are properly configured"""
        if not self.client:
            print("❌ Qdrant client not available for validation")
            return False
        
        try:
            # Check if collections exist
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            validation_passed = True
            
            # Validate chat collection
            if self.CHAT_COLLECTION not in collection_names:
                print(f"❌ Chat collection '{self.CHAT_COLLECTION}' not found")
                validation_passed = False
            else:
                try:
                    chat_collection = self.client.get_collection(self.CHAT_COLLECTION)
                    print(f"✅ Chat collection validated: {chat_collection.points_count} points")
                except Exception as e:
                    print(f"❌ Error validating chat collection: {e}")
                    validation_passed = False
            
            # Validate rooms collection
            if self.ROOMS_COLLECTION not in collection_names:
                print(f"❌ Rooms collection '{self.ROOMS_COLLECTION}' not found")
                validation_passed = False
            else:
                try:
                    rooms_collection = self.client.get_collection(self.ROOMS_COLLECTION)
                    print(f"✅ Rooms collection validated: {rooms_collection.points_count} points")
                except Exception as e:
                    print(f"❌ Error validating rooms collection: {e}")
                    validation_passed = False
            
            # Validate main document collection (for RAG)
            try:
                from src.config import QDRANT_COLLECTION
                if QDRANT_COLLECTION not in collection_names:
                    print(f"⚠️ Main document collection '{QDRANT_COLLECTION}' not found")
                    print("💡 This collection is used for RAG document retrieval")
                else:
                    try:
                        doc_collection = self.client.get_collection(QDRANT_COLLECTION)
                        print(f"✅ Document collection validated: {doc_collection.points_count} points")
                    except Exception as e:
                        print(f"❌ Error validating document collection: {e}")
            except ImportError:
                print("⚠️ Could not import QDRANT_COLLECTION config")
            
            if validation_passed:
                print("✅ All collections validated successfully")
            else:
                print("❌ Collection validation failed")
            
            return validation_passed
            
        except Exception as e:
            print(f"❌ Error during collection validation: {e}")
            return False
    
    def ensure_document_collection(self):
        """Ensure the main document collection exists for RAG functionality"""
        if not self.client:
            print("❌ Qdrant client not available")
            return False
        
        try:
            from src.config import QDRANT_COLLECTION
            
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if QDRANT_COLLECTION not in collection_names:
                print(f"🆕 Creating document collection: {QDRANT_COLLECTION}")
                self.client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created document collection: {QDRANT_COLLECTION}")
                return True
            else:
                print(f"✅ Document collection exists: {QDRANT_COLLECTION}")
                return True
                
        except ImportError:
            print("⚠️ Could not import QDRANT_COLLECTION config")
            return False
        except Exception as e:
            print(f"❌ Error ensuring document collection: {e}")
            return False
    
    def recreate_collections_with_indexes(self):
        """Recreate collections with proper payload indexes (use if filtering fails)"""
        if not self.client:
            print("❌ Qdrant client not available")
            return False
        
        try:
            # Delete existing collections
            try:
                self.client.delete_collection(self.CHAT_COLLECTION)
                print(f"🗑️ Deleted existing chat collection: {self.CHAT_COLLECTION}")
            except:
                pass
            
            try:
                self.client.delete_collection(self.ROOMS_COLLECTION)
                print(f"🗑️ Deleted existing rooms collection: {self.ROOMS_COLLECTION}")
            except:
                pass
            
            # Recreate collections
            self.client.create_collection(
                collection_name=self.CHAT_COLLECTION,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            
            self.client.create_collection(
                collection_name=self.ROOMS_COLLECTION,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            
            # Create payload indexes
            self.client.create_payload_index(
                collection_name=self.CHAT_COLLECTION,
                field_name="room_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            print(f"✅ Recreated collections with proper indexes")
            return True
            
        except Exception as e:
            print(f"❌ Error recreating collections: {e}")
            return False
    
    async def create_room(self, first_message: str, user_id: str = None) -> dict:
        """Create a new chat room with auto-generated ID and title"""
        room_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Generate room title from first message
        room_title = await self._generate_room_title(first_message)
        
        room_data = {
            "room_id": room_id,
            "room_title": room_title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "created_by": user_id,
            "message_count": 0
        }
        
        # Cache room data
        self.rooms[room_id] = room_data
        
        # Store room in Qdrant if available
        if self.client:
            try:
                # Embed room title for semantic search
                loop = asyncio.get_event_loop()
                vector = await loop.run_in_executor(self.executor, embed, room_title)
                
                point = {
                    "id": room_id,
                    "vector": vector,
                    "payload": room_data
                }
                
                await loop.run_in_executor(
                    self.executor,
                    partial(self.client.upsert, collection_name=self.ROOMS_COLLECTION, points=[point])
                )
                print(f"✅ Created room {room_id[:8]} with title: {room_title}")
            except Exception as e:
                print(f"⚠️ Failed to store room in Qdrant: {e}")
        
        return room_data
    
    async def _generate_room_title(self, first_message: str) -> str:
        """Generate room title from first user message using LLM"""
        try:
            # Dynamically resolve project root and add to path
            current_file = os.path.abspath(__file__)
            project_root = os.path.abspath(os.path.join(current_file, "../../../.."))
            src_path = os.path.join(project_root, "src")

            if src_path not in sys.path:
                sys.path.insert(0, src_path)

            from src.models.client_factory import get_llm_client, get_default_model
            from src.models.LLM import LLM
            
            # Prepare prompt for title generation
            system_prompt = "You are a helpful assistant that creates concise chat room titles."
            user_prompt = f"""Create a short, descriptive title for a chat room based on this first message:

            "{first_message}"

            Requirements:
            - Use Traditional Chinese.
            - Capture the main topic or intent.
            - No quotes or special characters but wrap the title in quotes.
            

            Examples:
            - "詢問技術問題" (for technical questions)
            - "文件分析討論" (for document analysis)
            - "產品功能諮詢" (for product inquiries)

            Title:"""
            
            # Get LLM client and generate title
            llm_client = get_llm_client()
            default_model = get_default_model()
            
            options = {"temperature": 0.3, "max_tokens": 20}
            
            # Generate title using LLM
            loop = asyncio.get_event_loop()
            title = await loop.run_in_executor(
                self.executor,
                lambda: LLM(llm_client, default_model, system_prompt, user_prompt, options=options, raw=True)
            )
            
            # Clean up the title - extract only the actual response
            # Remove the prompt part if it's included in the response
            if "Title:" in title:
                # Extract everything after "Title:" and take the last line
                title_part = title.split("Title:")[-1].strip()
                # Split by lines and take the last non-empty line
                lines = [line.strip() for line in title_part.splitlines() if line.strip()]
                if lines:
                    title = lines[-1]  # Take the last line
                else:
                    title = title_part.strip()
            elif "assistant" in title.lower():
                # Extract after "assistant" marker
                assistant_start = title.lower().find("assistant")
                if assistant_start != -1:
                    # Find the next newline or colon after "assistant"
                    response_start = title.find("\n", assistant_start)
                    if response_start == -1:
                        response_start = title.find(":", assistant_start)
                    if response_start != -1:
                        title = title[response_start:].strip()
                    else:
                        # If no separator found, take everything after "assistant"
                        title = title[assistant_start + len("assistant"):].strip()
            
            # Take only the first line if multiple lines
            title = title.splitlines()[0].strip()
            
            # Remove quotes if present
            title = title.strip('"\'')
            
            # Fallback if title is too long or empty
            if not title or len(title) > 30:
                title = "新對話"
            return title
            
        except Exception as e:
            print(f"⚠️ Failed to generate room title: {e}")
            # Fallback title
            return "新對話"
    
    async def get_room(self, room_id: str) -> Optional[dict]:
        """Get room metadata"""
        # Check cache first
        if room_id in self.rooms:
            return self.rooms[room_id].copy()
        
        # Try to fetch from Qdrant
        if self.client:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.retrieve(collection_name=self.ROOMS_COLLECTION, ids=[room_id])
                )
                
                if result:
                    room_data = result[0].payload
                    self.rooms[room_id] = room_data  # Cache it
                    return room_data.copy()
            except Exception as e:
                print(f"⚠️ Failed to retrieve room from Qdrant: {e}")
        
        return None
    
    async def update_room_timestamp(self, room_id: str):
        """Update room's last updated timestamp"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if room_id in self.rooms:
            self.rooms[room_id]["updated_at"] = timestamp
            self.rooms[room_id]["message_count"] = self.rooms[room_id].get("message_count", 0) + 1
            
            # Update in Qdrant if available
            if self.client:
                try:
                    loop = asyncio.get_event_loop()
                    # Embed room title for update
                    room_title = self.rooms[room_id]["room_title"]
                    vector = await loop.run_in_executor(self.executor, embed, room_title)
                    
                    point = {
                        "id": room_id,
                        "vector": vector,
                        "payload": self.rooms[room_id]
                    }
                    
                    await loop.run_in_executor(
                        self.executor,
                        partial(self.client.upsert, collection_name=self.ROOMS_COLLECTION, points=[point])
                    )
                except Exception as e:
                    print(f"⚠️ Failed to update room in Qdrant: {e}")
    
    async def add_message(self, room_id: str, role: str, content: str, user_id: str = None, files: List[str] = None) -> str:
        """Add message to Qdrant and buffer"""
        
        # Generate message data
        msg_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            # Embed message content
            loop = asyncio.get_event_loop()
            vector = await loop.run_in_executor(self.executor, embed, content)
            
            # Prepare message payload with new structure
            payload = {
                "room_id": room_id,
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "files": files or [],
                "user_id": user_id  # Keep user_id in payload for internal use
            }
            
            # Prepare Qdrant point
            point = {
                "id": msg_id,
                "vector": vector,
                "payload": payload
            }
            
            # Async Qdrant insertion (only if client is available)
            if self.client:
                await loop.run_in_executor(
                    self.executor,
                    partial(self.client.upsert, collection_name=self.CHAT_COLLECTION, points=[point])
                )
                print(f"💾 Saved message {msg_id} to Qdrant")
            else:
                print(f"⚠️ Qdrant not available, message {msg_id} only saved to buffer")
            
            # Prepare message for buffer and response (new structure)
            message_for_response = {
                "msg_id": msg_id,
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "files": files or [],
                "room_id": room_id  # Keep room_id for internal operations
            }
            
            # Add to recency buffer
            self.buffer.add_message(room_id, message_for_response)
            
            # Save to fallback database
            await loop.run_in_executor(
                self.executor,
                self.fallback.save_message,
                message_for_response
            )
            
            # Update room timestamp
            await self.update_room_timestamp(room_id)
            
            return msg_id
            
        except Exception as e:
            print(f"❌ Error adding message: {e}")
            raise
    
    async def get_chat_history(self, room_id: str, recency_k: int = 50) -> List[dict]:
        """Get recent messages from buffer, fallback to SQL if empty"""
        
        # Try buffer first
        messages = self.buffer.get_recent_messages(room_id, recency_k)
        
        if not messages:
            # Fallback to SQL
            loop = asyncio.get_event_loop()
            messages = await loop.run_in_executor(
                self.executor,
                self.fallback.get_recent_messages_fallback,
                room_id, recency_k
            )
            
            # Warm up buffer with fallback data
            for msg in messages:
                self.buffer.add_message(room_id, msg)
        
        # Return messages without internal fields (room_id, user_id)
        clean_messages = []
        for msg in messages:
            clean_msg = {
                "msg_id": msg["msg_id"],
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "files": msg.get("files", [])
            }
            clean_messages.append(clean_msg)
        
        return clean_messages
    
    async def search_messages(
        self, 
        room_id: str, 
        query: str, 
        semantic_k: int = 20,
        time_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> List[dict]:
        """Semantic search with time decay"""
        
        if not self.client:
            print("⚠️ Qdrant not available, semantic search disabled")
            return []
        
        try:
            loop = asyncio.get_event_loop()
            
            # Generate query vector
            query_vector = await loop.run_in_executor(self.executor, embed, query)
            
            # Search in Qdrant
            def _search():
                return self.client.search(
                    collection_name=self.CHAT_COLLECTION,
                    query_vector=query_vector,
                    query_filter=Filter(
                        must=[FieldCondition(key="room_id", match=MatchValue(value=room_id))]
                    ),
                    limit=semantic_k
                )
            
            hits = await loop.run_in_executor(self.executor, _search)
            
            # Apply time decay and re-rank
            scored_results = []
            for hit in hits:
                time_score = time_decay(hit.payload['timestamp'])
                semantic_score = hit.score
                combined_score = (semantic_score * semantic_weight + 
                                time_score * time_weight)
                
                # Clean up the message format
                clean_message = {
                    "msg_id": hit.id,
                    "role": hit.payload['role'],
                    "content": hit.payload['content'],
                    "timestamp": hit.payload['timestamp'],
                    "files": hit.payload.get('files', [])
                }
                
                scored_results.append({
                    'message': clean_message,
                    'score': combined_score,
                    'semantic_score': semantic_score,
                    'time_score': time_score
                })
            
            # Sort by combined score
            return sorted(scored_results, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            print(f"❌ Error in semantic search: {e}")
            return []
    
    async def delete_room(self, room_id: str) -> bool:
        """Delete room and all its messages completely"""
        try:
            loop = asyncio.get_event_loop()
            
            if self.client:
                # Try to delete using filter first
                try:
                    def _delete_messages():
                        return self.client.delete(
                            collection_name=self.CHAT_COLLECTION,
                            points_selector=Filter(
                                must=[FieldCondition(key="room_id", match=MatchValue(value=room_id))]
                            )
                        )
                    
                    def _delete_room():
                        return self.client.delete(
                            collection_name=self.ROOMS_COLLECTION,
                            points_selector=[room_id]
                        )
                    
                    # Execute deletions
                    await loop.run_in_executor(self.executor, _delete_messages)
                    await loop.run_in_executor(self.executor, _delete_room)
                    
                except Exception as filter_error:
                    print(f"⚠️ Filter-based deletion failed: {filter_error}")
                    print("🔄 Trying fallback deletion method...")
                    
                    # Fallback: Scroll and delete by IDs
                    try:
                        def _scroll_and_delete_messages():
                            # Scroll to get all message IDs for this room
                            scroll_result = self.client.scroll(
                                collection_name=self.CHAT_COLLECTION,
                                scroll_filter=Filter(
                                    must=[FieldCondition(key="room_id", match=MatchValue(value=room_id))]
                                ),
                                limit=1000,  # Get all messages
                                with_payload=False,
                                with_vectors=False
                            )
                            
                            if scroll_result[0]:  # If we found messages
                                message_ids = [point.id for point in scroll_result[0]]
                                if message_ids:
                                    self.client.delete(
                                        collection_name=self.CHAT_COLLECTION,
                                        points_selector=message_ids
                                    )
                                    print(f"🗑️ Deleted {len(message_ids)} messages using fallback method")
                            
                            # Try to delete room metadata
                            try:
                                self.client.delete(
                                    collection_name=self.ROOMS_COLLECTION,
                                    points_selector=[room_id]
                                )
                            except:
                                pass  # Room might not exist
                        
                        await loop.run_in_executor(self.executor, _scroll_and_delete_messages)
                        
                    except Exception as fallback_error:
                        print(f"❌ Fallback deletion also failed: {fallback_error}")
                        print("💡 Consider recreating collections with proper indexes")
            
            # Clear from memory caches
            self.buffer.clear_room(room_id)
            if room_id in self.rooms:
                del self.rooms[room_id]
            
            # Clear from SQLite fallback
            await loop.run_in_executor(
                self.executor,
                self.fallback.delete_room_messages,
                room_id
            )
            
            print(f"🗑️ Deleted room {room_id[:8]} and all its messages")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting room: {e}")
            return False
    
    async def clear_room_history(self, room_id: str) -> bool:
        """Clear all messages for a room but keep room metadata"""
        try:
            loop = asyncio.get_event_loop()
            
            if self.client:
                # Clear messages from Qdrant (delete by filter)
                def _delete_messages():
                    return self.client.delete(
                        collection_name=self.CHAT_COLLECTION,
                        points_selector=Filter(
                            must=[FieldCondition(key="room_id", match=MatchValue(value=room_id))]
                        )
                    )
                
                await loop.run_in_executor(self.executor, _delete_messages)
            
            # Clear from buffer
            self.buffer.clear_room(room_id)
            
            # Clear from SQLite fallback
            await loop.run_in_executor(
                self.executor,
                self.fallback.delete_room_messages,
                room_id
            )
            
            # Reset room message count but keep metadata
            if room_id in self.rooms:
                self.rooms[room_id]["message_count"] = 0
                self.rooms[room_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            print(f"🗑️ Cleared history for room {room_id[:8]}")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing room history: {e}")
            return False
    
    async def list_active_rooms(self, limit: int = 50) -> List[dict]:
        """List active chat rooms ordered by last updated"""
        try:
            rooms = []
            
            # Get rooms from memory cache first
            cached_rooms = list(self.rooms.values())
            if cached_rooms:
                # Sort by updated_at descending
                cached_rooms.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
                rooms.extend(cached_rooms[:limit])
            
            # If we need more rooms or cache is empty, query Qdrant
            if len(rooms) < limit and self.client:
                try:
                    loop = asyncio.get_event_loop()
                    
                    def _search_rooms():
                        # Scroll through all rooms in Qdrant
                        return self.client.scroll(
                            collection_name=self.ROOMS_COLLECTION,
                            limit=limit * 2,  # Get more to filter and sort
                            with_payload=True,
                            with_vectors=False
                        )
                    
                    scroll_result = await loop.run_in_executor(self.executor, _search_rooms)
                    
                    # Extract room data from Qdrant points
                    qdrant_rooms = []
                    for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
                        room_data = point.payload
                        room_data["room_id"] = point.id
                        qdrant_rooms.append(room_data)
                    
                    # Merge with cached rooms and deduplicate
                    existing_ids = {room["room_id"] for room in rooms}
                    for room in qdrant_rooms:
                        if room["room_id"] not in existing_ids:
                            rooms.append(room)
                            # Cache the room data
                            self.rooms[room["room_id"]] = room
                
                except Exception as e:
                    print(f"⚠️ Error querying rooms from Qdrant: {e}")
            
            # Sort all rooms by updated_at descending and limit
            rooms.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            rooms = rooms[:limit]
            
            # Clean up room data for response (remove internal fields)
            clean_rooms = []
            for room in rooms:
                clean_room = {
                    "room_id": room["room_id"],
                    "room_title": room["room_title"],
                    "createdAt": room["created_at"],
                    "updatedAt": room["updated_at"],
                    "message_count": room.get("message_count", 0)
                }
                clean_rooms.append(clean_room)
            
            return clean_rooms
            
        except Exception as e:
            print(f"❌ Error listing rooms: {e}")
            return []

# Global service instance
chat_service = AsyncChatService() 