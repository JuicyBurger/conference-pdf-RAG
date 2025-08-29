"""
Configuration module for document-analyzer.

This module provides all configuration settings used throughout the application.
Settings are organized into logical groups with clear section headers.

Only essential credentials and connection settings are loaded from environment variables.
Hyperparameters and internal settings are defined directly in this file for easier maintenance.
"""

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()  # reads .env

# =============================================================================
# CREDENTIALS - Loaded from environment variables
# =============================================================================

# Qdrant credentials and collections
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
QDRANT_CHAT_DB = os.getenv("QDRANT_CHAT_DB", "chat_history")
QDRANT_ROOMS_DB = os.getenv("QDRANT_ROOMS_DB", "chat_rooms")
QDRANT_QA_DB = os.getenv("QDRANT_QA_DB", "qa_suggestions")

# Neo4j credentials and database
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# LLM provider and model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "dengcao/Qwen3-Embedding-4B:Q8_0")
GRAPH_EXTRACT_MODEL = os.getenv("GRAPH_EXTRACT_MODEL", "qwen2.5:7b-instruct-q4_K_M")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")  # Default for Ollama (answering)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)  # For backward compatibility

# Machine-specific endpoints
M416_3090 = os.getenv("M416_3090")
M416_3090ti = os.getenv("M416_3090ti")
M416_4090 = os.getenv("M416_4090")
OLLAMA_HOST = os.getenv("PRODUCTION_OLLAMA")

# Training corpus ID
TRAINING_CORPUS_ID = "sinon"

# =============================================================================
# DOCUMENT CHUNKING PARAMETERS - Defined directly in code
# =============================================================================

# Pipeline settings - Optimized chunking for retrieval
# Smaller chunks improve recall and avoid overly long nodes in Qdrant
# Tune for finer splitting of long paragraphs
CHUNK_OVERLAP = 80
CHUNK_MAX_CHARS = 300

# Graph extraction chunking parameters
GRAPH_SPLIT_CHUNK_SIZE = 800
GRAPH_SPLIT_OVERLAP = 60

# =============================================================================
# FILE PATHS AND DIRECTORIES - Defined directly in code
# =============================================================================

# Prepared data directory for caching full-document extraction
PREPARED_DIR = "data/prepared"

# =============================================================================
# RAG (RETRIEVAL-AUGMENTED GENERATION) SETTINGS - Defined directly in code
# =============================================================================

# RAG routing defaults
RAG_DEFAULT_MODE = "hybrid"  # vector | graph | hybrid
RAG_DISABLE_QUERY_REWRITER = True

# Graph traversal / retrieval parameters
GRAPH_MAX_HOPS = 3
GRAPH_EXPANSION_TOP_K = 6

# Batch processing sizes
INDEX_BATCH_SIZE = 100
GRAPH_BATCH_SIZE = 100

# =============================================================================
# LLM GENERATION PARAMETERS - Defined directly in code
# =============================================================================

# Response generation parameters
RESPONSE_MAX_TOKENS = 512
RESPONSE_TEMPERATURE = 0.0
DEFAULT_REASONING_EFFORT = "low"  # "low" | "medium" | "high" | ""

# Chat title generation toggle
CHAT_USE_LLM_ROOM_TITLE = True

# =============================================================================
# KNOWLEDGE GRAPH EXTRACTION CONFIGURATION - Defined directly in code
# =============================================================================

# LLM extractor controls
GRAPH_EXTRACT_MAX_TRIPLETS = 6
GRAPH_EXTRACT_NUM_WORKERS = 2

# Whether to write chunks to Neo4j
GRAPH_WRITE_CHUNKS_TO_NEO4J = True

# Skip KG extraction during training ingestion (embedding-only runs)
SKIP_GRAPH_EXTRACTION = False

# 1: disable query relevance signal, 0: enable query relevance signal
DISABLE_QUERY_SIGNAL = 1

# Entity and relation type definitions
# Default entity types
DEFAULT_GRAPH_ALLOWED_ENTITY_TYPES = [
    # orgs & people
    "Company",            # 上市公司/關係企業/供應商/客戶(若已知)
    "Subsidiary",         # 子公司/轉投資
    "BusinessUnit",       # 事業部/部門/品牌線
    "Committee",          # 審計/薪酬/提名/永續等委員會
    "Person",             # 董事/經理人/會計師等個人
    "Role",               # 董事長/總經理/協理/發言人/會計師等職稱
    "Auditor",            # 會計師事務所/簽證會計師
    "Security",           # 股票/存託憑證等有價證券(含代號)
    # operations & footprint
    "Facility",           # 工廠/辦公室/賣場
    "Location",           # 國家/城市/市場/區域
    "Product",            # 產品/藥劑/肥料/品牌
    "ProductCategory",    # 產品分類(植物保護/塑膠/民生用品等)
    # events & rules
    "Event",              # 法說會/法人說明會/董事會/股東會/公告
    "Regulation",         # 法規/準則/政策(如IFRS/金管會規範)
    # reporting primitives
    "FinancialMetric",    # 指標(營收/毛利/淨利/EPS/現金流/ROE…)
    "AccountingItem",     # 報表科目(資產/負債/權益/合約負債…)
    "TimePeriod",         # 期間(Q1/2024、月/年/起迄日)
    "Currency",           # 幣別(TWD/USD…)
    "Value",              # 數值節點(金額/數量)
    "Percentage",         # 百分比節點(比重/毛利率…)
]

# Default relation types
DEFAULT_GRAPH_ALLOWED_RELATION_TYPES = [
    # structure & governance
    "SUBSIDIARY_OF",          # Subsidiary -> Company
    "BUSINESS_UNIT_OF",       # BusinessUnit -> Company
    "BRAND_OF",               # Product/Brand -> Company
    "AUDITED_BY",             # Company -> Auditor
    "HOLDS_ROLE",             # Person -> Role
    "ROLE_AT",                # Role -> Company/Committee
    "MEMBER_OF",              # Person -> Committee
    "LISTED_AS",              # Company -> Security (stock/ticker)
    # footprint & offerings
    "OPERATES_IN",            # Company/BusinessUnit -> Location
    "LOCATED_IN",             # Facility -> Location
    "HAS_FACILITY",           # Company -> Facility
    "PRODUCES",               # Company/BusinessUnit -> Product
    "IN_CATEGORY",            # Product -> ProductCategory
    # events
    "HELD_EVENT",             # Company -> Event
    "TOOK_PLACE_ON",          # Event -> TimePeriod
    # reporting
    "REPORTED",               # Company -> FinancialMetric
    "HAS_ACCOUNT_ITEM",       # FinancialStatement/Company -> AccountingItem
    "FOR_PERIOD",             # FinancialMetric/AccountingItem -> TimePeriod
    "HAS_VALUE",              # FinancialMetric/AccountingItem -> Value
    "DENOMINATED_IN",         # FinancialMetric/AccountingItem/Value -> Currency
    "HAS_PERCENTAGE",         # FinancialMetric/AccountingItem -> Percentage
    # policy & compliance
    "COMPLIES_WITH",          # Company -> Regulation/Standard
    "SUBJECT_TO",             # BusinessUnit/Product -> Regulation/Policy
    # relationships (keep sparing)
    "PARTNERS_WITH",          # Company -> Company/Institution
    # structural relationships (for graph traversal)
    "HAS_CHUNK",              # Document -> Chunk (structural)
    "HAS_DOCUMENT",           # Collection -> Document (structural)
    "HAS_ENTITY",             # Document -> Entity (structural, minimal wiring)
    # tables & observations
    "IN_TABLE",               # Observation -> Table
    "HAS_OBSERVATION",        # Chunk -> Observation (optional wiring)
]

# Parse from environment or use defaults - allow override via env if needed
_allowed_entities_raw = os.getenv("GRAPH_ALLOWED_ENTITY_TYPES", "")
_allowed_relations_raw = os.getenv("GRAPH_ALLOWED_RELATION_TYPES", "")

GRAPH_ALLOWED_ENTITY_TYPES = (
    [s.strip() for s in _allowed_entities_raw.split(",") if s.strip()]
    if _allowed_entities_raw.strip()
    else DEFAULT_GRAPH_ALLOWED_ENTITY_TYPES
)

GRAPH_ALLOWED_RELATION_TYPES = (
    [s.strip() for s in _allowed_relations_raw.split(",") if s.strip()]
    if _allowed_relations_raw.strip()
    else DEFAULT_GRAPH_ALLOWED_RELATION_TYPES
)

# =============================================================================
# TABLE EXTRACTION AND PROCESSING - Defined directly in code
# =============================================================================

# Table HTML chunking toggle
USE_TABLE_HTML_CHUNKS = True

# Retain saved table HTML files after ingestion
RETAIN_TABLE_HTML_FILES = True

# =============================================================================
# EXTERNAL API ENDPOINTS - Defined directly in code
# =============================================================================

# Remote OCR API for table extraction
OCR_API_URL = "http://192.168.100.32:9010/ocr/file"
OCR_CLEANUP_URL = ""

# LLM cleanup endpoint
LLM_CLEANUP_URL = ""