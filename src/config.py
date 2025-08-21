from dotenv import load_dotenv
import os

load_dotenv()  # reads .env

# Qdrant settings
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
QDRANT_QA_DB = os.getenv("QDRANT_QA_DB", "qa_suggestions")

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "production"
PRODUCTION_HOST = os.getenv("PRODUCTION_HOST", "http://localhost:8000")
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or os.getenv("PRODUCTION_OLLAMA")  # Prefer OLLAMA_HOST, fallback to legacy

# Model settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b-instruct-fp16")  # Default for Ollama (answering)
GRAPH_EXTRACT_MODEL = os.getenv("GRAPH_EXTRACT_MODEL", os.getenv("DEFAULT_GRAPH_EXTRACT_MODEL", "qwen2.5:7b-instruct-q4_K_M"))

# Embedding model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "dengcao/Qwen3-Embedding-4B:Q8_0")  # Default embedding model for Ollama

# OCR model settings
OCR_MODEL = os.getenv("OCR_MODEL", "benhaotang/Nanonets-OCR-s:latest")  # Default OCR model for table extraction

if LLM_PROVIDER == "production":
    # For vLLM, use the exact model name that vLLM serves (with forward slash and capital B)
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")  # Default for vLLM Production

# Pipeline settings
# Defaults are tuned for Chinese: ~800 tokens per chunk with 120-token overlap
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 120))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", 2000))

# Backward compatibility
OLLAMA_MODEL = DEFAULT_MODEL  # For existing code that uses OLLAMA_MODEL

# Graph / Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI")
# Support both NEO4J_USERNAME and legacy NEO4J_USER
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# Database name (Neo4j 4.x+ multi-DB). Prefer NEO4J_DATABASE; default 'neo4j'
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE") or "neo4j"
TRAINING_CORPUS_ID = os.getenv("TRAINING_CORPUS_ID", "default")

# RAG routing defaults
RAG_DEFAULT_MODE = os.getenv("RAG_DEFAULT_MODE", "vector")  # vector | graph | hybrid

# Graph traversal / retrieval knobs
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", 2))
GRAPH_EXPANSION_TOP_K = int(os.getenv("GRAPH_EXPANSION_TOP_K", 12))

# Graph extraction/performance knobs
# Chunking for graph extraction pipeline
GRAPH_SPLIT_CHUNK_SIZE = int(os.getenv("GRAPH_SPLIT_CHUNK_SIZE", 400))
GRAPH_SPLIT_OVERLAP = int(os.getenv("GRAPH_SPLIT_OVERLAP", 40))

# LLM extractor controls
GRAPH_EXTRACT_MAX_TRIPLETS = int(os.getenv("GRAPH_EXTRACT_MAX_TRIPLETS", 6))
GRAPH_EXTRACT_NUM_WORKERS = int(os.getenv("GRAPH_EXTRACT_NUM_WORKERS", 1))

# Whether to write (:Document)-[:HAS_CHUNK]->(:Chunk) to Neo4j.
# Default on for training pipeline to ensure Neo4j and Qdrant synchronization.
GRAPH_WRITE_CHUNKS_TO_NEO4J = bool(int(os.getenv("GRAPH_WRITE_CHUNKS_TO_NEO4J", "1")))

# Optional guidance types for DynamicLLMPathExtractor
# If env vars are not provided, use project defaults below.
_allowed_entities_raw = os.getenv("GRAPH_ALLOWED_ENTITY_TYPES", "")
_allowed_relations_raw = os.getenv("GRAPH_ALLOWED_RELATION_TYPES", "")

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
]

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