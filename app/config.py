import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")

# SerpApi Configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# TTS Configuration
TTS_SERVICE = os.getenv("TTS_SERVICE", "google")  # or "azure", "elevenlabs", etc.
TTS_API_KEY = os.getenv("TTS_API_KEY")

# STT Configuration
STT_SERVICE = os.getenv("STT_SERVICE", "google")  # or "azure", "whisper", etc.
STT_API_KEY = os.getenv("STT_API_KEY")

# RAG Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1536))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 5))
