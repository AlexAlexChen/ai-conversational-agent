# AI Conversational Agent

A powerful, stateful AI agent capable of natural language interaction in both text and voice forms. This system leverages LangChain with FastAPI to create an intelligent assistant with persistent memory, vector-based retrieval, and web search capabilities.

## Features

- **Natural Language Processing**: Supports both text and voice-based interactions
- **Persistent Memory**: Utilizes Redis to maintain context across user sessions
- **Retrieval-Augmented Generation (RAG)**: Enhances responses by retrieving relevant information from a knowledge base
- **Vector Storage**: Uses Qdrant to store and query high-dimensional embeddings for efficient semantic search
- **Web Search Integration**: Leverages SerpAPI to provide up-to-date information from the web
- **Voice Capabilities**: Includes text-to-speech (TTS) and speech-to-text (STT) functionality

## Architecture

The application is structured as follows:

```
project_root/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration settings
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── chat.py             # Text chat endpoints
│   │   └── voice.py            # Voice interaction endpoints
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py            # Main AI agent logic
│   │   ├── memory.py           # Redis memory implementation
│   │   ├── tts.py              # Text-to-speech service
│   │   └── stt.py              # Speech-to-text service
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py      # LLM integration
│   │   ├── redis_service.py    # Redis connection and operations
│   │   ├── qdrant_service.py   # Vector store operations
│   │   └── serp_service.py     # SerpApi integration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py       # Text to vector conversion utilities
│       ├── ranking.py          # Filtering and ranking of search results
│       └── response.py         # Response formatting utilities
│
├── data/
│   ├── knowledge_base/         # Offline data for RAG
│   ├── vectors/                # Storage for pre-computed vectors
│   └── conversation_logs/      # Optional storage for conversation history
│
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_memory.py
│   ├── test_rag.py
│   └── test_api.py
│
├── scripts/
│   ├── seed_knowledge.py       # Scripts to populate knowledge base
│   ├── generate_embeddings.py  # Pre-compute embeddings
│   └── benchmark.py            # Performance testing
│
├── .env                        # Environment variables (API keys, etc.)
├── .env.example                # Example environment variables
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker services (Redis, Qdrant)
├── Dockerfile                  # Application containerization
└── README.md                   # This file
```

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- API keys for:
  - Language Model provider
  - SerpAPI
  - TTS service (if applicable)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-conversational-agent.git
   cd ai-conversational-agent
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file to include your API keys and configuration.

3. Start the required services with Docker Compose:
   ```bash
   docker-compose up -d
   ```
   This will start Redis and Qdrant services.

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Seed the knowledge base and generate embeddings:
   ```bash
   python scripts/seed_knowledge.py
   python scripts/generate_embeddings.py
   ```

## Usage

### Starting the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### API Endpoints

- **Text Chat**: `POST /api/chat`
  ```json
  {
    "message": "What can you tell me about AI?",
    "session_id": "user123"
  }
  ```

- **Voice Interaction**: `POST /api/voice`
  - Accepts audio file upload
  - Returns both text response and audio response

### Interactive Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Key Components

### AI Agent (core/agent.py)
The central component that orchestrates all interactions, managing the conversation flow and coordinating between different services.

### Memory System (core/memory.py)
Redis-based memory implementation that stores conversation history, allowing the agent to maintain context across multiple interactions.

### Retrieval-Augmented Generation (services/qdrant_service.py)
- Converts text into high-dimensional vectors using pre-trained models
- Stores these embeddings in Qdrant vector database
- Retrieves relevant context to enhance the language model's responses

### Web Search Integration (services/serp_service.py)
Uses SerpAPI to augment the agent's knowledge with up-to-date information from the web, applying filtering and ranking to prioritize the most relevant results.

### Voice Capabilities (core/tts.py, core/stt.py)
Implements text-to-speech and speech-to-text functionality for voice-based interactions.

## Configuration

Key configuration options in `app/config.py`:

- LLM settings (model, temperature, etc.)
- Redis connection details
- Qdrant connection details
- RAG parameters (number of results, similarity threshold)
- SerpAPI settings
- Voice service configurations

## Testing

Run the test suite with:

```bash
pytest
```

## Performance Benchmarking

To evaluate system performance:

```bash
python scripts/benchmark.py
```

## License

[MIT License](LICENSE)

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Qdrant](https://qdrant.tech/)
- [Redis](https://redis.io/)
- [SerpAPI](https://serpapi.com/)
