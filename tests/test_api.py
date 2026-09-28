import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json

from app.main import app
from app.core.agent import AIAgent
from app.services.llm_service import LLMService
from app.core.tts import TextToSpeech
from app.core.stt import SpeechToText

@pytest.fixture
def test_client():
    """Create a FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def mock_ai_agent():
    """Create a mock AI agent for testing"""
    with patch('app.core.agent.AIAgent') as mock:
        mock_instance = mock.return_value
        mock_instance.process_message = AsyncMock(return_value=("Test response", [{"title": "Source", "url": "http://example.com"}]))
        mock_instance.memory_store = MagicMock()
        mock_instance.memory_store.clear_history = MagicMock()
        yield mock_instance

@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service for testing"""
    with patch('app.services.llm_service.get_llm_service') as mock:
        mock.return_value = MagicMock(spec=LLMService)
        yield mock.return_value

@pytest.fixture
def mock_tts():
    """Create a mock TTS service for testing"""
    with patch('app.core.tts.TextToSpeech') as mock:
        mock_instance = mock.return_value
        mock_instance.synthesize = MagicMock(return_value=b"Test audio data")
        yield mock_instance

@pytest.fixture
def mock_stt():
    """Create a mock STT service for testing"""
    with patch('app.core.stt.SpeechToText') as mock:
        mock_instance = mock.return_value
        mock_instance.transcribe = MagicMock(return_value="Test transcription")
        yield mock_instance

class TestChatEndpoints:
    """Test suite for chat endpoints"""

    @patch('app.routes.chat.AIAgent')
    async def test_chat_endpoint(self, mock_agent_class, test_client, mock_ai_agent, mock_llm_service):
        """Test the chat endpoint"""
        mock_agent_class.return_value = mock_ai_agent

        # Test request data
        request_data = {
            "message": "Hello, AI",
            "session_id": "test_session"
        }

        # Send the request
        response = test_client.post("/chat/", json=request_data)

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert len(data["sources"]) == 1
        assert data["session_id"] == "test_session"

        # Verify the agent call
        mock_ai_agent.process_message.assert_called_once_with(
            message="Hello, AI",
            session_id="test_session"
        )

    @patch('app.routes.chat.AIAgent')
    async def test_clear_chat_history_endpoint(self, mock_agent_class, test_client, mock_ai_agent):
        """Test the clear chat history endpoint"""
        mock_agent_class.return_value = mock_ai_agent

        # Send the request
        response = test_client.delete("/chat/test_session")

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

        # Verify the agent call
        mock_ai_agent.memory_store.clear_history.assert_called_once_with("test_session")

class TestVoiceEndpoints:
    """Test suite for voice endpoints"""

    @patch('app.routes.voice.AIAgent')
    @patch('app.routes.voice.SpeechToText')
    async def test_process_voice_endpoint(self, mock_stt_class, mock_agent_class, test_client, mock_ai_agent, mock_stt):
        """Test the process voice endpoint"""
        mock_agent_class.return_value = mock_ai_agent
        mock_stt_class.return_value = mock_stt

        # Create a test audio file
        test_audio = b"Test audio data"

        # Send the request
        response = test_client.post(
            "/voice/",
            files={"audio": ("test.mp3", test_audio, "audio/mp3")},
            data={"session_id": "test_session"}
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Test response"
        assert len(data["sources"]) == 1
        assert data["session_id"] == "test_session"

        # Verify the service calls
        mock_stt.transcribe.assert_called_once()
        mock_ai_agent.process_message.assert_called_once_with(
            message="Test transcription",
            session_id="test_session"
        )

    @patch('app.routes.voice.TextToSpeech')
    async def test_text_to_speech_endpoint(self, mock_tts_class, test_client, mock_tts):
        """Test the text to speech endpoint"""
        mock_tts_class.return_value = mock_tts

        # Test request data
        request_data = {
            "text": "Hello, this is a test",
        }

        # Send the request
        response = test_client.post("/voice/tts", data=request_data)

        # Verify the response
        assert response.status_code == 200
        assert response.content == b"Test audio data"

        # Verify the service calls
        mock_tts.synthesize.assert_called_once_with(
            "Hello, this is a test",
            "default"
        )

    @patch('app.routes.voice.SpeechToText')
    async def test_speech_to_text_endpoint(self, mock_stt_class, test_client, mock_stt):
        """Test the speech to text endpoint"""
        mock_stt_class.return_value = mock_stt

        # Create a test audio file
        test_audio = b"Test audio data"

        # Send the request
        response = test_client.post(
            "/voice/stt",
            files={"audio": ("test.mp3", test_audio, "audio/mp3")}
        )

        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Test transcription"

        # Verify the service calls
        mock_stt.transcribe.assert_called_once()
