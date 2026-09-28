import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from app.core.agent import AIAgent
from app.services.llm_service import LLMService

@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service for testing"""
    service = MagicMock(spec=LLMService)
    mock_llm = AsyncMock()
    mock_llm.arun = AsyncMock(return_value="This is a test response")
    service.get_llm.return_value = mock_llm
    return service

@pytest.fixture
def mock_serp_service():
    """Create a mock SerpApi service for testing"""
    with patch('app.services.serp_service.SerpService') as mock:
        mock_instance = mock.return_value
        mock_instance.search.return_value = [
            {
                "title": "Test Result",
                "content": "This is a test search result content.",
                "url": "https://example.com/test",
                "source": "web_search"
            }
        ]
        yield mock_instance

@pytest.fixture
def mock_qdrant_service():
    """Create a mock Qdrant service for testing"""
    with patch('app.services.qdrant_service.QdrantService') as mock:
        mock_instance = mock.return_value
        mock_instance.search.return_value = [
            {
                "title": "Knowledge Base Entry",
                "content": "This is a test entry from the knowledge base.",
                "score": 0.85
            }
        ]
        yield mock_instance

@pytest.fixture
def mock_memory_store():
    """Create a mock Redis memory store for testing"""
    with patch('app.core.memory.RedisMemoryStore') as mock:
        mock_instance = mock.return_value
        mock_instance.get_history.return_value = MagicMock()
        yield mock_instance

@pytest.fixture
def agent(mock_llm_service, mock_serp_service, mock_qdrant_service, mock_memory_store):
    """Create an AI agent with mock dependencies for testing"""
    with patch('app.core.agent.SerpService', return_value=mock_serp_service), \
         patch('app.core.agent.QdrantService', return_value=mock_qdrant_service), \
         patch('app.core.agent.RedisMemoryStore', return_value=mock_memory_store):
        agent = AIAgent(mock_llm_service)
        return agent

class TestAIAgent:
    """Test suite for the AI Agent core functionality"""

    @pytest.mark.asyncio
    async def test_process_message(self, agent):
        """Test basic message processing"""
        # Process a test message
        response, sources = await agent.process_message("What is artificial intelligence?", "test_session")

        # Verify the response
        assert response == "This is a test response"
        assert len(sources) > 0

        # Verify service calls
        agent.memory_store.get_history.assert_called_once_with("test_session")
        agent.qdrant_service.search.assert_called_once()
        agent.serp_service.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_no_web_search(self, agent):
        """Test message processing without web search"""
        # Mock _needs_web_search to return False
        agent._needs_web_search = MagicMock(return_value=False)

        # Process a test message
        response, sources = await agent.process_message("Hello there", "test_session")

        # Verify the response
        assert response == "This is a test response"

        # Verify service calls
        agent.memory_store.get_history.assert_called_once_with("test_session")
        agent.qdrant_service.search.assert_called_once()
        agent.serp_service.search.assert_not_called()

    def test_needs_web_search(self, agent):
        """Test the logic for determining if web search is needed"""
        # Questions should trigger web search
        assert agent._needs_web_search("What is the weather today?")
        assert agent._needs_web_search("How does a neural network work?")
        assert agent._needs_web_search("Tell me about recent AI developments")

        # Simple greetings should not trigger web search
        assert not agent._needs_web_search("Hello")
        assert not agent._needs_web_search("How are you?")

    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling during message processing"""
        # Force an error in the LLM
        agent.llm.arun.side_effect = Exception("Test error")

        # Process a message that should raise an exception
        with pytest.raises(Exception):
            await agent.process_message("Test error message", "test_session")

    @pytest.mark.asyncio
    async def test_memory_interaction(self, agent):
        """Test interaction with memory store"""
        # Process a message
        await agent.process_message("Test memory", "test_session")

        # Verify memory interactions
        agent.memory_store.get_history.assert_called_once_with("test_session")
        agent.memory_store.add_interaction.assert_called_once()

    def test_reset(self, agent):
        """Test resetting agent state"""
        # Reset the agent
        agent.reset("test_session")

        # Verify memory was cleared
        agent.memory_store.clear_history.assert_called_once_with("test_session")
