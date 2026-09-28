import pytest
from unittest.mock import MagicMock, patch
import redis

from app.core.memory import RedisMemoryStore
from langchain.memory import ConversationBufferMemory

@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing"""
    with patch('redis.Redis') as mock:
        mock_instance = mock.return_value
        mock_instance.ping.return_value = True
        mock_instance.llen.return_value = 10
        mock_instance.delete.return_value = True
        mock_instance.keys.return_value = [b"chat_history:session1", b"chat_history:session2"]
        yield mock_instance

@pytest.fixture
def memory_store(mock_redis_client):
    """Create a RedisMemoryStore with a mock Redis client"""
    with patch('redis.Redis', return_value=mock_redis_client):
        store = RedisMemoryStore()
        return store

@pytest.fixture
def mock_redis_chat_history():
    """Create a mock RedisChatMessageHistory for testing"""
    with patch('app.core.memory.RedisChatMessageHistory') as mock:
        mock_instance = mock.return_value
        yield mock_instance

@pytest.fixture
def mock_conversation_buffer_memory():
    """Create a mock ConversationBufferMemory for testing"""
    with patch('app.core.memory.ConversationBufferMemory') as mock:
        mock_instance = mock.return_value
        yield mock_instance

class TestRedisMemoryStore:
    """Test suite for the Redis memory store"""

    def test_init(self, mock_redis_client):
        """Test initialization of memory store"""
        # Initialize store with default settings
        store = RedisMemoryStore()

        # Verify Redis client initialization
        mock_redis_client.ping.assert_called_once()

    def test_init_connection_error(self):
        """Test handling of Redis connection errors"""
        # Set up Redis client to raise connection error
        with patch('redis.Redis') as mock:
            mock_instance = mock.return_value
            mock_instance.ping.side_effect = redis.ConnectionError("Test error")

            # Verify that the error is propagated
            with pytest.raises(redis.ConnectionError):
                RedisMemoryStore()

    def test_get_history(self, memory_store, mock_redis_chat_history, mock_conversation_buffer_memory):
        """Test retrieving conversation history"""
        with patch('app.core.memory.RedisChatMessageHistory', return_value=mock_redis_chat_history), \
             patch('app.core.memory.ConversationBufferMemory', return_value=mock_conversation_buffer_memory):

            # Get history for a session
            result = memory_store.get_history("test_session")

            # Verify the result and method calls
            assert result == mock_conversation_buffer_memory

    def test_clear_history(self, memory_store):
        """Test clearing conversation history"""
        # Clear history for a session
        memory_store.clear_history("test_session")

        # Verify the Redis client call
        memory_store.redis_client.delete.assert_called_once_with("chat_history:test_session")

    def test_get_all_sessions(self, memory_store):
        """Test retrieving all session IDs"""
        # Get all sessions
        sessions = memory_store.get_all_sessions()

        # Verify the Redis client call and result
        memory_store.redis_client.keys.assert_called_once_with("chat_history:*")
        assert sessions == ["session1", "session2"]

    def test_add_interaction(self, memory_store):
        """Test adding an interaction to history"""
        # Add an interaction
        memory_store.add_interaction("test_session", "Hello", "Hi there")

        # Verify the Redis client call
        memory_store.redis_client.llen.assert_called_once_with("chat_history:test_session")
