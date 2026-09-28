from langchain.memory import ConversationBufferMemory
from langchain.memory import RedisChatMessageHistory
import redis
import logging
from typing import List, Dict, Any

from app.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB

logger = logging.getLogger(__name__)

class RedisMemoryStore:
    """Memory store implementation using Redis for persistent storage"""

    def __init__(self):
        """Initialize the Redis memory store"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                db=REDIS_DB,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def get_history(self, session_id: str) -> ConversationBufferMemory:
        """Get conversation history for a session

        Args:
            session_id: Session identifier

        Returns:
            ConversationBufferMemory with the session's history
        """
        # Create a Redis-backed chat message history
        message_history = RedisChatMessageHistory(
            session_id=session_id,
            url=f"redis://{':' + REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
            key_prefix="chat_history:"
        )

        # Create memory with the message history
        return ConversationBufferMemory(
            memory_key="history",
            chat_memory=message_history,
            return_messages=True
        )

    def add_interaction(self, session_id: str, user_message: str, ai_response: str) -> None:
        """Add a user-AI interaction to the history

        This is typically handled automatically by the RedisChatMessageHistory
        when used with the ConversationChain, but we keep this method
        for explicit memory operations if needed.

        Args:
            session_id: Session identifier
            user_message: User's message
            ai_response: AI's response
        """
        key = f"chat_history:{session_id}"

        # Store as messages in Redis
        history_length = self.redis_client.llen(key)
        logger.debug(f"Current history length for {session_id}: {history_length}")

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session

        Args:
            session_id: Session identifier
        """
        # Get the Redis key for this session
        key = f"chat_history:{session_id}"

        # Delete the key
        result = self.redis_client.delete(key)
        logger.info(f"Cleared history for session {session_id}: {result}")

    def get_all_sessions(self) -> List[str]:
        """Get all active session IDs

        Returns:
            List of session IDs
        """
        # Use pattern matching to find all session keys
        keys = self.redis_client.keys("chat_history:*")

        # Extract session IDs from keys
        session_ids = [key.split(":", 1)[1] for key in keys]

        return session_ids
