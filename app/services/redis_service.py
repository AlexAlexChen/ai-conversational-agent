import redis
from typing import Any, Dict, List, Optional, Union
import json
import logging

from app.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB

logger = logging.getLogger(__name__)

class RedisService:
    """Service for Redis operations and connection management"""

    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 password: Optional[str] = None,
                 db: Optional[int] = None):
        """Initialize the Redis service

        Args:
            host: Redis server host
            port: Redis server port
            password: Redis server password
            db: Redis database number
        """
        self.host = host or REDIS_HOST
        self.port = port or REDIS_PORT
        self.password = password or REDIS_PASSWORD
        self.db = db or REDIS_DB

        # Initialize the Redis client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Redis client"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True
            )

            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def get_client(self) -> redis.Redis:
        """Get the Redis client

        Returns:
            Redis client instance
        """
        return self.client

    def ping(self) -> bool:
        """Test connection to Redis

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {str(e)}")
            return False

    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """Set a key-value pair in Redis

        Args:
            key: Key to set
            value: Value to set (will be JSON-encoded if not a string)
            expiry: Optional expiry time in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            # JSON encode non-string values
            if not isinstance(value, str):
                value = json.dumps(value)

            # Set the value
            if expiry:
                return self.client.setex(key, expiry, value)
            else:
                return self.client.set(key, value)
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {str(e)}")
            return False

    def get(self, key: str, as_json: bool = False) -> Any:
        """Get a value from Redis

        Args:
            key: Key to get
            as_json: Whether to parse the value as JSON

        Returns:
            Value if found, None otherwise
        """
        try:
            value = self.client.get(key)

            if value and as_json:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON for key {key}")
                    return value

            return value
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {str(e)}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from Redis

        Args:
            key: Key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {str(e)}")
            return False

    def list_push(self, key: str, value: Any) -> bool:
        """Push a value to a Redis list

        Args:
            key: List key
            value: Value to push (will be JSON-encoded if not a string)

        Returns:
            True if successful, False otherwise
        """
        try:
            # JSON encode non-string values
            if not isinstance(value, str):
                value = json.dumps(value)

            # Push to the list
            return bool(self.client.rpush(key, value))
        except Exception as e:
            logger.error(f"Redis list push failed for key {key}: {str(e)}")
            return False

    def list_get(self, key: str, as_json: bool = False) -> List[Any]:
        """Get all values from a Redis list

        Args:
            key: List key
            as_json: Whether to parse values as JSON

        Returns:
            List of values
        """
        try:
            values = self.client.lrange(key, 0, -1)

            if as_json:
                result = []
                for value in values:
                    try:
                        result.append(json.loads(value))
                    except json.JSONDecodeError:
                        result.append(value)
                return result

            return values
        except Exception as e:
            logger.error(f"Redis list get failed for key {key}: {str(e)}")
            return []

# Dependency for FastAPI
def get_redis_service():
    """Get a Redis service instance for dependency injection"""
    return RedisService()
