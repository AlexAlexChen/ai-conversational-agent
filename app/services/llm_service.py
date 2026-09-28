from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from typing import Any, Dict, Optional
import logging

from app.config import LLM_MODEL, LLM_API_KEY

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with language models"""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the LLM service

        Args:
            model_name: Name of the language model to use
            api_key: API key for accessing the language model
        """
        self.model_name = model_name or LLM_MODEL
        self.api_key = api_key or LLM_API_KEY

        if not self.api_key:
            logger.warning("No API key provided for LLM service")

    def get_llm(self, temperature: float = 0.7, streaming: bool = False) -> ChatOpenAI:
        """Get a configured language model instance

        Args:
            temperature: Model temperature (0.0 to 1.0)
            streaming: Whether to stream responses

        Returns:
            Configured ChatOpenAI instance
        """
        logger.debug(f"Creating LLM instance with model: {self.model_name}")

        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.api_key,
            temperature=temperature,
            streaming=streaming,
            verbose=True
        )

    def get_completion_llm(self, temperature: float = 0.7) -> OpenAI:
        """Get a configured completion-oriented language model

        This is useful for simpler completion tasks rather than chat.

        Args:
            temperature: Model temperature (0.0 to 1.0)

        Returns:
            Configured OpenAI instance
        """
        logger.debug(f"Creating completion LLM instance with model: {self.model_name}")

        return OpenAI(
            model_name=self.model_name,
            openai_api_key=self.api_key,
            temperature=temperature,
            verbose=True
        )

    async def generate_response(self,
                          messages: list,
                          temperature: float = 0.7,
                          max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate a response using the OpenAI API directly

        This method uses the API directly rather than LangChain for more control.

        Args:
            messages: List of message dictionaries (role, content)
            temperature: Model temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Response from the language model
        """
        import openai

        openai.api_key = self.api_key

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

# Dependency for FastAPI
def get_llm_service():
    """Get an LLM service instance for dependency injection"""
    return LLMService()
