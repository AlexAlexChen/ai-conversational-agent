from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from typing import List, Dict, Tuple, Any
import logging

from app.core.memory import RedisMemoryStore
from app.services.serp_service import SerpService
from app.services.qdrant_service import QdrantService
from app.utils.embeddings import get_embedding
from app.utils.ranking import rank_results, filter_results
from app.utils.response import format_response

logger = logging.getLogger(__name__)

class AIAgent:
    """Main AI agent that orchestrates the conversation flow and information retrieval"""

    def __init__(self, llm_service):
        """Initialize the AI agent with necessary services

        Args:
            llm_service: Service providing access to the language model
        """
        self.llm = llm_service.get_llm()
        self.memory_store = RedisMemoryStore()
        self.serp_service = SerpService()
        self.qdrant_service = QdrantService()

        # Define the prompt template for the LLM
        self.prompt_template = PromptTemplate.from_template(
            '''You are a humanized AI assistant capable of natural conversation in both text and voice forms.

            Context from previous conversations:
            {history}

            Relevant information from knowledge base and search results:
            {context}

            Human: {input}
            AI: '''
        )

    async def process_message(self, message: str, session_id: str) -> Tuple[str, List[Dict[str, str]]]:
        """Process a user message and generate a response

        Args:
            message: User's message
            session_id: Session ID for tracking conversation history

        Returns:
            Tuple containing the response text and list of sources
        """
        try:
            # Retrieve conversation history
            history = self.memory_store.get_history(session_id)

            # Generate embedding for the message
            embedding = get_embedding(message)

            # Retrieve relevant context from vector store (RAG)
            rag_results = self.qdrant_service.search(embedding)

            # Search the web if needed
            web_results = []
            if self._needs_web_search(message):
                logger.info(f"Performing web search for query: {message}")
                web_results = self.serp_service.search(message)

            # Combine and rank all results
            all_results = rag_results + web_results
            ranked_results = rank_results(all_results, embedding)
            filtered_results = filter_results(ranked_results)

            # Build context string from top results
            context = "\n\n".join([
                f"Source: {r.get('title', 'Unknown')}\n{r.get('content', '')}"
                for r in filtered_results[:5]
            ])

            # Create the chain
            chain = ConversationChain(
                llm=self.llm,
                prompt=self.prompt_template,
                memory=history
            )

            # Run the chain to generate response
            response = await chain.arun(input=message, context=context)

            # Extract source information
            sources = [
                {
                    "title": r.get("title", "Unknown"),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:150] + "..." if len(r.get("content", "")) > 150 else r.get("content", "")
                }
                for r in filtered_results[:5] if "url" in r
            ]

            # Store the interaction in memory
            self.memory_store.add_interaction(session_id, message, response)

            return response, sources

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise

    def _needs_web_search(self, message: str) -> bool:
        """Determine if web search is needed for this message

        Args:
            message: User's message

        Returns:
            Boolean indicating if web search should be performed
        """
        # Simple heuristic: check for question words, search-related phrases,
        # or words indicating current events/recent information
        query_indicators = [
            "what", "how", "who", "when", "where", "why",
            "tell me about", "find", "search", "lookup",
            "latest", "recent", "news", "current", "today",
            "yesterday", "last week", "this month"
        ]

        message_lower = message.lower()

        return any(indicator in message_lower for indicator in query_indicators)

    def reset(self, session_id: str) -> None:
        """Reset the agent's state for a session

        Args:
            session_id: Session ID to reset
        """
        self.memory_store.clear_history(session_id)
