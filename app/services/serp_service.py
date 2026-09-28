from serpapi import GoogleSearch
from typing import Dict, List, Any, Optional
import logging

from app.config import SERPAPI_API_KEY

logger = logging.getLogger(__name__)

class SerpService:
    """Service for search engine results via SerpApi"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the SerpApi service

        Args:
            api_key: SerpApi API key
        """
        self.api_key = api_key or SERPAPI_API_KEY

        if not self.api_key:
            logger.warning("No API key provided for SerpApi service")

    def search(self,
               query: str,
               num_results: int = 5,
               result_type: str = "organic") -> List[Dict[str, Any]]:
        """Search the web using SerpApi

        Args:
            query: Search query string
            num_results: Maximum number of results to return
            result_type: Type of results to return (organic, news, etc.)

        Returns:
            List of dictionaries with search results
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        logger.info(f"Performing web search: {query}")

        try:
            # Set up search parameters
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": num_results * 2  # Request more results in case some are filtered out
            }

            # Execute search
            search = GoogleSearch(params)
            results = search.get_dict()

            # Process results based on type
            formatted_results = []

            if result_type == "organic" and "organic_results" in results:
                # Extract organic search results
                for result in results["organic_results"][:num_results]:
                    formatted_results.append({
                        "content": result.get("snippet", ""),
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "source": "web_search",
                        "position": result.get("position")
                    })

            elif result_type == "news" and "news_results" in results:
                # Extract news results
                for result in results.get("news_results", [])[:num_results]:
                    formatted_results.append({
                        "content": result.get("snippet", ""),
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "source": "news_search",
                        "published": result.get("date", ""),
                        "source_name": result.get("source", "")
                    })

            elif result_type == "all":
                # Include all types of results

                # Organic results
                for result in results.get("organic_results", [])[:num_results]:
                    formatted_results.append({
                        "content": result.get("snippet", ""),
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "source": "web_search",
                        "position": result.get("position")
                    })

                # News results
                for result in results.get("news_results", [])[:num_results]:
                    formatted_results.append({
                        "content": result.get("snippet", ""),
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "source": "news_search",
                        "published": result.get("date", ""),
                        "source_name": result.get("source", "")
                    })

            logger.info(f"Retrieved {len(formatted_results)} search results")

            return formatted_results[:num_results]  # Limit to requested number

        except Exception as e:
            logger.error(f"SerpApi search failed: {str(e)}")
            return []

    def get_knowledge_graph(self, query: str) -> Dict[str, Any]:
        """Get knowledge graph information for a query

        Args:
            query: Search query string

        Returns:
            Dictionary with knowledge graph information
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        logger.info(f"Retrieving knowledge graph for: {query}")

        try:
            # Set up search parameters
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key
            }

            # Execute search
            search = GoogleSearch(params)
            results = search.get_dict()

            # Extract knowledge graph if present
            if "knowledge_graph" in results:
                return results["knowledge_graph"]
            else:
                logger.info(f"No knowledge graph found for query: {query}")
                return {}

        except Exception as e:
            logger.error(f"SerpApi knowledge graph retrieval failed: {str(e)}")
            return {}

    def search_news(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search for news articles

        Args:
            query: Search query string
            num_results: Maximum number of results to return

        Returns:
            List of dictionaries with news results
        """
        # Use the general search method with news result type
        return self.search(query, num_results, "news")

    def get_answer_box(self, query: str) -> Dict[str, Any]:
        """Get featured snippet/answer box for a query

        Args:
            query: Search query string

        Returns:
            Dictionary with answer box information
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        logger.info(f"Retrieving answer box for: {query}")

        try:
            # Set up search parameters
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key
            }

            # Execute search
            search = GoogleSearch(params)
            results = search.get_dict()

            # Extract answer box if present
            if "answer_box" in results:
                return results["answer_box"]
            else:
                logger.info(f"No answer box found for query: {query}")
                return {}

        except Exception as e:
            logger.error(f"SerpApi answer box retrieval failed: {str(e)}")
            return {}

# Dependency for FastAPI
def get_serp_service():
    """Get a SerpApi service instance for dependency injection"""
    return SerpService()
